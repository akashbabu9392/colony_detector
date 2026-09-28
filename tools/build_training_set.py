"""Build a tiled detection dataset (YOLO and/or COCO) for colony detectors.

Colonies are small, so detectors are trained on native-resolution tiles of
the plate, exactly as colony_detector.engines.tiled runs them at inference.

Label sources (combine freely):

* ``--dataset DIR``: a labelled dataset in COCO, YOLO (Roboflow export),
  AGAR or hand-count format, auto-detected (colony_detector/datasets.py).
  Its own train/valid split is kept; its test split is left out.
* ``--gold eval/gold --images data/plates``: hand counts from
  tools/hand_count.py (points + radius -> boxes).
* ``--yolo-labels DIR --images data/plates``: existing YOLO txt labels for
  whole-plate images (class 0 = colony, 1 = fuzzy_colony).
* ``--synthetic N``: generated plates with exact boxes (pre-training).
* ``--pseudo --images DIR``: label unlabelled plates with the current
  pipeline (review these with hand_count.py before trusting them).

Example::

    python tools/build_training_set.py --images data/plates --gold eval/gold \\
        --synthetic 400 --out datasets/tiles --coco
    python tools/train_yolo.py --data datasets/tiles/data.yaml

Plates are split into train/val *by plate* (never by tile) so validation
tiles never come from a plate seen in training.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from colony_detector.engines.tiled import make_tiles  # noqa: E402
from colony_detector.imageio import read_image  # noqa: E402
from colony_detector.plate import find_plate  # noqa: E402
from colony_detector.synth import make_plate  # noqa: E402

NAMES = ["colony", "fuzzy_colony"]
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def boxes_from_points(points, default_r: float = 6.0):
    out = []
    for p in points:
        x, y = p[0], p[1]
        r = p[2] if len(p) > 2 and p[2] else default_r
        cls = 1 if (len(p) > 3 and p[3]) else 0
        out.append((x - r, y - r, x + r, y + r, cls))
    return out


def boxes_from_yolo(txt: Path, w: int, h: int):
    out = []
    for line in txt.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        c, xc, yc, bw, bh = int(parts[0]), *map(float, parts[1:5])
        out.append(((xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h, c))
    return out


def collect(args):
    """Yields (name, image, boxes, split or None)."""
    if args.dataset:
        from colony_detector.datasets import load_dataset

        for smp in load_dataset(args.dataset):
            if smp.boxes is None:
                continue  # count-only images cannot train a detector
            if smp.split == "test":
                continue  # keep the provided test split held out for evaluate.py
            yield smp.image.stem, read_image(smp.image), smp.boxes, smp.split
    if args.images:
        imgs = sorted(p for p in Path(args.images).rglob("*") if p.suffix.lower() in IMAGE_EXT)
        counter = None
        for path in imgs:
            boxes = None
            g = Path(args.gold) / f"{path.stem}.json" if args.gold else None
            y = Path(args.yolo_labels) / f"{path.stem}.txt" if args.yolo_labels else None
            if g and g.exists():
                img = read_image(path)
                boxes = boxes_from_points(json.loads(g.read_text()).get("points", []))
            elif y and y.exists():
                img = read_image(path)
                boxes = boxes_from_yolo(y, img.shape[1], img.shape[0])
            elif args.pseudo:
                if counter is None:
                    from colony_detector.pipeline import ColonyCounter

                    counter = ColonyCounter()
                    counter.load()
                img = read_image(path)
                res = counter.count(img)
                boxes = [(*d.xyxy, 1 if d.class_name == "fuzzy_colony" else 0)
                         for d in res.detections if d.count == 1 and d.confidence >= 0.5]
            if boxes is not None:
                yield path.stem, img, boxes, None
    for i in range(args.synthetic):
        img, gt = make_plate(20_000 + i)
        boxes = [(c["x"] - c["r"], c["y"] - c["r"], c["x"] + c["r"], c["y"] + c["r"], int(c["fuzzy"]))
                 for c in gt["colonies"]]
        yield f"synth_{i:05d}", img, boxes, None


def tile_plate(img, boxes, tile: int, overlap: float, empty_keep: float, rng: random.Random):
    plate = find_plate(img)
    H, W = img.shape[:2]
    R = plate.radius
    if plate.found:
        x0, y0 = max(0, int(plate.cx - R)), max(0, int(plate.cy - R))
        x1, y1 = min(W, int(plate.cx + R)), min(H, int(plate.cy + R))
    else:
        # Already-cropped plates or image patches: use the whole frame.
        x0, y0, x1, y1 = 0, 0, W, H
    crop = img[y0:y1, x0:x1]
    b = np.array([(bx0 - x0, by0 - y0, bx1 - x0, by1 - y0, c) for bx0, by0, bx1, by1, c in boxes],
                 dtype=np.float64).reshape(-1, 5)
    for tx0, ty0, tx1, ty1 in make_tiles(crop.shape[1], crop.shape[0], tile, overlap):
        tw, th = tx1 - tx0, ty1 - ty0
        # Skip tiles that are almost entirely outside the dish.
        ccx, ccy = (tx0 + tx1) / 2 + x0, (ty0 + ty1) / 2 + y0
        if plate.found and np.hypot(ccx - plate.cx, ccy - plate.cy) > R + 0.5 * tile:
            continue
        labels = []
        for bx0, by0, bx1, by1, c in b:
            cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
            if not (tx0 <= cx < tx1 and ty0 <= cy < ty1):
                continue
            cx0, cy0 = max(bx0, tx0) - tx0, max(by0, ty0) - ty0
            cx1, cy1 = min(bx1, tx1) - tx0, min(by1, ty1) - ty0
            if cx1 - cx0 < 2 or cy1 - cy0 < 2:
                continue
            labels.append((int(c), cx0, cy0, cx1, cy1))
        if not labels and rng.random() > empty_keep:
            continue
        yield crop[ty0:ty1, tx0:tx1], labels, tw, th


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", help="labelled dataset folder (COCO / YOLO / AGAR / points), auto-detected")
    ap.add_argument("--images")
    ap.add_argument("--gold", default="eval/gold")
    ap.add_argument("--yolo-labels")
    ap.add_argument("--pseudo", action="store_true")
    ap.add_argument("--synthetic", type=int, default=0)
    ap.add_argument("--out", default="datasets/tiles")
    ap.add_argument("--tile", type=int, default=640)
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--val", type=float, default=0.15, help="fraction of plates for validation")
    ap.add_argument("--empty-keep", type=float, default=0.3, help="share of colony-free tiles to keep")
    ap.add_argument("--coco", action="store_true", help="also write COCO json (for RF-DETR)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out)
    coco = {s: {"images": [], "annotations": [], "categories": [
        {"id": i, "name": n, "supercategory": "colony"} for i, n in enumerate(NAMES)]} for s in ("train", "valid")}
    n_tiles = {"train": 0, "valid": 0}
    n_boxes = {"train": 0, "valid": 0}
    ann_id = 0
    for name, img, boxes, src_split in collect(args):
        split = src_split if src_split in ("train", "valid") else ("valid" if rng.random() < args.val else "train")
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for k, (tile, labels, tw, th) in enumerate(tile_plate(img, boxes, args.tile, args.overlap,
                                                                args.empty_keep, rng)):
            stem = f"{name}_{k:03d}"
            cv2.imwrite(str(out / "images" / split / f"{stem}.jpg"), tile, [cv2.IMWRITE_JPEG_QUALITY, 95])
            with open(out / "labels" / split / f"{stem}.txt", "w") as fh:
                for c, x0, y0, x1, y1 in labels:
                    fh.write(f"{c} {(x0 + x1) / 2 / tw:.6f} {(y0 + y1) / 2 / th:.6f} "
                             f"{(x1 - x0) / tw:.6f} {(y1 - y0) / th:.6f}\n")
            if args.coco:
                img_id = len(coco[split]["images"])
                coco[split]["images"].append({"id": img_id, "file_name": f"{stem}.jpg", "width": tw, "height": th})
                for c, x0, y0, x1, y1 in labels:
                    coco[split]["annotations"].append({
                        "id": ann_id, "image_id": img_id, "category_id": c, "iscrowd": 0,
                        "bbox": [x0, y0, x1 - x0, y1 - y0], "area": (x1 - x0) * (y1 - y0)})
                    ann_id += 1
            n_tiles[split] += 1
            n_boxes[split] += len(labels)
        print(f"{name}: {len(boxes)} colonies -> {split}")

    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/valid\n"
        f"names:\n  0: colony\n  1: fuzzy_colony\n")
    if args.coco:
        # RF-DETR expects <split>/_annotations.coco.json next to the images.
        for split in ("train", "valid"):
            d = out / "coco" / split
            d.mkdir(parents=True, exist_ok=True)
            for f in (out / "images" / split).glob("*.jpg"):
                target = d / f.name
                if not target.exists():
                    target.symlink_to(f.resolve())
            (d / "_annotations.coco.json").write_text(json.dumps(coco[split]))
        test = out / "coco" / "test"
        if not test.exists():
            test.symlink_to((out / "coco" / "valid").resolve())
    print(f"tiles: {n_tiles}, boxes: {n_boxes}, written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
