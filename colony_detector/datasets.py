"""Read labelled colony datasets in the common formats.

``load_dataset(root)`` auto-detects and yields :class:`Sample` records:

* **COCO** - any ``*.json`` with ``images`` + ``annotations`` (Roboflow COCO
  exports: ``train/_annotations.coco.json`` ...).
* **YOLO** - ``.../images/*.jpg`` with matching ``.../labels/*.txt``
  (Roboflow / Ultralytics layout; class names from ``data.yaml``).
* **AGAR** - one ``<id>.json`` per image with ``labels: [{x, y, width,
  height, class}]`` next to ``<id>.jpg``.
* **Points** - hand_count.py files: ``{"image": ..., "points": [[x, y, r]]}``.
* **Counts only** - a ``counts.csv`` with ``image,count`` columns.

The split (train / valid / test) is taken from the path when it contains
one of those folder names, so a provided test split stays held out.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from colony_detector.engines.tiled import map_class
from colony_detector.types import FUZZY

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
_SPLITS = {"train": "train", "training": "train", "valid": "valid", "val": "valid",
           "validation": "valid", "test": "test", "testing": "test"}


@dataclass
class Sample:
    image: Path
    boxes: list[tuple[float, float, float, float, int]] | None = None  # x0,y0,x1,y1,cls (0 colony, 1 fuzzy)
    count: int | None = None
    split: str | None = None
    extra: dict = field(default_factory=dict)

    @property
    def total(self) -> int | None:
        if self.count is not None:
            return self.count
        return len(self.boxes) if self.boxes is not None else None

    def points(self) -> list[dict] | None:
        if self.boxes is None:
            return None
        return [{"x": (b[0] + b[2]) / 2, "y": (b[1] + b[3]) / 2, "r": max(b[2] - b[0], b[3] - b[1]) / 2}
                for b in self.boxes]


def _split_of(path: Path) -> str | None:
    for part in reversed(path.parts):
        s = _SPLITS.get(part.lower())
        if s:
            return s
    return None


def _cls_id(name: str) -> int | None:
    """Dataset class name -> 0 colony / 1 fuzzy / None (not a colony: skip)."""
    mapped = map_class(name)
    if mapped is None:
        return None
    return 1 if mapped == FUZZY else 0


def _find_image(base: Path, name: str) -> Path | None:
    p = base / name
    if p.exists():
        return p
    stem = Path(name).stem
    for ext in IMAGE_EXT:
        q = base / f"{stem}{ext}"
        if q.exists():
            return q
    hits = list(base.rglob(Path(name).name))
    return hits[0] if hits else None


def _coco(js: Path, data: dict) -> Iterator[Sample]:
    cats = {c["id"]: c.get("name", "colony") for c in data.get("categories", [])}
    # Roboflow adds a supercategory placeholder class (e.g. "colonies") that
    # is never used by annotations; mapping by name handles it.
    by_img: dict[int, list] = {}
    for a in data["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)
    for im in data["images"]:
        path = _find_image(js.parent, im["file_name"])
        if path is None:
            continue
        boxes = []
        for a in by_img.get(im["id"], []):
            c = _cls_id(cats.get(a["category_id"], "colony"))
            if c is None:
                continue
            x, y, w, h = a["bbox"]
            boxes.append((x, y, x + w, y + h, c))
        yield Sample(path, boxes=boxes, split=_split_of(js))


def _yolo_names(root: Path) -> dict[int, str]:
    for y in list(root.rglob("data.yaml"))[:1]:
        try:
            import yaml

            names = yaml.safe_load(y.read_text()).get("names", {})
        except Exception:  # noqa: BLE001 - names are optional
            return {}
        return dict(enumerate(names)) if isinstance(names, list) else {int(k): v for k, v in names.items()}
    return {}


def _yolo(root: Path) -> Iterator[Sample]:
    import cv2

    names = _yolo_names(root)
    for img in sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXT and "images" in p.parts):
        parts = list(img.parts)
        i = len(parts) - 1 - parts[::-1].index("images")
        parts[i] = "labels"
        txt = Path(*parts).with_suffix(".txt")
        if not txt.exists():
            continue
        h, w = cv2.imread(str(img), cv2.IMREAD_REDUCED_GRAYSCALE_2).shape[:2]
        h, w = h * 2, w * 2
        boxes = []
        for line in txt.read_text().splitlines():
            v = line.split()
            if len(v) < 5:
                continue
            c = _cls_id(names.get(int(v[0]), "colony"))
            if c is None:
                continue
            if len(v) > 5:  # segmentation polygon -> its bounding box
                xs, ys = [float(t) for t in v[1::2]], [float(t) for t in v[2::2]]
                boxes.append((min(xs) * w, min(ys) * h, max(xs) * w, max(ys) * h, c))
            else:
                xc, yc, bw, bh = map(float, v[1:5])
                boxes.append(((xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h, c))
        yield Sample(img, boxes=boxes, split=_split_of(img))


def load_dataset(root: str | Path) -> list[Sample]:
    root = Path(root)
    samples: dict[Path, Sample] = {}

    for js in sorted(root.rglob("*.json")):
        try:
            data = json.loads(js.read_text())
        except (ValueError, UnicodeDecodeError):
            continue
        if isinstance(data, dict) and "images" in data and "annotations" in data:
            for s in _coco(js, data):
                samples[s.image] = s
        elif isinstance(data, dict) and "labels" in data and isinstance(data["labels"], list):
            img = _find_image(js.parent, js.stem)
            if img is None:
                continue
            boxes = []
            for lab in data["labels"]:
                c = _cls_id(str(lab.get("class", "colony")))
                if c is not None and {"x", "y", "width", "height"} <= set(lab):
                    x, y = float(lab["x"]), float(lab["y"])
                    boxes.append((x, y, x + float(lab["width"]), y + float(lab["height"]), c))
            samples[img] = Sample(img, boxes=boxes, split=_split_of(js),
                                  count=data.get("colonies_number"))
        elif isinstance(data, dict) and "points" in data:
            img = _find_image(root, data.get("image", js.stem))
            if img is None:
                continue
            boxes = []
            for p in data["points"]:
                r = p[2] if len(p) > 2 else 6.0
                boxes.append((p[0] - r, p[1] - r, p[0] + r, p[1] + r, 0))
            samples[img] = Sample(img, boxes=boxes, split=_split_of(js))

    for s in _yolo(root):
        samples.setdefault(s.image, s)

    for c in root.rglob("*.csv"):
        with open(c) as fh:
            reader = csv.DictReader(fh)
            if not reader.fieldnames or not {"image", "count"} <= set(reader.fieldnames):
                continue
            for row in reader:
                img = _find_image(c.parent, row["image"])
                if img is not None and img not in samples:
                    samples[img] = Sample(img, count=int(float(row["count"])), split=_split_of(img))
    return sorted(samples.values(), key=lambda s: str(s.image))
