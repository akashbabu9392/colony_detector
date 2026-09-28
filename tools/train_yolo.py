"""Train a YOLO colony detector on plate tiles.

    python tools/train_yolo.py --data datasets/tiles/data.yaml --model yolo11s.pt --epochs 150

Runs on CPU but is slow; on a single T4 (e.g. Google Colab) yolo11s trains in
roughly 1-3 hours for a few thousand tiles. The best checkpoint is copied to
``models/yolo11s_tiles.pt``, where the service picks it up automatically.

Augmentations are chosen for plates: rotations and both flips are free (a
plate has no up), colour/brightness jitter covers media and lighting, and
mosaic is kept because it mimics the tile boundaries seen at inference.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="datasets/tiles/data.yaml")
    ap.add_argument("--model", default="yolo11s.pt", help="starting weights (COCO-pretrained)")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="")
    ap.add_argument("--name", default="colony_tiles")
    ap.add_argument("--out", default=str(REPO / "models" / "yolo11s_tiles.pt"))
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=args.device or None, name=args.name, patience=40, cos_lr=True,
        degrees=180, flipud=0.5, fliplr=0.5, mosaic=1.0, close_mosaic=15,
        hsv_h=0.03, hsv_s=0.5, hsv_v=0.4, scale=0.3, translate=0.1,
        max_det=1000,
    )
    best = Path(model.trainer.best)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best, args.out)
    metrics = model.val(data=args.data, imgsz=args.imgsz, max_det=1000)
    print(f"best: {best} -> {args.out}")
    print(f"val mAP50={metrics.box.map50:.4f} mAP50-95={metrics.box.map:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
