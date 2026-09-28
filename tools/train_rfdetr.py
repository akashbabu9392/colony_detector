"""Fine-tune RF-DETR on plate tiles (COCO format from build_training_set.py --coco).

    pip install rfdetr
    python tools/train_rfdetr.py --dataset datasets/tiles/coco --size base --epochs 60

Needs a GPU in practice. The best checkpoint is copied to
``models/rfdetr_tiles.pth`` for the service.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="datasets/tiles/coco")
    ap.add_argument("--size", default="base", choices=["nano", "small", "medium", "base", "large"])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--output", default="runs/rfdetr")
    ap.add_argument("--out", default=str(REPO / "models" / "rfdetr_tiles.pth"))
    args = ap.parse_args()

    import rfdetr

    cls = {"nano": "RFDETRNano", "small": "RFDETRSmall", "medium": "RFDETRMedium",
           "base": "RFDETRBase", "large": "RFDETRLarge"}[args.size]
    model = getattr(rfdetr, cls)()
    model.train(dataset_dir=args.dataset, epochs=args.epochs, batch_size=args.batch,
                grad_accum_steps=args.grad_accum, lr=args.lr, output_dir=args.output)
    ckpts = sorted(Path(args.output).glob("checkpoint_best*.pth")) or sorted(Path(args.output).glob("*.pth"))
    if not ckpts:
        print("no checkpoint found in", args.output)
        return 1
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(ckpts[0], args.out)
    print(f"{ckpts[0]} -> {args.out}  (set CD_RFDETR_SIZE={args.size})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
