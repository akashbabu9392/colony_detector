"""Command line: count colonies in one or more plate images.

    python -m colony_detector plate1.jpg plate2.png --out results/
    python -m colony_detector plates/ --csv counts.csv --engines classical,yolo
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2

from colony_detector.config import Settings
from colony_detector.imageio import read_image
from colony_detector.pipeline import ColonyCounter

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def iter_images(paths: list[str]):
    for p in map(Path, paths):
        if p.is_dir():
            yield from sorted(f for f in p.rglob("*") if f.suffix.lower() in IMAGE_EXT)
        else:
            yield p


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="colony_detector", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="image files or folders")
    ap.add_argument("--out", help="folder for annotated images and per-image JSON")
    ap.add_argument("--csv", help="write a summary CSV here")
    ap.add_argument("--engines", help="override CD_ENGINES, e.g. classical,yolo")
    ap.add_argument("--polarity", choices=["auto", "dark", "bright", "both"])
    ap.add_argument("--json", action="store_true", help="print full JSON to stdout")
    args = ap.parse_args(argv)

    settings = Settings()
    if args.engines:
        settings.engines = args.engines
    if args.polarity:
        settings.classical.polarity = args.polarity
    counter = ColonyCounter(settings)
    counter.load()

    out = Path(args.out) if args.out else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in iter_images(args.inputs):
        try:
            res = counter.count(read_image(path), annotate=bool(out))
        except Exception as exc:  # noqa: BLE001 - keep going through a batch
            print(f"{path}: ERROR {exc}", file=sys.stderr)
            continue
        resp = res.to_response()
        rows.append({
            "image": str(path), "total": res.total, **res.counts, "tntc": res.tntc,
            "needs_review": res.confidence["needs_review"],
            "reasons": ";".join(res.confidence["reason_codes"]), "ms": res.inference_ms,
        })
        flag = " TNTC" if res.tntc else ""
        review = f" review:{','.join(res.confidence['reason_codes'])}" if res.confidence["needs_review"] else ""
        print(f"{path}: {res.total} CFU ({res.counts}){flag}{review} [{res.inference_ms} ms]")
        if out:
            (out / f"{path.stem}.json").write_text(json.dumps(resp, indent=1))
            if res.annotated is not None:
                cv2.imwrite(str(out / f"{path.stem}_annotated.jpg"), res.annotated,
                            [cv2.IMWRITE_JPEG_QUALITY, 90])
        if args.json:
            print(json.dumps(resp))
    if args.csv and rows:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
