"""Fit a trained detector's cut-offs on validation plates.

Runs the detector once per plate at a low confidence, caches the boxes, then
grid-searches the confidence cut-off, a minimum colony size (drops dust
specks) and a stricter cut-off for very large boxes (scratches, plate-wide
boxes) for the most exact per-plate counts. Writes ``models/detector.json``,
which the service applies at start-up.

    python tools/tune_detector.py --dataset data/raw --engine rfdetr --split valid

Fit on the valid split and report on the test split, never the same plates.
"""

from __future__ import annotations

import argparse
import itertools
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from colony_detector.config import Settings  # noqa: E402
from colony_detector.datasets import load_dataset  # noqa: E402
from colony_detector.imageio import read_image  # noqa: E402
from colony_detector.metrics import count_metrics, detection_metrics, match_points  # noqa: E402
from colony_detector.pipeline import build_engines  # noqa: E402
from colony_detector.plate import find_plate  # noqa: E402

CONFS = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]
MIN_SIZES = [0, 6, 8, 10, 12]
BIG_SIZES = [0, 200, 400]
BIG_CONFS = [0.7, 0.8, 0.9]


def cache_detections(engine, samples, cache_path: Path) -> dict:
    cache = pickle.loads(cache_path.read_bytes()) if cache_path.exists() else {}
    engine.conf = 0.05
    engine.min_size_px = engine.big_size_px = 0.0
    for i, s in enumerate(samples):
        if s.image.name in cache:
            continue
        img = read_image(s.image)
        dets = engine.detect(img, find_plate(img), {})
        cache[s.image.name] = {"gt": s.points(), "n": s.total,
                               "dets": [(d.cx, d.cy, 2 * d.radius, d.confidence) for d in dets]}
        if i % 10 == 0:
            cache_path.write_bytes(pickle.dumps(cache))
            print(f"{i}/{len(samples)}", flush=True)
    cache_path.write_bytes(pickle.dumps(cache))
    return cache


def score(cache: dict, conf: float, min_px: float, big_px: float, big_conf: float) -> tuple[dict, dict]:
    true, pred = [], []
    tp = fp = fn = 0
    for v in cache.values():
        keep = [(x, y, sz, c) for x, y, sz, c in v["dets"]
                if c >= conf and sz >= min_px and not (big_px and sz > big_px and c < big_conf)]
        true.append(v["n"])
        pred.append(len(keep))
        if v["gt"] is not None:
            a, b, c = match_points(v["gt"], [{"x": x, "y": y, "r": sz / 2, "count": 1} for x, y, sz, _ in keep])
            tp, fp, fn = tp + a, fp + b, fn + c
    return count_metrics(true, pred), detection_metrics(tp, fp, fn)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="valid")
    ap.add_argument("--engine", default="rfdetr", choices=["rfdetr", "yolo", "gdino"])
    ap.add_argument("--cache", help="detection cache file (default: runs/<engine>_<split>_dets.pkl)")
    ap.add_argument("--out", default=None, help="default: <models_dir>/detector.json")
    args = ap.parse_args()

    settings = Settings()
    settings.engines = args.engine
    (engine,) = build_engines(settings)
    engine.load()
    samples = [s for s in load_dataset(args.dataset) if s.split == args.split and s.total is not None]
    cache_path = Path(args.cache or f"runs/{args.engine}_{args.split}_dets.pkl")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = cache_detections(engine, samples, cache_path)

    best = None
    for conf, min_px, big_px, big_conf in itertools.product(CONFS, MIN_SIZES, BIG_SIZES, BIG_CONFS):
        if not big_px and big_conf != BIG_CONFS[0]:
            continue
        cm, dm = score(cache, conf, min_px, big_px, big_conf)
        key = (cm["exact"], -cm["mae"], dm["f1"])
        if best is None or key > best[0]:
            params = {"conf": conf, "min_size_px": min_px, "big_size_px": big_px, "big_conf": big_conf}
            best = (key, params, cm, dm)
    _, params, cm, dm = best
    out = Path(args.out) if args.out else settings.models_dir / "detector.json"
    data = json.loads(out.read_text()) if out.exists() else {}
    data[args.engine] = params
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2))
    print(f"\nbest on {args.split} ({len(cache)} plates): {params}")
    print(f"  counts: {cm}\n  detection: {dm}\nwritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
