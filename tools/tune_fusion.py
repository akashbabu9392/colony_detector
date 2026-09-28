"""Fit the fusion weights and threshold to your own gold-standard plates.

Every engine runs once per plate; the fusion of their cached detections is
then grid-searched for the lowest count error (ties broken by detection F1
when point labels exist). The winner is written to ``models/fusion.json``,
which the service loads at start-up.

    python tools/tune_fusion.py --images data/plates --gold eval/gold
    python tools/tune_fusion.py --dataset data/colony_dataset   # its valid split
    python tools/tune_fusion.py --synthetic 40          # sanity run

Tune on plates that are NOT in your final evaluation set.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluate import load_cases  # noqa: E402

from colony_detector.config import Settings  # noqa: E402
from colony_detector.fusion import fuse  # noqa: E402
from colony_detector.imageio import read_image  # noqa: E402
from colony_detector.metrics import count_metrics, detection_metrics, match_points  # noqa: E402
from colony_detector.pipeline import ColonyCounter  # noqa: E402
from colony_detector.plate import find_plate  # noqa: E402
from colony_detector.synth import make_plate  # noqa: E402

WEIGHTS = [0.25, 0.5, 0.8, 1.0, 1.25, 1.6, 2.0]
THRESHOLDS = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", help="labelled dataset folder; tunes on its valid split by default")
    ap.add_argument("--split", default="valid")
    ap.add_argument("--images")
    ap.add_argument("--gold", default="eval/gold")
    ap.add_argument("--csv")
    ap.add_argument("--synthetic", type=int, default=0)
    ap.add_argument("--engines", help="defaults to CD_ENGINES / auto")
    ap.add_argument("--out", default=None, help="default: <models_dir>/fusion.json")
    args = ap.parse_args()

    settings = Settings()
    if args.engines:
        settings.engines = args.engines
    counter = ColonyCounter(settings)
    counter.load()
    names = [e.name for e in counter.engines]
    if len(names) < 2:
        print(f"only one engine available ({names}); nothing to fuse")
        return 1

    cases = load_cases(args)
    cache = []
    for case in cases:
        if "seed" in case:
            img, gt = make_plate(case["seed"])
            case = {**case, "count": gt["count"], "points": gt["colonies"]}
        else:
            img = read_image(case["path"])
        plate = find_plate(img)
        ctx: dict = {}
        per = {e.name: e.detect(img, plate, ctx) for e in counter.engines}
        cache.append((case, per))
        print(f"{case['name']}: " + ", ".join(f"{k}={sum(d.count for d in v)}" for k, v in per.items()),
              f"true={case['count']}")

    best = None
    for ws in itertools.product(WEIGHTS, repeat=len(names)):
        weights = dict(zip(names, ws))
        for thr in THRESHOLDS:
            true, pred = [], []
            tp = fp = fn = 0
            for case, per in cache:
                dets = fuse(per, weights, thr)
                true.append(case["count"])
                pred.append(sum(d.count for d in dets))
                if case.get("points"):
                    a, b, c = match_points(case["points"], [
                        {"x": d.cx, "y": d.cy, "r": d.radius, "count": d.count} for d in dets])
                    tp, fp, fn = tp + a, fp + b, fn + c
            cm = count_metrics(true, pred)
            f1 = detection_metrics(tp, fp, fn)["f1"] if tp + fp + fn else 0.0
            key = (cm["mae"], -f1)
            if best is None or key < best[0]:
                best = (key, weights, thr, cm, f1)

    _, weights, thr, cm, f1 = best
    out = Path(args.out) if args.out else settings.models_dir / "fusion.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"weights": weights, "threshold": thr,
                               "fitted_on": len(cache), "metrics": cm, "f1": f1}, indent=2))
    print(f"\nbest: weights={weights} threshold={thr} -> {cm} f1={f1}\nwritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
