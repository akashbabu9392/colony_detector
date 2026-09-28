"""Measure counting accuracy against ground truth.

Ground truth sources (any combination):

* ``--gold eval/gold``: one JSON per image, written by tools/hand_count.py:
  ``{"image": "plate1.jpg", "count": 57, "points": [[x, y], ...]}``
  (``points`` optional; with points you also get precision/recall/F1).
* ``--csv counts.csv``: columns ``image,count`` for count-only gold.
* ``--synthetic N``: N generated plates with exact truth.

Examples::

    python tools/evaluate.py --images data/plates --gold eval/gold
    python tools/evaluate.py --synthetic 60 --per-engine
    python tools/evaluate.py --images data/plates --csv lab_counts.csv --engines classical,yolo

Writes a per-image CSV and prints summary metrics (MAE, RMSE, bias, MAPE,
share within 5%/10%, detection F1).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from colony_detector.config import Settings  # noqa: E402
from colony_detector.imageio import read_image  # noqa: E402
from colony_detector.metrics import count_metrics, detection_metrics, match_points  # noqa: E402
from colony_detector.pipeline import ColonyCounter  # noqa: E402
from colony_detector.synth import make_plate  # noqa: E402

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def load_cases(args) -> list[dict]:
    cases: list[dict] = []
    csv_gold: dict[str, int] = {}
    if args.csv:
        with open(args.csv) as fh:
            for row in csv.DictReader(fh):
                csv_gold[Path(row["image"]).stem] = int(float(row["count"]))
    if args.images:
        gold_dir = Path(args.gold) if args.gold else None
        for img in sorted(p for p in Path(args.images).rglob("*") if p.suffix.lower() in IMAGE_EXT):
            case = {"name": img.name, "path": img}
            g = gold_dir / f"{img.stem}.json" if gold_dir else None
            if g and g.exists():
                data = json.loads(g.read_text())
                case["count"] = int(data.get("count", len(data.get("points", []))))
                if data.get("points"):
                    case["points"] = [{"x": p[0], "y": p[1], "r": p[2] if len(p) > 2 else 6.0}
                                      for p in data["points"]]
            elif img.stem in csv_gold:
                case["count"] = csv_gold[img.stem]
            if "count" in case:
                cases.append(case)
    for seed in range(args.synthetic):
        cases.append({"name": f"synthetic_{seed:04d}", "seed": 10_000 + seed})
    return cases


def run(counter: ColonyCounter, cases: list[dict]) -> tuple[list[dict], dict]:
    rows, tc, pc = [], [], []
    tp = fp = fn = 0
    has_points = False
    for case in cases:
        if "seed" in case:
            img, gt = make_plate(case["seed"])
            case = {**case, "count": gt["count"], "points": gt["colonies"]}
        else:
            img = read_image(case["path"])
        res = counter.count(img)
        row = {"image": case["name"], "true": case["count"], "pred": res.total,
               "error": res.total - case["count"], "tntc": res.tntc,
               "needs_review": res.confidence["needs_review"], "ms": res.inference_ms}
        if case.get("points") is not None:
            has_points = True
            pred = [{"x": d.cx, "y": d.cy, "r": d.radius, "count": d.count} for d in res.detections]
            a, b, c = match_points(case["points"], pred)
            tp, fp, fn = tp + a, fp + b, fn + c
            row.update({"tp": a, "fp": b, "fn": c})
        rows.append(row)
        tc.append(case["count"])
        pc.append(res.total)
        print(f"{row['image']}: true={row['true']} pred={row['pred']} err={row['error']:+d}")
    summary = count_metrics(tc, pc)
    if has_points:
        summary["detection"] = detection_metrics(tp, fp, fn)
    return rows, summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", help="folder of plate images")
    ap.add_argument("--gold", default="eval/gold", help="folder of hand-count JSON files")
    ap.add_argument("--csv", help="CSV with image,count columns")
    ap.add_argument("--synthetic", type=int, default=0, help="number of synthetic plates")
    ap.add_argument("--engines", help="engines to evaluate, e.g. classical,yolo")
    ap.add_argument("--per-engine", action="store_true", help="also evaluate each engine alone")
    ap.add_argument("--out", default="eval/results.csv")
    args = ap.parse_args()

    cases = load_cases(args)
    if not cases:
        print("no evaluable images: give --images with --gold/--csv, or --synthetic N", file=sys.stderr)
        return 1

    settings = Settings()
    if args.engines:
        settings.engines = args.engines
    configs = [("ensemble", settings.engine_list())]
    if args.per_engine and len(configs[0][1]) > 1:
        configs += [(e, [e]) for e in configs[0][1]]

    report = {}
    for label, engines in configs:
        s = Settings()
        s.engines = ",".join(engines)
        counter = ColonyCounter(s)
        counter.load()
        print(f"\n=== {label}: {counter.model_version}")
        rows, summary = run(counter, cases)
        report[label] = summary
        out = Path(args.out)
        if label != "ensemble":
            out = out.with_name(f"{out.stem}_{label}{out.suffix}")
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="") as fh:
            fields = list(dict.fromkeys(k for r in rows for k in r))
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
    print("\n" + json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
