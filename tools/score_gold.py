"""Score colony counts against a reviewed answer key (tools/answer_key.py).

Each gold file lists the colonies that count under the rule (>= min mm), the
pin-point specks below it, and the spots a reviewer could not decide. Scoring
is fair to both readings of an undecided spot:

* a plate's count is correct when it lies in [count, count + unsure];
* a detection on an unsure spot or a speck is neither a hit nor a false
  positive;
* plates flagged TNTC / overgrown are reported separately (no exact count
  exists for them).

Predictions come from a detections cache (pickle {image name: {"dets": [(x,
y, radius, conf), ...]}}) or from running the service pipeline over the
images. A missing cache is built by running the configured detector
(CD_ENGINES / CD_RFDETR_WEIGHTS) once at a low confidence:

    python tools/score_gold.py --gold eval/gold_test_ruleB --images data/raw   # runs ColonyCounter
    python tools/score_gold.py --gold eval/gold_test_ruleB --images data/raw \
        --cache runs/test_dets.pkl --tune models/detector.json
    python tools/score_gold.py --gold eval/gold_test_ruleB --cache runs/test_dets.pkl --conf 0.5
    python tools/score_gold.py --gold eval/gold_test_ruleB --original runs/test_dets.pkl  # original labels

--tune fits the detector's confidence cut (and the crowded-plate cut, see
TiledBoxEngine.dense_conf) on the gold plates, reports a 2-fold
cross-validated estimate so the number is not flattered by fitting, and
writes the chosen values for the engine.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def match(gold: np.ndarray, pred: np.ndarray) -> list[tuple[int, int]]:
    """Hungarian matching of (x, y, r) rows; a pair must lie within
    max(8 px, 0.8 x the larger radius)."""
    if not len(gold) or not len(pred):
        return []
    d = np.linalg.norm(gold[:, None, :2] - pred[None, :, :2], axis=2)
    tol = np.maximum(8.0, 0.8 * np.maximum(gold[:, None, 2], pred[None, :, 2]))
    cost = np.where(d <= tol, d, 1e6)
    rows, cols = linear_sum_assignment(cost)
    return [(r, c) for r, c in zip(rows, cols) if cost[r, c] < 1e6]


def score_plate(g: dict, pred: np.ndarray, min_mm: float, review_mm: float) -> dict:
    """pred rows are (x, y, r) after the confidence cut; the size rule is
    applied here with the plate's own scale."""
    px_mm = g["px_per_mm"]
    size_mm = 2 * pred[:, 2] / px_mm if len(pred) else np.zeros(0)
    counted = pred[size_mm >= min_mm] if len(pred) else pred
    specks = int(((size_mm >= review_mm) & (size_mm < min_mm)).sum()) if len(pred) else 0
    gold = np.array(g["points"], float).reshape(-1, 3)
    ignore = np.array(g["unsure"] + g["small_specks"], float).reshape(-1, 3)
    pairs = match(gold, counted)
    hit_p = {c for _, c in pairs}
    rest = np.array([p for i, p in enumerate(counted) if i not in hit_p], float).reshape(-1, 3)
    ignored = len(match(ignore, rest))
    tp = len(pairs)
    fp = len(rest) - ignored
    fn = len(gold) - tp
    n, lo, hi = len(counted), g["count"], g["count"] + len(g["unsure"])
    err = 0 if lo <= n <= hi else (n - hi if n > hi else n - lo)
    return {"image": g["image"], "gold": lo, "unsure": hi - lo, "pred": n, "error": err,
            "tp": tp, "fp": fp, "fn": fn, "pred_specks": specks, "gold_specks": len(g["small_specks"]),
            "flag": g.get("plate_flag") or ""}


def summarise(rows: list[dict]) -> dict:
    if not rows:
        return {}
    err = np.array([r["error"] for r in rows])
    tp, fp, fn = (sum(r[k] for r in rows) for k in ("tp", "fp", "fn"))
    p = tp / max(tp + fp, 1)
    rc = tp / max(tp + fn, 1)
    return {
        "plates": len(rows),
        "exact": round(float((err == 0).mean()), 3),
        "within_1": round(float((np.abs(err) <= 1).mean()), 3),
        "within_2": round(float((np.abs(err) <= 2).mean()), 3),
        "mae": round(float(np.abs(err).mean()), 3),
        "bias": round(float(err.mean()), 3),
        "precision": round(p, 4), "recall": round(rc, 4), "f1": round(2 * p * rc / max(p + rc, 1e-9), 4),
        "tp": tp, "fp": fp, "fn": fn,
    }


def load_gold(folder: str) -> list[dict]:
    return [json.loads(f.read_text()) for f in sorted(Path(folder).glob("*.json"))]


def _cut(dets: np.ndarray, conf: float, dense_conf: float, dense_min: int) -> np.ndarray:
    """Same rule as TiledBoxEngine: a crowded plate keeps boxes down to dense_conf."""
    if not len(dets):
        return dets[:, :3]
    t = conf
    if dense_min and (dets[:, 3] >= conf).sum() >= dense_min:
        t = min(conf, dense_conf)
    return dets[dets[:, 3] >= t, :3]


def tune(gold: list[dict], cache_path: str, min_mm: float, review_mm: float) -> tuple[dict, dict]:
    cache = pickle.loads(Path(cache_path).read_bytes())
    plates = [g for g in gold if not g.get("plate_flag") and g["image"] in cache]
    dets = {g["image"]: np.array(cache[g["image"]]["dets"], float).reshape(-1, 4) for g in plates}
    grid = [(c, c, 0) for c in np.arange(0.3, 0.86, 0.05)]
    grid += list(itertools.product(np.arange(0.5, 0.91, 0.05), np.arange(0.3, 0.56, 0.05), (3, 5, 8, 10, 15, 20)))

    def run(params, subset):
        c, dc, dm = params
        return summarise([score_plate(g, _cut(dets[g["image"]], c, dc, dm), min_mm, review_mm) for g in subset])

    def key(sm):
        # Plate-level exactness alone favours sparse plates (most plates hold
        # 0-3 colonies) and would starve crowded ones; balance it with +-1
        # agreement and colony-level F1.
        return (sm["exact"] + sm["within_1"] + sm["f1"], -sm["mae"])

    best = max(grid, key=lambda p: key(run(p, plates)))
    cv = []
    for salt in range(5):
        def fold(g, salt=salt):
            return int(hashlib.sha1(f"{salt}:{g['image']}".encode()).hexdigest(), 16) % 2
        for k in (0, 1):
            fit = max(grid, key=lambda p: key(run(p, [g for g in plates if fold(g) != k])))
            cv.append(run(fit, [g for g in plates if fold(g) == k]))
    estimate = {m: round(float(np.mean([x[m] for x in cv])), 3) for m in ("exact", "within_1", "within_2", "mae", "f1")}
    c, dc, dm = best
    params = {"conf": round(float(c), 2), "dense_conf": round(float(dc), 2) if dm else 0.0, "dense_min": int(dm)}
    return params, {"fitted": run(best, plates), "cross_validated": estimate}


def predictions_from_original(path: str) -> dict[str, np.ndarray]:
    cache = pickle.loads(Path(path).read_bytes())
    return {k: np.array([(p["x"], p["y"], p["r"]) for p in v["gt"]], float).reshape(-1, 3)
            for k, v in cache.items()}


def build_cache(gold: list[dict], images: str, path: Path) -> None:
    """Run the configured detector once per gold plate at a low cut-off."""
    from colony_detector.config import Settings
    from colony_detector.datasets import load_dataset
    from colony_detector.engines.tiled import TiledBoxEngine
    from colony_detector.imageio import read_image
    from colony_detector.pipeline import build_engines
    from colony_detector.plate import find_plate

    engines = [e for e in build_engines(Settings()) if isinstance(e, TiledBoxEngine)]
    if not engines:
        raise SystemExit("no trained detector configured (CD_RFDETR_WEIGHTS / models/rfdetr_tiles.pth)")
    eng = engines[0]
    eng.load()
    eng.conf, eng.dense_conf, eng.dense_min = 0.05, 0.0, 0.0
    labels = {s.image.name: s for s in load_dataset(images)}
    cache = pickle.loads(path.read_bytes()) if path.exists() else {}
    for i, g in enumerate(gold):
        if g["image"] in cache:
            continue
        smp = labels[g["image"]]
        img = read_image(smp.image)
        dets = eng.detect(img, find_plate(img), {})
        cache[g["image"]] = {"gt": smp.points(), "n": smp.total,
                             "dets": [(d.cx, d.cy, d.radius, d.confidence) for d in dets]}
        if i % 10 == 0:
            path.write_bytes(pickle.dumps(cache))
            print(f"{i}/{len(gold)}", file=sys.stderr, flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(cache))


def predictions_from_pipeline(gold: list[dict], images: str) -> dict[str, np.ndarray]:
    from colony_detector.config import Settings
    from colony_detector.imageio import read_image
    from colony_detector.pipeline import ColonyCounter

    s = Settings()
    s.min_colony_mm = 0.0  # the size rule is applied by the scorer, per plate
    counter = ColonyCounter(s)
    index = {p.name: p for p in Path(images).rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")}
    out = {}
    for i, g in enumerate(gold):
        res = counter.count(read_image(index[g["image"]]))
        out[g["image"]] = np.array([(d.cx, d.cy, d.radius) for d in res.detections], float).reshape(-1, 3)
        print(f"{i + 1}/{len(gold)} {g['image']} {len(res.detections)}", file=sys.stderr, flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--cache", help="detections pickle (built with --images when missing)")
    ap.add_argument("--images", help="dataset root: images and original labels")
    ap.add_argument("--original", metavar="CACHE", help="score the original labels stored in a cache")
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--min-mm", type=float, default=None, help="default: the gold files' rule")
    ap.add_argument("--review-mm", type=float, default=None)
    ap.add_argument("--dense-conf", type=float, default=0.0)
    ap.add_argument("--dense-min", type=int, default=0)
    ap.add_argument("--detector", metavar="DETECTOR_JSON", help="take conf / dense cut-offs from this file")
    ap.add_argument("--csv", help="write per-plate rows here")
    ap.add_argument("--tune", metavar="DETECTOR_JSON", help="fit the cuts (needs --cache) and write them here")
    ap.add_argument("--engine", default="rfdetr", help="engine name in detector.json")
    args = ap.parse_args()

    gold = load_gold(args.gold)
    rule = gold[0]["rule"]
    min_mm = rule["min_colony_mm"] if args.min_mm is None else args.min_mm
    review_mm = rule["review_colony_mm"] if args.review_mm is None else args.review_mm
    if not (args.cache or args.images or args.original):
        ap.error("give --cache, --images or --original")
    if args.cache and not Path(args.cache).exists():
        if not args.images:
            ap.error(f"{args.cache} does not exist; add --images to build it")
        build_cache(gold, args.images, Path(args.cache))
    if args.tune:
        if not args.cache:
            ap.error("--tune needs --cache")
        params, report = tune(gold, args.cache, min_mm, review_mm)
        out = Path(args.tune)
        cfg = json.loads(out.read_text()) if out.exists() else {}
        cfg[args.engine] = {**cfg.get(args.engine, {}), **params}
        out.write_text(json.dumps(cfg, indent=2) + "\n")
        print(json.dumps({"params": params, **report}, indent=1))
        return 0
    if args.detector:
        cut = json.loads(Path(args.detector).read_text()).get(args.engine, {})
        args.conf = cut.get("conf", args.conf)
        args.dense_conf, args.dense_min = cut.get("dense_conf", 0.0), int(cut.get("dense_min", 0))
    if args.cache:
        cache = pickle.loads(Path(args.cache).read_bytes())
        preds = {k: _cut(np.array(v["dets"], float).reshape(-1, 4), args.conf, args.dense_conf, args.dense_min)
                 for k, v in cache.items()}
    elif args.original:
        preds = predictions_from_original(args.original)
    else:
        preds = predictions_from_pipeline(gold, args.images)

    rows = [score_plate(g, preds[g["image"]], min_mm, review_mm) for g in gold if g["image"] in preds]
    report = {
        "countable": summarise([r for r in rows if not r["flag"]]),
        "tntc_or_overgrown": summarise([r for r in rows if r["flag"]]),
        "rule": {"min_colony_mm": min_mm, "review_colony_mm": review_mm},
    }
    print(json.dumps(report, indent=1))
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
