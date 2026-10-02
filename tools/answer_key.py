"""Build a corrected answer key (gold counts) for a dataset split.

Original labels and model detections are compared plate by plate:

* where both agree, the colony is accepted automatically;
* every disagreement (labelled but not detected, detected but not labelled,
  or an implausibly large label such as a whole-plate lawn) becomes a
  numbered candidate on zoomed review sheets;
* a reviewer records a decision per candidate, and ``apply`` writes one gold
  file per plate that follows the counting rule (count >= --min-mm, list
  specks between --review-mm and --min-mm separately, ignore smaller).

    python tools/answer_key.py prepare --dataset data/raw --split test \\
        --cache runs/rfdetr_test_dets.pkl --out review/test
    #   ...review review/test/sheet_*.jpg, fill review/test/decisions.json...
    python tools/answer_key.py apply --out review/test --gold eval/gold_test

decisions.json maps candidate id -> "colony" | "not" | "small" | "unsure".
Unsure candidates are kept out of the count and listed in the gold file.
Two more key forms correct what the automatic step got wrong:

* ``"auto:<image name>:<index>"`` re-decides an auto-accepted colony (both
  the label and the model agreed on it, e.g. both boxed sticker text);
* ``"plate:<image name>": "tntc" | "overgrown"`` marks a plate whose exact
  count is not meaningful (confluent lawn, overlapping moulds).

``apply`` reads decisions.json, or merges every dec_*.json when it is absent.
Sizes use the dish (90 mm) as the ruler.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from colony_detector.datasets import load_dataset  # noqa: E402
from colony_detector.imageio import read_image  # noqa: E402
from colony_detector.plate import find_plate  # noqa: E402


def _match(labels: np.ndarray, dets: np.ndarray) -> list[tuple[int, int]]:
    if not len(labels) or not len(dets):
        return []
    lc = (labels[:, :2] + labels[:, 2:4]) / 2
    lr = (labels[:, 2] - labels[:, 0] + labels[:, 3] - labels[:, 1]) / 4
    d = np.linalg.norm(lc[:, None] - dets[None, :, :2], axis=2)
    tol = np.maximum(8.0, 0.8 * np.maximum(lr[:, None], dets[None, :, 2]))
    cost = np.where(d <= tol, d, 1e6)
    rows, cols = linear_sum_assignment(cost)
    return [(r, c) for r, c in zip(rows, cols) if cost[r, c] < 1e6]


def prepare(args) -> int:
    cache = pickle.loads(Path(args.cache).read_bytes())
    splits = set(args.split.split(","))
    samples = [s for s in load_dataset(args.dataset) if s.split in splits and s.boxes is not None]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plates, review = {}, []
    for s in samples:
        if s.image.name not in cache:
            continue
        img = read_image(s.image)
        plate = find_plate(img)
        px_mm = 2.0 * plate.radius / args.dish_mm
        L = np.array([b[:4] for b in s.boxes], float).reshape(-1, 4)
        D = np.array([(x, y, r, c) for x, y, r, c in cache[s.image.name]["dets"] if c >= args.min_conf],
                     float).reshape(-1, 4)
        pairs = _match(L, D)
        ml, md = {a for a, _ in pairs}, {b for _, b in pairs}
        auto = []
        for a, b in pairs:
            x0, y0, x1, y1 = L[a]
            size = ((x1 - x0) + (y1 - y0)) / 2
            item = {"x": (x0 + x1) / 2, "y": (y0 + y1) / 2, "r": size / 2, "mm": size / px_mm,
                    "source": "both", "conf": float(D[b, 3])}
            if item["mm"] > args.huge_mm:
                review.append({**item, "plate": s.image.name, "kind": "huge"})
            else:
                auto.append(item)
        for a in set(range(len(L))) - ml:
            x0, y0, x1, y1 = L[a]
            size = ((x1 - x0) + (y1 - y0)) / 2
            item = {"x": (x0 + x1) / 2, "y": (y0 + y1) / 2, "r": size / 2, "mm": size / px_mm, "source": "label"}
            if item["mm"] >= args.review_mm:
                review.append({**item, "plate": s.image.name, "kind": "label_only"})
        for b in set(range(len(D))) - md:
            x, y, r, c = D[b]
            item = {"x": x, "y": y, "r": r, "mm": 2 * r / px_mm, "source": "model", "conf": float(c)}
            if item["mm"] >= args.review_mm:
                review.append({**item, "plate": s.image.name, "kind": "model_only"})
        plates[s.image.name] = {"path": str(s.image), "px_per_mm": px_mm, "auto": auto,
                                "original_count": s.total, "split": s.split}
    for i, r in enumerate(review):
        r["id"] = i
    (out / "candidates.json").write_text(json.dumps({"plates": plates, "review": review,
                                                    "min_mm": args.min_mm, "review_mm": args.review_mm}))
    _sheets(review, plates, out)
    print(f"{len(plates)} plates, {sum(len(p['auto']) for p in plates.values())} auto-accepted, "
          f"{len(review)} to review -> {out}")
    return 0


def _sheets(review: list[dict], plates: dict, out: Path, per_sheet: int = 40, tile: int = 190) -> None:
    colour = {"label_only": (255, 160, 0), "model_only": (0, 0, 255), "huge": (255, 0, 255)}
    cache: dict[str, np.ndarray] = {}
    for start in range(0, len(review), per_sheet):
        ims = []
        for r in review[start:start + per_sheet]:
            img = cache.get(r["plate"])
            if img is None:
                cache.clear()
                img = cache[r["plate"]] = read_image(plates[r["plate"]]["path"])
            h = int(max(40, 3 * r["r"]))
            x, y = int(r["x"]), int(r["y"])
            crop = img[max(0, y - h):y + h, max(0, x - h):x + h].copy()
            cx, cy = x - max(0, x - h), y - max(0, y - h)
            cv2.circle(crop, (cx, cy), int(r["r"] + max(4, 0.15 * h)), colour[r["kind"]], max(1, h // 40))
            crop = cv2.resize(crop, (tile, tile), interpolation=cv2.INTER_CUBIC)
            label = f"#{r['id']} {r['mm']:.1f}mm"
            if "conf" in r:
                label += f" {r['conf']:.2f}"
            cv2.rectangle(crop, (0, 0), (tile, 18), (0, 0, 0), -1)
            cv2.putText(crop, label, (3, 13), 0, 0.45, (255, 255, 255), 1)
            ims.append(crop)
        while len(ims) % 8:
            ims.append(np.zeros((tile, tile, 3), np.uint8))
        sheet = np.vstack([np.hstack(ims[i:i + 8]) for i in range(0, len(ims), 8)])
        cv2.imwrite(str(out / f"sheet_{start // per_sheet:02d}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])


def _load_decisions(out: Path) -> dict[str, str]:
    path = out / "decisions.json"
    files = [path] if path.exists() else sorted(out.glob("dec_*.json"))
    merged: dict[str, str] = {}
    for f in files:
        merged.update(json.loads(f.read_text()))
    if not path.exists():
        path.write_text(json.dumps(merged, indent=0, sort_keys=True))
    return merged


def apply(args) -> int:
    out = Path(args.out)
    data = json.loads((out / "candidates.json").read_text())
    raw = _load_decisions(out)
    decisions = {int(k): v for k, v in raw.items() if k.isdigit()}
    auto_fix = {k[5:]: v for k, v in raw.items() if k.startswith("auto:")}
    plate_flag = {k[6:]: v for k, v in raw.items() if k.startswith("plate:")}
    min_mm, review_mm = data["min_mm"], data["review_mm"]
    gold = Path(args.gold)
    gold.mkdir(parents=True, exist_ok=True)
    per_plate = {name: {"colonies": [], "small": [], "unsure": []} for name in data["plates"]}
    for name, p in data["plates"].items():
        for i, c in enumerate(p["auto"]):
            fix = auto_fix.get(f"{name}:{i}")
            if fix == "not":
                continue
            if fix in ("unsure", "small"):
                per_plate[name][fix].append(c)
                continue
            bucket = "colonies" if c["mm"] >= min_mm else "small" if c["mm"] >= review_mm else None
            if bucket:
                per_plate[name][bucket].append(c)
    missing = [r["id"] for r in data["review"] if r["id"] not in decisions]
    if missing:
        print(f"no decision for {len(missing)} candidates (treated as 'not'): {missing[:20]}")
    for r in data["review"]:
        d = decisions.get(r["id"], "not")
        if d == "colony":
            per_plate[r["plate"]]["colonies" if r["mm"] >= min_mm else "small"].append(r)
        elif d == "small":
            per_plate[r["plate"]]["small"].append(r)
        elif d == "unsure":
            per_plate[r["plate"]]["unsure"].append(r)
    total = changed = 0
    for name, v in per_plate.items():
        pts = [[round(c["x"], 1), round(c["y"], 1), round(c["r"], 1)] for c in v["colonies"]]
        (gold / f"{Path(name).stem}.json").write_text(json.dumps({
            "image": name, "count": len(pts), "points": pts,
            "small_specks": [[round(c["x"], 1), round(c["y"], 1), round(c["r"], 1)] for c in v["small"]],
            "unsure": [[round(c["x"], 1), round(c["y"], 1), round(c["r"], 1)] for c in v["unsure"]],
            "original_count": data["plates"][name]["original_count"],
            "px_per_mm": round(data["plates"][name]["px_per_mm"], 3),
            "plate_flag": next((v for k, v in plate_flag.items() if name.startswith(k)), None),
            "rule": {"min_colony_mm": min_mm, "review_colony_mm": review_mm},
        }, indent=1))
        total += 1
        changed += len(pts) != data["plates"][name]["original_count"]
    print(f"wrote {total} gold files to {gold}; {changed} plates have a different count than the original labels")
    return 0


def _plate_order(data: dict) -> list[str]:
    """Plates in the order their candidates were numbered."""
    first: dict[str, int] = {}
    for r in data["review"]:
        first.setdefault(r["plate"], r["id"])
    return sorted(first, key=first.get)


def web(args) -> int:
    """Data for tools/review_page/index.html: plate photos with numbered
    rings, the crop sheets (used as sprites) and review.json."""
    import shutil

    out = Path(args.out)
    data = json.loads((out / "candidates.json").read_text())
    names = _plate_order(data)
    index = {n: i for i, n in enumerate(names)}
    (out / "ov").mkdir(exist_ok=True)
    colour = {"label_only": (255, 143, 30), "model_only": (42, 48, 224), "huge": (214, 58, 214)}
    plates = []
    for n in names:
        p = data["plates"][n]
        img = read_image(p["path"])
        for c in p["auto"]:
            cv2.circle(img, (int(c["x"]), int(c["y"])), int(c["r"] + 8), (76, 177, 34), 7)
        for r in data["review"]:
            if r["plate"] != n:
                continue
            x, y, rr = int(r["x"]), int(r["y"]), int(r["r"] + 12)
            cv2.circle(img, (x, y), rr, colour[r["kind"]], 9)
            for col, th in (((0, 0, 0), 16), ((255, 255, 255), 6)):
                cv2.putText(img, str(r["id"]), (x + rr + 6, y + 18), 0, 2.6, col, th)
        h, w = img.shape[:2]
        s = args.photo_px / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        rel = f"ov/p{index[n]:04d}.jpg"
        cv2.imwrite(str(out / rel), img, [cv2.IMWRITE_JPEG_QUALITY, 82])
        plates.append({"name": n, "split": p.get("split", ""), "orig": p["original_count"], "auto": len(p["auto"]),
                       "px_mm": p["px_per_mm"], "ov": rel})
    per, cols = 40, 8
    sheets, items = {}, []
    for r in data["review"]:
        k = r["id"] // per
        sheet = f"sheet_{k:02d}.jpg"
        n_on = min(per, len(data["review"]) - k * per)
        sheets[sheet] = {"cols": cols, "rows": -(-n_on // cols)}
        j = r["id"] % per
        items.append({"id": r["id"], "p": index[r["plate"]], "kind": r["kind"], "mm": round(r["mm"], 2),
                      "conf": round(r["conf"], 2) if "conf" in r else None,
                      "sheet": sheet, "col": j % cols, "row": j // cols})
    (out / "review.json").write_text(json.dumps({"rule": {"min_mm": data["min_mm"], "review_mm": data["review_mm"]},
                                                 "plates": plates, "sheets": sheets, "items": items}))
    shutil.copy(Path(__file__).resolve().parent / "review_page" / "index.html", out / "index.html")
    print(f"{len(plates)} plates, {len(items)} spots -> {out}/index.html")
    return 0


def from_web(args) -> int:
    """Shared decisions of the review page (the `plates` docs, as a JSON list
    of {k, calls, flag}) -> decisions.json for `apply`."""
    out = Path(args.out)
    data = json.loads((out / "candidates.json").read_text())
    names = _plate_order(data)
    docs = json.loads(Path(args.docs).read_text())
    decisions: dict[str, str] = {}
    for d in docs:
        decisions.update({str(k): v for k, v in (d.get("calls") or {}).items()})
        if d.get("flag") in ("tntc", "overgrown"):
            decisions[f"plate:{names[int(d['k'])]}"] = d["flag"]
    (out / "decisions.json").write_text(json.dumps(decisions, indent=0, sort_keys=True))
    print(f"{len(decisions)} decisions -> {out / 'decisions.json'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--dataset", required=True)
    p.add_argument("--split", default="test", help="comma list, e.g. train,valid")
    p.add_argument("--cache", required=True, help="pickle {image name: {'dets': [(x, y, r, conf), ...]}}")
    p.add_argument("--out", required=True)
    p.add_argument("--min-conf", type=float, default=0.3)
    p.add_argument("--min-mm", type=float, default=0.5)
    p.add_argument("--review-mm", type=float, default=0.3)
    p.add_argument("--huge-mm", type=float, default=25.0)
    p.add_argument("--dish-mm", type=float, default=90.0)
    a = sub.add_parser("apply")
    a.add_argument("--out", required=True)
    a.add_argument("--gold", required=True)
    w = sub.add_parser("web", help="build the click-through review page")
    w.add_argument("--out", required=True)
    w.add_argument("--photo-px", type=int, default=1100)
    f = sub.add_parser("from-web", help="review page decisions -> decisions.json")
    f.add_argument("--out", required=True)
    f.add_argument("--docs", required=True, help="JSON list of the page's plates documents")
    args = ap.parse_args()
    return {"prepare": prepare, "apply": apply, "web": web, "from-web": from_web}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
