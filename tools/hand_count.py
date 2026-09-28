"""Hand-count plates to build gold-standard labels.

Opens each image with the model's detections pre-filled so you only fix
mistakes, then saves the corrected colony positions to ``eval/gold``.

    python tools/hand_count.py data/plates            # all images in a folder
    python tools/hand_count.py plate1.jpg --empty     # start from zero

Mouse / keys:
    left click      add a colony
    right click     remove the nearest colony
    z               undo
    c               clear all
    + / -           zoom in / out (window scale)
    Enter or s      save and go to next image
    n               skip without saving
    q / Esc         quit

The JSON written per image is what tools/evaluate.py and
tools/build_training_set.py read:
    {"image": "plate1.jpg", "count": 57, "points": [[x, y, r], ...]}
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from colony_detector.imageio import read_image  # noqa: E402

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def label_image(img: np.ndarray, points: list[tuple[float, float, float]]):
    """Interactive editing of one plate. Returns the points to save, None to
    skip, or "quit"."""
    points = list(points)
    default_r = float(np.median([p[2] for p in points])) if points else 6.0
    history: list[list] = []
    h, w = img.shape[:2]
    view = {"scale": min(1.0, 1000 / max(h, w))}
    win = "hand count"
    lw = max(1, int(max(h, w) / 800))

    def redraw():
        vis = img.copy()
        for x, y, r in points:
            cv2.circle(vis, (int(x), int(y)), int(max(r, 3)) + lw, (0, 255, 0), lw, cv2.LINE_AA)
        cv2.putText(vis, f"{len(points)}  (Enter=save, n=skip, q=quit)", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, max(0.8, max(h, w) / 1500), (0, 0, 255), lw * 2)
        sc = view["scale"]
        cv2.imshow(win, cv2.resize(vis, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA))

    def on_mouse(event, x, y, flags, param):
        fx, fy = x / view["scale"], y / view["scale"]
        if event == cv2.EVENT_LBUTTONDOWN:
            history.append(list(points))
            points.append((round(fx, 1), round(fy, 1), round(default_r, 1)))
            redraw()
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            i = int(np.argmin([math.hypot(px - fx, py - fy) for px, py, _ in points]))
            history.append(list(points))
            points.pop(i)
            redraw()

    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, on_mouse)
    redraw()
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key in (13, 10, ord("s")):
            return points
        if key == ord("n"):
            return None
        if key in (ord("q"), 27):
            return "quit"
        if key == ord("z") and history:
            points[:] = history.pop()
        elif key == ord("c"):
            history.append(list(points))
            points.clear()
        elif key in (ord("+"), ord("=")):
            view["scale"] *= 1.25
        elif key == ord("-"):
            view["scale"] /= 1.25
        redraw()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--gold", default="eval/gold")
    ap.add_argument("--empty", action="store_true", help="do not pre-fill with model detections")
    ap.add_argument("--redo", action="store_true", help="re-open images that already have gold")
    args = ap.parse_args()

    images = []
    for p in map(Path, args.inputs):
        images += sorted(f for f in p.rglob("*") if f.suffix.lower() in IMAGE_EXT) if p.is_dir() else [p]
    gold = Path(args.gold)
    gold.mkdir(parents=True, exist_ok=True)

    counter = None
    if not args.empty:
        from colony_detector.pipeline import ColonyCounter

        counter = ColonyCounter()
        counter.load()

    for path in images:
        out = gold / f"{path.stem}.json"
        if out.exists() and not args.redo:
            points = [tuple(p) for p in json.loads(out.read_text()).get("points", [])]
            print(f"{path.name}: already labelled ({len(points)}), use --redo to edit")
            continue
        img = read_image(path)
        points: list[tuple[float, float, float]] = []
        if out.exists():
            points = [tuple(p) for p in json.loads(out.read_text()).get("points", [])]
        elif counter is not None:
            res = counter.count(img)
            for d in res.detections:
                if d.count == 1:
                    points.append((round(d.cx, 1), round(d.cy, 1), round(d.radius, 1)))
                else:  # spread clump estimates around the clump centre
                    for k in range(d.count):
                        a = 2 * math.pi * k / d.count
                        points.append((round(d.cx + 0.5 * d.radius * math.cos(a), 1),
                                       round(d.cy + 0.5 * d.radius * math.sin(a), 1), round(d.radius / 2, 1)))
        result = label_image(img, points)
        if result == "quit":
            break
        if result is not None:
            out.write_text(json.dumps({"image": path.name, "count": len(result),
                                       "points": [list(p) for p in result]}, indent=1))
            print(f"{path.name}: saved {len(result)}")
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
