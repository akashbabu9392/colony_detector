"""Accuracy regression tests on synthetic plates with exact ground truth.

Seeds here are disjoint from the ones used while tuning, so these are
held-out numbers. Thresholds sit a little below measured performance; a
drop below them means a real regression.
"""

import math

import numpy as np
import pytest

from colony_detector.metrics import count_metrics, detection_metrics, match_points
from colony_detector.plate import find_plate
from colony_detector.synth import SynthConfig, make_plate


def _pred(res):
    return [{"x": d.cx, "y": d.cy, "r": d.radius, "count": d.count} for d in res.detections]


def test_heldout_accuracy(counter):
    true, pred = [], []
    tp = fp = fn = 0
    for seed in range(500, 516):
        img, gt = make_plate(seed)
        res = counter.count(img)
        a, b, c = match_points(gt["colonies"], _pred(res))
        tp, fp, fn = tp + a, fp + b, fn + c
        true.append(gt["count"])
        pred.append(res.total)
    cm = count_metrics(true, pred)
    dm = detection_metrics(tp, fp, fn)
    print(cm, dm)
    assert dm["precision"] >= 0.93
    assert dm["recall"] >= 0.82
    assert dm["f1"] >= 0.88
    assert cm["mape_pct"] <= 20


@pytest.mark.parametrize("style", ["transmitted", "reflected"])
def test_sparse_plates_are_exact_or_near(counter, style):
    """Well separated colonies, no touching: the count should be within 1."""
    cfg = SynthConfig(touching_frac=0.0, fuzzy_frac=0.0, n_min=10, n_max=40)
    for seed in range(3):
        img, gt = make_plate(700 + seed, cfg, style=style)
        res = counter.count(img)
        assert abs(res.total - gt["count"]) <= max(1, round(0.05 * gt["count"])), (seed, res.total, gt["count"])


def test_empty_plate_counts_zero(counter):
    cfg = SynthConfig(bubbles=(0, 0), pen_marks=(0, 0), dust=(0, 0))
    img, _ = make_plate(11, cfg, n=0)
    res = counter.count(img, annotate=True)
    assert res.total == 0
    # TRD: no annotated image for a zero count.
    assert res.annotated is None
    assert "annotated_image" not in res.to_response(include_image=True)


def test_artifacts_are_not_counted(counter):
    """Bubbles, pen strokes and dust on an otherwise empty plate."""
    cfg = SynthConfig(bubbles=(4, 4), pen_marks=(1, 1), dust=(20, 20))
    worst = 0
    for seed in range(4):
        img, _ = make_plate(900 + seed, cfg, n=0, style="transmitted")
        worst = max(worst, counter.count(img).total)
    assert worst <= 2


def test_touching_pair_is_split(counter):
    import cv2

    S = 800
    img = np.full((S, S, 3), 235, np.uint8)
    cv2.circle(img, (400, 400), 380, (150, 205, 225), -1, cv2.LINE_AA)
    r = 14
    centres = [(300, 400), (300 + int(1.7 * r), 400), (500, 300), (500, 300 + int(1.8 * r))]
    for x, y in centres:
        cv2.circle(img, (x, y), r, (60, 90, 120), -1, cv2.LINE_AA)
    img = cv2.GaussianBlur(img, (0, 0), 0.8)
    res = counter.count(img)
    assert res.total == 4


def test_plate_detection_is_accurate():
    good = 0
    for seed in range(300, 330):
        img, gt = make_plate(seed)
        p = find_plate(img)
        g = gt["plate"]
        centre_ok = math.hypot(p.cx - g["cx"], p.cy - g["cy"]) < 0.03 * g["r"]
        # Either the dish wall or the agar edge is a correct plate boundary.
        radius_ok = min(abs(p.radius - g["r"]), abs(p.radius - g["agar_r"])) < 0.06 * g["r"]
        good += p.found and centre_ok and radius_ok
    assert good >= 28


def test_high_density_flags_tntc():
    from colony_detector.config import Settings
    from colony_detector.pipeline import ColonyCounter

    s = Settings(engines="classical")
    s.tntc_limit = 50
    c = ColonyCounter(s)
    img, gt = make_plate(42, SynthConfig(n_min=120, n_max=120))
    res = c.count(img)
    assert res.tntc
    assert "TNTC" in res.confidence["reason_codes"]
    assert res.confidence["needs_review"]
