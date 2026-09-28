"""Accuracy metrics for colony counting.

Count metrics are what the lab reports (and what the TRD's CFU number is
judged on). Detection metrics check that the count is right for the right
reasons: each true colony matched to exactly one detection.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import linear_sum_assignment


def match_points(gt: list[dict], pred: list[dict], tol_frac: float = 0.8,
                 tol_min: float = 4.0) -> tuple[int, int, int]:
    """One-to-one matching of predicted centres to true centres.

    A prediction matches a true colony when its centre lies within
    ``max(tol_min, tol_frac * r_true)`` px. Predictions representing a clump
    (``count`` > 1) may match that many true colonies.
    Returns (true positives, false positives, false negatives).
    """
    if not gt:
        return 0, int(sum(p.get("count", 1) for p in pred)), 0
    if not pred:
        return 0, 0, len(gt)
    # Expand clumps into slots.
    slots = []
    for p in pred:
        slots.extend([p] * int(p.get("count", 1)))
    g = np.array([[c["x"], c["y"]] for c in gt], np.float64)
    s = np.array([[p["x"], p["y"]] for p in slots], np.float64)
    tol = np.array([max(tol_min, tol_frac * c.get("r", 5.0)) for c in gt])
    # Clump slots may be anywhere within the clump's radius.
    slack = np.array([p.get("r", 0.0) if p.get("count", 1) > 1 else 0.0 for p in slots])
    d = np.linalg.norm(g[:, None, :] - s[None, :, :], axis=2)
    allowed = d <= (tol[:, None] + slack[None, :])
    cost = np.where(allowed, d, 1e6)
    rows, cols = linear_sum_assignment(cost)
    tp = int(sum(allowed[r, c] for r, c in zip(rows, cols)))
    return tp, len(slots) - tp, len(gt) - tp


def count_metrics(true_counts: list[int], pred_counts: list[int]) -> dict:
    t = np.asarray(true_counts, np.float64)
    p = np.asarray(pred_counts, np.float64)
    if len(t) == 0:
        return {}
    err = p - t
    ape = np.abs(err) / np.maximum(t, 1.0)
    return {
        "n_plates": int(len(t)),
        "mae": round(float(np.mean(np.abs(err))), 3),
        "rmse": round(float(math.sqrt(np.mean(err ** 2))), 3),
        "bias": round(float(np.mean(err)), 3),
        "mape_pct": round(float(np.mean(ape) * 100), 2),
        "within_5pct": round(float(np.mean(ape <= 0.05)), 3),
        "within_10pct": round(float(np.mean(ape <= 0.10)), 3),
        "exact": round(float(np.mean(err == 0)), 3),
    }


def detection_metrics(tp: int, fp: int, fn: int) -> dict:
    prec = tp / (tp + fp) if tp + fp else 1.0
    rec = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": round(prec, 4),
            "recall": round(rec, 4), "f1": round(f1, 4)}
