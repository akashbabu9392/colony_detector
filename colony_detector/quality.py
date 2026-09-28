"""Image quality signals and review rules (TRD sections 9 and 12)."""

from __future__ import annotations

import math

import cv2
import numpy as np

from colony_detector.plate import Plate


def image_quality(bgr: np.ndarray, plate: Plate, roi_radius: float | None = None) -> dict:
    """Focus and glare measured on the agar only."""
    h, w = bgr.shape[:2]
    s = min(1.0, 1200.0 / max(h, w))
    small = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else bgr
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    r = (roi_radius or plate.radius * 0.95) * s
    mask = np.zeros(gray.shape, np.uint8)
    cv2.circle(mask, (int(plate.cx * s), int(plate.cy * s)), max(1, int(r)), 1, -1)
    m = mask.astype(bool)
    if m.sum() < 100:
        m = np.ones_like(m)

    # Focus: high-percentile Laplacian response (colony edges), robust to
    # plates that are mostly empty agar. ~0.5 is a usable phone photo.
    lap = np.abs(cv2.Laplacian(cv2.GaussianBlur(gray, (3, 3), 0), cv2.CV_32F))
    p99 = float(np.percentile(lap[m], 99.5))
    focus = 1.0 - math.exp(-p99 / 25.0)

    # Glare: saturated specular highlights, i.e. clipped pixels well above
    # the agar's own brightness (a near-white agar is not glare).
    agar = float(np.median(gray[m]))
    glare = float(((gray[m] >= 250) & (gray[m] > agar + 25)).mean())
    return {"focus_score": round(focus, 3), "glare_score": round(glare, 4)}


def review(total: int, detections_conf: list[float], quality: dict, plate: Plate,
           coverage: float, tntc_limit: int, min_focus: float, max_glare: float,
           low_conf: float) -> tuple[dict, bool]:
    """Returns (confidence block, tntc flag)."""
    reasons: list[str] = []
    overgrowth = coverage > 0.35
    tntc = total > tntc_limit or overgrowth
    if not plate.found:
        reasons.append("PLATE_NOT_FOUND")
    if quality["focus_score"] < min_focus:
        reasons.append("LOW_FOCUS")
    if quality["glare_score"] > max_glare:
        reasons.append("GLARE")
    if overgrowth:
        reasons.append("OVERGROWTH")
    if total > tntc_limit:
        reasons.append("TNTC")
    mean_conf = float(np.mean(detections_conf)) if detections_conf else 1.0
    low = sum(1 for c in detections_conf if c < low_conf)
    if detections_conf and low / len(detections_conf) > 0.25:
        reasons.append("MANY_LOW_CONFIDENCE")

    penalty = 1.0
    penalty *= 0.85 if "PLATE_NOT_FOUND" in reasons else 1.0
    penalty *= 0.8 if "LOW_FOCUS" in reasons else 1.0
    penalty *= 0.85 if "GLARE" in reasons else 1.0
    penalty *= 0.6 if tntc else 1.0
    overall = round(mean_conf * penalty, 3)
    return {
        "overall_score": overall,
        "needs_review": bool(reasons),
        "reason_codes": reasons,
    }, tntc
