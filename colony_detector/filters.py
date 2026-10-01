"""Shape checks applied to detections after the detector.

Detectors trained on plate photos still fire on scratches in the agar, lint
and fibres: thin, elongated objects. Colonies are compact (roughly round,
or lobed for spreading growth). ``elongation`` isolates the object under a
detection and measures how long and thin it is.
"""

from __future__ import annotations

import cv2
import numpy as np

from colony_detector.types import Detection


def elongation(bgr: np.ndarray, det: Detection, margin: float = 0.6) -> float:
    """Length / width of the object at the detection's centre (1 = round).

    The object is separated from the agar by its colour distance (Lab) to
    the window's border, which is assumed to be agar. Returns 1.0 when no
    object can be isolated (no evidence of a streak).
    """
    h, w = bgr.shape[:2]
    r = max(det.radius, 3.0) * (1.0 + margin)
    x0, x1 = int(max(0, det.cx - r)), int(min(w, det.cx + r + 1))
    y0, y1 = int(max(0, det.cy - r)), int(min(h, det.cy + r + 1))
    win = bgr[y0:y1, x0:x1]
    if win.shape[0] < 7 or win.shape[1] < 7:
        return 1.0
    lab = cv2.cvtColor(win, cv2.COLOR_BGR2LAB).astype(np.float32)
    border = np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])
    dist = np.linalg.norm(lab - np.median(border, axis=0), axis=2)
    dist = cv2.GaussianBlur(dist, (0, 0), 1.0)
    d8 = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, labels = cv2.connectedComponents(mask)
    if n <= 1:
        return 1.0
    cy, cx = int(det.cy - y0), int(det.cx - x0)
    cy, cx = min(max(cy, 0), mask.shape[0] - 1), min(max(cx, 0), mask.shape[1] - 1)
    lab_id = labels[cy, cx]
    if lab_id == 0:
        # Centre fell on background: take the component closest to it.
        ys, xs = np.nonzero(labels)
        k = int(np.argmin((ys - cy) ** 2 + (xs - cx) ** 2))
        lab_id = labels[ys[k], xs[k]]
    pts = np.column_stack(np.nonzero(labels == lab_id))[:, ::-1].astype(np.float32)
    if len(pts) < 5:
        return 1.0
    (_, _), (a, b), _ = cv2.minAreaRect(pts)
    lo, hi = sorted((a, b))
    return float(hi / max(lo, 1.0))
