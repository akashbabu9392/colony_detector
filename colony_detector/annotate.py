"""Annotated image rendering (TRD section 10)."""

from __future__ import annotations

import cv2
import numpy as np

from colony_detector.plate import Plate
from colony_detector.types import FUZZY, Detection

_COLORS = {"colony": (50, 125, 46), FUZZY: (37, 168, 249)}  # BGR, matches MicroID UI
_CLUMP = (40, 40, 220)


def render(bgr: np.ndarray, detections: list[Detection], plate: Plate | None,
           total: int, label_indices: bool = True) -> np.ndarray:
    img = bgr.copy()
    h, w = img.shape[:2]
    lw = max(1, int(round(max(h, w) / 700)))
    fs = max(0.35, max(h, w) / 2400)

    if plate is not None:
        cv2.circle(img, (int(plate.cx), int(plate.cy)), int(plate.radius), (200, 120, 0), lw, cv2.LINE_AA)

    for i, d in enumerate(detections, start=1):
        color = _CLUMP if d.count > 1 else _COLORS.get(d.class_name, (160, 160, 160))
        pts = d.polygon().astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], True, color, lw, cv2.LINE_AA)
        if label_indices or d.count > 1:
            text = f"{i}" if d.count == 1 else f"{i}(x{d.count})"
            org = (int(d.cx + d.radius + 2), int(d.cy - d.radius))
            cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), lw + 2, cv2.LINE_AA)
            cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), lw, cv2.LINE_AA)

    label = f"CFU: {total}"
    scale = fs * 2.2
    (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, lw * 2)
    cv2.rectangle(img, (8, 8), (8 + tw + 16, 8 + th + base + 16), (0, 0, 0), -1)
    cv2.putText(img, label, (16, 16 + th), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), lw * 2, cv2.LINE_AA)
    return img
