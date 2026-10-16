"""Annotated image rendering (TRD section 10)."""

from __future__ import annotations

import cv2
import numpy as np

from colony_detector.plate import Plate
from colony_detector.types import FUZZY, Detection

_COLORS = {"colony": (255, 115, 60), FUZZY: (255, 115, 60)}  # BGR: vibrant royal blue matching screenshot
_CLUMP = (255, 115, 60)


def render(bgr: np.ndarray, detections: list[Detection], plate: Plate | None,
           total: int, label_indices: bool = False) -> np.ndarray:
    img = bgr.copy()
    h, w = img.shape[:2]
    # Line width tuned to a clean, balanced sweet spot
    lw = max(2, int(round(max(h, w) / 500)))

    for i, d in enumerate(detections, start=1):
        color = _COLORS.get(d.class_name, (255, 115, 60))
        pts = d.polygon().astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], True, color, lw, cv2.LINE_AA)
        if label_indices or d.count > 1:
            text = f"{i}" if d.count == 1 else f"x{d.count}"
            org = (int(d.cx + d.radius + 2), int(d.cy - d.radius))
            fs = max(0.4, max(h, w) / 3000)
            cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), lw + 1, cv2.LINE_AA)
            cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), max(1, lw - 1), cv2.LINE_AA)

    return img

