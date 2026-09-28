"""Cellpose instance segmentation engine.

Cellpose (CPnet) models fine-tuned on colony images produce one mask per
colony, which handles touching colonies far better than thresholding. The
MicroID project already holds such a fine-tuned model
(``colony_project_model``); point ``CD_CELLPOSE_WEIGHTS`` at it or drop it in
``models/``.

Requires ``cellpose<4`` (the CPnet architecture; Cellpose 4 switched to SAM).
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from colony_detector.engines.base import Engine, EngineUnavailable, file_digest
from colony_detector.plate import Plate
from colony_detector.types import COLONY, Detection


class CellposeEngine(Engine):
    name = "cellpose"
    kind = "mask"
    weight = 1.0

    def __init__(self, weights: str, device: str = "", max_side: int = 1536):
        self.weights = weights
        self.device = device
        self.max_side = max_side
        self.model = None

    def version(self) -> str:
        return f"cellpose:{file_digest(self.weights)}"

    def load(self) -> None:
        try:
            from cellpose import models
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise EngineUnavailable("cellpose is not installed (pip install 'cellpose<4')") from exc
        use_gpu = self.device.startswith("cuda")
        self.model = models.CellposeModel(gpu=use_gpu, pretrained_model=self.weights)

    def detect(self, bgr: np.ndarray, plate: Plate, ctx: dict) -> list[Detection]:
        if self.model is None:
            self.load()
        H, W = bgr.shape[:2]
        R = plate.radius
        x0, y0 = max(0, int(plate.cx - R)), max(0, int(plate.cy - R))
        x1, y1 = min(W, int(plate.cx + R)), min(H, int(plate.cy + R))
        crop = bgr[y0:y1, x0:x1]
        s = min(1.0, self.max_side / max(crop.shape[:2]))
        work = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else crop
        rgb = cv2.cvtColor(work, cv2.COLOR_BGR2RGB)

        # Use the classical engine's colony size estimate when available so
        # Cellpose rescales the image to the size it was trained on.
        diameter = None
        typ = ctx.get("typical_radius_px")
        if typ:
            diameter = max(6.0, 2.0 * typ * s)

        masks, flows, _ = self.model.eval(rgb, diameter=diameter, channels=[0, 0])[:3]
        cellprob = flows[2] if len(flows) > 2 else None

        roi_r = ctx.get("roi_radius_px", 0.97 * R)
        roi_c = ctx.get("roi_center", (plate.cx, plate.cy))
        dets: list[Detection] = []
        for lab in range(1, int(masks.max()) + 1):
            m = (masks == lab).astype(np.uint8)
            area = int(m.sum())
            if area < 6:
                continue
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            cnt = max(cnts, key=cv2.contourArea)
            mo = cv2.moments(m, binaryImage=True)
            cx, cy = mo["m10"] / mo["m00"], mo["m01"] / mo["m00"]
            gx, gy = cx / s + x0, cy / s + y0
            if math.hypot(gx - roi_c[0], gy - roi_c[1]) > roi_r:
                continue
            conf = 0.8
            if cellprob is not None:
                p = float(np.mean(cellprob[m.astype(bool)]))
                conf = float(1.0 / (1.0 + math.exp(-p)))
            pts = cnt[:, 0, :].astype(np.float64) / s + np.array([x0, y0], dtype=np.float64)
            dets.append(Detection(
                cx=gx, cy=gy, radius=math.sqrt(area / math.pi) / s, confidence=conf,
                class_name=COLONY, contour=pts, source=self.name,
            ))
        return dets
