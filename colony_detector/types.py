"""Core data types shared by every engine."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

COLONY = "colony"
FUZZY = "fuzzy_colony"
CLASS_NAMES = (COLONY, FUZZY)


@dataclass
class Detection:
    """One counted object, in absolute pixel coordinates of the input image.

    ``count`` is 1 for a single colony. A merged clump that could not be split
    into individual outlines keeps one outline but carries its estimated
    number of colonies, so totals stay honest without inventing geometry.
    """

    cx: float
    cy: float
    radius: float
    confidence: float
    class_name: str = COLONY
    contour: np.ndarray | None = None  # (N, 2) float, x/y
    source: str = "classical"
    shape_source: str = "contour"  # contour | box | circle | cluster_estimate
    count: int = 1
    extra: dict = field(default_factory=dict)

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        if self.contour is not None and len(self.contour) >= 3:
            x0, y0 = self.contour.min(axis=0)
            x1, y1 = self.contour.max(axis=0)
            return float(x0), float(y0), float(x1), float(y1)
        r = self.radius
        return self.cx - r, self.cy - r, self.cx + r, self.cy + r

    @property
    def area(self) -> float:
        if self.contour is not None and len(self.contour) >= 3:
            x, y = self.contour[:, 0], self.contour[:, 1]
            return float(0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))
        return math.pi * self.radius * self.radius

    def polygon(self, n: int = 24) -> np.ndarray:
        """Outline to draw/return; synthesises a circle for box detections."""
        if self.contour is not None and len(self.contour) >= 3:
            return self.contour
        t = np.linspace(0, 2 * math.pi, n, endpoint=False)
        return np.stack([self.cx + self.radius * np.cos(t), self.cy + self.radius * np.sin(t)], 1)

    def scaled(self, s: float, dx: float = 0.0, dy: float = 0.0) -> "Detection":
        """Map from a working frame to image frame: p_img = p_work / s + d."""
        contour = None
        if self.contour is not None:
            contour = self.contour / s + np.array([dx, dy], dtype=np.float64)
        return Detection(
            cx=self.cx / s + dx,
            cy=self.cy / s + dy,
            radius=self.radius / s,
            confidence=self.confidence,
            class_name=self.class_name,
            contour=contour,
            source=self.source,
            shape_source=self.shape_source,
            count=self.count,
            extra=dict(self.extra),
        )
