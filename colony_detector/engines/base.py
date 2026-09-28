"""Engine interface.

An engine turns a plate image into a list of :class:`Detection`. Engines are
combined by :mod:`colony_detector.fusion`, so each one only has to be good at
finding colonies; it does not decide the final count on its own.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from colony_detector.plate import Plate
from colony_detector.types import Detection


class EngineUnavailable(RuntimeError):
    """The engine's optional dependency or weights are missing."""


class Engine(ABC):
    name: str = "engine"
    # "mask" engines return real outlines; "box" engines return rectangles.
    kind: str = "mask"
    # Vote weight in fusion.
    weight: float = 1.0

    def load(self) -> None:  # noqa: B027 - optional hook
        """Load weights. Called once, before the first ``detect``."""

    @abstractmethod
    def detect(self, bgr: np.ndarray, plate: Plate, ctx: dict) -> list[Detection]:
        """Detect colonies. ``ctx`` is shared across engines for one image:
        earlier engines may leave hints (typical colony radius, agar ROI)."""

    def version(self) -> str:
        return self.name


def file_digest(path: str | Path, n: int = 8) -> str:
    h = hashlib.sha256()
    p = Path(path)
    if p.is_dir():
        for f in sorted(p.rglob("*")):
            if f.is_file():
                h.update(f.read_bytes())
    else:
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()[:n]
