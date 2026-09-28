"""Petri-dish localisation.

Finds the dish as a circle so that the table, labels, the dish wall and
everything outside the agar never reach the counter.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

_DETECT_DIM = 800


@dataclass
class Plate:
    cx: float
    cy: float
    radius: float
    found: bool
    method: str
    support: float = 0.0

    def mask(self, shape: tuple[int, int], radius: float | None = None) -> np.ndarray:
        h, w = shape[:2]
        m = np.zeros((h, w), np.uint8)
        r = self.radius if radius is None else radius
        cv2.circle(m, (int(round(self.cx)), int(round(self.cy))), max(int(round(r)), 1), 255, -1)
        return m

    def contains(self, x: float, y: float, radius: float | None = None) -> bool:
        r = self.radius if radius is None else radius
        return math.hypot(x - self.cx, y - self.cy) <= r

    def as_dict(self) -> dict:
        return {
            "found": self.found,
            "method": self.method,
            "center": [round(self.cx, 1), round(self.cy, 1)],
            "radius_px": round(self.radius, 1),
            "support": round(self.support, 3),
        }


def _edge_support(edges: np.ndarray, cx: float, cy: float, r: float) -> float:
    """Fraction of the circle (inside the frame) that lies on an edge."""
    h, w = edges.shape
    t = np.linspace(0, 2 * math.pi, 540, endpoint=False)
    xs = np.round(cx + r * np.cos(t)).astype(int)
    ys = np.round(cy + r * np.sin(t)).astype(int)
    inside = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    if inside.mean() < 0.6:
        return 0.0
    return float(edges[ys[inside], xs[inside]].mean() / 255.0) * float(inside.mean())


def _hough_candidates(gray: np.ndarray) -> list[tuple[float, float, float]]:
    h, w = gray.shape
    short, long_ = min(h, w), max(h, w)
    out: list[tuple[float, float, float]] = []
    for method, p1, p2 in (
        (cv2.HOUGH_GRADIENT_ALT, 150, 0.75),
        (cv2.HOUGH_GRADIENT_ALT, 60, 0.6),
        (cv2.HOUGH_GRADIENT, 100, 40),
    ):
        circles = cv2.HoughCircles(
            gray,
            method,
            dp=1.5,
            minDist=short / 8,
            param1=p1,
            param2=p2,
            minRadius=int(short * 0.22),
            maxRadius=int(long_ * 0.62),
        )
        if circles is not None:
            out.extend((float(x), float(y), float(r)) for x, y, r in circles[0][:12])
    return out


def _round_regions(mask: np.ndarray) -> list[tuple[float, float, float]]:
    out = []
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:2]:
        area = cv2.contourArea(c)
        (x, y), r = cv2.minEnclosingCircle(c)
        if r > 0 and area / (math.pi * r * r) > 0.75:
            out.append((float(x), float(y), float(r)))
    return out


def _contour_candidate(gray: np.ndarray, small_bgr: np.ndarray) -> list[tuple[float, float, float]]:
    """Largest roughly circular region, by brightness (Otsu) and by colour
    difference from the table around it (catches dark dishes on dark tables,
    where edges are too weak for Hough)."""
    out = []
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    for mask in (th, 255 - th):
        out += _round_regions(mask)
    lab = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    border = np.concatenate([lab[:4].reshape(-1, 3), lab[-4:].reshape(-1, 3),
                             lab[:, :4].reshape(-1, 3), lab[:, -4:].reshape(-1, 3)])
    dist = np.linalg.norm(lab - np.median(border, axis=0), axis=2)
    # Colonies are far more different from the table than the agar is; clip
    # so Otsu separates dish from table rather than colonies from the rest.
    dist = np.minimum(dist, np.percentile(dist, 75))
    dist = cv2.GaussianBlur(dist, (0, 0), 4)
    d8 = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, th = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    out += _round_regions(th)
    return out


def _outermost_ring(edges, cx, cy, r, support, w, h) -> float:
    """Largest concentric ring (up to +25%) with comparable edge support.
    Angled photos show the meniscus reflection well inside the dish wall,
    and colonies grow all the way out to the wall."""
    best = r
    for grow in np.arange(1.005, 1.25, 0.005):
        rr = r * grow
        if rr - min(cx, cy, w - cx, h - cy) > 0.05 * rr:
            break
        if _edge_support(edges, cx, cy, rr) >= 0.5 * support:
            best = rr
    return best


def find_plate(bgr: np.ndarray) -> Plate:
    """Locate the dish. Falls back to the inscribed circle of the frame,
    which is exactly right for MicroID's pre-cropped disk images."""
    h, w = bgr.shape[:2]
    s = min(1.0, _DETECT_DIM / max(h, w))
    small = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else bgr
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # Stretch contrast so a dark dish on a dark table still has edges.
    lo, hi = np.percentile(gray, (1, 99))
    if hi - lo < 120:
        gray = np.clip((gray.astype(np.float32) - lo) * (200.0 / max(hi - lo, 1.0)) + 20, 0, 255).astype(np.uint8)
    sh, sw = gray.shape

    edges = cv2.Canny(gray, 30, 90)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8))

    fallback = Plate(w / 2.0, h / 2.0, min(w, h) / 2.0 * 0.98, False, "frame")

    hough = _hough_candidates(gray)
    regions = _contour_candidate(gray, small)
    best, best_score = None, 0.0
    short = min(sh, sw)
    for cx, cy, r in hough + regions:
        # Dish must be mostly in frame and reasonably central.
        if r < 0.22 * short:
            continue
        overflow = max(0.0, r - min(cx, cy, sw - cx, sh - cy))
        if overflow > 0.12 * r:
            continue
        support = _edge_support(edges, cx, cy, r)
        score = support * math.sqrt(r / (0.5 * short))
        if score > best_score:
            best, best_score = (cx, cy, r, support), score

    if best is None or best[3] < 0.35:
        # Edges too weak for a confident circle: accept a large, round,
        # central region that differs from the table in colour/brightness.
        ok = [(cx, cy, r) for cx, cy, r in regions
              if r >= 0.3 * short and r - min(cx, cy, sw - cx, sh - cy) <= 0.12 * r]
        if not ok:
            return fallback
        cx, cy, r = max(ok, key=lambda c: c[2])
        return Plate(float(cx / s), float(cy / s), float(r / s), True, "region", 0.0)

    cx, cy, r, support = best
    # Dishes show several concentric rings (lid, base wall, meniscus) and the
    # Hough winner is often an inner one. Take the outermost ring that is
    # still well supported; the counter finds the agar edge inside it.
    r_out = _outermost_ring(edges, cx, cy, r, support, sw, sh)
    return Plate(float(cx / s), float(cy / s), float(r_out / s), True, "hough", float(support))
