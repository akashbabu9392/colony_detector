"""Consensus fusion of several engines.

Each engine votes for colonies; detections from different engines that sit
on the same spot are one colony. A colony is counted when the weighted vote

    score = sum(weight_e * confidence_e for engines that saw it) / sum(weight_e)

reaches ``threshold``. Agreement therefore raises confidence, and an object
seen by only one engine must be seen with conviction. Outlines come from the
best mask engine that saw the colony, so the overlay stays precise even when a
box detector contributed the vote.
"""

from __future__ import annotations

import math
from collections import defaultdict

import cv2
import numpy as np

from colony_detector.types import COLONY, Detection


def _absorb_clumps(results: dict[str, list[Detection]]) -> dict[str, list[Detection]]:
    """Clumps the classical engine could only estimate (count > 1) either get
    replaced by a learned engine's individual colonies, or swallow the
    detections that fall inside them so nothing is double counted."""
    clumps = [(e, d) for e, ds in results.items() for d in ds if d.count > 1 and d.contour is not None]
    if not clumps:
        return results
    out = {e: list(ds) for e, ds in results.items()}
    for eng, clump in clumps:
        poly = clump.contour.astype(np.float32).reshape(-1, 1, 2)
        inside: dict[str, list[Detection]] = defaultdict(list)
        for e, ds in out.items():
            if e == eng:
                continue
            for d in ds:
                if cv2.pointPolygonTest(poly, (float(d.cx), float(d.cy)), False) >= 0:
                    inside[e].append(d)
        if any(len(v) >= 2 for v in inside.values()):
            # A learned engine separated the clump: trust its individuals.
            out[eng] = [d for d in out[eng] if d is not clump]
        else:
            for e, ds in inside.items():
                out[e] = [d for d in out[e] if all(d is not x for x in ds)]
            clump.extra.setdefault("absorbed", sorted(inside))
    return out


def fuse(results: dict[str, list[Detection]], weights: dict[str, float],
         threshold: float = 0.3) -> list[Detection]:
    engines = [e for e in results]
    if not engines:
        return []
    if len(engines) == 1:
        return list(results[engines[0]])

    results = _absorb_clumps(results)
    total_w = sum(weights.get(e, 1.0) for e in engines)
    pool = sorted(((d, e) for e, ds in results.items() for d in ds), key=lambda t: -t[0].confidence)

    # Spatial hash for neighbour lookup.
    radii = [d.radius for d, _ in pool] or [5.0]
    cell = max(4.0, 2.5 * float(np.median(radii)))
    grid: dict[tuple[int, int], list[int]] = defaultdict(list)
    clusters: list[dict] = []

    for d, e in pool:
        gx, gy = int(d.cx // cell), int(d.cy // cell)
        best, best_dist = None, math.inf
        for ix in (gx - 1, gx, gx + 1):
            for iy in (gy - 1, gy, gy + 1):
                for ci in grid.get((ix, iy), ()):
                    cl = clusters[ci]
                    if e in cl["members"]:
                        continue
                    dist = math.hypot(d.cx - cl["cx"], d.cy - cl["cy"])
                    lim = 0.6 * max(d.radius, cl["r"]) + 1.0
                    size_ok = 1 / 3 <= d.radius / max(cl["r"], 1e-6) <= 3
                    if dist <= lim and size_ok and dist < best_dist:
                        best, best_dist = ci, dist
        if best is None:
            clusters.append({"members": {e: d}, "cx": d.cx, "cy": d.cy, "r": d.radius})
            grid[(gx, gy)].append(len(clusters) - 1)
        else:
            clusters[best]["members"][e] = d

    fused: list[Detection] = []
    for cl in clusters:
        members: dict[str, Detection] = cl["members"]
        score = sum(weights.get(e, 1.0) * d.confidence for e, d in members.items()) / total_w
        if score < threshold:
            continue
        # Geometry from the most trusted engine with a real outline.
        masks = [(e, d) for e, d in members.items() if d.contour is not None]
        pick = masks or list(members.items())
        _, rep = max(pick, key=lambda t: weights.get(t[0], 1.0) * t[1].confidence)
        votes: dict[str, float] = defaultdict(float)
        for e, d in members.items():
            votes[d.class_name] += weights.get(e, 1.0) * d.confidence
        cls = max(votes, key=votes.get) if votes else COLONY
        fused.append(Detection(
            cx=rep.cx, cy=rep.cy, radius=rep.radius, confidence=float(min(0.99, score)),
            class_name=cls, contour=rep.contour, source="+".join(sorted(members)),
            shape_source=rep.shape_source, count=max(d.count for d in members.values()),
            extra={"votes": {e: round(d.confidence, 3) for e, d in members.items()}},
        ))
    return fused
