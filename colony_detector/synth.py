"""Synthetic agar plates with exact ground truth.

Used for regression tests, for measuring the counter with known answers, and
to pre-train tile detectors before fine-tuning on real hand-labelled plates.
Plates vary lighting (transmitted / reflected), agar colour, colony colour,
size, density, touching pairs, blur, noise, and include artifacts that must
*not* be counted: air bubbles, pen marks, dust and a visible dish wall.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

AGARS = [  # BGR
    (150, 205, 225), (120, 180, 215), (170, 215, 235), (90, 130, 200),
    (160, 190, 200), (60, 70, 170), (190, 215, 215), (70, 75, 75),
]


@dataclass
class SynthConfig:
    size: int = 1024
    n_min: int = 5
    n_max: int = 180
    r_min_frac: float = 0.004
    r_max_frac: float = 0.012
    touching_frac: float = 0.15
    fuzzy_frac: float = 0.08
    bubbles: tuple[int, int] = (0, 5)
    pen_marks: tuple[int, int] = (0, 1)
    dust: tuple[int, int] = (0, 25)
    noise: tuple[float, float] = (1.5, 4.0)
    blur: tuple[float, float] = (0.0, 1.0)


def _soft_disk(canvas, alpha, cx, cy, r, color, edge=1.2, core=0.0, wobble=None):
    """Alpha-blend a soft-edged disk (optionally irregular) onto canvas."""
    h, w = canvas.shape[:2]
    pad = int(r * 1.6 + 4)
    x0, x1 = max(0, int(cx) - pad), min(w, int(cx) + pad + 1)
    y0, y1 = max(0, int(cy) - pad), min(h, int(cy) + pad + 1)
    if x0 >= x1 or y0 >= y1:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    dx, dy = xx - cx, yy - cy
    d = np.sqrt(dx * dx + dy * dy)
    rr = r
    if wobble is not None:
        ang = np.arctan2(dy, dx)
        rr = r * (1.0 + sum(a * np.sin(k * ang + p) for k, a, p in wobble))
    a = np.clip((rr - d) / edge + 0.5, 0, 1) * alpha
    col = np.array(color, np.float32)
    patch = canvas[y0:y1, x0:x1]
    if core:
        # Denser centre (colonies are thicker in the middle).
        shade = 1.0 - core * np.clip(1 - d / max(r, 1e-3), 0, 1)
        col_px = col[None, None, :] * shade[..., None]
    else:
        col_px = col[None, None, :]
    patch[:] = patch * (1 - a[..., None]) + col_px * a[..., None]


def make_plate(seed: int, cfg: SynthConfig | None = None, n: int | None = None,
               style: str | None = None, artifacts: bool = True) -> tuple[np.ndarray, dict]:
    cfg = cfg or SynthConfig()
    rng = np.random.default_rng(seed)
    S = cfg.size
    style = style or ("reflected" if rng.random() < 0.25 else "transmitted")

    if style == "reflected":
        table = rng.uniform(20, 70) * np.ones(3)
    else:
        table = rng.uniform(200, 250) * rng.uniform(0.92, 1.0, 3)
    img = np.empty((S, S, 3), np.float32)
    img[:] = table
    cx = S / 2 + rng.uniform(-0.03, 0.03) * S
    cy = S / 2 + rng.uniform(-0.03, 0.03) * S
    R = S * rng.uniform(0.44, 0.48)

    # Dish wall and agar.
    wall = np.clip(table * rng.uniform(0.75, 0.95), 0, 255)
    cv2.circle(img, (int(cx), int(cy)), int(R), wall.tolist(), -1, cv2.LINE_AA)
    agar = np.array(AGARS[rng.integers(len(AGARS))], np.float32)
    if style == "reflected":
        agar = agar * rng.uniform(0.25, 0.45)
    agar = np.clip(agar + rng.normal(0, 8, 3), 0, 255)
    R_agar = R * rng.uniform(0.94, 0.97)
    cv2.circle(img, (int(cx), int(cy)), int(R_agar), agar.tolist(), -1, cv2.LINE_AA)
    cv2.circle(img, (int(cx), int(cy)), int(R * 0.985), (wall * 0.8).tolist(), 2, cv2.LINE_AA)

    # Colonies.
    n = int(n if n is not None else rng.integers(cfg.n_min, cfg.n_max + 1))
    r_mean = S * rng.uniform(cfg.r_min_frac, cfg.r_max_frac)
    if style == "reflected":
        colony_col = np.clip(agar + rng.uniform(70, 140), 0, 255) * np.array([0.8, 1.0, 1.1])
    else:
        dark = rng.uniform(0.35, 0.7)
        tint = rng.uniform(0.7, 1.2, 3)
        colony_col = np.clip(agar * dark * tint, 0, 255)
    colonies: list[dict] = []
    place_r = R_agar * 0.93
    attempts = 0
    while len(colonies) < n and attempts < n * 60:
        attempts += 1
        r = float(np.clip(rng.lognormal(math.log(r_mean), 0.25), 2.5, 2.2 * r_mean))
        if colonies and rng.random() < cfg.touching_frac:
            ref = colonies[rng.integers(len(colonies))]
            ang = rng.uniform(0, 2 * math.pi)
            dist = (ref["r"] + r) * rng.uniform(0.8, 1.0)
            x, y = ref["x"] + dist * math.cos(ang), ref["y"] + dist * math.sin(ang)
            min_sep = 0.75
        else:
            rho = place_r * math.sqrt(rng.random())
            ang = rng.uniform(0, 2 * math.pi)
            x, y = cx + rho * math.cos(ang), cy + rho * math.sin(ang)
            min_sep = 1.15
        if math.hypot(x - cx, y - cy) > place_r - r:
            continue
        if any(math.hypot(x - c["x"], y - c["y"]) < min_sep * (r + c["r"]) for c in colonies):
            continue
        fuzzy = rng.random() < cfg.fuzzy_frac
        colonies.append({"x": x, "y": y, "r": r, "fuzzy": bool(fuzzy)})

    for c in colonies:
        col = np.clip(colony_col * rng.uniform(0.9, 1.1, 3), 0, 255)
        if c["fuzzy"]:
            wob = [(k, rng.uniform(0.05, 0.15), rng.uniform(0, 6.28)) for k in (3, 5, 7)]
            _soft_disk(img, 0.85, c["x"], c["y"], c["r"] * 1.1, col,
                       edge=max(2.0, c["r"] * 0.35), wobble=wob)
        else:
            _soft_disk(img, 0.95, c["x"], c["y"], c["r"], col, edge=rng.uniform(0.8, 1.6),
                       core=rng.uniform(0.0, 0.25))

    arts = {"bubbles": 0, "pen": 0, "dust": 0}
    if artifacts:
        for _ in range(rng.integers(cfg.bubbles[0], cfg.bubbles[1] + 1)):
            br = S * rng.uniform(0.006, 0.02)
            rho = place_r * math.sqrt(rng.random())
            ang = rng.uniform(0, 2 * math.pi)
            bx, by = cx + rho * math.cos(ang), cy + rho * math.sin(ang)
            if style == "reflected":
                cv2.circle(img, (int(bx), int(by)), int(br), np.clip(agar * 0.6, 0, 255).tolist(), -1, cv2.LINE_AA)
                cv2.ellipse(img, (int(bx), int(by)), (int(br), int(br)), 0, 200, 320,
                            np.clip(agar + 150, 0, 255).tolist(), max(1, int(br * 0.2)), cv2.LINE_AA)
            else:
                cv2.circle(img, (int(bx), int(by)), int(br), np.clip(agar * 0.45, 0, 255).tolist(),
                           max(2, int(br * 0.25)), cv2.LINE_AA)
                _soft_disk(img, 0.8, bx, by, br * 0.55, np.clip(agar + 35, 0, 255), edge=2.0)
            arts["bubbles"] += 1
        for _ in range(rng.integers(cfg.pen_marks[0], cfg.pen_marks[1] + 1)):
            pts = [(cx + rng.uniform(-0.6, 0.6) * R, cy + rng.uniform(-0.6, 0.6) * R)]
            for _ in range(rng.integers(2, 5)):
                px, py = pts[-1]
                pts.append((px + rng.uniform(-0.25, 0.25) * R, py + rng.uniform(-0.25, 0.25) * R))
            ink = (90, 30, 20) if rng.random() < 0.5 else (25, 25, 25)
            cv2.polylines(img, [np.array(pts, np.int32)], False, ink, max(2, int(S * 0.004)), cv2.LINE_AA)
            arts["pen"] += 1
        for _ in range(rng.integers(cfg.dust[0], cfg.dust[1] + 1)):
            rho = place_r * math.sqrt(rng.random())
            ang = rng.uniform(0, 2 * math.pi)
            cv2.circle(img, (int(cx + rho * math.cos(ang)), int(cy + rho * math.sin(ang))), 0,
                       np.clip(agar * 0.5, 0, 255).tolist(), -1)
            arts["dust"] += 1

    # Lighting: gradient + vignette, then blur and sensor noise.
    yy, xx = np.mgrid[:S, :S].astype(np.float32)
    g = 1.0 + rng.uniform(-0.12, 0.12) * (xx - S / 2) / S + rng.uniform(-0.12, 0.12) * (yy - S / 2) / S
    vig = 1.0 - rng.uniform(0.0, 0.15) * (((xx - cx) ** 2 + (yy - cy) ** 2) / (R * R))
    img *= (g * vig)[..., None]
    sigma = rng.uniform(*cfg.blur)
    if sigma > 0.2:
        img = cv2.GaussianBlur(img, (0, 0), sigma)
    img += rng.normal(0, rng.uniform(*cfg.noise), img.shape).astype(np.float32)
    out = np.clip(img, 0, 255).astype(np.uint8)

    gt = {
        "count": len(colonies),
        "colonies": [{"x": round(c["x"], 1), "y": round(c["y"], 1), "r": round(c["r"], 2),
                      "fuzzy": c["fuzzy"]} for c in colonies],
        "plate": {"cx": cx, "cy": cy, "r": R, "agar_r": R_agar},
        "style": style,
        "artifacts": arts,
        "seed": seed,
    }
    return out, gt
