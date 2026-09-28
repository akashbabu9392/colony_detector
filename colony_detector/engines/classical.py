"""Training-free colony detector.

Pipeline (all inside the dish ROI):

1. Estimate the agar background with a large median filter (two passes, the
   second with colonies masked out) and take the per-pixel Lab colour
   distance to it. This removes vignetting, uneven lighting and shadows and
   works for dark colonies on light agar, light colonies on dark agar and
   coloured colonies on coloured agar alike.
2. Threshold that contrast map with hysteresis in robust-noise units, so
   strong colony cores seed regions that grow out to their faint edges.
3. Find where the agar ends (the dish wall shows up as a ring of contrast)
   and ignore everything outside it.
4. Reject artifacts: air bubbles (hollow rings), pen marks / scratches / hairs
   (long strokes of constant width) and objects of the minority polarity
   (e.g. marker ink on a plate of light colonies).
5. Split touching colonies with an h-maxima distance-transform watershed and
   estimate the size of clumps that still cannot be separated.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import apply_hysteresis_threshold
from skimage.measure import regionprops
from skimage.morphology import convex_hull_image, h_maxima, skeletonize
from skimage.segmentation import watershed

from colony_detector.config import ClassicalParams
from colony_detector.engines.base import Engine
from colony_detector.plate import Plate
from colony_detector.types import COLONY, FUZZY, Detection

VERSION = "classical-1.0"


def _odd(n: float) -> int:
    n = max(3, int(round(n)))
    return n if n % 2 else n + 1


def _robust_stats(values: np.ndarray) -> tuple[float, float]:
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    return med, max(1.4826 * mad, 0.5)


def _circle_mask(shape, cx, cy, r) -> np.ndarray:
    m = np.zeros(shape[:2], np.uint8)
    cv2.circle(m, (int(round(cx)), int(round(cy))), max(1, int(round(r))), 1, -1)
    return m.astype(bool)


def _fill_small_holes(mask: np.ndarray, max_area: float) -> np.ndarray:
    filled = ndi.binary_fill_holes(mask)
    holes = filled & ~mask
    lab, n = ndi.label(holes)
    if n:
        sizes = ndi.sum(holes, lab, np.arange(1, n + 1))
        big = np.flatnonzero(sizes > max_area) + 1
        if big.size:
            filled &= ~np.isin(lab, big)
    return filled


def _merge_wide_necks(segs: list[np.ndarray], dcore: np.ndarray, ratio: float = 0.7) -> list[np.ndarray]:
    """Undo watershed cuts that run through a colony's body. Between two
    real colonies the core distance map has a saddle clearly lower than the
    smaller colony's own core radius."""
    n = len(segs)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    kernel = np.ones((3, 3), np.uint8)
    peak = [float(dcore[sg].max()) if sg.any() else 0.0 for sg in segs]
    grown = [cv2.dilate(sg.astype(np.uint8), kernel).astype(bool) for sg in segs]
    for i in range(n):
        for j in range(i + 1, n):
            line = grown[i] & segs[j]
            if not line.any():
                continue
            saddle = float(dcore[line].max())
            if saddle > ratio * min(peak[i], peak[j]):
                parent[find(j)] = find(i)
    groups: dict[int, np.ndarray] = {}
    for i, sg in enumerate(segs):
        root = find(i)
        groups[root] = sg if root not in groups else (groups[root] | sg)
    return list(groups.values())


class ClassicalEngine(Engine):
    name = "classical"
    kind = "mask"
    weight = 0.8

    def __init__(self, params: ClassicalParams | None = None):
        self.p = params or ClassicalParams()

    def version(self) -> str:
        return VERSION

    # ------------------------------------------------------------------ utils
    def _background(self, lab: np.ndarray, plate_m: np.ndarray, r: float,
                    exclude: np.ndarray | None = None) -> np.ndarray:
        """Per-channel median background at low resolution."""
        h, w = lab.shape[:2]
        small_side = 256.0
        f = min(1.0, small_side / (2 * r))
        src = lab.copy()
        valid = plate_m.copy()
        if exclude is not None:
            valid &= ~exclude
        if valid.sum() < 100:
            valid = plate_m
        for c in range(3):
            ch = src[..., c]
            ch[~valid] = np.uint8(np.median(lab[..., c][valid]))
        # Keep only the agar's own values inside the dish: colonies and the
        # outside world were replaced by the agar median above.
        small = cv2.resize(src, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        k = _odd(2 * r * f / 7.0)
        bg_small = np.stack([cv2.medianBlur(small[..., c], k) for c in range(3)], -1)
        return cv2.resize(bg_small, (w, h), interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def _flatten_rim(d: np.ndarray, cx: float, cy: float, r: float, rho0: float = 0.78) -> np.ndarray:
        """Remove concentric structure near the wall (lid edge, wall shading,
        meniscus) by subtracting the median of each thin ring within each
        angular sector. Colonies are local in angle, so they survive."""
        h, w = d.shape[:2]
        yy, xx = np.mgrid[:h, :w]
        rho = np.hypot(xx - cx, yy - cy) / r
        band = (rho >= rho0) & (rho < 1.0)
        if band.sum() < 500:
            return d
        span = 1.0 - rho0
        n_r = max(10, int(span * r / 3.0))  # ~3 px rings
        ri = np.clip(((rho[band] - rho0) / span * n_r).astype(int), 0, n_r - 1)
        ang = np.arctan2(yy[band] - cy, xx[band] - cx)
        n_a = 36
        ai = ((ang + np.pi) / (2 * np.pi) * n_a).astype(int) % n_a
        lab_idx = ri * n_a + ai + 1
        idx = np.arange(1, n_r * n_a + 1)
        out = d.copy()
        for c in range(3):
            vals = d[..., c][band]
            med = np.asarray(ndi.median(vals, lab_idx, idx), dtype=np.float32)
            med = np.nan_to_num(med).reshape(n_r, n_a)
            # Smooth across neighbouring sectors (wrap-around).
            med = (np.roll(med, 1, 1) + 2 * med + np.roll(med, -1, 1)) / 4.0
            # Fade in so the interior is untouched.
            fade = np.clip((rho[band] - rho0) / 0.03, 0, 1).astype(np.float32)
            out[..., c][band] = vals - fade * med[ri, ai]
        return out

    def _contrast(self, lab: np.ndarray, bg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self._contrast_from_diff(lab.astype(np.float32) - bg.astype(np.float32))

    def _threshold(self, norm, thr_lo, thr_hi, plate_m, r):
        mask = apply_hysteresis_threshold(norm, thr_lo, thr_hi) & plate_m
        mask = ndi.binary_opening(mask, structure=np.ones((2, 2), bool))
        ro = max(2, int(round(0.004 * r)))
        disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ro + 1, 2 * ro + 1))
        mask_open = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, disk).astype(bool)
        return mask, mask_open

    def _contrast_from_diff(self, d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        dl = d[..., 0] * (100.0 / 255.0) * 2.0  # OpenCV L is 0..255; ~2x L* units
        chroma = np.hypot(d[..., 1], d[..., 2])
        c = np.sqrt(dl * dl + (self.p.chroma_weight * chroma) ** 2)
        c = cv2.GaussianBlur(c, (0, 0), 1.0)
        dl = cv2.GaussianBlur(dl, (0, 0), 1.0)
        return c, dl

    def _inner_radius(self, fg_open: np.ndarray, cx, cy, r) -> float:
        """Walk outwards from 75% radius; the agar ends where a whole annulus
        lights up (dish wall / meniscus). ``fg_open`` has thin lines (lid
        edge, wall reflections) removed, so colonies seen through the lid
        rim are not cut off."""
        h, w = fg_open.shape
        yy, xx = np.ogrid[:h, :w]
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / r
        inner = (dist > 0.4) & (dist < 0.75)
        base = float(fg_open[inner].mean()) if inner.any() else 0.0
        limit = max(0.35, 3.0 * base + 0.1)
        ts = np.arange(0.75, 1.0, 0.005)
        frac = np.zeros(len(ts))
        for i, t in enumerate(ts):
            ring = (dist >= t) & (dist < t + 0.005)
            frac[i] = fg_open[ring].mean() if ring.sum() >= 20 else 1.0
        edge = 1.0
        # The wall is a band; a lid edge or scratch ring is a single line with
        # clear agar beyond it, so require the next 3% of radius to be busy.
        for i in range(len(ts)):
            if frac[i] > limit and frac[i:i + 6].mean() > limit:
                edge = ts[i]
                break
        return r * max(0.75, edge - self.p.rim_margin_frac)

    @staticmethod
    def _local_sigma(contrast, med, plate_m, r) -> np.ndarray:
        h, w = contrast.shape
        f = min(1.0, 192.0 / (2 * r))
        dev = np.abs(contrast - med)
        dev[~plate_m] = 0
        small = cv2.resize(dev, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        small8 = np.clip(small * 4.0, 0, 255).astype(np.uint8)
        k = _odd(2 * r * f / 8.0)
        loc = cv2.medianBlur(small8, k).astype(np.float32) / 4.0 * 1.4826
        return cv2.resize(loc, (w, h), interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def _is_bubble(img: np.ndarray, dl: np.ndarray, thr: float) -> bool:
        """A bubble's outline and centre have opposite (or no) polarity: a
        dark ring around a clear lens, or a bright ring around a dark centre.
        A colony is at least as strong in its centre as at its edge."""
        dist = ndi.distance_transform_edt(np.pad(img, 1))[1:-1, 1:-1]
        dmax = float(dist.max())
        if dmax < 3:
            return False
        core = dist >= 0.55 * dmax
        ring = img & (dist <= 0.4 * dmax)
        if core.sum() < 4 or ring.sum() < 8:
            return False
        ring_dl = float(dl[ring].mean())
        core_dl = float(dl[core].mean())
        if abs(ring_dl) < 0.6 * thr:
            return False
        # Reflected light: only the bright rim of a bubble is detected, as an
        # open arc, while its dark lens sits in the arc's concavity.
        hull = convex_hull_image(img)
        gap = hull & ~img
        if gap.sum() >= 0.25 * img.sum():
            if float(dl[gap].mean()) * np.sign(ring_dl) < -0.5 * thr:
                return True
        # A lens: clearly opposite-polarity pixels well inside the outline.
        inner = dist >= max(2.0, 0.35 * dmax)
        if inner.any():
            opposite = (dl[inner] * np.sign(ring_dl)) < -0.5 * thr
            if opposite.sum() >= max(3, 0.03 * inner.sum()):
                return True
        # Colonies: centre >= edge (ratio > 1, typically ~2-3). Bubbles: centre
        # is clear or reversed (ratio < ~0.5).
        return core_dl * np.sign(ring_dl) < 0.6 * abs(ring_dl)

    @staticmethod
    def _is_stroke(region_img: np.ndarray, dist: np.ndarray) -> bool:
        """Pen strokes, scratches and hairs: long skeleton, constant width."""
        skel = skeletonize(region_img)
        n = int(skel.sum())
        if n < 8:
            return False
        widths = dist[skel]
        thick = float(np.median(widths))
        if thick <= 0:
            return False
        length_ratio = n / (2.0 * thick)
        if length_ratio < 5.0:
            return False
        q20, q80 = np.percentile(widths, (20, 80))
        # Chains of touching colonies have narrow necks between wide bodies.
        return (q20 / max(q80, 1e-6)) > 0.55 or length_ratio > 14.0

    # --------------------------------------------------------------- detect
    def detect(self, bgr: np.ndarray, plate: Plate, ctx: dict) -> list[Detection]:
        p = self.p
        H, W = bgr.shape[:2]
        R = plate.radius
        x0 = max(0, int(plate.cx - R - 4))
        y0 = max(0, int(plate.cy - R - 4))
        x1 = min(W, int(math.ceil(plate.cx + R + 4)))
        y1 = min(H, int(math.ceil(plate.cy + R + 4)))
        crop = bgr[y0:y1, x0:x1]
        s = min(1.0, p.work_plate_diameter / (2.0 * R))
        work = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else crop
        cx, cy, r = (plate.cx - x0) * s, (plate.cy - y0) * s, R * s

        lab = cv2.cvtColor(work, cv2.COLOR_BGR2LAB)
        plate_m = _circle_mask(lab.shape, cx, cy, r * 0.97)

        # Pass 1 background, then mask obvious objects and re-estimate.
        bg = self._background(lab, plate_m, r)
        contrast, dl = self._contrast(lab, bg)
        med, sig = _robust_stats(contrast[plate_m])
        rough = contrast > max(p.floor_low, med + p.k_low * sig)
        rough = cv2.dilate(rough.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        bg = self._background(lab, plate_m, r, exclude=rough)
        contrast, dl = self._contrast(lab, bg)

        med, sig = _robust_stats(contrast[plate_m])
        thr_hi = max(p.floor_high, med + p.k_high * sig)
        thr_lo = max(p.floor_low, med + p.k_low * sig)
        thr_lo = min(thr_lo, 0.8 * thr_hi)
        min_r = max(p.min_radius_px, p.min_radius_frac * r)

        # Local noise: textured or grainy patches of agar get proportionally
        # higher thresholds instead of turning into hundreds of specks.
        sig_map = self._local_sigma(contrast, med, plate_m, r)
        scale = np.maximum(1.0, sig_map / sig)

        mask, mask_open = self._threshold(contrast / scale, thr_lo, thr_hi, plate_m, r)
        # Where does the dish wall start? (first busy band from the centre)
        wall = self._inner_radius(mask_open, cx, cy, r) / r + p.rim_margin_frac
        if wall < 0.96:
            # A wall/meniscus band is visible. Colonies do grow in the
            # meniscus, so flatten concentric structure there and look again.
            d = self._flatten_rim(
                lab.astype(np.float32) - bg.astype(np.float32), cx, cy, r, max(0.6, wall - 0.04))
            contrast, dl = self._contrast_from_diff(d)
            mask, mask_open = self._threshold(contrast / scale, thr_lo, thr_hi, plate_m, r)
            roi_r = self._inner_radius(mask_open, cx, cy, r)
        else:
            roi_r = r * max(0.75, wall - p.rim_margin_frac)
        # The outer band of the dish (wall shading, lid edge, condensation,
        # meniscus) is always held to stricter rules.
        meniscus_r = min(roi_r, wall * r - 0.01 * r, p.strict_band_start * r)
        roi = _circle_mask(lab.shape, cx, cy, roi_r)
        # Near the wall keep only blobs thicker than the wall lines.
        core = _circle_mask(lab.shape, cx, cy, min(0.9 * roi_r, meniscus_r))
        mask = np.where(core, mask, mask_open) & roi
        a_min = math.pi * min_r * min_r
        a_max = p.max_area_frac * math.pi * r * r
        # Fill colony-sized holes only: a closed ring (agar edge, lid line)
        # must not turn the whole plate into one object.
        filled = _fill_small_holes(mask, a_max)

        labels, n = ndi.label(filled)
        rejected = {"small": 0, "large": 0, "bubble": 0, "stroke": 0, "polarity": 0, "rim": 0}
        cands = []
        for reg in regionprops(labels, intensity_image=contrast):
            if reg.area < a_min:
                rejected["small"] += 1
                continue
            if reg.area > a_max:
                rejected["large"] += 1
                continue
            sl = reg.slice
            img = reg.image
            raw_area = int(mask[sl][img].sum())
            hollow = raw_area / float(reg.area)
            eq_r = reg.equivalent_diameter_area / 2.0
            if p.reject_bubbles and eq_r > 2.0 * min_r:
                # Air bubbles are either hollow rings, or a bright lens inside
                # a dark ring (transmitted light) / dark inside bright ring.
                if hollow < 0.6 or self._is_bubble(img, dl[sl], thr_lo):
                    rejected["bubble"] += 1
                    continue
            dist = ndi.distance_transform_edt(np.pad(img, 1))[1:-1, 1:-1]
            if p.reject_strokes and self._is_stroke(img, dist):
                rejected["stroke"] += 1
                continue
            sdl = float(dl[sl][img].mean())
            mean_c = float(reg.intensity_mean)
            rc = math.hypot(reg.centroid[1] - cx, reg.centroid[0] - cy)
            in_meniscus = rc > meniscus_r
            touches_edge = False
            if in_meniscus:
                ys, xs = reg.coords[:, 0], reg.coords[:, 1]
                touches_edge = float(np.max(np.hypot(xs - cx, ys - cy))) >= roi_r - 1.5
            if in_meniscus and (mean_c < 0.5 * (thr_lo + thr_hi) or reg.solidity < 0.85 or reg.eccentricity > 0.8
                                or eq_r < 2.5 * min_r or touches_edge):
                # Meniscus: bubbles, condensation and wall glints live here;
                # only clear, compact, round objects are counted.
                rejected["rim"] += 1
                continue
            pol = 0
            if abs(sdl) > 0.35 * mean_c:
                pol = 1 if sdl > 0 else -1
            cands.append({"reg": reg, "dist": dist, "pol": pol, "eq_r": eq_r, "mean_c": mean_c})

        # Dominant polarity among colony-shaped objects.
        polarity = p.polarity
        if polarity == "auto":
            votes = [c["pol"] for c in cands if c["reg"].solidity > 0.85]
            dark, bright = votes.count(-1), votes.count(1)
            if dark + bright == 0 or min(dark, bright) > 0.4 * max(dark, bright):
                polarity = "both"
            else:
                polarity = "dark" if dark > bright else "bright"
        keep_pol = {"dark": {-1, 0}, "bright": {1, 0}, "both": {-1, 0, 1}}.get(polarity, {-1, 0, 1})
        kept = []
        for c in cands:
            if c["pol"] in keep_pol:
                kept.append(c)
            else:
                rejected["polarity"] += 1

        singles = [c["eq_r"] for c in kept if c["reg"].solidity > 0.9 and c["reg"].eccentricity < 0.7]
        if not singles:
            singles = [c["eq_r"] for c in kept]
        r_typ = float(np.median(singles)) if singles else 3.0 * min_r
        a_typ = math.pi * r_typ * r_typ

        dets: list[Detection] = []
        for c in kept:
            parts = self._split_and_build(c, r_typ, a_typ, a_min, thr_lo, thr_hi, contrast, dl, min_r)
            for d, seg_img, seg_dl in parts:
                # Touching bubbles only separate after the split.
                if p.reject_bubbles and d.radius > 2.0 * min_r and self._is_bubble(seg_img, seg_dl, thr_lo):
                    rejected["bubble"] += 1
                    continue
                # Dust and debris: specks far smaller than this plate's colonies.
                if d.radius < max(min_r, p.min_relative_radius * r_typ):
                    rejected["small"] += 1
                    continue
                dets.append(d)

        # Fuzzy = edge markedly softer than this plate's typical colony, so
        # an out-of-focus photo does not turn every colony fuzzy.
        softness = [d.extra["edge_width"] for d in dets if d.radius >= 4]
        if len(softness) >= 5:
            ref = float(np.median(softness))
            for d in dets:
                if d.radius >= 4 and d.extra["edge_width"] > max(2.0 * ref, 0.5):
                    d.class_name = FUZZY
        dets = [d.scaled(s, x0, y0) for d in dets]

        cov = float(filled[roi].mean()) if roi.any() else 0.0
        ctx.update({
            "typical_radius_px": r_typ / s,
            "roi_radius_px": roi_r / s,
            "meniscus_radius_px": meniscus_r / s,
            "roi_center": (cx / s + x0, cy / s + y0),
            "polarity": polarity,
            "foreground_coverage": cov,
            "rejected": rejected,
            "thresholds": {"noise_median": round(med, 2), "noise_sigma": round(sig, 2),
                           "low": round(thr_lo, 2), "high": round(thr_hi, 2)},
            "work_scale": s,
        })
        return dets

    # ------------------------------------------------------------ splitting
    def _split_and_build(self, c, r_typ, a_typ, a_min, thr_lo, thr_hi, contrast, dl, min_r):
        """Yields (detection in work coords, segment mask, dl crop)."""
        p = self.p
        reg = c["reg"]
        img = reg.image
        dist = c["dist"]
        oy, ox = reg.slice[0].start, reg.slice[1].start
        segments: list[np.ndarray] = [img]

        # Every blob is a split candidate: a round single colony has exactly
        # one distance-transform peak, touching colonies have one per body
        # separated by a neck. h sets how deep a neck must be to count.
        if p.split_touching and reg.area >= 2 * a_min:
            # Markers come from the colony cores (above half of this blob's
            # peak contrast): the soft skirt around touching colonies makes
            # necks look thick, the cores do not.
            cc = contrast[reg.slice]
            peak = float(np.percentile(cc[img], 95))
            core = img & (cc >= thr_lo + 0.8 * max(peak - thr_lo, 0.0))
            dcore = cv2.GaussianBlur(
                ndi.distance_transform_edt(np.pad(core, 1))[1:-1, 1:-1].astype(np.float32), (0, 0), 0.7)
            h = max(0.5, 0.08 * r_typ)
            markers, nm = ndi.label(h_maxima(dcore, h) & core)
            if nm > 1:
                smooth = cv2.GaussianBlur(dist.astype(np.float32), (0, 0), 0.7)
                ws = watershed(-(smooth + dcore), markers, mask=img)
                segments = _merge_wide_necks([ws == i for i in range(1, nm + 1)], dcore)
                segments = [sg for sg in segments if sg.sum() >= 0.5 * a_min]
                if not segments:
                    segments = [img]

        out = []
        crop_c = contrast[reg.slice]
        crop_dl = dl[reg.slice]
        for seg in segments:
            area = float(seg.sum())
            cnts, _ = cv2.findContours(seg.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not cnts:
                continue
            cnt = max(cnts, key=cv2.contourArea)
            peri = cv2.arcLength(cnt, True)
            hull_a = cv2.contourArea(cv2.convexHull(cnt))
            cont_a = max(cv2.contourArea(cnt), 1.0)
            solidity = min(1.0, cont_a / hull_a) if hull_a > 0 else 1.0
            circ = min(1.0, 4 * math.pi * (area) / max(peri + math.pi, 1e-6) ** 2)
            m = cv2.moments(seg.astype(np.uint8), binaryImage=True)
            if m["m00"] == 0:
                continue
            scx, scy = m["m10"] / m["m00"], m["m01"] / m["m00"]
            eq_r = math.sqrt(area / math.pi)

            count = 1
            shape_source = "contour"
            # Overlapping colonies that watershed cannot separate still form a
            # lumpy but compact blob; very low solidity means an irregular
            # object (arc, smear) rather than a clump, so it is not multiplied.
            if p.estimate_clusters and 0.6 <= solidity < 0.8 and circ >= 0.45 and area > 2.5 * a_typ:
                count = int(np.clip(round(area / a_typ), 2, 60))
                shape_source = "cluster_estimate"

            vals = crop_c[seg]
            mean_c = float(vals.mean())
            # Edge softness: how wide the colony's edge ramp is, in px.
            er = cv2.erode(seg.astype(np.uint8), np.ones((3, 3), np.uint8))
            ring = seg & ~er.astype(bool)
            gy, gx = np.gradient(crop_c)
            grad = float(np.hypot(gx, gy)[ring].mean()) if ring.any() else 0.0
            edge_w = mean_c / max(grad, 1e-3)
            irregular = eq_r >= 6 and circ < 0.5 and solidity < 0.85

            c_contrast = np.clip((mean_c - thr_lo) / max(2 * thr_hi - thr_lo, 1e-6), 0, 1)
            c_shape = 0.5 * circ + 0.5 * solidity
            conf = 0.3 + 0.45 * c_contrast + 0.25 * c_shape
            conf *= float(np.clip(eq_r / (2.0 * min_r), 0.55, 1.0))

            pts = cnt[:, 0, :].astype(np.float64)
            eps = max(0.4, 0.008 * peri)
            approx = cv2.approxPolyDP(cnt, eps, True)[:, 0, :].astype(np.float64)
            if len(approx) >= 6:
                pts = approx
            pts = pts + np.array([ox, oy], dtype=np.float64)

            out.append((Detection(
                cx=scx + ox, cy=scy + oy, radius=eq_r,
                confidence=float(np.clip(conf, 0.05, 0.99)),
                class_name=FUZZY if irregular else COLONY,
                contour=pts, source=self.name, shape_source=shape_source, count=count,
                extra={"solidity": round(solidity, 3), "circularity": round(circ, 3),
                       "contrast": round(mean_c, 2), "edge_width": round(edge_w / max(eq_r, 1.0), 3)},
            ), seg, crop_dl))
        return out
