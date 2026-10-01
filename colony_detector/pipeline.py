"""End-to-end plate counting."""

from __future__ import annotations

import base64
import logging
import time
from dataclasses import dataclass, field

import numpy as np

from colony_detector import __version__
from colony_detector.annotate import render
from colony_detector.config import Settings
from colony_detector.engines.base import Engine, EngineUnavailable
from colony_detector.engines.classical import ClassicalEngine
from colony_detector.fusion import fuse
from colony_detector.imageio import decode_image, encode_png
from colony_detector.plate import Plate, find_plate
from colony_detector.quality import image_quality, review
from colony_detector.types import CLASS_NAMES, Detection

log = logging.getLogger("colony_detector")


def build_engines(settings: Settings) -> list[Engine]:
    tiled = {"tile_size": settings.tile_size, "overlap": settings.tile_overlap,
             "conf": settings.detector_conf, "device": settings.device,
             "tile_scale": settings.tile_scale}
    engines: list[Engine] = []
    for name in settings.engine_list():
        if name == "classical":
            engines.append(ClassicalEngine(settings.classical))
        elif name == "yolo":
            from colony_detector.engines.tiled import YoloEngine
            if not settings.yolo_weights:
                raise EngineUnavailable("yolo requested but CD_YOLO_WEIGHTS is not set")
            engines.append(YoloEngine(settings.yolo_weights, **tiled))
        elif name == "rfdetr":
            from colony_detector.engines.tiled import RFDETREngine
            if not settings.rfdetr_weights:
                raise EngineUnavailable("rfdetr requested but CD_RFDETR_WEIGHTS is not set")
            engines.append(RFDETREngine(settings.rfdetr_weights, size=settings.rfdetr_size,
                                        **{**tiled, "conf": settings.rfdetr_conf}))
        elif name == "cellpose":
            from colony_detector.engines.cellpose_engine import CellposeEngine
            if not settings.cellpose_weights:
                raise EngineUnavailable("cellpose requested but CD_CELLPOSE_WEIGHTS is not set")
            engines.append(CellposeEngine(settings.cellpose_weights, device=settings.device))
        elif name == "gdino":
            from colony_detector.engines.tiled import GroundingDinoEngine
            engines.append(GroundingDinoEngine(settings.gdino_model, settings.gdino_prompt, **tiled))
        else:
            raise ValueError(f"unknown engine {name!r}")
    for e in engines:
        if e.name in settings.engine_weights:
            e.weight = settings.engine_weights[e.name]
        for key, value in settings.detector_filters.get(e.name, {}).items():
            if hasattr(e, key):
                setattr(e, key, float(value))
    # The classical engine runs first: it measures the agar ROI and colony
    # size that the learned engines reuse.
    engines.sort(key=lambda e: 0 if e.name == "classical" else 1)
    return engines


@dataclass
class CountResult:
    total: int
    detections: list[Detection]
    plate: Plate
    quality: dict
    confidence: dict
    tntc: bool
    image_size: tuple[int, int]
    inference_ms: int
    model_version: str
    engines: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)
    annotated: np.ndarray | None = None

    @property
    def counts(self) -> dict[str, int]:
        out = {c: 0 for c in CLASS_NAMES}
        for d in self.detections:
            out[d.class_name] = out.get(d.class_name, 0) + d.count
        return out

    def to_response(self, include_image: bool = False) -> dict:
        """The cfu-model-2 ``/predict`` contract MicroID consumes, plus the
        TRD quality/confidence blocks."""
        boxes = []
        for d in self.detections:
            x0, y0, x1, y1 = d.xyxy
            boxes.append({
                "class_name": d.class_name,
                "confidence": round(d.confidence, 4),
                "xyxy": [int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))],
                "centroid": [int(round(d.cx)), int(round(d.cy))],
                "radius_px": int(max(1, round(d.radius))),
                "contour": np.round(d.polygon()).astype(int).tolist(),
                "shape_source": d.shape_source,
                "count": d.count,
                "sources": d.source.split("+"),
            })
        w, h = self.image_size
        resp = {
            "model_version": self.model_version,
            "image": {"width": w, "height": h},
            "counts": self.counts,
            "total_count": self.total,
            "tntc": self.tntc,
            "inference_ms": self.inference_ms,
            "boxes": boxes,
            "plate": {**self.plate.as_dict(),
                      "agar_radius_px": round(self.diagnostics.get("roi_radius_px", self.plate.radius), 1)},
            "quality": self.quality,
            "confidence": self.confidence,
            "engines": self.engines,
            "diagnostics": {k: v for k, v in self.diagnostics.items() if k not in ("roi_center",)},
        }
        # TRD rule: annotated image iff there is at least one detection.
        if include_image and self.detections and self.annotated is not None:
            resp["annotated_image"] = base64.b64encode(encode_png(self.annotated)).decode()
        return resp


class ColonyCounter:
    def __init__(self, settings: Settings | None = None, engines: list[Engine] | None = None):
        self.settings = settings or Settings()
        self.engines = engines if engines is not None else build_engines(self.settings)
        self.engine_errors: dict[str, str] = {}
        self._loaded = False

    def load(self) -> None:
        ready = []
        for e in self.engines:
            try:
                e.load()
                ready.append(e)
            except Exception as exc:  # noqa: BLE001 - keep serving with what loads
                log.warning("engine %s unavailable: %s", e.name, exc)
                self.engine_errors[e.name] = str(exc)
        if not ready:
            raise EngineUnavailable(f"no engine could be loaded: {self.engine_errors}")
        self.engines = ready
        self._loaded = True

    @property
    def model_version(self) -> str:
        parts = [e.version() for e in self.engines]
        return f"colony-detector-{__version__}[{'+'.join(parts)}]"

    def count_bytes(self, data: bytes, annotate: bool = False) -> CountResult:
        return self.count(decode_image(data), annotate=annotate)

    def count(self, bgr: np.ndarray, annotate: bool = False) -> CountResult:
        if not self._loaded:
            self.load()
        t0 = time.perf_counter()
        s = self.settings
        plate = find_plate(bgr)
        ctx: dict = {}
        results: dict[str, list[Detection]] = {}
        engine_info: dict[str, dict] = {}
        for e in self.engines:
            te = time.perf_counter()
            try:
                dets = e.detect(bgr, plate, ctx)
            except Exception as exc:  # noqa: BLE001
                log.exception("engine %s failed", e.name)
                engine_info[e.name] = {"status": "error", "error": str(exc)[:300]}
                continue
            results[e.name] = dets
            engine_info[e.name] = {
                "status": "ok",
                "count": int(sum(d.count for d in dets)),
                "ms": int((time.perf_counter() - te) * 1000),
            }
        if not results:
            raise EngineUnavailable(f"every engine failed: {engine_info}")

        # Sanity guard: an engine whose count is wildly off the training-free
        # classical count is out of its domain on this image (wrong
        # magnification, unseen medium...). It is left out of the vote and
        # the plate goes to review instead of silently skewing the count.
        disagreement = []
        ref = results.get("classical")
        if ref is not None and len(results) > 1:
            n_ref = sum(d.count for d in ref)
            for name in [n for n in results if n != "classical"]:
                n_e = sum(d.count for d in results[name])
                lo, hi = sorted((n_ref, n_e))
                if hi - lo > 10 and hi > s.outlier_ratio * max(lo, 1):
                    disagreement.append(name)
                    engine_info[name]["status"] = "excluded_out_of_domain"
                    del results[name]

        weights = {e.name: e.weight for e in self.engines}
        dets = fuse(results, weights, s.fuse_threshold)

        # Counting rule by physical size, measured with the dish as ruler.
        px_per_mm = 2.0 * plate.radius / s.dish_mm
        small_specks = 0
        if s.min_colony_mm > 0:
            kept = []
            for d in dets:
                size_mm = 2.0 * d.radius / px_per_mm
                if size_mm >= s.min_colony_mm:
                    kept.append(d)
                elif size_mm >= s.review_colony_mm:
                    small_specks += 1
            dets = kept
        # Reading order (row bands, then left to right) so colony #N in the
        # overlay is easy to find by eye.
        band = 4.0 * float(np.median([d.radius for d in dets])) if dets else 1.0
        dets.sort(key=lambda d: (int(d.cy // max(band, 1.0)), d.cx))
        total = int(sum(d.count for d in dets))

        quality = image_quality(bgr, plate, ctx.get("roi_radius_px"))
        coverage = float(ctx.get("foreground_coverage", 0.0))
        confidence, tntc = review(
            total, [d.confidence for d in dets], quality, plate, coverage,
            s.tntc_limit, s.min_focus_score, s.max_glare_score, s.low_confidence,
        )
        if disagreement:
            confidence["needs_review"] = True
            confidence["reason_codes"].append("ENGINE_DISAGREEMENT")
        if small_specks:
            # Pin-point specks below the counting size: a person decides.
            confidence["needs_review"] = True
            confidence["reason_codes"].append("SMALL_SPECKS")
        ctx["counting_rule"] = {"min_colony_mm": s.min_colony_mm, "review_colony_mm": s.review_colony_mm,
                                "px_per_mm": round(px_per_mm, 2), "small_specks": small_specks}
        quality = {**quality, "plate_found": plate.found, "overgrowth_detected": coverage > 0.35,
                   "foreground_coverage": round(coverage, 4)}

        h, w = bgr.shape[:2]
        result = CountResult(
            total=total, detections=dets, plate=plate, quality=quality,
            confidence=confidence, tntc=tntc, image_size=(w, h),
            inference_ms=int((time.perf_counter() - t0) * 1000),
            model_version=self.model_version, engines=engine_info,
            diagnostics={k: v for k, v in ctx.items()},
        )
        if annotate and dets:
            result.annotated = render(bgr, dets, plate, total)
        return result
