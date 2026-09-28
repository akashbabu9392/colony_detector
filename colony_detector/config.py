"""Runtime configuration.

Everything is read from environment variables (prefix ``CD_``) so the same
image runs locally, in Docker and behind MicroID without code changes.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODELS_DIR = REPO_ROOT / "models"


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _env_float(name: str, default: float) -> float:
    raw = _env(name)
    return float(raw) if raw else default


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    return int(raw) if raw else default


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name).lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _first_existing(*candidates: Path | str | None) -> str:
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(candidate)
    return ""


@dataclass
class ClassicalParams:
    """Tuning knobs for the training-free engine.

    Defaults were tuned on synthetic plates (tools/synth_plates.py) and checked
    visually on the OpenCFU sample plates; see README "Accuracy".
    """

    # Plate diameter (px) the engine works at. Larger keeps tiny colonies,
    # smaller is faster. Images are never upscaled.
    work_plate_diameter: int = 1600
    # Colony polarity relative to agar: auto | dark | bright | both.
    polarity: str = "auto"
    # Hysteresis thresholds in robust-sigma units above the agar noise floor.
    k_high: float = 7.0
    k_low: float = 3.5
    # Absolute contrast floors (Lab units) so a perfectly flat plate does not
    # promote sensor noise to colonies.
    floor_high: float = 14.0
    floor_low: float = 7.0
    # Weight of chroma (a*, b*) relative to lightness in the contrast map.
    chroma_weight: float = 1.6
    # Smallest colony radius, as a fraction of plate radius and absolute px.
    min_radius_frac: float = 0.004
    min_radius_px: float = 1.5
    # ...and relative to the plate's typical colony (dust / debris filter).
    min_relative_radius: float = 0.25
    # Largest single object, as a fraction of plate area.
    max_area_frac: float = 0.08
    # Fraction of the plate radius treated as wall/rim and ignored.
    rim_margin_frac: float = 0.025
    # Beyond this fraction of the dish radius only clear, round, compact
    # objects count (the wall zone is full of glints and condensation).
    strict_band_start: float = 0.9
    # Split touching colonies with distance-transform watershed.
    split_touching: bool = True
    # Estimate counts of blobs that cannot be split cleanly.
    estimate_clusters: bool = True
    # Drop hollow rings (air bubbles) and thin strokes (pen, scratches, hair).
    reject_bubbles: bool = True
    reject_strokes: bool = True


@dataclass
class Settings:
    # Which engines to run: "auto" = every engine whose weights are present,
    # plus the classical engine. Or a comma list, e.g. "classical,yolo".
    engines: str = field(default_factory=lambda: _env("CD_ENGINES", "auto"))

    models_dir: Path = field(
        default_factory=lambda: Path(_env("CD_MODELS_DIR", str(DEFAULT_MODELS_DIR)))
    )

    yolo_weights: str = field(default_factory=lambda: _env("CD_YOLO_WEIGHTS"))
    rfdetr_weights: str = field(default_factory=lambda: _env("CD_RFDETR_WEIGHTS"))
    rfdetr_size: str = field(default_factory=lambda: _env("CD_RFDETR_SIZE", "base"))
    cellpose_weights: str = field(default_factory=lambda: _env("CD_CELLPOSE_WEIGHTS"))
    # Zero-shot engine (Colony Grounded SAM2 style). Off unless requested,
    # because it downloads large foundation models on first use.
    gdino_model: str = field(
        default_factory=lambda: _env("CD_GDINO_MODEL", "IDEA-Research/grounding-dino-base")
    )
    gdino_prompt: str = field(
        default_factory=lambda: _env("CD_GDINO_PROMPT", "bacterial colony. round colony.")
    )

    # Tiled inference for box detectors (small colonies vanish when a whole
    # plate is resized to 640 px, so plates are cut into overlapping tiles).
    tile_size: int = field(default_factory=lambda: _env_int("CD_TILE_SIZE", 640))
    tile_overlap: float = field(default_factory=lambda: _env_float("CD_TILE_OVERLAP", 0.25))
    detector_conf: float = field(default_factory=lambda: _env_float("CD_DETECTOR_CONF", 0.20))
    # DETR-style scores are calibrated differently from YOLO's: 0.4 gave the
    # lowest count error for RF-DETR on held-out plates (0.2 over-counts).
    rfdetr_conf: float = field(default_factory=lambda: _env_float("CD_RFDETR_CONF", 0.40))
    device: str = field(default_factory=lambda: _env("CD_DEVICE", ""))

    # Fusion: minimum consensus score for a detection to be counted.
    fuse_threshold: float = field(default_factory=lambda: _env_float("CD_FUSE_THRESHOLD", 0.25))
    # A learned engine whose count differs from the classical count by more
    # than this factor is treated as out of domain for that image.
    outlier_ratio: float = field(default_factory=lambda: _env_float("CD_OUTLIER_RATIO", 2.5))

    # Plates above this are reported as TNTC (too numerous to count).
    tntc_limit: int = field(default_factory=lambda: _env_int("CD_TNTC_LIMIT", 300))
    # Quality gates for needs_review.
    min_focus_score: float = field(default_factory=lambda: _env_float("CD_MIN_FOCUS", 0.25))
    max_glare_score: float = field(default_factory=lambda: _env_float("CD_MAX_GLARE", 0.05))
    low_confidence: float = field(default_factory=lambda: _env_float("CD_LOW_CONF", 0.35))

    api_key: str = field(default_factory=lambda: _env("CD_API_KEY"))
    max_upload_mb: int = field(default_factory=lambda: _env_int("CD_MAX_UPLOAD_MB", 40))
    warmup: bool = field(default_factory=lambda: _env_bool("CD_WARMUP", True))

    classical: ClassicalParams = field(default_factory=ClassicalParams)

    def __post_init__(self) -> None:
        self.models_dir = Path(self.models_dir)
        md = self.models_dir
        # Conventional file names (same as the cfu-model-2 training notebooks)
        # are picked up automatically when no explicit path is given.
        self.yolo_weights = self.yolo_weights or _first_existing(
            md / "yolo11s_tiles.pt", md / "yolo_tiles.pt", md / "best.pt"
        )
        self.rfdetr_weights = self.rfdetr_weights or _first_existing(
            md / "rfdetr_tiles.pth", md / "rfdetr.pth"
        )
        self.cellpose_weights = self.cellpose_weights or _first_existing(
            md / "cellpose_colony", md / "colony_project_model"
        )
        polarity = _env("CD_POLARITY")
        if polarity:
            self.classical.polarity = polarity
        # Fusion weights fitted on gold plates by tools/tune_fusion.py.
        self.engine_weights: dict[str, float] = {}
        fusion = md / "fusion.json"
        if fusion.is_file():
            data = json.loads(fusion.read_text())
            self.engine_weights = {k: float(v) for k, v in data.get("weights", {}).items()}
            if "threshold" in data and not _env("CD_FUSE_THRESHOLD"):
                self.fuse_threshold = float(data["threshold"])

    def engine_list(self) -> list[str]:
        raw = (self.engines or "auto").lower().replace(" ", "")
        if raw != "auto":
            return [name for name in raw.split(",") if name]
        names = ["classical"]
        if self.yolo_weights:
            names.append("yolo")
        if self.rfdetr_weights:
            names.append("rfdetr")
        if self.cellpose_weights:
            names.append("cellpose")
        return names
