"""Box detectors run with sliced (tiled) inference.

A 90 mm plate photographed at 3000 px has 5-20 px colonies. Resizing the whole
plate to a detector's 640 px input shrinks them to 1-4 px and they vanish, so
the plate is cut into overlapping tiles at native resolution (SAHI-style),
each tile is detected independently and duplicates along tile seams are
merged. Detectors must be *trained* on tiles of the same scale
(tools/build_training_set.py does that).
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from colony_detector.engines.base import Engine, EngineUnavailable, file_digest
from colony_detector.plate import Plate
from colony_detector.types import COLONY, FUZZY, Detection

# Classes a detector may be trained with to learn what *not* to count.
IGNORED_CLASSES = {"bubble", "artifact", "artefact", "debris", "scratch", "text", "marker", "dust"}

RawBox = tuple[float, float, float, float, float, str]  # x0, y0, x1, y1, conf, class


def make_tiles(w: int, h: int, tile: int, overlap: float) -> list[tuple[int, int, int, int]]:
    """Tile origins covering a w x h image; the last row/column is snapped to
    the border so every tile is full size (when the image is large enough)."""
    if w <= tile and h <= tile:
        return [(0, 0, w, h)]
    step = max(1, int(tile * (1.0 - overlap)))

    def starts(n: int) -> list[int]:
        if n <= tile:
            return [0]
        out = list(range(0, n - tile, step))
        out.append(n - tile)
        return sorted(set(out))

    return [(x, y, min(x + tile, w), min(y + tile, h)) for y in starts(h) for x in starts(w)]


def merge_boxes(boxes: list[RawBox], iou_thr: float = 0.5, ios_thr: float = 0.75) -> list[RawBox]:
    """Greedy NMS that also removes a box mostly contained in a better one
    (intersection over the smaller box), which is what tile seams produce."""
    if not boxes:
        return []
    arr = np.array([b[:5] for b in boxes], dtype=np.float64)
    order = np.argsort(-arr[:, 4])
    keep: list[int] = []
    areas = (arr[:, 2] - arr[:, 0]).clip(0) * (arr[:, 3] - arr[:, 1]).clip(0)
    suppressed = np.zeros(len(arr), bool)
    for i in order:
        if suppressed[i]:
            continue
        keep.append(i)
        xx0 = np.maximum(arr[i, 0], arr[:, 0])
        yy0 = np.maximum(arr[i, 1], arr[:, 1])
        xx1 = np.minimum(arr[i, 2], arr[:, 2])
        yy1 = np.minimum(arr[i, 3], arr[:, 3])
        inter = (xx1 - xx0).clip(0) * (yy1 - yy0).clip(0)
        union = areas[i] + areas - inter
        iou = inter / np.maximum(union, 1e-9)
        ios = inter / np.maximum(np.minimum(areas[i], areas), 1e-9)
        suppressed |= (iou > iou_thr) | (ios > ios_thr)
    return [boxes[i] for i in keep]


def map_class(name: str) -> str | None:
    n = (name or "").strip().lower()
    if n in IGNORED_CLASSES:
        return None
    return FUZZY if "fuzzy" in n or "spread" in n else COLONY


class TiledBoxEngine(Engine):
    kind = "box"
    weight = 1.0

    def __init__(self, tile_size: int = 640, overlap: float = 0.25, conf: float = 0.2,
                 device: str = "", global_pass: bool = True):
        self.tile_size = tile_size
        self.overlap = overlap
        self.conf = conf
        self.device = device
        self.global_pass = global_pass

    def _predict(self, tiles: list[np.ndarray]) -> list[list[RawBox]]:
        """BGR tiles -> boxes in tile pixel coordinates."""
        raise NotImplementedError

    def detect(self, bgr: np.ndarray, plate: Plate, ctx: dict) -> list[Detection]:
        H, W = bgr.shape[:2]
        R = plate.radius
        x0, y0 = max(0, int(plate.cx - R)), max(0, int(plate.cy - R))
        x1, y1 = min(W, int(math.ceil(plate.cx + R))), min(H, int(math.ceil(plate.cy + R)))
        crop = bgr[y0:y1, x0:x1]
        ch, cw = crop.shape[:2]

        tiles = make_tiles(cw, ch, self.tile_size, self.overlap)
        preds = self._predict([crop[ty0:ty1, tx0:tx1] for tx0, ty0, tx1, ty1 in tiles])
        boxes: list[RawBox] = []
        for (tx0, ty0, _, _), tile_boxes in zip(tiles, preds):
            boxes.extend((bx0 + tx0, by0 + ty0, bx1 + tx0, by1 + ty0, c, n)
                         for bx0, by0, bx1, by1, c, n in tile_boxes)

        # A downscaled whole-plate pass catches colonies larger than a tile.
        if self.global_pass and len(tiles) > 1:
            f = self.tile_size / max(ch, cw)
            small = cv2.resize(crop, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
            for bx0, by0, bx1, by1, c, n in self._predict([small])[0]:
                if max(bx1 - bx0, by1 - by0) / f > 0.5 * self.tile_size:
                    boxes.append((bx0 / f, by0 / f, bx1 / f, by1 / f, c, n))

        boxes = merge_boxes(boxes)
        roi_r = ctx.get("roi_radius_px", 0.97 * R)
        rc = ctx.get("roi_center", (plate.cx, plate.cy))
        dets = []
        for bx0, by0, bx1, by1, c, n in boxes:
            cls = map_class(n)
            if cls is None or c < self.conf:
                continue
            cx, cy = (bx0 + bx1) / 2 + x0, (by0 + by1) / 2 + y0
            if math.hypot(cx - rc[0], cy - rc[1]) > roi_r:
                continue
            dets.append(Detection(
                cx=cx, cy=cy, radius=((bx1 - bx0) + (by1 - by0)) / 4.0, confidence=float(c),
                class_name=cls, source=self.name, shape_source="box",
            ))
        return dets


class YoloEngine(TiledBoxEngine):
    """Ultralytics YOLO (v8/11/12) trained on plate tiles."""

    name = "yolo"
    weight = 1.2

    def __init__(self, weights: str, **kw):
        super().__init__(**kw)
        self.weights = weights
        self.model = None

    def version(self) -> str:
        return f"yolo:{file_digest(self.weights)}"

    def load(self) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover
            raise EngineUnavailable("ultralytics is not installed") from exc
        self.model = YOLO(self.weights)

    def _predict(self, tiles):
        if self.model is None:
            self.load()
        kw = {"imgsz": self.tile_size, "conf": min(self.conf, 0.1), "iou": 0.5,
              "verbose": False, "max_det": 1000}
        if self.device:
            kw["device"] = self.device
        results = self.model.predict(tiles, **kw)
        names = self.model.names
        out = []
        for res in results:
            b = res.boxes
            xyxy = b.xyxy.cpu().numpy() if len(b) else np.zeros((0, 4))
            conf = b.conf.cpu().numpy() if len(b) else np.zeros(0)
            cls = b.cls.cpu().numpy().astype(int) if len(b) else np.zeros(0, int)
            out.append([(*map(float, xyxy[i]), float(conf[i]), names.get(int(cls[i]), "colony")
                         if isinstance(names, dict) else names[int(cls[i])]) for i in range(len(conf))])
        return out


class RFDETREngine(TiledBoxEngine):
    """RF-DETR (Roboflow) fine-tuned on plate tiles."""

    name = "rfdetr"
    weight = 1.2

    def __init__(self, weights: str, size: str = "base", **kw):
        super().__init__(**kw)
        self.weights = weights
        self.size = size
        self.model = None
        self.class_names: list[str] = []

    def version(self) -> str:
        return f"rfdetr-{self.size}:{file_digest(self.weights)}"

    def load(self) -> None:
        try:
            import rfdetr
        except ImportError as exc:  # pragma: no cover
            raise EngineUnavailable("rfdetr is not installed (pip install rfdetr)") from exc
        cls = {
            "nano": "RFDETRNano", "small": "RFDETRSmall", "medium": "RFDETRMedium",
            "base": "RFDETRBase", "large": "RFDETRLarge",
        }.get(self.size.lower(), "RFDETRBase")
        self.model = getattr(rfdetr, cls)(pretrain_weights=self.weights)
        names = getattr(self.model, "class_names", None)
        self.class_names = list(names.values()) if isinstance(names, dict) else list(names or [])

    def _predict(self, tiles):  # pragma: no cover - needs trained weights
        if self.model is None:
            self.load()
        from PIL import Image

        out = []
        for t in tiles:
            det = self.model.predict(Image.fromarray(cv2.cvtColor(t, cv2.COLOR_BGR2RGB)),
                                     threshold=min(self.conf, 0.1))
            boxes = []
            for i in range(len(det.xyxy)):
                cid = int(det.class_id[i]) if det.class_id is not None else 0
                name = self.class_names[cid] if 0 <= cid < len(self.class_names) else COLONY
                boxes.append((*map(float, det.xyxy[i]), float(det.confidence[i]), name))
            out.append(boxes)
        return out


class GroundingDinoEngine(TiledBoxEngine):
    """Zero-shot detection with a text prompt (Grounding DINO).

    This is the detection half of "Colony Grounded SAM2" (arXiv 2603.13393):
    no training data is needed, which makes it a good bootstrapping tool for
    pseudo-labels. Set ``CD_GDINO_MODEL`` to a colony fine-tuned checkpoint
    for the paper's accuracy; the stock model is noticeably weaker.
    """

    name = "gdino"
    weight = 0.8

    def __init__(self, model_id: str, prompt: str, text_threshold: float = 0.2, **kw):
        super().__init__(**kw)
        self.model_id = model_id
        self.prompt = prompt
        self.text_threshold = text_threshold
        self.model = None
        self.processor = None

    def version(self) -> str:
        return f"gdino:{self.model_id}"

    def load(self) -> None:  # pragma: no cover - downloads a large model
        try:
            import torch  # noqa: F401
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except ImportError as exc:
            raise EngineUnavailable("transformers/torch are not installed") from exc
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id)
        if self.device:
            self.model.to(self.device)
        self.model.eval()

    def _predict(self, tiles):  # pragma: no cover - needs model download
        import torch
        from PIL import Image

        if self.model is None:
            self.load()
        out = []
        for t in tiles:
            h, w = t.shape[:2]
            img = Image.fromarray(cv2.cvtColor(t, cv2.COLOR_BGR2RGB))
            inputs = self.processor(images=img, text=self.prompt, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
            post = self.processor.post_process_grounded_object_detection
            try:
                res = post(outputs, inputs["input_ids"], threshold=self.conf,
                           text_threshold=self.text_threshold, target_sizes=[(h, w)])[0]
            except TypeError:  # older transformers
                res = post(outputs, inputs["input_ids"], box_threshold=self.conf,
                           text_threshold=self.text_threshold, target_sizes=[(h, w)])[0]
            boxes = []
            for box, score in zip(res["boxes"].cpu().numpy(), res["scores"].cpu().numpy()):
                bx0, by0, bx1, by1 = map(float, box)
                # The whole plate or a large region is not a colony.
                if (bx1 - bx0) * (by1 - by0) > 0.25 * w * h:
                    continue
                boxes.append((bx0, by0, bx1, by1, float(score), COLONY))
            out.append(boxes)
        return out
