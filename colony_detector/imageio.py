"""Image decoding that tolerates what labs actually upload."""

from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np


class ImageDecodeError(ValueError):
    pass


def _to_bgr8(img: np.ndarray) -> np.ndarray:
    if img.dtype != np.uint8:
        img = img.astype(np.float64)
        lo, hi = np.percentile(img, (0.1, 99.9))
        img = np.clip((img - lo) / max(hi - lo, 1e-6) * 255.0, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def decode_image(data: bytes) -> np.ndarray:
    """Bytes -> BGR uint8. EXIF orientation is applied (phone photos)."""
    if not data:
        raise ImageDecodeError("empty image")
    buf = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    if img is None:
        # Some TIFF flavours (e.g. ImageJ big-endian) only open with Pillow.
        try:
            from PIL import Image, ImageOps

            pil = ImageOps.exif_transpose(Image.open(io.BytesIO(data)))
            if pil.mode not in ("RGB", "L", "I;16", "I", "F"):
                pil = pil.convert("RGB")
            arr = np.asarray(pil)
            if arr.ndim == 3:
                arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            img = arr
        except Exception as exc:  # noqa: BLE001 - any decoder failure is a bad upload
            raise ImageDecodeError(f"unsupported or corrupt image: {exc}") from exc
    img = _to_bgr8(img)
    if min(img.shape[:2]) < 64:
        raise ImageDecodeError(f"image too small: {img.shape[1]}x{img.shape[0]}")
    return img


def read_image(path: str | Path) -> np.ndarray:
    return decode_image(Path(path).read_bytes())


def encode_png(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return buf.tobytes()


def encode_jpeg(img: np.ndarray, quality: int = 90) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buf.tobytes()
