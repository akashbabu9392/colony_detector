"""HTTP inference service.

Drop-in for MicroID's ``AI_SERVICE_URL``: ``POST /predict`` takes a multipart
``file`` and returns the cfu-model-2 response contract
(``backend/app/services/colony_counter/remote_client.py`` in MicroID).

Run:  uvicorn colony_detector.service:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import hmac
import logging
from contextlib import asynccontextmanager

import numpy as np
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile

from colony_detector import __version__
from colony_detector.config import Settings
from colony_detector.engines.base import EngineUnavailable
from colony_detector.imageio import ImageDecodeError, decode_image
from colony_detector.pipeline import ColonyCounter

log = logging.getLogger("colony_detector")

_state: dict = {"counter": None, "error": None}
# Models are not re-entrant on one device; serialise inference, keep the
# event loop free by running it in a worker thread.
_lock = asyncio.Lock()


def _load(settings: Settings) -> None:
    try:
        counter = ColonyCounter(settings)
        counter.load()
        if settings.warmup:
            blank = np.full((512, 512, 3), 200, np.uint8)
            counter.count(blank)
        _state["counter"] = counter
        _state["error"] = None
    except Exception as exc:  # noqa: BLE001 - reported via /health and 503s
        log.exception("model load failed")
        _state["error"] = str(exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings()
    app.state.settings = settings
    await asyncio.to_thread(_load, settings)
    yield


app = FastAPI(title="Colony Detector", version=__version__, lifespan=lifespan)


def _auth(authorization: str | None = Header(default=None)) -> None:
    key = app.state.settings.api_key if hasattr(app.state, "settings") else ""
    if not key:
        return
    token = (authorization or "").removeprefix("Bearer ").strip()
    if not hmac.compare_digest(token, key):
        raise HTTPException(status_code=401, detail="invalid API key")


def _counter() -> ColonyCounter:
    counter = _state["counter"]
    if counter is None:
        raise HTTPException(status_code=503, detail=f"model not loaded: {_state['error'] or 'starting'}")
    return counter


@app.get("/health")
def health() -> dict:
    counter = _state["counter"]
    return {
        "status": "ok" if counter else "unavailable",
        "version": __version__,
        "model_version": counter.model_version if counter else None,
        "engines": [e.name for e in counter.engines] if counter else [],
        "engine_errors": counter.engine_errors if counter else {},
        "error": _state["error"],
    }


@app.post("/predict", dependencies=[Depends(_auth)])
async def predict(
    file: UploadFile = File(...),
    include_image: bool = Form(True),
) -> dict:
    counter = _counter()
    limit = app.state.settings.max_upload_mb * 1024 * 1024
    data = await file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(status_code=413, detail="image too large")
    try:
        bgr = decode_image(data)
    except ImageDecodeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    async with _lock:
        try:
            result = await asyncio.to_thread(counter.count, bgr, include_image)
        except EngineUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
    return result.to_response(include_image=include_image)


# Alias with a more descriptive name for direct API users.
app.post("/v1/count", dependencies=[Depends(_auth)])(predict)
