"""HTTP contract tests.

``microid_to_contract`` mirrors how MicroID's RemoteColonyCounter reads the
``/predict`` response (backend/app/services/colony_counter/remote_client.py),
so a change that would break MicroID fails here.
"""

import base64
from statistics import mean

import cv2
import pytest
from fastapi.testclient import TestClient

from colony_detector.synth import make_plate


def microid_to_contract(payload: dict) -> dict:
    boxes = payload.get("boxes") or []
    total = payload.get("total_count", len(boxes))
    details = []
    for i, box in enumerate(boxes):
        centroid = box.get("centroid") or [0, 0]
        details.append({
            "colony_id": i + 1, "centroid_x": int(centroid[0]), "centroid_y": int(centroid[1]),
            "class_name": box.get("class_name", "unknown"),
            "confidence": float(box.get("confidence", 0.0)),
            "radius_px": box.get("radius_px", 0), "contour": box.get("contour") or [],
        })
    confs = [float(b.get("confidence", 0.0)) for b in boxes]
    return {"cfu_count_total": total, "colony_details": details,
            "mean_confidence": round(mean(confs), 3) if confs else 0.0,
            "model_version": payload.get("model_version"), "image": payload.get("image")}


@pytest.fixture(scope="module")
def client(monkeypatch_module):
    from colony_detector import service

    with TestClient(service.app) as c:
        yield c


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    mp.setenv("CD_ENGINES", "classical")
    mp.setenv("CD_WARMUP", "false")
    mp.delenv("CD_API_KEY", raising=False)
    yield mp
    mp.undo()


def _jpeg(seed=1, n=None):
    img, gt = make_plate(seed, n=n)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return buf.tobytes(), gt, img.shape


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["engines"] == ["classical"]


def test_predict_matches_microid_contract(client):
    data, gt, shape = _jpeg(3)
    r = client.post("/predict", files={"file": ("disk.jpg", data, "image/jpeg")},
                    data={"include_image": "false"})
    assert r.status_code == 200, r.text
    p = r.json()
    for key in ("model_version", "image", "counts", "total_count", "inference_ms", "boxes"):
        assert key in p
    assert p["image"] == {"width": shape[1], "height": shape[0]}
    assert p["total_count"] == sum(p["counts"].values())
    assert set(p["counts"]) >= {"colony", "fuzzy_colony"}
    assert "annotated_image" not in p
    b = p["boxes"][0]
    assert {"class_name", "confidence", "xyxy", "centroid", "radius_px", "contour", "shape_source"} <= set(b)
    assert len(b["xyxy"]) == 4 and len(b["centroid"]) == 2 and len(b["contour"]) >= 3
    mapped = microid_to_contract(p)
    assert mapped["cfu_count_total"] == p["total_count"]
    assert len(mapped["colony_details"]) == len(p["boxes"])
    assert abs(p["total_count"] - gt["count"]) <= max(3, 0.25 * gt["count"])
    # TRD blocks.
    assert {"focus_score", "glare_score", "plate_found", "overgrowth_detected"} <= set(p["quality"])
    assert {"overall_score", "needs_review", "reason_codes"} <= set(p["confidence"])


def test_predict_returns_annotated_image(client):
    data, _, _ = _jpeg(4)
    r = client.post("/predict", files={"file": ("disk.jpg", data, "image/jpeg")})
    assert r.status_code == 200
    png = base64.b64decode(r.json()["annotated_image"])
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_predict_rejects_garbage(client):
    r = client.post("/predict", files={"file": ("x.jpg", b"not an image", "image/jpeg")})
    assert r.status_code == 422


def test_api_key(client, monkeypatch):
    from colony_detector import service

    monkeypatch.setattr(service.app.state.settings, "api_key", "s3cret")
    data, _, _ = _jpeg(5, n=3)
    files = {"file": ("disk.jpg", data, "image/jpeg")}
    assert client.post("/predict", files=files).status_code == 401
    ok = client.post("/predict", files=files, headers={"Authorization": "Bearer s3cret"})
    assert ok.status_code == 200
