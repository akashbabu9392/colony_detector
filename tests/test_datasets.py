import json

import cv2
import numpy as np

from colony_detector.datasets import load_dataset


def _img(path, w=200, h=100):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.full((h, w, 3), 128, np.uint8))


def test_coco_roboflow_layout(tmp_path):
    d = tmp_path / "ds" / "test"
    _img(d / "a.jpg")
    json.dump({
        "images": [{"id": 0, "file_name": "a.jpg", "width": 200, "height": 100}],
        "categories": [{"id": 0, "name": "colonies"}, {"id": 1, "name": "colony"},
                       {"id": 2, "name": "bubble"}, {"id": 3, "name": "fuzzy colony"}],
        "annotations": [
            {"id": 0, "image_id": 0, "category_id": 1, "bbox": [10, 10, 8, 8]},
            {"id": 1, "image_id": 0, "category_id": 2, "bbox": [50, 50, 8, 8]},
            {"id": 2, "image_id": 0, "category_id": 3, "bbox": [90, 20, 10, 10]},
        ],
    }, open(d / "_annotations.coco.json", "w"))
    (s,) = load_dataset(tmp_path / "ds")
    assert s.split == "test"
    assert s.total == 2  # the bubble is not a colony
    assert sorted(b[4] for b in s.boxes) == [0, 1]
    assert s.points()[0]["x"] == 14


def test_yolo_layout(tmp_path):
    root = tmp_path / "ds"
    _img(root / "train" / "images" / "p1.png", 200, 100)
    (root / "train" / "labels").mkdir(parents=True)
    (root / "train" / "labels" / "p1.txt").write_text("0 0.5 0.5 0.1 0.2\n0 0.1 0.1 0.05 0.1\n")
    (root / "data.yaml").write_text("names: ['colony']\n")
    (s,) = load_dataset(root)
    assert s.split == "train" and s.total == 2
    x0, y0, x1, y1, c = s.boxes[0]
    assert abs(x0 - 90) < 1e-6 and abs(y1 - 60) < 1e-6 and c == 0


def test_agar_and_counts(tmp_path):
    root = tmp_path / "agar"
    _img(root / "17.jpg")
    json.dump({"colonies_number": 2, "labels": [
        {"x": 5, "y": 5, "width": 10, "height": 10, "class": "E.coli"},
        {"x": 50, "y": 5, "width": 10, "height": 10, "class": "E.coli"}]}, open(root / "17.json", "w"))
    _img(root / "18.jpg")
    (root / "counts.csv").write_text("image,count\n18.jpg,41\n")
    got = {s.image.name: s for s in load_dataset(root)}
    assert got["17.jpg"].total == 2 and len(got["17.jpg"].boxes) == 2
    assert got["18.jpg"].total == 41 and got["18.jpg"].boxes is None
