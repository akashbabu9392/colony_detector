"""Crowded-plate confidence cut and answer-key scoring."""

import sys
from pathlib import Path

import numpy as np

from colony_detector.engines.tiled import TiledBoxEngine
from colony_detector.plate import Plate

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from score_gold import _cut, score_plate  # noqa: E402


class FakeEngine(TiledBoxEngine):
    name = "fake"

    def __init__(self, scores, **kw):
        super().__init__(global_pass=False, **kw)
        self.scores = scores

    def _predict(self, tiles):
        out = [[] for _ in tiles]
        out[0] = [(20 + 30 * i, 300, 34 + 30 * i, 314, c, "colony") for i, c in enumerate(self.scores)]
        return out


def _detect(scores, **kw):
    img = np.zeros((600, 600, 3), np.uint8)
    eng = FakeEngine(scores, tile_size=640, **kw)
    return [round(d.confidence, 2) for d in eng.detect(img, Plate(300, 300, 290, True, "test"), {})]


def test_dense_cut_only_on_crowded_plates():
    crowded = [0.9] * 5 + [0.4, 0.2]
    sparse = [0.9] * 2 + [0.4, 0.2]
    assert _detect(crowded, conf=0.85, dense_conf=0.35, dense_min=5) == [0.9] * 5 + [0.4]
    assert _detect(sparse, conf=0.85, dense_conf=0.35, dense_min=5) == [0.9] * 2
    # The scorer applies the same rule to cached detections.
    dets = np.array([(0, 0, 5, c) for c in crowded])
    assert len(_cut(dets, 0.85, 0.35, 5)) == 6
    assert len(_cut(dets[2:], 0.85, 0.35, 5)) == 3


def test_unsure_spots_and_specks_are_not_penalised():
    gold = {"image": "p.jpg", "px_per_mm": 20.0, "count": 2, "plate_flag": None,
            "points": [[100, 100, 20], [300, 300, 20]], "unsure": [[500, 500, 20]],
            "small_specks": [[700, 700, 4]]}
    # Both colonies + the unsure spot: count 3 is inside [2, 3], no false positive.
    pred = np.array([[100, 100, 20], [300, 300, 20], [500, 500, 20]], float)
    row = score_plate(gold, pred, 0.5, 0.3)
    assert row["error"] == 0 and row["fp"] == 0 and row["tp"] == 2
    # A speck below 0.5 mm (radius 4 px = 0.4 mm) is flagged, not counted.
    row = score_plate(gold, np.array([[100, 100, 20], [700, 700, 4]], float), 0.5, 0.3)
    assert row["pred"] == 1 and row["pred_specks"] == 1 and row["error"] == -1 and row["fn"] == 1
