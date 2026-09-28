import numpy as np

from colony_detector.engines.tiled import TiledBoxEngine, make_tiles, map_class, merge_boxes
from colony_detector.fusion import fuse
from colony_detector.plate import Plate
from colony_detector.types import COLONY, FUZZY, Detection


def det(x, y, r=5, c=0.9, src="a", count=1, contour=None):
    return Detection(cx=x, cy=y, radius=r, confidence=c, source=src, count=count, contour=contour)


def test_fusion_consensus_and_veto():
    a = [det(10, 10, src="a"), det(50, 50, c=0.3, src="a"), det(90, 90, src="a")]
    b = [det(11, 10, src="b"), det(90.5, 91, src="b"), det(200, 200, c=0.95, src="b")]
    out = fuse({"a": a, "b": b}, {"a": 1.0, "b": 1.0}, threshold=0.3)
    def near(x, y):
        return any(abs(d.cx - x) <= 2 and abs(d.cy - y) <= 2 for d in out)

    # Agreed colonies kept; weak single vote (0.15) dropped; confident single kept.
    assert near(10, 10) and near(90, 90) and near(200, 200)
    assert not near(50, 50)
    assert len(out) == 3
    agreed = [d for d in out if abs(d.cx - 10) <= 2][0]
    assert set(agreed.extra["votes"]) == {"a", "b"}


def test_fusion_single_engine_passthrough():
    a = [det(10, 10, c=0.1)]
    assert len(fuse({"a": a}, {"a": 1.0}, threshold=0.9)) == 1


def test_fusion_clump_absorbs_or_yields():
    square = np.array([[0, 0], [40, 0], [40, 40], [0, 40]], float)
    clump = det(20, 20, r=20, count=3, contour=square, src="classical")
    # One learned detection inside: the clump estimate stands, no double count.
    out = fuse({"classical": [clump], "yolo": [det(20, 20, r=6, src="yolo")]},
               {"classical": 0.8, "yolo": 1.2}, 0.25)
    assert sum(d.count for d in out) == 3
    # Learned engine separates it into 3: individuals replace the estimate.
    clump2 = det(20, 20, r=20, count=3, contour=square.copy(), src="classical")
    yolo = [det(8, 8, r=6, src="yolo"), det(30, 10, r=6, src="yolo"), det(20, 32, r=6, src="yolo")]
    out = fuse({"classical": [clump2], "yolo": yolo}, {"classical": 0.8, "yolo": 1.2}, 0.25)
    assert sum(d.count for d in out) == 3
    assert all(d.count == 1 for d in out)


def test_make_tiles_cover_image():
    tiles = make_tiles(1500, 1000, 640, 0.25)
    cover = np.zeros((1000, 1500), bool)
    for x0, y0, x1, y1 in tiles:
        assert x1 - x0 == 640 and y1 - y0 == 640
        cover[y0:y1, x0:x1] = True
    assert cover.all()
    assert make_tiles(300, 200, 640, 0.25) == [(0, 0, 300, 200)]


def test_merge_boxes_removes_seam_duplicates():
    boxes = [(100, 100, 120, 120, 0.9, "colony"), (101, 100, 121, 120, 0.8, "colony"),
             (100, 100, 110, 120, 0.7, "colony"), (300, 300, 320, 320, 0.6, "colony")]
    out = merge_boxes(boxes)
    assert len(out) == 2
    assert out[0][4] == 0.9


def test_class_mapping():
    assert map_class("colony") == COLONY
    assert map_class("Fuzzy_Colony") == FUZZY
    assert map_class("bubble") is None


class FakeDetector(TiledBoxEngine):
    """Predicts a box wherever a dark pixel blob is centred (per tile)."""

    name = "fake"

    def _predict(self, tiles):
        out = []
        for t in tiles:
            ys, xs = np.nonzero(t[..., 0] < 50)
            boxes = []
            if len(xs):
                from scipy import ndimage as ndi

                lab, n = ndi.label(t[..., 0] < 50)
                for sl in ndi.find_objects(lab):
                    boxes.append((sl[1].start, sl[0].start, sl[1].stop, sl[0].stop, 0.9, "colony"))
            out.append(boxes)
        return out


def test_tiled_engine_global_coordinates():
    img = np.full((1400, 1400, 3), 200, np.uint8)
    pts = [(300, 300), (700, 700), (1100, 400), (640, 650), (660, 1000)]
    for x, y in pts:
        img[y - 5:y + 5, x - 5:x + 5] = 0
    eng = FakeDetector(tile_size=512, overlap=0.25, conf=0.2, global_pass=False)
    plate = Plate(700, 700, 690, True, "test")
    dets = eng.detect(img, plate, {})
    got = sorted((round(d.cx), round(d.cy)) for d in dets)
    assert len(got) == len(pts)
    for (x, y), (gx, gy) in zip(sorted(pts), got):
        assert abs(gx - x) <= 1 and abs(gy - y) <= 1
