import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json

from conftest import MockDetection, MockObjectResult, blank_bgr

from visionkit.lib.object_detector import ObjectDetector


def det():
    return ObjectDetector.__new__(ObjectDetector)


def make_result(*args):
    return MockObjectResult([MockDetection(*a) for a in args])


def test_filter_by_confidence_removes_low():
    d = det()
    result = make_result(("cat", 0.9), ("dog", 0.3))
    filtered = d.filter_by_confidence(result, threshold=0.5)
    assert len(filtered) == 1
    assert filtered[0].categories[0].category_name == "cat"


def test_get_bounding_boxes_structure():
    d = det()
    result = make_result(("person", 0.8, 10, 20, 50, 60))
    boxes = d.get_bounding_boxes(result)
    assert len(boxes) == 1
    assert boxes[0]["label"] == "person"
    assert boxes[0]["bbox"] == (10, 20, 50, 60)


def test_is_crowded_true():
    d = det()
    result = make_result(*[("p", 0.9)] * 6)
    assert d.is_crowded(result, threshold=5) is True


def test_is_crowded_false():
    d = det()
    result = make_result(("p", 0.9), ("q", 0.8))
    assert d.is_crowded(result, threshold=5) is False


def test_get_objects_by_size_sorted():
    d = det()
    result = make_result(("a", 0.9, 0, 0, 10, 10), ("b", 0.9, 0, 0, 50, 50))
    sorted_dets = d.get_objects_by_size(result)
    assert sorted_dets[0].bounding_box.width == 50


def test_get_proximity_between_two_dets():
    d = det()
    a = MockDetection(x=0, y=0, w=10, h=10)  # center (5,5)
    b = MockDetection(x=90, y=0, w=10, h=10)  # center (95,5)
    prox = d.get_proximity(a, b)
    assert abs(prox - 90.0) < 1.0


def test_detect_line_crossing_returns_crossing():
    d = det()
    result = make_result(("car", 0.9, 90, 40, 20, 20))  # center (100, 50)
    crossing = d.detect_line_crossing(result, (100, 0), (100, 200), line_threshold=15)
    assert len(crossing) == 1


def test_export_to_json_valid():
    d = det()
    result = make_result(("cat", 0.75, 5, 10, 30, 40))
    data = d.export_to_json(result)
    assert "detections" in data
    json.dumps(data)  # must serialize


def test_batch_detect_returns_per_frame(monkeypatch):
    d = det()
    monkeypatch.setattr(d, "detect", lambda img, **kw: (MockObjectResult([]), None))
    results = d.batch_detect([blank_bgr(), blank_bgr()])
    assert len(results) == 2
