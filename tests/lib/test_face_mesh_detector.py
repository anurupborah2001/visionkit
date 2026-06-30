import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from conftest import blank_bgr, make_blend, make_face_478

from openvisionkit.lib.face_mesh_detector import FaceMeshDetector


def det():
    return FaceMeshDetector.__new__(FaceMeshDetector)


FACE = make_face_478()


def test_is_smiling_true_when_high_scores():
    d = det()
    blend = make_blend(mouthSmileLeft=0.6, mouthSmileRight=0.55)
    assert d.is_smiling(blend, threshold=0.4) is True


def test_is_smiling_false_when_neutral():
    d = det()
    assert d.is_smiling(make_blend(), threshold=0.4) is False


def test_is_yawning_delegates_to_mouth_ratio(monkeypatch):
    d = det()
    monkeypatch.setattr(d, "get_mouth_openness_ratio", lambda face: 0.7)
    assert d.is_yawning(FACE, ratio_threshold=0.5) is True


def test_is_yawning_false_when_closed(monkeypatch):
    d = det()
    monkeypatch.setattr(d, "get_mouth_openness_ratio", lambda face: 0.2)
    assert d.is_yawning(FACE, ratio_threshold=0.5) is False


def test_is_surprised_both_conditions_met(monkeypatch):
    d = det()
    monkeypatch.setattr(d, "get_eyebrow_raise", lambda b: 0.5)
    monkeypatch.setattr(d, "get_mouth_openness_ratio", lambda f: 0.4)
    assert d.is_surprised(make_blend(), FACE) is True


def test_is_surprised_false_without_brow(monkeypatch):
    d = det()
    monkeypatch.setattr(d, "get_eyebrow_raise", lambda b: 0.1)
    monkeypatch.setattr(d, "get_mouth_openness_ratio", lambda f: 0.4)
    assert d.is_surprised(make_blend(), FACE) is False


def test_get_eyebrow_raise_value():
    d = det()
    blend = make_blend(browInnerUp=0.42)
    assert abs(d.get_eyebrow_raise(blend) - 0.42) < 1e-6


def test_is_eyes_closed_both_low(monkeypatch):
    d = det()
    monkeypatch.setattr(d, "get_eye_aspect_ratio", lambda face, eye="left": 0.15)
    assert d.is_eyes_closed(FACE, ear_threshold=0.22) is True


def test_is_eyes_closed_one_open(monkeypatch):
    d = det()

    def mock_ear(face, eye="left"):
        return 0.15 if eye == "left" else 0.35

    monkeypatch.setattr(d, "get_eye_aspect_ratio", mock_ear)
    assert d.is_eyes_closed(FACE, ear_threshold=0.22) is False


def test_is_drowsy_mirrors_is_eyes_closed(monkeypatch):
    d = det()
    monkeypatch.setattr(d, "get_eye_aspect_ratio", lambda face, eye="left": 0.10)
    assert d.is_drowsy(FACE) is True


def test_get_face_bounding_box_returns_xywh():
    d = det()
    face = [[100, 200], [150, 250], [120, 220]] + [[130, 230]] * 475
    x, y, w, h = d.get_face_bounding_box(face)
    assert x == 100
    assert y == 200
    assert w == 50
    assert h == 50


def test_get_face_symmetry_score_range():
    d = det()
    score = d.get_face_symmetry_score(FACE)
    assert 0.0 <= score <= 1.0


def test_draw_face_oval_returns_same_shape():
    d = det()
    img = blank_bgr()
    result = d.draw_face_oval(img, FACE)
    assert result.shape == img.shape
    assert result is not img


def test_get_attention_level_range(monkeypatch):
    d = det()
    monkeypatch.setattr(d, "is_looking_at_camera", lambda face, **kw: True)
    monkeypatch.setattr(d, "is_eyes_closed", lambda face, **kw: False)
    score = d.get_attention_level(FACE, make_blend())
    assert 0.0 <= score <= 1.0


def test_get_lip_separation_positive():
    d = det()
    face = [[i * 2, i * 2] for i in range(478)]
    sep = d.get_lip_separation(face)
    assert sep >= 0.0
