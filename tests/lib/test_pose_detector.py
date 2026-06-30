import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


from conftest import MockLandmark, MockPoseResult, blank_bgr

from openvisionkit.lib.pose_detector import PoseDetector


def det():
    return PoseDetector.__new__(PoseDetector)


def make_upright():
    lms = [MockLandmark(x=0.5, y=0.5) for _ in range(33)]
    lms[11] = MockLandmark(x=0.4, y=0.3)  # left shoulder
    lms[12] = MockLandmark(x=0.6, y=0.3)  # right shoulder
    lms[23] = MockLandmark(x=0.4, y=0.6)  # left hip
    lms[24] = MockLandmark(x=0.6, y=0.6)  # right hip
    return lms


def test_get_spine_angle_upright_near_zero():
    d = det()
    result = MockPoseResult(make_upright())
    angle = d.get_spine_angle(result)
    assert abs(angle) < 5.0


def test_get_torso_tilt_level_near_zero():
    d = det()
    result = MockPoseResult(make_upright())
    tilt = d.get_torso_tilt(result)
    assert abs(tilt) < 5.0


def test_is_hunching_false_for_upright():
    d = det()
    result = MockPoseResult(make_upright())
    assert d.is_hunching(result, threshold=20) is False


def test_is_hunching_true_for_tilted():
    d = det()
    lms = make_upright()
    lms[11] = MockLandmark(x=0.3, y=0.2)
    lms[12] = MockLandmark(x=0.7, y=0.5)
    result = MockPoseResult(lms)
    assert d.is_hunching(result, threshold=20) is True


def test_get_symmetry_score_range():
    d = det()
    result = MockPoseResult(make_upright())
    score = d.get_symmetry_score(result)
    assert 0.0 <= score <= 1.0


def test_get_symmetry_score_empty_returns_zero():
    r = MockPoseResult.__new__(MockPoseResult)
    r.pose_landmarks = []
    assert det().get_symmetry_score(r) == 0.0


def make_arms_raised():
    lms = make_upright()
    lms[15] = MockLandmark(x=0.4, y=0.1, visibility=0.9)
    lms[16] = MockLandmark(x=0.6, y=0.1, visibility=0.9)
    return lms


def test_is_arms_raised_true():
    d = det()
    assert d.is_arms_raised(MockPoseResult(make_arms_raised())) is True


def test_is_arms_raised_false_for_normal():
    d = det()
    assert d.is_arms_raised(MockPoseResult(make_upright())) is False


def test_detect_fall_false_for_standing():
    d = det()
    lms = make_upright()
    lms[0] = MockLandmark(x=0.5, y=0.1)
    result = MockPoseResult(lms)
    assert d.detect_fall(result) is False


def test_detect_fall_true_when_head_below_hips():
    d = det()
    lms = make_upright()
    lms[0] = MockLandmark(x=0.5, y=0.9)
    result = MockPoseResult(lms)
    assert d.detect_fall(result) is True


def test_is_arms_crossed_true():
    d = det()
    lms = make_upright()
    lms[15] = MockLandmark(x=0.7, y=0.5, visibility=0.9)
    lms[16] = MockLandmark(x=0.3, y=0.5, visibility=0.9)
    assert d.is_arms_crossed(MockPoseResult(lms)) is True


def test_is_arms_crossed_false_for_normal():
    d = det()
    lms = make_upright()
    lms[15] = MockLandmark(x=0.3, y=0.5, visibility=0.9)
    lms[16] = MockLandmark(x=0.7, y=0.5, visibility=0.9)
    assert d.is_arms_crossed(MockPoseResult(lms)) is False


def test_get_knee_angle_returns_float():
    d = det()
    lms = make_upright()
    lms[23] = MockLandmark(x=0.4, y=0.6)
    lms[25] = MockLandmark(x=0.4, y=0.75)
    lms[27] = MockLandmark(x=0.4, y=0.9)
    angle = d.get_knee_angle(MockPoseResult(lms), side="left")
    assert 0.0 <= angle <= 180.0


def test_get_body_bounding_box_returns_tuple():
    d = det()
    img = blank_bgr(480, 640)
    result = MockPoseResult(make_upright())
    bbox = d.get_body_bounding_box(result, img)
    assert len(bbox) == 4
    x, y, w, h = bbox
    assert w >= 0 and h >= 0
