import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json

from conftest import blank_bgr, make_hand_landmarks

from visionkit.lib.hand_detector import HandDetector


def det():
    return HandDetector.__new__(HandDetector)


def make_fingers(thumb=0, index=0, middle=0, ring=0, little=0):
    return [thumb, index, middle, ring, little]


def test_is_ok_sign_true_when_close_and_others_up(monkeypatch):
    d = det()
    lms = make_hand_landmarks()
    # Thumb tip (idx 4) and index tip (idx 8) are close in fixture
    monkeypatch.setattr(d, "fingers_up", lambda h: make_fingers(0, 0, 1, 1, 1))
    assert d.is_ok_sign(lms) is True


def test_is_ok_sign_false_when_fingers_down(monkeypatch):
    d = det()
    lms = make_hand_landmarks()
    monkeypatch.setattr(d, "fingers_up", lambda h: make_fingers(0, 0, 0, 0, 0))
    assert d.is_ok_sign(lms) is False


def test_is_call_me_true(monkeypatch):
    d = det()
    lms = make_hand_landmarks()
    monkeypatch.setattr(d, "fingers_up", lambda h: make_fingers(1, 0, 0, 0, 1))
    assert d.is_call_me(lms) is True


def test_is_call_me_false(monkeypatch):
    d = det()
    lms = make_hand_landmarks()
    monkeypatch.setattr(d, "fingers_up", lambda h: make_fingers(1, 1, 0, 0, 1))
    assert d.is_call_me(lms) is False


def test_is_rock_sign_true(monkeypatch):
    d = det()
    lms = make_hand_landmarks()
    monkeypatch.setattr(d, "fingers_up", lambda h: make_fingers(0, 1, 0, 0, 1))
    assert d.is_rock_sign(lms) is True


def test_is_rock_sign_false(monkeypatch):
    d = det()
    lms = make_hand_landmarks()
    monkeypatch.setattr(d, "fingers_up", lambda h: make_fingers(0, 1, 1, 0, 1))
    assert d.is_rock_sign(lms) is False


def test_recognize_number_delegates(monkeypatch):
    d = det()
    lms = make_hand_landmarks()
    monkeypatch.setattr(d, "get_finger_count", lambda h: 3)
    assert d.recognize_number(lms) == 3


def make_oriented_landmarks(direction="up"):
    lms = [[i, 0.5, 0.5, 0.0] for i in range(21)]
    if direction == "up":
        lms[0][1], lms[0][2] = 0.5, 0.8  # wrist low
        lms[9][1], lms[9][2] = 0.5, 0.2  # MCP high
    elif direction == "right":
        lms[0][1], lms[0][2] = 0.2, 0.5
        lms[9][1], lms[9][2] = 0.8, 0.5
    return lms


def test_get_hand_orientation_pointing_up():
    d = det()
    lms = make_oriented_landmarks("up")
    assert d.get_hand_orientation(lms) == "palm_up"


def test_get_hand_orientation_pointing_right():
    d = det()
    lms = make_oriented_landmarks("right")
    assert d.get_hand_orientation(lms) == "palm_right"


def test_get_swipe_direction_right():
    d = det()
    assert d.get_swipe_direction((100, 200), (150, 205), threshold=20) == "right"


def test_get_swipe_direction_left():
    d = det()
    assert d.get_swipe_direction((200, 200), (150, 202), threshold=20) == "left"


def test_get_swipe_direction_none_below_threshold():
    d = det()
    assert d.get_swipe_direction((100, 100), (105, 103), threshold=20) == "none"


def test_get_all_finger_angles_has_five_keys(monkeypatch):
    d = det()
    lms = make_hand_landmarks()
    monkeypatch.setattr(d, "get_angle_between_landmarks", lambda lm, a, b, c: 90.0)
    angles = d.get_all_finger_angles(lms)
    assert set(angles.keys()) == {"thumb", "index", "middle", "ring", "little"}


def test_draw_gesture_label_same_shape():
    d = det()
    img = blank_bgr()
    hand_data = {"bounding_box": (50, 60, 100, 120), "center_point": (100, 120)}
    result = d.draw_gesture_label(img, hand_data, "OK")
    assert result.shape == img.shape


def test_to_json_serializable():
    d = det()
    lms = make_hand_landmarks()
    hand_data = {
        "landmarks_list": lms,
        "bounding_box": (50, 60, 100, 120),
        "center_point": (100, 120),
        "hand_type": "Right",
    }
    data = d.to_json(hand_data)
    assert json.dumps(data)
    assert data["hand_type"] == "Right"
    assert len(data["landmarks"]) == 21
