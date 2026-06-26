# tests/conftest.py
import numpy as np


def blank_bgr(h=300, w=400):
    return np.full((h, w, 3), 128, dtype=np.uint8)


def make_face_detection(score=0.9, x=100, y=80, w=80, h=90):
    return {
        "id": 0,
        "score": score,
        "bbox": (x, y, w, h),
        "bbox_xyxy": (x, y, x + w, y + h),
        "center": (x + w // 2, y + h // 2),
        "area": w * h,
        "normalized_keypoints": [],
    }


def make_blend(**kwargs):
    base = {
        "mouthSmileLeft": 0.0,
        "mouthSmileRight": 0.0,
        "browInnerUp": 0.0,
        "browDownLeft": 0.0,
        "browDownRight": 0.0,
        "eyeBlinkLeft": 0.0,
        "eyeBlinkRight": 0.0,
        "jawOpen": 0.0,
        "mouthFrownLeft": 0.0,
        "mouthFrownRight": 0.0,
    }
    base.update(kwargs)
    return base


def make_face_478(img_w=640, img_h=480):
    """478 landmarks as [[x_px, y_px]] centered on image."""
    rng = np.random.default_rng(42)
    return [
        [
            img_w // 2 + int(rng.integers(-120, 120)),
            img_h // 2 + int(rng.integers(-120, 120)),
        ]
        for _ in range(478)
    ]


def make_hand_landmarks(n=21):
    """21 landmarks as [[id, x_norm, y_norm, z_norm]]."""
    rng = np.random.default_rng(0)
    lms = [[i, float(rng.random()), float(rng.random()), 0.0] for i in range(n)]
    # Force thumb tip (4) close to index tip (8) for OK-sign fixture
    lms[4][1], lms[4][2] = 0.5, 0.5
    lms[8][1], lms[8][2] = 0.53, 0.51
    return lms


class MockLandmark:
    def __init__(self, x=0.5, y=0.5, z=0.0, visibility=0.9):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility


class MockPoseResult:
    def __init__(self, landmarks=None):
        lms = [MockLandmark() for _ in range(33)] if landmarks is None else landmarks
        self.pose_landmarks = [lms]


class MockBBox:
    def __init__(self, x=50, y=40, w=60, h=70):
        self.origin_x = x
        self.origin_y = y
        self.width = w
        self.height = h


class MockCategory:
    def __init__(self, name="person", score=0.9):
        self.category_name = name
        self.score = score


class MockDetection:
    def __init__(self, name="person", score=0.9, x=50, y=40, w=60, h=70):
        self.categories = [MockCategory(name, score)]
        self.bounding_box = MockBBox(x, y, w, h)


class MockObjectResult:
    def __init__(self, detections=None):
        self.detections = detections or []
