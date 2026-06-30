# OpenVisionKit Utility Methods Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 87 utility methods across 11 detector/segmentation classes in `openvisionkit/lib/` to cover real-world use cases without model inference in tests.

**Architecture:** All methods added in-place to existing class files. New methods operate on already-parsed results (detection dicts, numpy arrays, landmark lists) — no model files needed in tests. Tests use `ClassName.__new__(ClassName)` to skip `__init__` and avoid model loading.

**Tech Stack:** Python 3.11+, OpenCV (`cv2`), NumPy, MediaPipe, pytest, uv

---

## Task 0: Test infrastructure

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/lib/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p tests/lib
touch tests/__init__.py tests/lib/__init__.py
```

- [ ] **Step 2: Write conftest.py**

```python
# tests/conftest.py
import numpy as np
import pytest


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
        "mouthSmileLeft": 0.0, "mouthSmileRight": 0.0,
        "browInnerUp": 0.0, "browDownLeft": 0.0, "browDownRight": 0.0,
        "eyeBlinkLeft": 0.0, "eyeBlinkRight": 0.0, "jawOpen": 0.0,
        "mouthFrownLeft": 0.0, "mouthFrownRight": 0.0,
    }
    base.update(kwargs)
    return base


def make_face_478(img_w=640, img_h=480):
    """478 landmarks as [[x_px, y_px]] centered on image."""
    rng = np.random.default_rng(42)
    return [
        [img_w // 2 + int(rng.integers(-120, 120)),
         img_h // 2 + int(rng.integers(-120, 120))]
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
        if landmarks is None:
            lms = [MockLandmark() for _ in range(33)]
        else:
            lms = landmarks
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
```

- [ ] **Step 3: Verify pytest discovers tests**

```bash
uv run pytest tests/ --collect-only
```

Expected: `no tests ran` (no test files yet — just confirming collection works)

- [ ] **Step 4: Commit**

```bash
git add tests/
git commit -m "test: add test infrastructure and shared fixtures"
```

---

## Task 1: FaceDetector — privacy and crop utilities

**Spec:** `pixelate_faces`, `is_frontal`, `get_padded_crop`

**Files:**
- Create: `tests/lib/test_face_detector.py`
- Modify: `openvisionkit/lib/face_detector.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/lib/test_face_detector.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np
import pytest
from openvisionkit.lib.face_detector import FaceDetector
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from conftest import blank_bgr, make_face_detection


def det():
    return FaceDetector.__new__(FaceDetector)


def test_pixelate_faces_changes_face_region():
    d = det()
    img = blank_bgr()
    img[80:170, 100:180] = (200, 100, 50)  # distinct face color
    detection = make_face_detection(x=100, y=80, w=80, h=90)
    result = d.pixelate_faces(img, [detection], block_size=8)
    assert result.shape == img.shape
    # Pixelated region should differ from original
    assert not np.array_equal(result[80:170, 100:180], img[80:170, 100:180])


def test_pixelate_faces_no_detections_returns_copy():
    d = det()
    img = blank_bgr()
    result = d.pixelate_faces(img, [])
    assert np.array_equal(result, img)
    assert result is not img


def test_is_frontal_above_threshold():
    d = det()
    assert d.is_frontal(make_face_detection(score=0.9), threshold=0.8) is True


def test_is_frontal_below_threshold():
    d = det()
    assert d.is_frontal(make_face_detection(score=0.7), threshold=0.8) is False


def test_get_padded_crop_shape():
    d = det()
    img = blank_bgr(300, 400)
    detection = make_face_detection(x=100, y=80, w=80, h=90)
    crop = d.get_padded_crop(img, detection, pad_ratio=0.2)
    assert crop.ndim == 3
    assert crop.shape[0] > 90  # taller than unpadded
    assert crop.shape[1] > 80


def test_get_padded_crop_clipped_at_boundary():
    d = det()
    img = blank_bgr(100, 100)
    detection = make_face_detection(x=0, y=0, w=50, h=50)
    crop = d.get_padded_crop(img, detection, pad_ratio=0.5)
    assert crop.shape[0] <= 100
    assert crop.shape[1] <= 100
```

- [ ] **Step 2: Run to confirm failure**

```bash
uv run pytest tests/lib/test_face_detector.py -v 2>&1 | head -30
```

Expected: `AttributeError: pixelate_faces` or similar

- [ ] **Step 3: Implement methods in `openvisionkit/lib/face_detector.py`**

Add after the last existing method:

```python
    def pixelate_faces(self, image, detections, block_size=10):
        out = image.copy()
        for det in detections:
            x, y, w, h = det["bbox"]
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(image.shape[1], x + w)
            y2 = min(image.shape[0], y + h)
            if x2 <= x1 or y2 <= y1:
                continue
            roi = out[y1:y2, x1:x2]
            small_w = max(1, (x2 - x1) // block_size)
            small_h = max(1, (y2 - y1) // block_size)
            small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
            out[y1:y2, x1:x2] = cv2.resize(small, (x2 - x1, y2 - y1),
                                            interpolation=cv2.INTER_NEAREST)
        return out

    def is_frontal(self, detection, threshold=0.8):
        return detection["score"] >= threshold

    def get_padded_crop(self, image, detection, pad_ratio=0.2):
        x, y, w, h = detection["bbox"]
        pad_x = int(w * pad_ratio)
        pad_y = int(h * pad_ratio)
        h_img, w_img = image.shape[:2]
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w_img, x + w + pad_x)
        y2 = min(h_img, y + h + pad_y)
        return image[y1:y2, x1:x2].copy()
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
uv run pytest tests/lib/test_face_detector.py::test_pixelate_faces_changes_face_region tests/lib/test_face_detector.py::test_is_frontal_above_threshold tests/lib/test_face_detector.py::test_get_padded_crop_shape -v
```

Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add openvisionkit/lib/face_detector.py tests/lib/test_face_detector.py
git commit -m "feat(face_detector): add pixelate_faces, is_frontal, get_padded_crop"
```

---

## Task 2: FaceDetector — tracking and batch utilities

**Spec:** `draw_face_ids`, `get_attention_score`, `batch_detect`, `save_crops`

**Files:**
- Modify: `tests/lib/test_face_detector.py`
- Modify: `openvisionkit/lib/face_detector.py`

- [ ] **Step 1: Append failing tests**

```python
# Append to tests/lib/test_face_detector.py

def test_draw_face_ids_returns_image_with_same_shape():
    d = det()
    img = blank_bgr()
    tracked = [{"bbox": (100, 80, 80, 90), "id": 1}]
    result = d.draw_face_ids(img, tracked)
    assert result.shape == img.shape
    assert result is not img


def test_get_attention_score_zero_when_no_detections():
    d = det()
    assert d.get_attention_score([], 640, 480) == 0.0


def test_get_attention_score_returns_float_0_to_1():
    d = det()
    score = d.get_attention_score([make_face_detection(x=280, y=200, w=80, h=80)], 640, 480)
    assert 0.0 <= score <= 1.0


def test_get_attention_score_centered_face_higher_than_corner():
    d = det()
    center = d.get_attention_score([make_face_detection(x=280, y=200, w=80, h=80)], 640, 480)
    corner = d.get_attention_score([make_face_detection(x=0, y=0, w=20, h=20)], 640, 480)
    assert center > corner


def test_batch_detect_returns_list_per_frame():
    d = det()
    # Patch detect_faces to avoid model dependency
    d.detect_faces = lambda img, **kw: (img, [make_face_detection()])
    imgs = [blank_bgr(), blank_bgr()]
    results = d.batch_detect(imgs)
    assert len(results) == 2
    assert isinstance(results[0], list)


def test_save_crops_creates_files(tmp_path):
    d = det()
    img = blank_bgr(300, 400)
    detections = [make_face_detection(x=100, y=80, w=80, h=90)]
    paths = d.save_crops(img, detections, str(tmp_path), prefix="face")
    assert len(paths) == 1
    assert os.path.exists(paths[0])
    assert paths[0].endswith("face_0.png")
```

- [ ] **Step 2: Run to confirm failure**

```bash
uv run pytest tests/lib/test_face_detector.py -v 2>&1 | tail -15
```

Expected: new tests fail with `AttributeError`

- [ ] **Step 3: Implement methods**

```python
    def draw_face_ids(self, image, tracked_faces):
        out = image.copy()
        for face in tracked_faces:
            x, y, w, h = face["bbox"]
            face_id = face.get("id", 0)
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(out, f"ID:{face_id}", (x, max(y - 10, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return out

    def get_attention_score(self, detections, img_w, img_h):
        if not detections:
            return 0.0
        img_area = img_w * img_h
        img_cx, img_cy = img_w / 2.0, img_h / 2.0
        max_dist = (img_cx ** 2 + img_cy ** 2) ** 0.5
        scores = []
        for det in detections:
            area_ratio = min(det["area"] / img_area, 1.0)
            cx, cy = det["center"]
            dist = ((cx - img_cx) ** 2 + (cy - img_cy) ** 2) ** 0.5
            centrality = 1.0 - min(dist / max_dist, 1.0) if max_dist > 0 else 1.0
            scores.append(0.5 * area_ratio + 0.5 * centrality)
        return float(max(scores))

    def batch_detect(self, images):
        return [self.detect_faces(img)[1] for img in images]

    def save_crops(self, image, detections, output_dir, prefix="face"):
        import os
        os.makedirs(output_dir, exist_ok=True)
        paths = []
        for i, det in enumerate(detections):
            crop = self.get_padded_crop(image, det, pad_ratio=0.1)
            path = os.path.join(output_dir, f"{prefix}_{i}.png")
            cv2.imwrite(path, crop)
            paths.append(path)
        return paths
```

- [ ] **Step 4: Run all FaceDetector tests**

```bash
uv run pytest tests/lib/test_face_detector.py -v
```

Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add openvisionkit/lib/face_detector.py tests/lib/test_face_detector.py
git commit -m "feat(face_detector): add draw_face_ids, get_attention_score, batch_detect, save_crops"
```

---

## Task 3: FaceMeshDetector — expression detection

**Spec:** `is_smiling`, `is_yawning`, `is_surprised`, `get_eyebrow_raise`, `is_eyes_closed`, `is_drowsy`

**Files:**
- Create: `tests/lib/test_face_mesh_detector.py`
- Modify: `openvisionkit/lib/face_mesh_detector.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/lib/test_face_mesh_detector.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from openvisionkit.lib.face_mesh_detector import FaceMeshDetector
from conftest import make_blend, make_face_478


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
```

- [ ] **Step 2: Run to confirm failure**

```bash
uv run pytest tests/lib/test_face_mesh_detector.py -v 2>&1 | tail -15
```

- [ ] **Step 3: Implement methods in `openvisionkit/lib/face_mesh_detector.py`**

```python
    def is_smiling(self, blend, threshold=0.4):
        left = blend.get("mouthSmileLeft", 0.0)
        right = blend.get("mouthSmileRight", 0.0)
        return (left + right) / 2.0 > threshold

    def is_yawning(self, face, ratio_threshold=0.5):
        return self.get_mouth_openness_ratio(face) > ratio_threshold

    def is_surprised(self, blend, face, brow_threshold=0.3, mouth_threshold=0.3):
        return (self.get_eyebrow_raise(blend) > brow_threshold and
                self.get_mouth_openness_ratio(face) > mouth_threshold)

    def get_eyebrow_raise(self, blend):
        return float(blend.get("browInnerUp", 0.0))

    def is_eyes_closed(self, face, ear_threshold=0.22):
        left_ear = self.get_eye_aspect_ratio(face, eye="left")
        right_ear = self.get_eye_aspect_ratio(face, eye="right")
        return left_ear < ear_threshold and right_ear < ear_threshold

    def is_drowsy(self, face, ear_threshold=0.22):
        return self.is_eyes_closed(face, ear_threshold=ear_threshold)
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
uv run pytest tests/lib/test_face_mesh_detector.py -v
```

- [ ] **Step 5: Commit**

```bash
git add openvisionkit/lib/face_mesh_detector.py tests/lib/test_face_mesh_detector.py
git commit -m "feat(face_mesh_detector): add expression detection methods"
```

---

## Task 4: FaceMeshDetector — geometry and composite

**Spec:** `get_face_bounding_box`, `get_face_symmetry_score`, `draw_face_oval`, `get_attention_level`, `get_lip_separation`

**Files:**
- Modify: `tests/lib/test_face_mesh_detector.py`
- Modify: `openvisionkit/lib/face_mesh_detector.py`

- [ ] **Step 1: Append failing tests**

```python
# Append to tests/lib/test_face_mesh_detector.py
import numpy as np
from conftest import blank_bgr


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
```

- [ ] **Step 2: Run to confirm failure**

```bash
uv run pytest tests/lib/test_face_mesh_detector.py -v 2>&1 | tail -10
```

- [ ] **Step 3: Implement methods**

```python
    _SYMMETRY_PAIRS = [
        (33, 263), (160, 387), (158, 385), (133, 362),
        (144, 374), (145, 375), (153, 380), (154, 381),
        (61, 291), (185, 409), (40, 270), (37, 267),
    ]

    def get_face_bounding_box(self, face):
        xs = [p[0] for p in face]
        ys = [p[1] for p in face]
        x = int(min(xs))
        y = int(min(ys))
        w = int(max(xs)) - x
        h = int(max(ys)) - y
        return (x, y, w, h)

    def get_face_symmetry_score(self, face):
        if not face:
            return 0.0
        xs = [p[0] for p in face]
        ys = [p[1] for p in face]
        cx = sum(xs) / len(xs)
        y_range = max(max(ys) - min(ys), 1)
        diffs = []
        for l_idx, r_idx in self._SYMMETRY_PAIRS:
            if l_idx < len(face) and r_idx < len(face):
                lx, ly = face[l_idx]
                rx, ry = face[r_idx]
                mirrored_lx = 2 * cx - lx
                dx = abs(mirrored_lx - rx) / max(cx, 1)
                dy = abs(ly - ry) / y_range
                diffs.append((dx + dy) / 2)
        if not diffs:
            return 0.0
        return float(max(0.0, 1.0 - sum(diffs) / len(diffs)))

    def draw_face_oval(self, image, face):
        out = image.copy()
        x, y, w, h = self.get_face_bounding_box(face)
        cx, cy = x + w // 2, y + h // 2
        cv2.ellipse(out, (cx, cy), (max(1, w // 2), max(1, h // 2)),
                    0, 0, 360, (0, 255, 0), 2)
        return out

    def get_attention_level(self, face, blend):
        looking = self.is_looking_at_camera(face)
        gaze_score = 1.0 if looking else 0.3
        eye_penalty = 0.5 if self.is_eyes_closed(face) else 0.0
        return float(max(0.0, gaze_score - eye_penalty))

    def get_lip_separation(self, face):
        if len(face) < 15:
            return 0.0
        upper = face[self.UPPER_LIP_CENTER]
        lower = face[self.LOWER_LIP_CENTER]
        return float(self.euclidean_distance(upper, lower))
```

- [ ] **Step 4: Run all FaceMeshDetector tests**

```bash
uv run pytest tests/lib/test_face_mesh_detector.py -v
```

- [ ] **Step 5: Commit**

```bash
git add openvisionkit/lib/face_mesh_detector.py tests/lib/test_face_mesh_detector.py
git commit -m "feat(face_mesh_detector): add geometry and composite methods"
```

---

## Task 5: HandDetector — gesture recognition

**Spec:** `is_ok_sign`, `is_call_me`, `is_rock_sign`, `recognize_number`

**Files:**
- Create: `tests/lib/test_hand_detector.py`
- Modify: `openvisionkit/lib/hand_detector.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/lib/test_hand_detector.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from openvisionkit.lib.hand_detector import HandDetector
from conftest import make_hand_landmarks


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
```

- [ ] **Step 2: Run to confirm failure**

```bash
uv run pytest tests/lib/test_hand_detector.py -v 2>&1 | tail -15
```

- [ ] **Step 3: Implement methods in `openvisionkit/lib/hand_detector.py`**

```python
    def is_ok_sign(self, hand_landmarks):
        fingers = self.fingers_up(hand_landmarks)
        t = hand_landmarks[4][1:3]
        i = hand_landmarks[8][1:3]
        dist = self.euclidean_distance(t, i)
        return dist < 0.08 and fingers[2] == 1 and fingers[3] == 1 and fingers[4] == 1

    def is_call_me(self, hand_landmarks):
        fingers = self.fingers_up(hand_landmarks)
        return (fingers[0] == 1 and fingers[1] == 0 and
                fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 1)

    def is_rock_sign(self, hand_landmarks):
        fingers = self.fingers_up(hand_landmarks)
        return (fingers[0] == 0 and fingers[1] == 1 and
                fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 1)

    def recognize_number(self, hand_landmarks):
        return self.get_finger_count(hand_landmarks)
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/lib/test_hand_detector.py -v
```

- [ ] **Step 5: Commit**

```bash
git add openvisionkit/lib/hand_detector.py tests/lib/test_hand_detector.py
git commit -m "feat(hand_detector): add gesture recognition methods"
```

---

## Task 6: HandDetector — hand analysis and utilities

**Spec:** `get_hand_orientation`, `get_swipe_direction`, `get_all_finger_angles`, `draw_gesture_label`, `to_json`

**Files:**
- Modify: `tests/lib/test_hand_detector.py`
- Modify: `openvisionkit/lib/hand_detector.py`

- [ ] **Step 1: Append failing tests**

```python
# Append to tests/lib/test_hand_detector.py
import numpy as np
from conftest import blank_bgr


def make_oriented_landmarks(direction="up"):
    lms = [[i, 0.5, 0.5, 0.0] for i in range(21)]
    # Wrist at idx 0, middle MCP at idx 9
    if direction == "up":
        lms[0][1], lms[0][2] = 0.5, 0.8   # wrist low
        lms[9][1], lms[9][2] = 0.5, 0.2   # MCP high (fingers up)
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
    monkeypatch.setattr(d, "get_angle_between_landmarks",
                        lambda lm, a, b, c: 90.0)
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
    import json
    lms = make_hand_landmarks()
    hand_data = {
        "landmarks_list": lms,
        "bounding_box": (50, 60, 100, 120),
        "center_point": (100, 120),
        "hand_type": "Right",
    }
    data = d.to_json(hand_data)
    assert json.dumps(data)  # must be JSON-safe
    assert data["hand_type"] == "Right"
    assert len(data["landmarks"]) == 21
```

- [ ] **Step 2: Run to confirm failure**

```bash
uv run pytest tests/lib/test_hand_detector.py -v 2>&1 | tail -15
```

- [ ] **Step 3: Implement methods**

```python
    def get_hand_orientation(self, hand_landmarks):
        wrist = hand_landmarks[0][1:3]
        middle_mcp = hand_landmarks[9][1:3]
        dx = middle_mcp[0] - wrist[0]
        dy = middle_mcp[1] - wrist[1]
        if abs(dx) >= abs(dy):
            return "palm_right" if dx > 0 else "palm_left"
        return "palm_up" if dy < 0 else "palm_down"

    def get_swipe_direction(self, prev_wrist, curr_wrist, threshold=20):
        dx = curr_wrist[0] - prev_wrist[0]
        dy = curr_wrist[1] - prev_wrist[1]
        if max(abs(dx), abs(dy)) < threshold:
            return "none"
        if abs(dx) >= abs(dy):
            return "right" if dx > 0 else "left"
        return "down" if dy > 0 else "up"

    def get_all_finger_angles(self, hand_landmarks):
        joints = {
            "thumb":  (1, 2, 3),
            "index":  (5, 6, 7),
            "middle": (9, 10, 11),
            "ring":   (13, 14, 15),
            "little": (17, 18, 19),
        }
        return {
            name: self.get_angle_between_landmarks(hand_landmarks, a, b, c)
            for name, (a, b, c) in joints.items()
        }

    def draw_gesture_label(self, image, hand_data, label):
        out = image.copy()
        x, y, w, h = hand_data["bounding_box"]
        cv2.putText(out, label, (x, max(y - 10, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        return out

    def to_json(self, hand_data):
        return {
            "hand_type": hand_data.get("hand_type", "Unknown"),
            "center_point": list(hand_data.get("center_point", (0, 0))),
            "bounding_box": list(hand_data.get("bounding_box", (0, 0, 0, 0))),
            "landmarks": [list(lm) for lm in hand_data.get("landmarks_list", [])],
        }
```

- [ ] **Step 4: Run all HandDetector tests**

```bash
uv run pytest tests/lib/test_hand_detector.py -v
```

- [ ] **Step 5: Commit**

```bash
git add openvisionkit/lib/hand_detector.py tests/lib/test_hand_detector.py
git commit -m "feat(hand_detector): add orientation, swipe, angle, label, json utilities"
```

---

## Task 7: PoseDetector — posture analysis

**Spec:** `get_spine_angle`, `get_torso_tilt`, `is_hunching`, `get_symmetry_score`

**Files:**
- Create: `tests/lib/test_pose_detector.py`
- Modify: `openvisionkit/lib/pose_detector.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/lib/test_pose_detector.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
import pytest
from openvisionkit.lib.pose_detector import PoseDetector
from conftest import MockLandmark, MockPoseResult


def det():
    return PoseDetector.__new__(PoseDetector)


def make_upright():
    """33 landmarks: person standing upright, centred."""
    lms = [MockLandmark(x=0.5, y=0.5) for _ in range(33)]
    # Shoulders level at y=0.3
    lms[11] = MockLandmark(x=0.4, y=0.3)   # left shoulder
    lms[12] = MockLandmark(x=0.6, y=0.3)   # right shoulder
    # Hips level at y=0.6
    lms[23] = MockLandmark(x=0.4, y=0.6)   # left hip
    lms[24] = MockLandmark(x=0.6, y=0.6)   # right hip
    return lms


def test_get_spine_angle_upright_near_zero():
    d = det()
    result = MockPoseResult(make_upright())
    angle = d.get_spine_angle(result)
    assert abs(angle) < 5.0  # nearly vertical spine


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
    lms[11] = MockLandmark(x=0.3, y=0.2)  # left shoulder much lower
    lms[12] = MockLandmark(x=0.7, y=0.5)  # right shoulder higher
    result = MockPoseResult(lms)
    assert d.is_hunching(result, threshold=20) is True


def test_get_symmetry_score_range():
    d = det()
    result = MockPoseResult(make_upright())
    score = d.get_symmetry_score(result)
    assert 0.0 <= score <= 1.0


def test_get_symmetry_score_empty_returns_zero():
    d = det()
    r = MockPoseResult.__new__(MockPoseResult)
    r.pose_landmarks = []
    assert d.get_symmetry_score(r) == 0.0
```

- [ ] **Step 2: Run to confirm failure**

```bash
uv run pytest tests/lib/test_pose_detector.py -v 2>&1 | tail -15
```

- [ ] **Step 3: Implement methods in `openvisionkit/lib/pose_detector.py`**

```python
    def get_spine_angle(self, detection_result):
        if not detection_result.pose_landmarks:
            return 0.0
        import math
        lms = detection_result.pose_landmarks[0]
        shoulder_mid_x = (lms[11].x + lms[12].x) / 2
        shoulder_mid_y = (lms[11].y + lms[12].y) / 2
        hip_mid_x = (lms[23].x + lms[24].x) / 2
        hip_mid_y = (lms[23].y + lms[24].y) / 2
        dx = shoulder_mid_x - hip_mid_x
        dy = shoulder_mid_y - hip_mid_y
        return float(math.degrees(math.atan2(abs(dx), abs(dy) + 1e-6)))

    def get_torso_tilt(self, detection_result):
        if not detection_result.pose_landmarks:
            return 0.0
        import math
        lms = detection_result.pose_landmarks[0]
        dx = lms[12].x - lms[11].x
        dy = lms[12].y - lms[11].y
        return float(math.degrees(math.atan2(dy, dx + 1e-6)))

    def is_hunching(self, detection_result, threshold=20):
        return abs(self.get_torso_tilt(detection_result)) > threshold

    def get_symmetry_score(self, detection_result):
        if not detection_result.pose_landmarks:
            return 0.0
        lms = detection_result.pose_landmarks[0]
        pairs = [(11, 12), (13, 14), (15, 16), (23, 24), (25, 26), (27, 28)]
        mid_x = (lms[11].x + lms[12].x) / 2
        diffs = []
        for l_idx, r_idx in pairs:
            l, r = lms[l_idx], lms[r_idx]
            if l.visibility < 0.5 or r.visibility < 0.5:
                continue
            mirrored_lx = 2 * mid_x - l.x
            dx = abs(mirrored_lx - r.x)
            dy = abs(l.y - r.y)
            diffs.append((dx + dy) / 2)
        if not diffs:
            return 0.0
        return float(max(0.0, 1.0 - sum(diffs) / len(diffs) * 10))
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/lib/test_pose_detector.py -v
```

- [ ] **Step 5: Commit**

```bash
git add openvisionkit/lib/pose_detector.py tests/lib/test_pose_detector.py
git commit -m "feat(pose_detector): add posture analysis methods"
```

---

## Task 8: PoseDetector — action and spatial detection

**Spec:** `is_arms_raised`, `detect_fall`, `is_arms_crossed`, `get_knee_angle`, `get_hip_angle`, `get_body_bounding_box`

**Files:**
- Modify: `tests/lib/test_pose_detector.py`
- Modify: `openvisionkit/lib/pose_detector.py`

- [ ] **Step 1: Append failing tests**

```python
# Append to tests/lib/test_pose_detector.py
import numpy as np
from conftest import blank_bgr


def make_arms_raised():
    lms = make_upright()
    lms[15] = MockLandmark(x=0.4, y=0.1, visibility=0.9)  # left wrist above shoulder
    lms[16] = MockLandmark(x=0.6, y=0.1, visibility=0.9)  # right wrist above shoulder
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
    lms[0] = MockLandmark(x=0.5, y=0.1)   # nose at top
    result = MockPoseResult(lms)
    assert d.detect_fall(result) is False


def test_detect_fall_true_when_head_below_hips():
    d = det()
    lms = make_upright()
    lms[0] = MockLandmark(x=0.5, y=0.9)   # nose below hips (y=0.6)
    result = MockPoseResult(lms)
    assert d.detect_fall(result) is True


def test_is_arms_crossed_true():
    d = det()
    lms = make_upright()
    # left wrist crosses to right side, right wrist to left side
    lms[15] = MockLandmark(x=0.7, y=0.5, visibility=0.9)  # left wrist right of center
    lms[16] = MockLandmark(x=0.3, y=0.5, visibility=0.9)  # right wrist left of center
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
    lms[23] = MockLandmark(x=0.4, y=0.6)  # hip
    lms[25] = MockLandmark(x=0.4, y=0.75)  # knee
    lms[27] = MockLandmark(x=0.4, y=0.9)   # ankle
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
```

- [ ] **Step 2: Run to confirm failure**

```bash
uv run pytest tests/lib/test_pose_detector.py -v 2>&1 | tail -15
```

- [ ] **Step 3: Implement methods**

```python
    def is_arms_raised(self, detection_result, threshold=0.2):
        if not detection_result.pose_landmarks:
            return False
        lms = detection_result.pose_landmarks[0]
        return (lms[15].y < lms[11].y - threshold and
                lms[16].y < lms[12].y - threshold)

    def detect_fall(self, detection_result):
        if not detection_result.pose_landmarks:
            return False
        lms = detection_result.pose_landmarks[0]
        hip_y = (lms[23].y + lms[24].y) / 2
        return lms[0].y > hip_y

    def is_arms_crossed(self, detection_result):
        if not detection_result.pose_landmarks:
            return False
        lms = detection_result.pose_landmarks[0]
        mid_x = (lms[11].x + lms[12].x) / 2
        return lms[15].x > mid_x and lms[16].x < mid_x

    def get_knee_angle(self, detection_result, side="left"):
        if not detection_result.pose_landmarks:
            return 0.0
        import math
        import numpy as np
        lms = detection_result.pose_landmarks[0]
        if side == "left":
            hip, knee, ankle = lms[23], lms[25], lms[27]
        else:
            hip, knee, ankle = lms[24], lms[26], lms[28]
        a = np.array([hip.x, hip.y])
        b = np.array([knee.x, knee.y])
        c = np.array([ankle.x, ankle.y])
        ba, bc = a - b, c - b
        cos_a = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        return float(math.degrees(math.acos(np.clip(cos_a, -1.0, 1.0))))

    def get_hip_angle(self, detection_result, side="left"):
        if not detection_result.pose_landmarks:
            return 0.0
        import math
        import numpy as np
        lms = detection_result.pose_landmarks[0]
        if side == "left":
            shoulder, hip, knee = lms[11], lms[23], lms[25]
        else:
            shoulder, hip, knee = lms[12], lms[24], lms[26]
        a = np.array([shoulder.x, shoulder.y])
        b = np.array([hip.x, hip.y])
        c = np.array([knee.x, knee.y])
        ba, bc = a - b, c - b
        cos_a = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        return float(math.degrees(math.acos(np.clip(cos_a, -1.0, 1.0))))

    def get_body_bounding_box(self, detection_result, image):
        if not detection_result.pose_landmarks:
            return (0, 0, 0, 0)
        lms = detection_result.pose_landmarks[0]
        h, w = image.shape[:2]
        visible = [(lm.x * w, lm.y * h) for lm in lms if lm.visibility > 0.5]
        if not visible:
            return (0, 0, 0, 0)
        xs = [p[0] for p in visible]
        ys = [p[1] for p in visible]
        x1, y1 = int(min(xs)), int(min(ys))
        x2, y2 = int(max(xs)), int(max(ys))
        return (x1, y1, x2 - x1, y2 - y1)
```

- [ ] **Step 4: Run all PoseDetector tests**

```bash
uv run pytest tests/lib/test_pose_detector.py -v
```

- [ ] **Step 5: Commit**

```bash
git add openvisionkit/lib/pose_detector.py tests/lib/test_pose_detector.py
git commit -m "feat(pose_detector): add action and spatial detection methods"
```

---

## Task 9: ObjectDetector — filtering, querying, spatial, export

**Spec:** `filter_by_confidence`, `get_bounding_boxes`, `is_crowded`, `get_objects_by_size`, `get_proximity`, `detect_line_crossing`, `export_to_json`, `batch_detect`

**Files:**
- Create: `tests/lib/test_object_detector.py`
- Modify: `openvisionkit/lib/object_detector.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/lib/test_object_detector.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import pytest
from openvisionkit.lib.object_detector import ObjectDetector
from conftest import blank_bgr, MockDetection, MockObjectResult


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
    a = MockDetection(x=0, y=0, w=10, h=10)   # center (5,5)
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
```

- [ ] **Step 2: Run to confirm failure**

```bash
uv run pytest tests/lib/test_object_detector.py -v 2>&1 | tail -15
```

- [ ] **Step 3: Implement methods**

```python
    def filter_by_confidence(self, detection_result, threshold=0.5):
        return [
            d for d in detection_result.detections
            if d.categories and d.categories[0].score >= threshold
        ]

    def get_bounding_boxes(self, detection_result):
        boxes = []
        for det in detection_result.detections:
            bb = det.bounding_box
            label = det.categories[0].category_name if det.categories else "unknown"
            score = det.categories[0].score if det.categories else 0.0
            boxes.append({"label": label, "score": score,
                          "bbox": (bb.origin_x, bb.origin_y, bb.width, bb.height)})
        return boxes

    def is_crowded(self, detection_result, threshold=5):
        return len(detection_result.detections) >= threshold

    def get_objects_by_size(self, detection_result):
        return sorted(
            detection_result.detections,
            key=lambda d: d.bounding_box.width * d.bounding_box.height,
            reverse=True,
        )

    def get_proximity(self, det_a, det_b):
        bb_a, bb_b = det_a.bounding_box, det_b.bounding_box
        cx_a = bb_a.origin_x + bb_a.width / 2
        cy_a = bb_a.origin_y + bb_a.height / 2
        cx_b = bb_b.origin_x + bb_b.width / 2
        cy_b = bb_b.origin_y + bb_b.height / 2
        return float(((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2) ** 0.5)

    def detect_line_crossing(self, detection_result, line_start, line_end,
                              line_threshold=10):
        import numpy as np
        p1 = np.array(line_start, dtype=float)
        p2 = np.array(line_end, dtype=float)
        line_len = np.linalg.norm(p2 - p1)
        crossing = []
        for det in detection_result.detections:
            bb = det.bounding_box
            cx = bb.origin_x + bb.width / 2
            cy = bb.origin_y + bb.height / 2
            p = np.array([cx, cy])
            if line_len == 0:
                dist = float(np.linalg.norm(p - p1))
            else:
                t = float(np.clip(np.dot(p - p1, p2 - p1) / (line_len ** 2), 0, 1))
                proj = p1 + t * (p2 - p1)
                dist = float(np.linalg.norm(p - proj))
            if dist <= line_threshold:
                label = det.categories[0].category_name if det.categories else "unknown"
                crossing.append({"label": label, "center": (cx, cy), "distance": dist})
        return crossing

    def export_to_json(self, detection_result):
        return {
            "detections": [
                {
                    "label": d.categories[0].category_name if d.categories else "unknown",
                    "score": d.categories[0].score if d.categories else 0.0,
                    "bbox": {
                        "x": d.bounding_box.origin_x,
                        "y": d.bounding_box.origin_y,
                        "width": d.bounding_box.width,
                        "height": d.bounding_box.height,
                    },
                }
                for d in detection_result.detections
            ]
        }

    def batch_detect(self, images):
        return [self.detect(img)[0] for img in images]
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/lib/test_object_detector.py -v
```

- [ ] **Step 5: Commit**

```bash
git add openvisionkit/lib/object_detector.py tests/lib/test_object_detector.py
git commit -m "feat(object_detector): add filtering, spatial, and export utilities"
```

---

## Task 10: SelfieSegmentation — presence, bounds, and visual effects

**Spec:** `is_person_present`, `get_person_center`, `get_foreground_bounds`, `measure_foreground_height`, `create_green_screen`, `extract_foreground_on_white`, `apply_bokeh_effect`, `apply_edge_glow`

**Files:**
- Create: `tests/lib/test_selfie_segmentation.py`
- Modify: `openvisionkit/lib/selfie_segmentation.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/lib/test_selfie_segmentation.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from unittest.mock import patch
from openvisionkit.lib.selfie_segmentation import SelfieSegmentation
from conftest import blank_bgr


def det():
    return SelfieSegmentation.__new__(SelfieSegmentation)


def patch_mask(d, fg=True):
    """Patch process+_get_mask to return a foreground mask."""
    h, w = 300, 400
    mask = np.zeros((h, w), dtype=np.uint8)
    if fg:
        mask[50:250, 100:300] = 255  # large foreground blob
    d.process = lambda img: None
    d._get_mask = lambda result, **kw: mask
    return mask


def test_is_person_present_with_foreground():
    d = det()
    patch_mask(d, fg=True)
    assert d.is_person_present(blank_bgr()) is True


def test_is_person_present_empty_mask():
    d = det()
    patch_mask(d, fg=False)
    assert d.is_person_present(blank_bgr()) is False


def test_get_person_center_in_expected_region():
    d = det()
    patch_mask(d, fg=True)
    cx, cy = d.get_person_center(blank_bgr())
    assert 100 <= cx <= 300
    assert 50 <= cy <= 250


def test_get_foreground_bounds_nonzero():
    d = det()
    patch_mask(d, fg=True)
    x, y, w, h = d.get_foreground_bounds(blank_bgr())
    assert w > 0 and h > 0


def test_measure_foreground_height_matches_mask():
    d = det()
    patch_mask(d, fg=True)
    height = d.measure_foreground_height(blank_bgr())
    assert height == 200  # mask rows 50-250


def test_create_green_screen_background_is_green():
    d = det()
    patch_mask(d, fg=True)
    img = blank_bgr()  # background area is gray (128,128,128)
    result = d.create_green_screen(img)
    assert result.shape == img.shape
    # Top-left corner is background → should be green
    assert result[0, 0, 1] == 255   # G channel
    assert result[0, 0, 0] == 0     # B channel
    assert result[0, 0, 2] == 0     # R channel


def test_extract_foreground_on_white_background_is_white():
    d = det()
    patch_mask(d, fg=True)
    result = d.extract_foreground_on_white(blank_bgr())
    assert result[0, 0, 0] == 255  # top-left is background → white


def test_apply_bokeh_effect_returns_same_shape():
    d = det()
    patch_mask(d, fg=True)
    result = d.apply_bokeh_effect(blank_bgr(), blur_radius=15)
    assert result.shape == blank_bgr().shape


def test_apply_edge_glow_returns_same_shape():
    d = det()
    patch_mask(d, fg=True)
    result = d.apply_edge_glow(blank_bgr(), color=(0, 255, 0), thickness=2)
    assert result.shape == blank_bgr().shape
```

- [ ] **Step 2: Run to confirm failure**

```bash
uv run pytest tests/lib/test_selfie_segmentation.py -v 2>&1 | tail -15
```

- [ ] **Step 3: Implement methods**

```python
    def is_person_present(self, image, min_area_ratio=0.01):
        result = self.process(image)
        mask = self._get_mask(result)
        fg_area = np.count_nonzero(mask > 128)
        total = mask.shape[0] * mask.shape[1]
        return (fg_area / total) > min_area_ratio

    def get_person_center(self, image):
        result = self.process(image)
        mask = self._get_mask(result)
        binary = (mask > 128).astype(np.uint8)
        M = cv2.moments(binary)
        if M["m00"] == 0:
            return (image.shape[1] // 2, image.shape[0] // 2)
        return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

    def get_foreground_bounds(self, image):
        result = self.process(image)
        mask = self._get_mask(result)
        binary = (mask > 128).astype(np.uint8)
        pts = cv2.findNonZero(binary)
        if pts is None:
            return (0, 0, 0, 0)
        return cv2.boundingRect(pts)

    def measure_foreground_height(self, image):
        return self.get_foreground_bounds(image)[3]

    def create_green_screen(self, image):
        result = self.process(image)
        mask = self._get_mask(result)
        fg = (mask > 128)[..., np.newaxis]
        bg = np.zeros_like(image)
        bg[:] = (0, 255, 0)
        return np.where(fg, image, bg)

    def extract_foreground_on_white(self, image):
        result = self.process(image)
        mask = self._get_mask(result)
        fg = (mask > 128)[..., np.newaxis]
        bg = np.full_like(image, 255)
        return np.where(fg, image, bg)

    def apply_bokeh_effect(self, image, blur_radius=25):
        result = self.process(image)
        mask = self._get_mask(result)
        r = blur_radius | 1  # ensure odd
        blurred = cv2.GaussianBlur(image, (r, r), 0)
        fg = (mask > 128)[..., np.newaxis]
        return np.where(fg, image, blurred)

    def apply_edge_glow(self, image, color=(0, 255, 0), thickness=3):
        result = self.process(image)
        mask = self._get_mask(result)
        binary = (mask > 128).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        out = image.copy()
        cv2.drawContours(out, contours, -1, color, thickness)
        return out
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/lib/test_selfie_segmentation.py -v
```

- [ ] **Step 5: Commit**

```bash
git add openvisionkit/lib/selfie_segmentation.py tests/lib/test_selfie_segmentation.py
git commit -m "feat(selfie_segmentation): add presence, bounds, and visual effects"
```

---

## Task 11: HairSegmentation — analysis and color effects

**Spec:** `get_hair_bounding_box`, `get_hair_top_position`, `detect_hair_length_estimate`, `get_hair_density_map`, `apply_gradient_color`, `apply_highlights`

**Files:**
- Create: `tests/lib/test_hair_segmentation.py`
- Modify: `openvisionkit/lib/hair_segmentation.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/lib/test_hair_segmentation.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from openvisionkit.lib.hair_segmentation import HairSegmentation
from conftest import blank_bgr


def det():
    return HairSegmentation.__new__(HairSegmentation)


def patch_hair_mask(d, top=20, bottom=100):
    mask = np.zeros((300, 400), dtype=np.uint8)
    mask[top:bottom, 100:300] = 255
    d.get_hair_mask = lambda img: mask
    return mask


def test_get_hair_bounding_box():
    d = det()
    patch_hair_mask(d, top=20, bottom=100)
    x, y, w, h = d.get_hair_bounding_box(blank_bgr())
    assert x == 100
    assert y == 20
    assert w == 200
    assert h == 80


def test_get_hair_top_position():
    d = det()
    patch_hair_mask(d, top=20, bottom=100)
    assert d.get_hair_top_position(blank_bgr()) == 20


def test_detect_hair_length_short():
    d = det()
    patch_hair_mask(d, top=20, bottom=50)   # h=30 / img_h=300 = 10%
    assert d.detect_hair_length_estimate(blank_bgr()) == "short"


def test_detect_hair_length_medium():
    d = det()
    patch_hair_mask(d, top=0, bottom=70)   # h=70 / 300 = 23%
    assert d.detect_hair_length_estimate(blank_bgr()) == "medium"


def test_detect_hair_length_long():
    d = det()
    patch_hair_mask(d, top=0, bottom=150)  # h=150 / 300 = 50%
    assert d.detect_hair_length_estimate(blank_bgr()) == "long"


def test_get_hair_density_map_shape():
    d = det()
    patch_hair_mask(d)
    dm = d.get_hair_density_map(blank_bgr())
    assert dm.shape == (300, 400)
    assert dm.dtype == np.uint8


def test_apply_gradient_color_returns_same_shape():
    d = det()
    patch_hair_mask(d)
    result = d.apply_gradient_color(blank_bgr(), (255, 0, 0), (0, 0, 255))
    assert result.shape == blank_bgr().shape


def test_apply_highlights_returns_same_shape():
    d = det()
    patch_hair_mask(d)
    result = d.apply_highlights(blank_bgr())
    assert result.shape == blank_bgr().shape
```

- [ ] **Step 2: Run to confirm failure**

```bash
uv run pytest tests/lib/test_hair_segmentation.py -v 2>&1 | tail -15
```

- [ ] **Step 3: Implement methods**

```python
    def get_hair_bounding_box(self, bgr_image):
        mask = self.get_hair_mask(bgr_image)
        pts = cv2.findNonZero(mask)
        if pts is None:
            return (0, 0, 0, 0)
        return cv2.boundingRect(pts)

    def get_hair_top_position(self, bgr_image):
        mask = self.get_hair_mask(bgr_image)
        rows = np.any(mask > 0, axis=1)
        if not rows.any():
            return 0
        return int(np.argmax(rows))

    def detect_hair_length_estimate(self, bgr_image):
        x, y, w, h = self.get_hair_bounding_box(bgr_image)
        if h == 0:
            return "none"
        ratio = h / bgr_image.shape[0]
        if ratio < 0.15:
            return "short"
        if ratio < 0.35:
            return "medium"
        return "long"

    def get_hair_density_map(self, bgr_image):
        mask = self.get_hair_mask(bgr_image)
        density = cv2.GaussianBlur(mask.astype(np.float32), (31, 31), 0)
        max_val = density.max()
        if max_val > 0:
            density = (density / max_val * 255)
        return density.astype(np.uint8)

    def apply_gradient_color(self, bgr_image, color1, color2):
        mask = self.get_hair_mask(bgr_image)
        x, y, w, h = self.get_hair_bounding_box(bgr_image)
        if h == 0:
            return bgr_image.copy()
        out = bgr_image.copy()
        for row in range(y, min(y + h, bgr_image.shape[0])):
            t = (row - y) / h
            c = tuple(int(color1[i] * (1 - t) + color2[i] * t) for i in range(3))
            row_mask = mask[row] > 0
            out[row, row_mask] = c
        return out

    def apply_highlights(self, bgr_image, highlight_color=(255, 255, 200),
                         intensity=0.4):
        mask = self.get_hair_mask(bgr_image)
        out = bgr_image.copy()
        hair_pixels = np.argwhere(mask > 0)
        if len(hair_pixels) == 0:
            return out
        rng = np.random.default_rng(0)
        n = max(1, len(hair_pixels) // 5)
        indices = rng.choice(len(hair_pixels), size=n, replace=False)
        sparse = np.zeros_like(mask)
        for idx in indices:
            r, c = hair_pixels[idx]
            sparse[r, c] = 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        sparse = cv2.dilate(sparse, kernel, iterations=2)
        hl = np.array(highlight_color, dtype=np.uint8)
        blend_mask = (sparse > 0) & (mask > 0)
        out[blend_mask] = (out[blend_mask] * (1 - intensity) +
                           hl * intensity).astype(np.uint8)
        return out
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/lib/test_hair_segmentation.py -v
```

- [ ] **Step 5: Commit**

```bash
git add openvisionkit/lib/hair_segmentation.py tests/lib/test_hair_segmentation.py
git commit -m "feat(hair_segmentation): add analysis and color effect methods"
```

---

## Task 12: ImageDetector — preprocessing and analysis

**Spec:** `resize_to_fit`, `pad_to_square`, `normalize`, `create_thumbnail`, `batch_crop`, `get_dominant_colors`, `overlay_image`, `compare_histograms`, `to_base64`

**Files:**
- Create: `tests/lib/test_image_detector.py`
- Modify: `openvisionkit/lib/image_detector.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/lib/test_image_detector.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import base64
import numpy as np
import pytest
from openvisionkit.lib.image_detector import ImageDetector


def det(h=300, w=400):
    img = np.full((h, w, 3), 100, dtype=np.uint8)
    return ImageDetector(img)


def test_resize_to_fit_respects_bounds():
    d = det(300, 400)
    result = d.resize_to_fit(200, 200)
    assert result.shape[0] <= 200
    assert result.shape[1] <= 200
    # Aspect ratio preserved: 400/300 = 1.33 → 200x150
    assert result.shape[1] == 200
    assert result.shape[0] == 150


def test_pad_to_square_result_is_square():
    d = det(100, 200)
    result = d.pad_to_square()
    assert result.shape[0] == result.shape[1] == 200


def test_normalize_output_range():
    d = det()
    result = d.normalize()
    assert result.dtype == np.float32
    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_create_thumbnail_exact_size():
    d = det()
    thumb = d.create_thumbnail(size=(64, 64))
    assert thumb.shape == (64, 64, 3)


def test_batch_crop_returns_correct_count():
    d = det(300, 400)
    boxes = [(10, 10, 50, 50), (100, 100, 30, 30)]
    crops = d.batch_crop(boxes)
    assert len(crops) == 2
    assert crops[0].shape == (50, 50, 3)


def test_get_dominant_colors_count():
    d = det()
    colors = d.get_dominant_colors(k=3)
    assert len(colors) == 3
    assert all(len(c) == 3 for c in colors)


def test_overlay_image_same_shape():
    d = det(300, 400)
    overlay = np.zeros((50, 50, 3), dtype=np.uint8)
    result = d.overlay_image(overlay, x=10, y=10, alpha=0.5)
    assert result.shape == (300, 400, 3)


def test_compare_histograms_identical_returns_one():
    img = np.full((100, 100, 3), 128, dtype=np.uint8)
    d = ImageDetector(img)
    score = d.compare_histograms(img)
    assert abs(score - 1.0) < 0.01


def test_to_base64_decodable():
    d = det()
    b64 = d.to_base64()
    assert isinstance(b64, str)
    decoded = base64.b64decode(b64)
    assert len(decoded) > 0
```

- [ ] **Step 2: Run to confirm failure**

```bash
uv run pytest tests/lib/test_image_detector.py -v 2>&1 | tail -15
```

- [ ] **Step 3: Implement methods**

```python
    def resize_to_fit(self, max_width, max_height):
        h, w = self.image.shape[:2]
        scale = min(max_width / w, max_height / h)
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(self.image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def pad_to_square(self, fill=0):
        h, w = self.image.shape[:2]
        size = max(h, w)
        pad_h = size - h
        pad_w = size - w
        top, left = pad_h // 2, pad_w // 2
        return cv2.copyMakeBorder(self.image, top, pad_h - top,
                                  left, pad_w - left,
                                  cv2.BORDER_CONSTANT, value=fill)

    def normalize(self, mean=(0, 0, 0), std=(1, 1, 1)):
        img = self.image.astype(np.float32) / 255.0
        return ((img - np.array(mean)) / np.array(std)).astype(np.float32)

    def create_thumbnail(self, size=(128, 128)):
        return cv2.resize(self.image, size, interpolation=cv2.INTER_AREA)

    def batch_crop(self, boxes):
        h, w = self.image.shape[:2]
        crops = []
        for (x, y, bw, bh) in boxes:
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(w, x + bw), min(h, y + bh)
            crops.append(self.image[y1:y2, x1:x2].copy())
        return crops

    def get_dominant_colors(self, k=5):
        pixels = self.image.reshape(-1, 3).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, _, centers = cv2.kmeans(pixels, k, None, criteria, 10,
                                   cv2.KMEANS_RANDOM_CENTERS)
        return [tuple(int(c) for c in color) for color in centers]

    def overlay_image(self, overlay, x, y, alpha=1.0):
        out = self.image.copy()
        h, w = overlay.shape[:2]
        y2 = min(y + h, out.shape[0])
        x2 = min(x + w, out.shape[1])
        oh, ow = y2 - y, x2 - x
        if oh > 0 and ow > 0:
            roi = out[y:y2, x:x2]
            out[y:y2, x:x2] = cv2.addWeighted(roi, 1 - alpha,
                                               overlay[:oh, :ow], alpha, 0)
        return out

    def compare_histograms(self, other_image):
        def hist(img):
            h = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8],
                              [0, 256, 0, 256, 0, 256])
            cv2.normalize(h, h, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            return h
        return float(cv2.compareHist(hist(self.image), hist(other_image),
                                     cv2.HISTCMP_CORREL))

    def to_base64(self):
        import base64
        _, buf = cv2.imencode(".png", self.image)
        return base64.b64encode(buf).decode("utf-8")
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/lib/test_image_detector.py -v
```

- [ ] **Step 5: Commit**

```bash
git add openvisionkit/lib/image_detector.py tests/lib/test_image_detector.py
git commit -m "feat(image_detector): add preprocessing and analysis utilities"
```

---

## Task 13: TextDetector — extraction and analysis utilities

**Spec:** `is_text_present`, `extract_dates`, `extract_phone_numbers`, `extract_emails`, `get_reading_order`, `get_text_density`, `redact_sensitive`, `detect_language`

**Files:**
- Create: `tests/lib/test_text_detector.py`
- Modify: `openvisionkit/lib/text_detector.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/lib/test_text_detector.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np
import pytest
from openvisionkit.lib.text_detector import TextDetector


def det(text="Hello world"):
    img = np.full((100, 400, 3), 255, dtype=np.uint8)
    d = TextDetector.__new__(TextDetector)
    d.image = img
    d.detect_text = lambda: text
    d.filter_words_by_confidence = lambda conf: [{"text": w} for w in text.split()]
    d.detect_words = lambda: (None, [
        {"text": w, "left": i * 50, "top": 0, "width": 40, "height": 20}
        for i, w in enumerate(text.split())
    ])
    return d


def test_is_text_present_true_with_words():
    d = det("Hello world")
    assert d.is_text_present() is True


def test_is_text_present_false_empty():
    d = det("")
    d.filter_words_by_confidence = lambda conf: []
    assert d.is_text_present() is False


def test_extract_dates_dd_mm_yyyy():
    d = det()
    dates = d.extract_dates("Invoice date: 15/06/2024")
    assert "15/06/2024" in dates


def test_extract_dates_iso():
    d = det()
    dates = d.extract_dates("Created: 2024-01-31")
    assert "2024-01-31" in dates


def test_extract_phone_numbers():
    d = det()
    phones = d.extract_phone_numbers("Call us at +65 9123 4567 or 6789 0123")
    assert len(phones) >= 1


def test_extract_emails():
    d = det()
    emails = d.extract_emails("Contact: alice@example.com or bob.jones@corp.org")
    assert "alice@example.com" in emails
    assert "bob.jones@corp.org" in emails


def test_get_reading_order_sorted():
    d = det()
    words = [
        {"top": 50, "left": 200, "text": "B"},
        {"top": 10, "left": 100, "text": "A"},
        {"top": 10, "left": 300, "text": "C"},
    ]
    ordered = d.get_reading_order(words)
    assert ordered[0]["text"] == "A"
    assert ordered[1]["text"] == "C"
    assert ordered[2]["text"] == "B"


def test_get_text_density_returns_float():
    d = det("Hello")
    density = d.get_text_density()
    assert isinstance(density, float)
    assert density >= 0.0


def test_redact_sensitive_returns_same_shape():
    d = det("email@test.com")
    result = d.redact_sensitive()
    assert result.shape == d.image.shape


def test_detect_language_unknown_when_no_langdetect(monkeypatch):
    d = det("Hello")
    monkeypatch.setattr("builtins.__import__",
        lambda name, *a, **kw: (_ for _ in ()).throw(ImportError()) if name == "langdetect" else __import__(name, *a, **kw))
    lang = d.detect_language("hello world")
    assert lang in ("en", "unknown")
```

- [ ] **Step 2: Run to confirm failure**

```bash
uv run pytest tests/lib/test_text_detector.py -v 2>&1 | tail -15
```

- [ ] **Step 3: Implement methods**

```python
    def is_text_present(self, min_confidence=60.0):
        try:
            return len(self.filter_words_by_confidence(min_confidence)) > 0
        except Exception:
            return False

    def extract_dates(self, text=None):
        import re
        text = text if text is not None else self.detect_text()
        patterns = [
            r'\b\d{1,2}/\d{1,2}/\d{4}\b',
            r'\b\d{4}-\d{2}-\d{2}\b',
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}\b',
            r'\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}\b',
        ]
        found = []
        for pat in patterns:
            found.extend(re.findall(pat, text, re.IGNORECASE))
        return list(dict.fromkeys(found))

    def extract_phone_numbers(self, text=None):
        import re
        text = text if text is not None else self.detect_text()
        pattern = r'(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)?\d{3,4}[-.\s]?\d{4,6}'
        return re.findall(pattern, text)

    def extract_emails(self, text=None):
        import re
        text = text if text is not None else self.detect_text()
        return re.findall(r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b', text)

    def get_reading_order(self, words):
        return sorted(words, key=lambda w: (w.get("top", 0), w.get("left", 0)))

    def get_text_density(self):
        text = self.detect_text()
        char_count = len(text.replace(" ", "").replace("\n", ""))
        h, w = self.image.shape[:2]
        area = w * h
        return char_count / area if area > 0 else 0.0

    def redact_sensitive(self, patterns=None):
        import re
        out = self.image.copy()
        _, words = self.detect_words()
        default_patterns = [
            r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b',
            r'(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)?\d{3,4}[-.\s]?\d{4,6}',
        ]
        active = patterns or default_patterns
        for word in words:
            if any(re.search(p, word.get("text", ""), re.IGNORECASE) for p in active):
                x = word.get("left", 0)
                y = word.get("top", 0)
                w = word.get("width", 0)
                h = word.get("height", 0)
                cv2.rectangle(out, (x, y), (x + w, y + h), (0, 0, 0), -1)
        return out

    def detect_language(self, text=None):
        try:
            from langdetect import detect
            text = text if text is not None else self.detect_text()
            if not text.strip():
                return "unknown"
            return detect(text)
        except ImportError:
            import warnings
            warnings.warn("langdetect not installed; returning 'unknown'", stacklevel=2)
            return "unknown"
        except Exception:
            return "unknown"
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/lib/test_text_detector.py -v
```

- [ ] **Step 5: Commit**

```bash
git add openvisionkit/lib/text_detector.py tests/lib/test_text_detector.py
git commit -m "feat(text_detector): add extraction and analysis utilities"
```

---

## Task 14: FormROIDetector — field analysis utilities

**Spec:** `get_empty_fields` (unchecked checkbox/radio), `validate_required_fields`, `get_field_by_label`, `get_form_completion_score`, `highlight_empty_fields`, `extract_all_text`

**Note:** `ROIRegion` has `field_type`, `label`, `checked: Optional[bool]`, `confidence`. There is no `fill_state` attribute. "Empty" means: checkbox/radio where `checked` is `False` or `None`.

**Files:**
- Create: `tests/lib/test_form_roi_detector.py`
- Modify: `openvisionkit/lib/form_roi_detector.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/lib/test_form_roi_detector.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np
import pytest
from openvisionkit.lib.form_roi_detector import FormROIDetector, ROIRegion


def det():
    return FormROIDetector.__new__(FormROIDetector)


def make_region(field_type="checkbox", label="agree", checked=None,
                x1=10, y1=10, x2=50, y2=30):
    r = ROIRegion.__new__(ROIRegion)
    r.field_type = field_type
    r.label = label
    r.checked = checked
    r.confidence = 1.0
    r.x1, r.y1, r.x2, r.y2 = x1, y1, x2, y2
    return r


def test_get_empty_fields_returns_unchecked():
    d = det()
    regions = [
        make_region("checkbox", "agree", checked=False),
        make_region("checkbox", "terms", checked=True),
        make_region("radio", "opt1", checked=None),
    ]
    empty = d.get_empty_fields(regions)
    assert len(empty) == 2
    labels = [r.label for r in empty]
    assert "agree" in labels
    assert "opt1" in labels


def test_get_empty_fields_ignores_non_checkbox():
    d = det()
    regions = [make_region("text", "name", checked=None)]
    assert len(d.get_empty_fields(regions)) == 0


def test_validate_required_fields():
    d = det()
    regions = [
        make_region("checkbox", "agree", checked=True),
        make_region("checkbox", "terms", checked=False),
    ]
    result = d.validate_required_fields(regions, ["agree", "terms", "signature"])
    assert "agree" in result["filled"]
    assert "terms" in result["missing"]
    assert "signature" in result["missing"]


def test_get_field_by_label_found():
    d = det()
    regions = [make_region("text", "Name"), make_region("text", "Email")]
    found = d.get_field_by_label(regions, "email")
    assert found is not None
    assert found.label == "Email"


def test_get_field_by_label_not_found():
    d = det()
    assert d.get_field_by_label([], "missing") is None


def test_get_form_completion_score():
    d = det()
    regions = [
        make_region("checkbox", "a", checked=True),
        make_region("checkbox", "b", checked=True),
        make_region("checkbox", "c", checked=False),
        make_region("checkbox", "d", checked=None),
    ]
    score = d.get_form_completion_score(regions)
    assert abs(score - 0.5) < 0.01


def test_highlight_empty_fields_returns_image():
    d = det()
    img = np.full((100, 200, 3), 255, dtype=np.uint8)
    regions = [make_region("checkbox", "a", checked=False)]
    result = d.highlight_empty_fields(img, regions)
    assert result.shape == img.shape


def test_extract_all_text_returns_dict(monkeypatch):
    d = det()
    img = np.full((100, 200, 3), 255, dtype=np.uint8)
    d._ocr_text = lambda crop: "sample"
    regions = [make_region("text", "Name")]
    result = d.extract_all_text(img, regions)
    assert result == {"Name": "sample"}
```

- [ ] **Step 2: Run to confirm failure**

```bash
uv run pytest tests/lib/test_form_roi_detector.py -v 2>&1 | tail -15
```

- [ ] **Step 3: Implement methods in `openvisionkit/lib/form_roi_detector.py`**

```python
    def get_empty_fields(self, regions):
        return [
            r for r in regions
            if r.field_type in ("checkbox", "radio") and not r.checked
        ]

    def validate_required_fields(self, regions, required_labels):
        checked_labels = {
            r.label.lower() for r in regions
            if r.field_type in ("checkbox", "radio") and r.checked
        }
        missing = [l for l in required_labels if l.lower() not in checked_labels]
        filled = [l for l in required_labels if l.lower() in checked_labels]
        return {"missing": missing, "filled": filled}

    def get_field_by_label(self, regions, label):
        label_lower = label.lower()
        for r in regions:
            if r.label.lower() == label_lower:
                return r
        return None

    def get_form_completion_score(self, regions):
        if not regions:
            return 0.0
        checkable = [r for r in regions if r.field_type in ("checkbox", "radio")]
        if not checkable:
            return 0.0
        filled = sum(1 for r in checkable if r.checked)
        return filled / len(checkable)

    def highlight_empty_fields(self, image, regions, color=(0, 0, 255), thickness=2):
        out = image.copy()
        for r in self.get_empty_fields(regions):
            cv2.rectangle(out, (r.x1, r.y1), (r.x2, r.y2), color, thickness)
        return out

    def extract_all_text(self, image, regions):
        result = {}
        for r in regions:
            x1, y1, x2, y2 = r.x1, r.y1, r.x2, r.y2
            crop = image[y1:y2, x1:x2]
            result[r.label] = self._ocr_text(crop)
        return result
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/lib/test_form_roi_detector.py -v
```

- [ ] **Step 5: Commit**

```bash
git add openvisionkit/lib/form_roi_detector.py tests/lib/test_form_roi_detector.py
git commit -m "feat(form_roi_detector): add field analysis utilities"
```

---

## Task 15: FormROIAnnotator — annotation utilities

**Spec:** `get_annotations_by_type`, `export_to_json`, `get_annotation_count`, `merge_annotations`, `draw_annotation_summary`

**Note:** Annotation format: `[(x1,y1), (x2,y2), "type", "label", "category"]`

**Files:**
- Create: `tests/lib/test_form_roi_annotator.py`
- Modify: `openvisionkit/lib/form_roi_annotator.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/lib/test_form_roi_annotator.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import json
import numpy as np
import pytest
from openvisionkit.lib.form_roi_annotator import FormROIAnnotator


def ann(x1=10, y1=10, x2=60, y2=40, typ="checkbox", label="opt1", cat="form"):
    return [(x1, y1), (x2, y2), typ, label, cat]


def det():
    d = FormROIAnnotator.__new__(FormROIAnnotator)
    d.rois = []
    return d


def test_get_annotations_by_type_filters():
    d = det()
    anns = [ann(typ="checkbox"), ann(typ="text"), ann(typ="checkbox")]
    result = d.get_annotations_by_type(anns, "checkbox")
    assert len(result) == 2


def test_export_to_json_creates_file(tmp_path):
    d = det()
    anns = [ann()]
    path = str(tmp_path / "out.json")
    d.export_to_json(anns, path)
    with open(path) as f:
        data = json.load(f)
    assert len(data) == 1
    assert data[0]["type"] == "checkbox"


def test_get_annotation_count():
    d = det()
    anns = [ann(typ="checkbox"), ann(typ="checkbox"), ann(typ="text")]
    counts = d.get_annotation_count(anns)
    assert counts["checkbox"] == 2
    assert counts["text"] == 1


def test_merge_annotations_no_overlap():
    d = det()
    a = [ann(x1=0, y1=0, x2=50, y2=50)]
    b = [ann(x1=200, y1=200, x2=250, y2=250)]
    merged = d.merge_annotations(a, b)
    assert len(merged) == 2


def test_merge_annotations_deduplicates_overlap():
    d = det()
    a = [ann(x1=0, y1=0, x2=50, y2=50)]
    b = [ann(x1=5, y1=5, x2=45, y2=45)]  # heavily overlaps a
    merged = d.merge_annotations(a, b)
    assert len(merged) == 1


def test_draw_annotation_summary_returns_image():
    d = det()
    img = np.full((200, 300, 3), 255, dtype=np.uint8)
    anns = [ann(typ="checkbox"), ann(typ="text")]
    result = d.draw_annotation_summary(img, anns)
    assert result.shape == img.shape
```

- [ ] **Step 2: Run to confirm failure**

```bash
uv run pytest tests/lib/test_form_roi_annotator.py -v 2>&1 | tail -15
```

- [ ] **Step 3: Implement methods in `openvisionkit/lib/form_roi_annotator.py`**

```python
    def get_annotations_by_type(self, annotations, field_type):
        return [a for a in annotations if a[2] == field_type]

    def export_to_json(self, annotations, path):
        data = [
            {"x1": a[0][0], "y1": a[0][1], "x2": a[1][0], "y2": a[1][1],
             "type": a[2], "label": a[3], "category": a[4]}
            for a in annotations
        ]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def get_annotation_count(self, annotations):
        from collections import Counter
        return dict(Counter(a[2] for a in annotations))

    def merge_annotations(self, ann_list_a, ann_list_b):
        def iou(a, b):
            ax1, ay1 = a[0]
            ax2, ay2 = a[1]
            bx1, by1 = b[0]
            bx2, by2 = b[1]
            ix1 = max(ax1, bx1)
            iy1 = max(ay1, by1)
            ix2 = min(ax2, bx2)
            iy2 = min(ay2, by2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            if inter == 0:
                return 0.0
            area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
            area_b = max(1, (bx2 - bx1) * (by2 - by1))
            return inter / (area_a + area_b - inter)

        merged = list(ann_list_a)
        for b in ann_list_b:
            if not any(iou(a, b) > 0.5 for a in merged):
                merged.append(b)
        return merged

    def draw_annotation_summary(self, image, annotations):
        out = image.copy()
        counts = self.get_annotation_count(annotations)
        lines = [f"{t}: {c}" for t, c in counts.items()]
        box_h = 20 + len(lines) * 22
        cv2.rectangle(out, (5, 5), (180, box_h), (0, 0, 0), -1)
        for i, line in enumerate(lines):
            cv2.putText(out, line, (10, 22 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return out
```

- [ ] **Step 4: Run all FormROIAnnotator tests**

```bash
uv run pytest tests/lib/test_form_roi_annotator.py -v
```

- [ ] **Step 5: Commit**

```bash
git add openvisionkit/lib/form_roi_annotator.py tests/lib/test_form_roi_annotator.py
git commit -m "feat(form_roi_annotator): add annotation utility methods"
```

---

## Task 16: Full test suite + final commit

- [ ] **Step 1: Run full test suite**

```bash
uv run pytest tests/ -v --tb=short
```

Expected: all tests pass

- [ ] **Step 2: Fix any failures**

For each failure, read the error message, apply the minimal fix to the relevant source file, re-run only that test file before moving on.

- [ ] **Step 3: Final commit if any fixes applied**

```bash
git add -u
git commit -m "fix: resolve test failures from full suite run"
```

- [ ] **Step 4: Verify git log**

```bash
git log --oneline
```

Expected: 16+ commits showing incremental feature additions

---

## Spec Coverage Checklist

| Class | Methods | Tasks |
|---|---|---|
| FaceDetector | pixelate_faces, is_frontal, get_padded_crop, draw_face_ids, get_attention_score, batch_detect, save_crops | Task 1, 2 |
| FaceMeshDetector | is_smiling, is_yawning, is_surprised, get_eyebrow_raise, is_eyes_closed, is_drowsy, get_face_bounding_box, get_face_symmetry_score, draw_face_oval, get_attention_level, get_lip_separation | Task 3, 4 |
| HandDetector | is_ok_sign, is_call_me, is_rock_sign, recognize_number, get_hand_orientation, get_swipe_direction, get_all_finger_angles, draw_gesture_label, to_json | Task 5, 6 |
| PoseDetector | get_spine_angle, get_torso_tilt, is_hunching, get_symmetry_score, is_arms_raised, detect_fall, is_arms_crossed, get_knee_angle, get_hip_angle, get_body_bounding_box | Task 7, 8 |
| ObjectDetector | filter_by_confidence, get_bounding_boxes, is_crowded, get_objects_by_size, get_proximity, detect_line_crossing, export_to_json, batch_detect | Task 9 |
| SelfieSegmentation | is_person_present, get_person_center, get_foreground_bounds, measure_foreground_height, create_green_screen, extract_foreground_on_white, apply_bokeh_effect, apply_edge_glow | Task 10 |
| HairSegmentation | get_hair_bounding_box, get_hair_top_position, detect_hair_length_estimate, get_hair_density_map, apply_gradient_color, apply_highlights | Task 11 |
| ImageDetector | resize_to_fit, pad_to_square, normalize, create_thumbnail, batch_crop, get_dominant_colors, overlay_image, compare_histograms, to_base64 | Task 12 |
| TextDetector | is_text_present, extract_dates, extract_phone_numbers, extract_emails, get_reading_order, get_text_density, redact_sensitive, detect_language | Task 13 |
| FormROIDetector | get_empty_fields, validate_required_fields, get_field_by_label, get_form_completion_score, highlight_empty_fields, extract_all_text | Task 14 |
| FormROIAnnotator | get_annotations_by_type, export_to_json, get_annotation_count, merge_annotations, draw_annotation_summary | Task 15 |
