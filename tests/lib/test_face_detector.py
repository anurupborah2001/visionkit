import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np
import pytest
from visionkit.lib.face_detector import FaceDetector
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from conftest import blank_bgr, make_face_detection


def det():
    return FaceDetector.__new__(FaceDetector)


def test_pixelate_faces_changes_face_region():
    d = det()
    img = blank_bgr()
    # Give the face region a gradient so downscale→upscale produces a different result
    rng = np.random.default_rng(42)
    img[80:170, 100:180] = rng.integers(0, 256, (90, 80, 3), dtype=np.uint8)
    detection = make_face_detection(x=100, y=80, w=80, h=90)
    result = d.pixelate_faces(img, [detection], block_size=8)
    assert result.shape == img.shape
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
    assert crop.shape[0] > 90
    assert crop.shape[1] > 80


def test_get_padded_crop_clipped_at_boundary():
    d = det()
    img = blank_bgr(100, 100)
    detection = make_face_detection(x=0, y=0, w=50, h=50)
    crop = d.get_padded_crop(img, detection, pad_ratio=0.5)
    assert crop.shape[0] <= 100
    assert crop.shape[1] <= 100


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
