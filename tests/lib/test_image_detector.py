import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import base64

import numpy as np

from openvisionkit.lib.image_detector import ImageDetector


def det(h=300, w=400):
    img = np.full((h, w, 3), 100, dtype=np.uint8)
    return ImageDetector(img)


def test_resize_to_fit_respects_bounds():
    d = det(300, 400)
    result = d.resize_to_fit(200, 200)
    assert result.shape[0] <= 200
    assert result.shape[1] <= 200
    # Aspect ratio 400/300 = 1.33 → fits at 200x150
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
