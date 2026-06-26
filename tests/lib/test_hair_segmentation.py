import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from conftest import blank_bgr

from visionkit.lib.hair_segmentation import HairSegmentation


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
    patch_hair_mask(d, top=20, bottom=50)  # h=30 / img_h=300 = 10%
    assert d.detect_hair_length_estimate(blank_bgr()) == "short"


def test_detect_hair_length_medium():
    d = det()
    patch_hair_mask(d, top=0, bottom=70)  # h=70 / 300 = 23%
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
