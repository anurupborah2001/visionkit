import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from conftest import blank_bgr

from openvisionkit.lib.selfie_segmentation import SelfieSegmentation


def det():
    return SelfieSegmentation.__new__(SelfieSegmentation)


def patch_mask(d, fg=True):
    h, w = 300, 400
    mask = np.zeros((h, w), dtype=np.uint8)
    if fg:
        mask[50:250, 100:300] = 255
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
    img = blank_bgr()
    result = d.create_green_screen(img)
    assert result.shape == img.shape
    # Top-left corner is background -> should be green
    assert result[0, 0, 1] == 255  # G channel
    assert result[0, 0, 0] == 0  # B channel
    assert result[0, 0, 2] == 0  # R channel


def test_extract_foreground_on_white_background_is_white():
    d = det()
    patch_mask(d, fg=True)
    result = d.extract_foreground_on_white(blank_bgr())
    assert result[0, 0, 0] == 255


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
