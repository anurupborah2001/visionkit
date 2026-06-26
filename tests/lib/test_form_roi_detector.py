import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np

from visionkit.lib.form_roi_detector import FormROIDetector, ROIRegion


def det():
    return FormROIDetector.__new__(FormROIDetector)


def make_region(
    field_type="checkbox", label="agree", checked=None, x1=10, y1=10, x2=50, y2=30
):
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
