import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import json

import numpy as np

from visionkit.lib.form_roi_annotator import FormROIAnnotator


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
