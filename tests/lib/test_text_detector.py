import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np

from openvisionkit.lib.text_detector import TextDetector


def det(text="Hello world"):
    img = np.full((100, 400, 3), 255, dtype=np.uint8)
    d = TextDetector.__new__(TextDetector)
    d.image = img
    d.detect_text = lambda: text
    d.filter_words_by_confidence = lambda conf: (
        [{"text": w} for w in text.split()] if text.strip() else []
    )
    d.detect_words = lambda: (
        None,
        [
            {"text": w, "left": i * 50, "top": 0, "width": 40, "height": 20}
            for i, w in enumerate(text.split())
        ],
    )
    return d


def test_is_text_present_true_with_words():
    d = det("Hello world")
    assert d.is_text_present() is True


def test_is_text_present_false_empty():
    d = det("")
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


def test_detect_language_returns_string():
    d = det("Hello world")
    lang = d.detect_language("hello world")
    assert isinstance(lang, str)
    assert len(lang) > 0
