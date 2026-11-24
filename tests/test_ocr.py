"""
Tests for OCR Interface and CER Metrics
"""
from src.ocr.base import calculate_cer, OCRTextBox, OCRResult
from src.ocr.ocr_comparison import get_ocr_engine


def test_calculate_cer():
    # Exact match -> CER = 0.0
    assert calculate_cer("INVOICE 12345", "INVOICE 12345") == 0.0

    # Completely empty hypothesis -> CER = 1.0
    assert calculate_cer("INVOICE", "") == 1.0

    # 1 substitution in 4 chars -> CER = 0.25
    assert calculate_cer("TEST", "TEXT") == 0.25


def test_ocr_textbox_serialization():
    box = OCRTextBox(text="Kaveri Industrial Supplies", bbox=[10, 20, 100, 30], confidence=0.985)
    d = box.to_dict()
    assert d["text"] == "Kaveri Industrial Supplies"
    assert d["confidence"] == 0.985
    assert d["bbox"] == [10, 20, 100, 30]


def test_ocr_engine_factory():
    engine = get_ocr_engine("easyocr")
    assert engine is not None
    assert "EasyOCR" in engine.engine_name
