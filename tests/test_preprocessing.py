"""
Tests for Image Preprocessing Pipeline
"""
import numpy as np
import pytest
from src.preprocessing.image_pipeline import preprocessor


def test_load_and_preprocess_synthetic_image():
    # Create synthetic test invoice image (white background with black text-like blocks)
    img = np.ones((600, 800, 3), dtype=np.uint8) * 255
    # Add a horizontal line
    img[200:205, 100:700] = 0

    binary, meta = preprocessor.preprocess(img)

    assert binary is not None
    assert meta["success"] is True
    assert "width" in meta
    assert "height" in meta
    assert meta["width"] > 0
    assert meta["height"] > 0


def test_estimate_skew_angle():
    # Perfectly horizontal line
    gray = np.ones((400, 400), dtype=np.uint8) * 255
    gray[200:204, 50:350] = 0

    angle = preprocessor.estimate_skew_angle(gray)
    assert isinstance(angle, float)
    assert abs(angle) < 5.0  # Should be near 0 degrees
