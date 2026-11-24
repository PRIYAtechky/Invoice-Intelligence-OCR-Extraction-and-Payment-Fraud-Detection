"""Tests for Image Preprocessing Module."""

import pytest
import numpy as np
import cv2
from src.preprocessing.image_processor import ImageProcessor

def test_image_processor_initialization(tmp_path):
    proc = ImageProcessor(output_dir=str(tmp_path))
    assert proc.output_dir == tmp_path

def test_deskew_and_rotation():
    proc = ImageProcessor()
    # Create a synthetic white image with a horizontal black line
    img = np.ones((200, 200), dtype=np.uint8) * 255
    cv2.line(img, (20, 100), (180, 100), 0, 3)
    
    # Skew angle estimation should be close to 0
    angle = proc.estimate_skew_angle(img)
    assert abs(angle) <= 15.0

def test_contrast_enhancement():
    proc = ImageProcessor()
    gray = np.full((100, 100), 128, dtype=np.uint8)
    enhanced = proc.enhance_contrast(gray)
    assert enhanced.shape == gray.shape

def test_denoise_and_binarize():
    proc = ImageProcessor()
    noisy = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    denoised = proc.denoise(noisy)
    binarized = proc.binarize(denoised)
    assert binarized.shape == (100, 100)
    # Check binary values (0 or 255)
    unique_vals = set(np.unique(binarized))
    assert unique_vals.issubset({0, 255})

def test_invalid_image_path():
    proc = ImageProcessor()
    with pytest.raises(FileNotFoundError):
        proc.read_image("non_existent_path.jpg")
