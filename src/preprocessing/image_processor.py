"""Image Preprocessing Module for Scanned Invoices.

Includes deskewing, orientation detection, noise reduction,
binarization, contrast enhancement, and degraded scan handling.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
from config.logging_config import logger

class ImageProcessor:
    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir) if output_dir else Path("data/processed")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def read_image(self, image_path: str) -> np.ndarray:
        """Load image from path supporting Unicode paths on Windows."""
        path = str(image_path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image not found at: {path}")
        
        # Use cv2.imdecode to safely read paths with spaces or unicode
        with open(path, "rb") as f:
            bytes_data = bytearray(f.read())
            numpy_array = np.asarray(bytes_data, dtype=np.uint8)
            img = cv2.imdecode(numpy_array, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError(f"Failed to decode image from: {path}")
        return img

    def estimate_skew_angle(self, gray: np.ndarray) -> float:
        """Estimate skew angle using Hough line transform and text contours."""
        try:
            # Invert and threshold to get text lines
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            angles = []
            
            for cnt in contours:
                if cv2.contourArea(cnt) > 80:
                    min_rect = cv2.minAreaRect(cnt)
                    width, height = min_rect[1]
                    angle = min_rect[2]
                    
                    # Normalize orientation
                    if width < height:
                        angle = angle + 90
                    if -45 <= angle <= 45 and abs(angle) > 0.3:
                        angles.append(angle)
            
            if angles:
                median_angle = float(np.median(angles))
                # Only deskew if angle is significant and <= 15 degrees
                if abs(median_angle) <= 15.0:
                    return median_angle
        except Exception as e:
            logger.warning(f"Skew detection fallback: {e}")
        return 0.0

    def rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image by given angle with border padding."""
        if abs(angle) < 0.1:
            return image
        
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calculate new bounding dimensions
        cos_val = np.abs(rot_mat[0, 0])
        sin_val = np.abs(rot_mat[0, 1])
        new_w = int((h * sin_val) + (w * cos_val))
        new_h = int((h * cos_val) + (w * sin_val))
        
        rot_mat[0, 2] += (new_w / 2) - center[0]
        rot_mat[1, 2] += (new_h / 2) - center[1]
        
        rotated = cv2.warpAffine(
            image, rot_mat, (new_w, new_h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255)
        )
        return rotated

    def enhance_contrast(self, gray: np.ndarray) -> np.ndarray:
        """Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)."""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)

    def denoise(self, gray: np.ndarray) -> np.ndarray:
        """Apply bilateral filtering to remove background noise while preserving edges."""
        return cv2.bilateralFilter(gray, d=5, sigmaColor=50, sigmaSpace=50)

    def binarize(self, gray: np.ndarray) -> np.ndarray:
        """Otsu thresholding for crisp text extraction."""
        return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    def remove_stamps_and_artifacts(self, bgr_img: np.ndarray) -> np.ndarray:
        """Lighten colored stamps (red/blue ink) that obstruct black text."""
        try:
            hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
            # Detect red/blue stamp ink
            mask_red1 = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([10, 255, 255]))
            mask_red2 = cv2.inRange(hsv, np.array([170, 70, 50]), np.array([180, 255, 255]))
            mask_blue = cv2.inRange(hsv, np.array([100, 70, 50]), np.array([130, 255, 255]))
            stamp_mask = mask_red1 | mask_red2 | mask_blue
            
            cleaned = bgr_img.copy()
            # Fade stamp pixels to reduce interference with black text
            cleaned[stamp_mask > 0] = [240, 240, 240]
            return cleaned
        except Exception as e:
            logger.warning(f"Stamp removal skipped: {e}")
            return bgr_img

    def process(self, image_path: str, save_output: bool = True) -> Dict[str, Any]:
        """Execute full preprocessing pipeline."""
        original_img = self.read_image(image_path)
        img_name = Path(image_path).name
        
        # 1. Stamp attenuation
        unstamped = self.remove_stamps_and_artifacts(original_img)
        
        # 2. Convert to grayscale
        gray = cv2.cvtColor(unstamped, cv2.COLOR_BGR2GRAY)
        
        # 3. Detect and correct skew
        skew_angle = self.estimate_skew_angle(gray)
        if abs(skew_angle) > 0.3:
            deskewed_color = self.rotate_image(unstamped, skew_angle)
            deskewed_gray = cv2.cvtColor(deskewed_color, cv2.COLOR_BGR2GRAY)
        else:
            deskewed_color = unstamped
            deskewed_gray = gray
            
        # 4. Enhance contrast
        enhanced_gray = self.enhance_contrast(deskewed_gray)
        
        # 5. Denoise
        denoised_gray = self.denoise(enhanced_gray)
        
        # 6. Binarize
        binarized = self.binarize(denoised_gray)
        
        # Save processed image
        processed_path = None
        if save_output:
            out_filename = f"proc_{img_name}"
            out_file = self.output_dir / out_filename
            # Save binarized image for OCR
            cv2.imwrite(str(out_file), binarized)
            processed_path = str(out_file)
            
        return {
            "original_path": str(image_path),
            "processed_path": processed_path,
            "original_shape": original_img.shape,
            "processed_shape": binarized.shape,
            "skew_angle_corrected": float(round(skew_angle, 2)),
            "binarized_image": binarized,
            "processed_color_image": deskewed_color
        }
