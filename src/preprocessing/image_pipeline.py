"""
Image Preprocessing Pipeline for Invoice Documents
Implements deskewing, noise reduction, adaptive binarization, rotation correction,
contrast enhancement, and PDF conversion.
"""
import os
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from PIL import Image
import fitz  # PyMuPDF


class ImagePreprocessor:
    def __init__(self):
        pass

    def load_image(self, file_path_or_bytes, file_name: Optional[str] = None) -> Tuple[np.ndarray, Optional[str]]:
        """
        Load an image from a file path or bytes, supporting JPG, JPEG, PNG, and PDF.
        Returns:
            Tuple of (original_cv2_image_bgr, error_message)
        """
        try:
            if isinstance(file_path_or_bytes, (str, Path)):
                path = Path(file_path_or_bytes)
                if not path.exists():
                    return None, f"File not found: {path}"
                
                # Check for PDF
                if path.suffix.lower() == ".pdf":
                    return self._load_pdf_first_page(path)
                
                # Standard image
                img = cv2.imread(str(path))
                if img is None:
                    return None, f"Failed to decode image: {path}"
                return img, None
                
            elif isinstance(file_path_or_bytes, bytes):
                # Check if it's a PDF by magic bytes or file_name
                is_pdf = (file_name and file_name.lower().endswith(".pdf")) or file_path_or_bytes[:4] == b"%PDF"
                if is_pdf:
                    return self._load_pdf_bytes_first_page(file_path_or_bytes)
                
                nparr = np.frombuffer(file_path_or_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is None:
                    return None, "Failed to decode image from bytes"
                return img, None
            else:
                return None, f"Unsupported input type: {type(file_path_or_bytes)}"
        except Exception as e:
            return None, f"Error loading image: {str(e)}"

    def _load_pdf_first_page(self, pdf_path: Path) -> Tuple[np.ndarray, Optional[str]]:
        try:
            doc = fitz.open(str(pdf_path))
            if len(doc) == 0:
                return None, "PDF has no pages"
            page = doc.load_page(0)
            pix = page.get_pixmap(dpi=200)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
            if pix.n == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            elif pix.n == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            elif pix.n == 1:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            return img, None
        except Exception as e:
            return None, f"Failed to render PDF: {str(e)}"

    def _load_pdf_bytes_first_page(self, pdf_bytes: bytes) -> Tuple[np.ndarray, Optional[str]]:
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            if len(doc) == 0:
                return None, "PDF has no pages"
            page = doc.load_page(0)
            pix = page.get_pixmap(dpi=200)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
            if pix.n == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            elif pix.n == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            elif pix.n == 1:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            return img, None
        except Exception as e:
            return None, f"Failed to render PDF bytes: {str(e)}"

    def estimate_skew_angle(self, gray: np.ndarray) -> float:
        """Estimate skew angle using Hough transform and text orientation."""
        try:
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=20)
            if lines is None:
                return 0.0

            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                # Only keep near-horizontal lines (-45 to 45 deg)
                if -45 < angle < 45:
                    angles.append(angle)

            if not angles:
                return 0.0
            return float(np.median(angles))
        except Exception:
            return 0.0

    def rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image by given angle in degrees, expanding bounding box to prevent clipping."""
        if abs(angle) < 0.2:
            return image
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calculate new bounding dimensions
        cos = np.abs(rot_mat[0, 0])
        sin = np.abs(rot_mat[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        
        rot_mat[0, 2] += (new_w / 2) - center[0]
        rot_mat[1, 2] += (new_h / 2) - center[1]
        
        return cv2.warpAffine(image, rot_mat, (new_w, new_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    def detect_and_correct_orientation(self, image: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Detect if image is rotated by 90, 180, or 270 degrees.
        Returns rotated image and degrees rotated.
        """
        # For typical invoices, width is usually <= height (portrait).
        # If landscape, check aspect ratio.
        h, w = image.shape[:2]
        if w > h * 1.3:
            # Likely 90 degrees rotated landscape scan
            rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            return rotated, 90
        return image, 0

    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Execute full preprocessing pipeline:
        1. Orientation correction
        2. Deskewing
        3. Grayscale conversion
        4. Noise filtering (bilateral filter preserving text edges)
        5. Contrast stretching (CLAHE)
        6. Adaptive binarization
        Returns:
            Tuple of (processed_image, metadata)
        """
        metadata = {}
        try:
            # 1. Orientation check
            oriented_img, rot_deg = self.detect_and_correct_orientation(image)
            metadata["orientation_rotation_degrees"] = rot_deg

            # 2. Grayscale
            if len(oriented_img.shape) == 3:
                gray = cv2.cvtColor(oriented_img, cv2.COLOR_BGR2GRAY)
            else:
                gray = oriented_img.copy()

            # 3. Deskewing
            skew_angle = self.estimate_skew_angle(gray)
            metadata["skew_angle_degrees"] = round(skew_angle, 2)
            if abs(skew_angle) >= 0.5:
                gray = self.rotate_image(gray, skew_angle)

            # 4. Bilateral filter for noise reduction while keeping text edges crisp
            denoised = cv2.bilateralFilter(gray, 9, 75, 75)

            # 5. Contrast Limited Adaptive Histogram Equalization (CLAHE)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(denoised)

            # 6. Adaptive thresholding for clean binarization
            binary = cv2.adaptiveThreshold(
                enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 8
            )

            metadata["width"] = binary.shape[1]
            metadata["height"] = binary.shape[0]
            metadata["success"] = True

            return binary, metadata
        except Exception as e:
            # Fail gracefully, return original grayscale or raw
            metadata["success"] = False
            metadata["error"] = str(e)
            if len(image.shape) == 3:
                return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), metadata
            return image, metadata


# Global instance
preprocessor = ImagePreprocessor()
