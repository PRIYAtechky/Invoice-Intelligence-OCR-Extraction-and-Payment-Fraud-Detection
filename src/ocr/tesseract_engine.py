"""
PyTesseract Engine Implementation with Automatic Path Discovery
"""
import os
import shutil
import time
import numpy as np
from pathlib import Path
from typing import Optional
from src.ocr.base import BaseOCREngine, OCRResult, OCRTextBox
from src.config import settings


class TesseractEngine(BaseOCREngine):
    def __init__(self, tesseract_cmd: Optional[str] = None):
        self.engine_name = "Tesseract"
        self.tesseract_cmd = tesseract_cmd or self._discover_tesseract()

    def _discover_tesseract(self) -> Optional[str]:
        if settings.tesseract_cmd and Path(settings.tesseract_cmd).exists():
            return settings.tesseract_cmd
        
        # Check PATH
        path_cmd = shutil.which("tesseract")
        if path_cmd:
            return path_cmd
            
        # Common Windows locations
        win_candidates = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            os.path.expanduser(r"~\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"),
            os.path.expanduser(r"~\tesseract-ocr\tesseract.exe")
        ]
        for c in win_candidates:
            if Path(c).exists():
                return c
        return None

    def is_available(self) -> bool:
        return self.tesseract_cmd is not None and Path(self.tesseract_cmd).exists()

    def extract_text(self, image: np.ndarray) -> OCRResult:
        if not self.is_available():
            raise RuntimeError(
                "Tesseract executable not found. Please install Tesseract or configure its path in Settings."
            )

        import pytesseract
        pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd

        t0 = time.time()
        # Extract word data with bounding boxes and confidence
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        latency = time.time() - t0

        boxes = []
        lines_dict = {}
        for i in range(len(data["text"])):
            word = data["text"][i].strip()
            conf = float(data["conf"][i])
            if word and conf > 0:
                x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
                line_num = data["line_num"][i]
                block_num = data["block_num"][i]
                line_key = (block_num, line_num)
                if line_key not in lines_dict:
                    lines_dict[line_key] = []
                lines_dict[line_key].append(word)
                boxes.append(OCRTextBox(text=word, bbox=[x, y, w, h], confidence=conf / 100.0))

        # Reconstruct lines
        full_text = "\n".join([" ".join(words) for words in lines_dict.values()])
        return OCRResult(full_text=full_text, boxes=boxes, engine_name=self.engine_name, latency_seconds=latency)
