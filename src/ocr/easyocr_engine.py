"""
EasyOCR Engine Implementation
"""
import time
import numpy as np
from typing import Optional
from src.ocr.base import BaseOCREngine, OCRResult, OCRTextBox


class EasyOCREngine(BaseOCREngine):
    def __init__(self, gpu: bool = False):
        self.engine_name = "EasyOCR"
        self.gpu = gpu
        self._reader = None

    def _get_reader(self):
        if self._reader is None:
            import easyocr
            self._reader = easyocr.Reader(["en"], gpu=self.gpu)
        return self._reader

    def extract_text(self, image: np.ndarray) -> OCRResult:
        t0 = time.time()
        reader = self._get_reader()
        
        # EasyOCR works on BGR, RGB, or grayscale
        raw_results = reader.readtext(image)
        latency = time.time() - t0

        boxes = []
        text_lines = []
        for bbox, text, conf in raw_results:
            text_str = str(text).strip()
            if text_str:
                text_lines.append(text_str)
                # bbox is list of 4 points: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                int_bbox = [[int(pt[0]), int(pt[1])] for pt in bbox] if isinstance(bbox, (list, np.ndarray)) else []
                boxes.append(OCRTextBox(text=text_str, bbox=int_bbox, confidence=float(conf)))

        full_text = "\n".join(text_lines)
        return OCRResult(full_text=full_text, boxes=boxes, engine_name=self.engine_name, latency_seconds=latency)
