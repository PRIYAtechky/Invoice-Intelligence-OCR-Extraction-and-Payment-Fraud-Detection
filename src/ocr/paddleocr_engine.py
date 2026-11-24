"""
PaddleOCR Engine Implementation
"""
import time
import numpy as np
from src.ocr.base import BaseOCREngine, OCRResult, OCRTextBox


class PaddleOCREngine(BaseOCREngine):
    def __init__(self, use_gpu: bool = False):
        self.engine_name = "PaddleOCR"
        self.use_gpu = use_gpu
        self._ocr = None

    def _get_ocr(self):
        if self._ocr is None:
            try:
                from paddleocr import PaddleOCR
                # Disable angle classification to speed up inference if not needed
                self._ocr = PaddleOCR(use_angle_cls=False, lang="en")
            except Exception as e:
                raise RuntimeError(f"Failed to initialize PaddleOCR: {e}")
        return self._ocr

    def extract_text(self, image: np.ndarray) -> OCRResult:
        t0 = time.time()
        ocr = self._get_ocr()
        
        raw_results = ocr.ocr(image)
        latency = time.time() - t0

        boxes = []
        text_lines = []
        
        if raw_results and isinstance(raw_results, list) and len(raw_results) > 0 and raw_results[0] is not None:
            for item in raw_results[0]:
                if len(item) == 2:
                    bbox, (text, conf) = item
                    text_str = str(text).strip()
                    if text_str:
                        text_lines.append(text_str)
                        int_bbox = [[int(pt[0]), int(pt[1])] for pt in bbox] if isinstance(bbox, list) else []
                        boxes.append(OCRTextBox(text=text_str, bbox=int_bbox, confidence=float(conf)))

        full_text = "\n".join(text_lines)
        return OCRResult(full_text=full_text, boxes=boxes, engine_name=self.engine_name, latency_seconds=latency)
