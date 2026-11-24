"""EasyOCR Engine."""

import time
from typing import Dict, Any, List
from src.ocr.base import BaseOCREngine
from config.logging_config import logger

class EasyOCREngine(BaseOCREngine):
    def __init__(self, languages: List[str] = None, gpu: bool = False):
        try:
            import easyocr
            self.reader = easyocr.Reader(languages or ["en"], gpu=gpu)
            self._available = True
        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR: {e}")
            self.reader = None
            self._available = False

    @property
    def name(self) -> str:
        return "EasyOCR"

    def extract_text_and_boxes(self, image_path: str) -> Dict[str, Any]:
        if not self._available or self.reader is None:
            raise RuntimeError("EasyOCR is not available or failed to initialize.")
        
        start_time = time.time()
        results = self.reader.readtext(image_path)
        latency_ms = (time.time() - start_time) * 1000.0
        
        lines: List[str] = []
        boxes: List[Dict[str, Any]] = []
        confidences: List[float] = []
        
        for item in results:
            box_pts = item[0]
            text = str(item[1]).strip()
            conf = float(item[2])
            if text:
                lines.append(text)
                confidences.append(conf)
                boxes.append({
                    "box": box_pts,
                    "text": text,
                    "confidence": conf
                })
                
        raw_text = "\n".join(lines)
        mean_conf = float(sum(confidences) / len(confidences)) if confidences else 0.0
        
        return {
            "engine": self.name,
            "raw_text": raw_text,
            "lines": lines,
            "boxes": boxes,
            "mean_confidence": round(mean_conf, 4),
            "latency_ms": round(latency_ms, 2)
        }
