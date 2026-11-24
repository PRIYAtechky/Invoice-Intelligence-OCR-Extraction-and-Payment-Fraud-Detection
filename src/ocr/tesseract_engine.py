"""Tesseract OCR Engine."""

import time
import shutil
from typing import Dict, Any, List
from src.ocr.base import BaseOCREngine
from config.logging_config import logger

class TesseractOCREngine(BaseOCREngine):
    def __init__(self, tesseract_cmd: str = None):
        import pytesseract
        self.pytesseract = pytesseract
        
        # Check standard installation paths if not on PATH
        candidates = [
            tesseract_cmd,
            shutil.which("tesseract"),
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
        ]
        
        self.cmd_path = None
        for c in candidates:
            if c and shutil.which(c):
                self.cmd_path = c
                self.pytesseract.pytesseract.tesseract_cmd = c
                break
                
        self._available = self.cmd_path is not None
        if not self._available:
            logger.info("Tesseract binary not found on system PATH. RapidOCR and EasyOCR are active.")

    @property
    def name(self) -> str:
        return "Tesseract OCR"

    @property
    def is_available(self) -> bool:
        return self._available

    def extract_text_and_boxes(self, image_path: str) -> Dict[str, Any]:
        if not self._available:
            raise RuntimeError("Tesseract binary is not installed on this system.")
            
        start_time = time.time()
        data = self.pytesseract.image_to_data(image_path, output_type=self.pytesseract.Output.DICT)
        latency_ms = (time.time() - start_time) * 1000.0
        
        n_boxes = len(data["text"])
        lines_dict = {}
        boxes = []
        confidences = []
        
        for i in range(n_boxes):
            word = data["text"][i].strip()
            conf = float(data["conf"][i])
            if word and conf > 0:
                line_num = data["line_num"][i]
                lines_dict.setdefault(line_num, []).append(word)
                confidences.append(conf / 100.0)
                boxes.append({
                    "box": [data["left"][i], data["top"][i], data["left"][i] + data["width"][i], data["top"][i] + data["height"][i]],
                    "text": word,
                    "confidence": round(conf / 100.0, 4)
                })
                
        lines = [" ".join(words) for _, words in sorted(lines_dict.items())]
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
