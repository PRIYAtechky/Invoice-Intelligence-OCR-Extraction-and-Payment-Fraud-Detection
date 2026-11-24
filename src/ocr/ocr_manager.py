"""OCR Engine Manager and Factory."""

from typing import Dict, Any, Optional
from src.ocr.base import BaseOCREngine
from src.ocr.rapid_engine import RapidOCREngine
from src.ocr.easyocr_engine import EasyOCREngine
from src.ocr.tesseract_engine import TesseractOCREngine
from config.settings import settings
from config.logging_config import logger

class OCRManager:
    _instance = None
    _engines: Dict[str, BaseOCREngine] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(OCRManager, cls).__new__(cls)
            cls._instance._init_engines()
        return cls._instance

    def _init_engines(self):
        """Initialize available OCR engines."""
        # RapidOCR (PaddleOCR ONNX)
        try:
            self._engines["rapidocr"] = RapidOCREngine()
        except Exception as e:
            logger.warning(f"RapidOCR initialization failed: {e}")

        # EasyOCR
        try:
            self._engines["easyocr"] = EasyOCREngine()
        except Exception as e:
            logger.warning(f"EasyOCR initialization failed: {e}")

        # Tesseract
        try:
            tess = TesseractOCREngine()
            if tess.is_available:
                self._engines["tesseract"] = tess
        except Exception as e:
            logger.debug(f"Tesseract engine not loaded: {e}")

    def get_available_engines(self) -> Dict[str, str]:
        """Return dict of engine_key -> display_name."""
        return {key: eng.name for key, eng in self._engines.items()}

    def get_engine(self, engine_name: Optional[str] = None) -> BaseOCREngine:
        """Get an OCR engine by key, defaulting to system default."""
        target = (engine_name or settings.DEFAULT_OCR_ENGINE).lower()
        if target in self._engines:
            return self._engines[target]
        
        # Fallback order
        for fallback in ["rapidocr", "easyocr", "tesseract"]:
            if fallback in self._engines:
                logger.warning(f"Requested OCR engine '{target}' not available, falling back to '{fallback}'")
                return self._engines[fallback]
                
        raise RuntimeError("No OCR engines are available on this environment.")

    def extract(self, image_path: str, engine_name: Optional[str] = None) -> Dict[str, Any]:
        """Convenience method to run extraction with selected or default engine."""
        engine = self.get_engine(engine_name)
        return engine.extract_text_and_boxes(image_path)

ocr_manager = OCRManager()
