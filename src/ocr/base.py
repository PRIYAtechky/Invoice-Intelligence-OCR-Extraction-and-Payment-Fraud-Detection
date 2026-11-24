"""Base OCR Engine Abstract Class."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List

class BaseOCREngine(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the OCR engine."""
        pass

    @abstractmethod
    def extract_text_and_boxes(self, image_path: str) -> Dict[str, Any]:
        """
        Extract OCR text, line segments, bounding boxes, and confidence.
        
        Returns:
            Dict containing:
                - engine: str
                - raw_text: str
                - lines: List[str]
                - boxes: List[Dict[str, Any]] with keys (box, text, confidence)
                - mean_confidence: float
                - latency_ms: float
        """
        pass
