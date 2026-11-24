"""
Base OCR Engine Interface and Evaluation Metrics (CER)
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np


class OCRTextBox:
    def __init__(self, text: str, bbox: Optional[List[int]] = None, confidence: float = 1.0):
        self.text = text
        self.bbox = bbox or []  # [x, y, w, h] or list of [x, y] coordinates
        self.confidence = confidence

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "bbox": self.bbox,
            "confidence": round(self.confidence, 4)
        }


class OCRResult:
    def __init__(self, full_text: str, boxes: List[OCRTextBox], engine_name: str, latency_seconds: float = 0.0):
        self.full_text = full_text
        self.boxes = boxes
        self.engine_name = engine_name
        self.latency_seconds = round(latency_seconds, 3)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "full_text": self.full_text,
            "engine_name": self.engine_name,
            "latency_seconds": self.latency_seconds,
            "box_count": len(self.boxes),
            "boxes": [b.to_dict() for b in self.boxes]
        }


class BaseOCREngine(ABC):
    @abstractmethod
    def extract_text(self, image: np.ndarray) -> OCRResult:
        """Process image array and return standardized OCRResult."""
        pass


def calculate_cer(reference: str, hypothesis: str) -> float:
    """
    Calculate Character Error Rate (CER) using standard Levenshtein distance:
    CER = (Substitutions + Deletions + Insertions) / len(reference)
    Returns float in range [0.0, 1.0] (or >1.0 if severe insertions occur).
    """
    r = reference.strip()
    h = hypothesis.strip()
    
    if not r:
        return 0.0 if not h else 1.0
    if not h:
        return 1.0

    n, m = len(r), len(h)
    # Distance matrix optimization (two rows)
    dp = list(range(m + 1))
    
    for i in range(1, n + 1):
        prev_dp = dp[:]
        dp[0] = i
        for j in range(1, m + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            dp[j] = min(
                prev_dp[j] + 1,      # deletion
                dp[j - 1] + 1,       # insertion
                prev_dp[j - 1] + cost  # substitution
            )
            
    dist = dp[m]
    cer = dist / max(1, n)
    return round(float(cer), 4)
