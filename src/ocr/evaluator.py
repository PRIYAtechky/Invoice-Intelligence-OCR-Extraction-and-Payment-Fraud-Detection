"""OCR Comparison and Evaluation Module (CER, WER, Latency)."""

import os
import json
import time
from typing import Dict, Any, List
from pathlib import Path
from src.ocr.rapid_engine import RapidOCREngine
from src.ocr.easyocr_engine import EasyOCREngine
from config.settings import settings
from config.logging_config import logger

def calculate_levenshtein(s1: str, s2: str) -> int:
    """Compute standard Levenshtein edit distance."""
    if len(s1) < len(s2):
        return calculate_levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def calculate_cer(reference: str, hypothesis: str) -> float:
    """Calculate Character Error Rate: EditDistance / len(Reference)."""
    ref_clean = "".join(reference.split())
    hyp_clean = "".join(hypothesis.split())
    if not ref_clean:
        return 0.0 if not hyp_clean else 1.0
    dist = calculate_levenshtein(ref_clean, hyp_clean)
    return min(1.0, dist / len(ref_clean))

def evaluate_ocr_engines(
    sample_size: int = 15,
    images_dir: str = None,
    ground_truth_path: str = None
) -> Dict[str, Any]:
    """Run evaluation comparing RapidOCR (PaddleOCR) and EasyOCR."""
    gt_file = ground_truth_path or settings.LABELED_GROUND_TRUTH_PATH
    img_dir = Path(images_dir or (settings.BASE_DIR / "data" / "invoice_images"))
    if not img_dir.exists():
        # Fallback to starter pack directory if not copied
        img_dir = Path(r"c:\Users\Badmapriya S\Downloads\T01_Invoice_Intelligence_Starter_Pack\T01_Invoice_Intelligence_Starter_Pack\01_initial_data\invoice_images")

    with open(gt_file, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    # Stratified selection by degradation level (clean <= 0.6, degraded > 0.6)
    clean_items = [x for x in ground_truth if x.get("degradation_level", 0) <= 0.6][:sample_size // 2]
    degraded_items = [x for x in ground_truth if x.get("degradation_level", 0) > 0.6][:sample_size // 2]
    eval_items = clean_items + degraded_items

    rapid = RapidOCREngine()
    easy = EasyOCREngine()

    results = {
        "PaddleOCR (RapidOCR)": {"clean_cer": [], "degraded_cer": [], "latencies": [], "avg_conf": []},
        "EasyOCR": {"clean_cer": [], "degraded_cer": [], "latencies": [], "avg_conf": []}
    }

    logger.info(f"Evaluating {len(eval_items)} invoices across OCR engines...")
    
    for item in eval_items:
        img_file = img_dir / item["file"]
        if not img_file.exists():
            continue

        # Build expected text reference from ground truth fields
        ref_tokens = [
            item.get("vendor_name", ""),
            item.get("vendor_gstin", ""),
            item.get("invoice_number", ""),
            item.get("invoice_date", ""),
            item.get("po_number", ""),
            str(item.get("subtotal", "")),
            str(item.get("grand_total", ""))
        ]
        reference_text = " ".join([t for t in ref_tokens if t])
        is_clean = item.get("degradation_level", 0) <= 0.6

        # Test RapidOCR
        try:
            r_out = rapid.extract_text_and_boxes(str(img_file))
            r_cer = calculate_cer(reference_text, r_out["raw_text"])
            if is_clean:
                results["PaddleOCR (RapidOCR)"]["clean_cer"].append(r_cer)
            else:
                results["PaddleOCR (RapidOCR)"]["degraded_cer"].append(r_cer)
            results["PaddleOCR (RapidOCR)"]["latencies"].append(r_out["latency_ms"])
            results["PaddleOCR (RapidOCR)"]["avg_conf"].append(r_out["mean_confidence"])
        except Exception as e:
            logger.error(f"RapidOCR eval error on {item['file']}: {e}")

        # Test EasyOCR
        try:
            e_out = easy.extract_text_and_boxes(str(img_file))
            e_cer = calculate_cer(reference_text, e_out["raw_text"])
            if is_clean:
                results["EasyOCR"]["clean_cer"].append(e_cer)
            else:
                results["EasyOCR"]["degraded_cer"].append(e_cer)
            results["EasyOCR"]["latencies"].append(e_out["latency_ms"])
            results["EasyOCR"]["avg_conf"].append(e_out["mean_confidence"])
        except Exception as e:
            logger.error(f"EasyOCR eval error on {item['file']}: {e}")

    summary = {}
    for engine_name, data in results.items():
        clean_avg = sum(data["clean_cer"]) / len(data["clean_cer"]) if data["clean_cer"] else 0.0
        degraded_avg = sum(data["degraded_cer"]) / len(data["degraded_cer"]) if data["degraded_cer"] else 0.0
        overall_avg = (clean_avg + degraded_avg) / 2.0
        avg_lat = sum(data["latencies"]) / len(data["latencies"]) if data["latencies"] else 0.0
        avg_conf = sum(data["avg_conf"]) / len(data["avg_conf"]) if data["avg_conf"] else 0.0
        summary[engine_name] = {
            "clean_cer": round(clean_avg * 100, 2),
            "degraded_cer": round(degraded_avg * 100, 2),
            "overall_cer": round(overall_avg * 100, 2),
            "avg_latency_ms": round(avg_lat, 1),
            "avg_confidence": round(avg_conf * 100, 2)
        }

    return summary
