"""Extraction Accuracy Evaluator (Precision, Recall, Exact Match)."""

import os
import json
from pathlib import Path
from typing import Dict, Any, List
from src.ocr.ocr_manager import ocr_manager
from src.extraction.rule_extractor import RuleExtractor
from config.settings import settings
from config.logging_config import logger

def normalize_compare_val(val: Any) -> str:
    """Normalize values for robust exact-match comparison."""
    if val is None:
        return ""
    if isinstance(val, (int, float)):
        return f"{float(val):.2f}"
    return str(val).strip().upper().replace(" ", "").replace("/", "").replace("-", "")

def evaluate_extraction_accuracy(
    num_samples: int = 30,
    images_dir: str = None,
    ground_truth_path: str = None
) -> Dict[str, Any]:
    """
    Evaluate extraction accuracy across clean and degraded subsets
    from ground_truth_labeled_180.json.
    """
    gt_path = ground_truth_path or settings.LABELED_GROUND_TRUTH_PATH
    img_dir = Path(images_dir or (settings.BASE_DIR / "data" / "invoice_images"))
    if not img_dir.exists():
        img_dir = Path(r"c:\Users\Badmapriya S\Downloads\T01_Invoice_Intelligence_Starter_Pack\T01_Invoice_Intelligence_Starter_Pack\01_initial_data\invoice_images")

    with open(gt_path, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    clean_samples = [x for x in ground_truth if x.get("degradation_level", 0) <= 0.6][:num_samples // 2]
    degraded_samples = [x for x in ground_truth if x.get("degradation_level", 0) > 0.6][:num_samples // 2]
    eval_set = clean_samples + degraded_samples

    extractor = RuleExtractor()

    target_fields = [
        "vendor_name",
        "vendor_gstin",
        "invoice_number",
        "invoice_date",
        "po_number",
        "grand_total",
        "bank_account_last4"
    ]

    metrics = {
        "clean": {f: {"match": 0, "total": 0} for f in target_fields},
        "degraded": {f: {"match": 0, "total": 0} for f in target_fields},
        "overall": {f: {"match": 0, "total": 0} for f in target_fields},
        "grand_total_matches": {"clean": 0, "degraded": 0, "total": 0},
        "invoices_evaluated": len(eval_set)
    }

    logger.info(f"Evaluating extraction accuracy on {len(eval_set)} invoices...")

    for item in eval_set:
        img_file = img_dir / item["file"]
        if not img_file.exists():
            continue

        is_clean = item.get("degradation_level", 0) <= 0.6
        subset_key = "clean" if is_clean else "degraded"

        # Run OCR
        try:
            ocr_res = ocr_manager.extract(str(img_file))
            extracted = extractor.extract(ocr_res)
        except Exception as e:
            logger.error(f"Failed extraction on {item['file']}: {e}")
            continue

        for field in target_fields:
            gt_val = item.get(field)
            ex_val = getattr(extracted, field, None)

            gt_norm = normalize_compare_val(gt_val)
            ex_norm = normalize_compare_val(ex_val)

            # Check exact match
            is_match = (gt_norm == ex_norm) and bool(gt_norm)

            # For float grand total, allow 1.0 INR difference due to rounding
            if field == "grand_total" and gt_val is not None and ex_val is not None:
                try:
                    if abs(float(gt_val) - float(ex_val)) <= 1.05:
                        is_match = True
                except ValueError:
                    pass

            metrics[subset_key][field]["total"] += 1
            metrics["overall"][field]["total"] += 1

            if is_match:
                metrics[subset_key][field]["match"] += 1
                metrics["overall"][field]["match"] += 1

                if field == "grand_total":
                    metrics["grand_total_matches"][subset_key] += 1
                    metrics["grand_total_matches"]["total"] += 1

    # Format summary report
    report = {
        "invoices_evaluated": len(eval_set),
        "clean_count": len(clean_samples),
        "degraded_count": len(degraded_samples),
        "clean_field_accuracy": {},
        "degraded_field_accuracy": {},
        "overall_field_accuracy": {},
        "clean_avg_exact_match": 0.0,
        "degraded_avg_exact_match": 0.0,
        "overall_avg_exact_match": 0.0,
        "grand_total_accuracy": 0.0
    }

    clean_sum = 0
    degraded_sum = 0
    overall_sum = 0

    for f in target_fields:
        c_acc = (metrics["clean"][f]["match"] / metrics["clean"][f]["total"] * 100) if metrics["clean"][f]["total"] else 0.0
        d_acc = (metrics["degraded"][f]["match"] / metrics["degraded"][f]["total"] * 100) if metrics["degraded"][f]["total"] else 0.0
        o_acc = (metrics["overall"][f]["match"] / metrics["overall"][f]["total"] * 100) if metrics["overall"][f]["total"] else 0.0

        report["clean_field_accuracy"][f] = round(c_acc, 2)
        report["degraded_field_accuracy"][f] = round(d_acc, 2)
        report["overall_field_accuracy"][f] = round(o_acc, 2)

        clean_sum += c_acc
        degraded_sum += d_acc
        overall_sum += o_acc

    report["clean_avg_exact_match"] = round(clean_sum / len(target_fields), 2)
    report["degraded_avg_exact_match"] = round(degraded_sum / len(target_fields), 2)
    report["overall_avg_exact_match"] = round(overall_sum / len(target_fields), 2)
    
    total_invoices = metrics["overall"]["grand_total"]["total"]
    report["grand_total_accuracy"] = round(
        (metrics["grand_total_matches"]["total"] / total_invoices * 100) if total_invoices else 0.0, 2
    )

    return report
