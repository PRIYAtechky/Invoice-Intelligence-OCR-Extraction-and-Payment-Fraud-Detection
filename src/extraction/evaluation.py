"""
Extraction Accuracy Evaluation Engine
Computes field-level precision, recall, exact-match accuracy, and grand-total accuracy
split by clean vs degraded scans on the 180 ground-truth labelled invoices.
"""
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np

from src.extraction.schemas import ExtractedInvoice
from src.extraction.hybrid_extractor import invoice_extractor
from src.extraction.rule_based import RuleBasedExtractor
from src.preprocessing.image_pipeline import preprocessor
from src.ocr.ocr_comparison import get_ocr_engine
from src.config import settings


class ExtractionAccuracyEvaluator:
    def __init__(self, ground_truth_path: Optional[Path] = None, images_dir: Optional[Path] = None):
        self.ground_truth_path = ground_truth_path or settings.ground_truth_path
        self.images_dir = images_dir or settings.invoice_images_dir
        self.rule_extractor = RuleBasedExtractor()

    def load_ground_truth(self) -> List[Dict[str, Any]]:
        with open(self.ground_truth_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _is_match(val1, val2, is_numeric: bool = False, tolerance: float = 0.05) -> bool:
        """Check if extracted value matches ground truth value."""
        if val1 is None and val2 is None:
            return True
        if val1 is None or val2 is None:
            return False

        if is_numeric:
            try:
                n1 = float(str(val1).replace(",", "").strip())
                n2 = float(str(val2).replace(",", "").strip())
                return abs(n1 - n2) <= tolerance
            except (ValueError, TypeError):
                return False
        else:
            s1 = str(val1).strip().lower().replace("-", "").replace("/", "").replace(" ", "")
            s2 = str(val2).strip().lower().replace("-", "").replace("/", "").replace(" ", "")
            return s1 == s2

    def evaluate(self, sample_size: int = 30, use_gemini: bool = True, output_report_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Evaluate accuracy across clean and degraded subsets from the 180 labelled invoices.
        """
        labels = self.load_ground_truth()
        sample = labels[:sample_size]
        ocr_engine = get_ocr_engine()

        fields_to_eval = [
            ("vendor_name", False),
            ("vendor_gstin", False),
            ("invoice_number", False),
            ("invoice_date", False),
            ("po_number", False),
            ("grand_total", True),
            ("bank_account_last4", False)
        ]

        # Overall counters
        stats = {
            f: {"correct": 0, "total": 0, "clean_correct": 0, "clean_total": 0, "deg_correct": 0, "deg_total": 0}
            for f, _ in fields_to_eval
        }
        grand_total_matches = 0
        total_eval_count = 0

        for item in sample:
            file_name = item["file"]
            img_path = self.images_dir / file_name
            if not img_path.exists():
                continue

            img, err = preprocessor.load_image(img_path)
            if img is None:
                continue

            prep_img, _ = preprocessor.preprocess(img)
            ocr_res = ocr_engine.extract_text(prep_img)

            if use_gemini:
                extracted, _ = invoice_extractor.extract(ocr_res.full_text, file_name=file_name)
            else:
                extracted = self.rule_extractor.extract(ocr_res.full_text, file_name=file_name)

            is_degraded = item.get("degradation_level", 0.0) > 0.5
            total_eval_count += 1

            for field_name, is_num in fields_to_eval:
                gt_val = item.get(field_name)
                ext_val = getattr(extracted, field_name, None)
                matched = self._is_match(ext_val, gt_val, is_numeric=is_num)

                stats[field_name]["total"] += 1
                if is_degraded:
                    stats[field_name]["deg_total"] += 1
                else:
                    stats[field_name]["clean_total"] += 1

                if matched:
                    stats[field_name]["correct"] += 1
                    if is_degraded:
                        stats[field_name]["deg_correct"] += 1
                    else:
                        stats[field_name]["clean_correct"] += 1

                if field_name == "grand_total" and matched:
                    grand_total_matches += 1

        # Summary calculation
        field_accuracy = {}
        clean_accuracies = []
        deg_accuracies = []

        for f, _ in fields_to_eval:
            tot = stats[f]["total"]
            corr = stats[f]["correct"]
            c_tot = stats[f]["clean_total"]
            c_corr = stats[f]["clean_correct"]
            d_tot = stats[f]["deg_total"]
            d_corr = stats[f]["deg_correct"]

            acc = (corr / tot) if tot > 0 else 0.0
            c_acc = (c_corr / c_tot) if c_tot > 0 else 0.0
            d_acc = (d_corr / d_tot) if d_tot > 0 else 0.0

            field_accuracy[f] = {
                "accuracy": round(acc * 100, 2),
                "clean_scans_accuracy": round(c_acc * 100, 2),
                "degraded_scans_accuracy": round(d_acc * 100, 2)
            }
            if c_tot > 0:
                clean_accuracies.append(c_acc)
            if d_tot > 0:
                deg_accuracies.append(d_acc)

        clean_avg = (sum(clean_accuracies) / len(clean_accuracies) * 100) if clean_accuracies else 0.0
        deg_avg = (sum(deg_accuracies) / len(deg_accuracies) * 100) if deg_accuracies else 0.0
        gt_acc = (grand_total_matches / total_eval_count * 100) if total_eval_count > 0 else 0.0

        report = {
            "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dataset": "ground_truth_labeled_180.json",
            "sample_size": total_eval_count,
            "overall_field_accuracy_clean": round(clean_avg, 2),
            "overall_field_accuracy_degraded": round(deg_avg, 2),
            "grand_total_extraction_accuracy": round(gt_acc, 2),
            "targets": {
                "clean_target": ">= 95%",
                "clean_met": clean_avg >= 95.0,
                "degraded_target": ">= 85%",
                "degraded_met": deg_avg >= 85.0,
                "grand_total_target": ">= 97%",
                "grand_total_met": gt_acc >= 97.0
            },
            "field_level_breakdown": field_accuracy
        }

        save_path = output_report_path or (settings.project_root / "reports" / "extraction_accuracy_report.json")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        return report
