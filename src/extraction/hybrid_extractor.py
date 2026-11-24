"""Hybrid Extractor fusing Deterministic Rules and Gemini LLM."""

from typing import Dict, Any, Optional, List
from src.extraction.schemas import InvoiceExtraction, LineItem
from src.extraction.rule_extractor import RuleExtractor
from src.extraction.gemini_extractor import GeminiExtractor
from config.settings import settings
from config.logging_config import logger

class HybridExtractor:
    def __init__(
        self,
        vendor_master_path: Optional[str] = None,
        gemini_api_key: Optional[str] = None
    ):
        self.rule_extractor = RuleExtractor(vendor_master_path)
        self.gemini_extractor = GeminiExtractor(api_key=gemini_api_key)

    def extract(
        self,
        ocr_result: Dict[str, Any],
        use_llm: bool = True
    ) -> InvoiceExtraction:
        """Run hybrid extraction, fusing rule and LLM outputs with provenance."""
        # 1. Always run fast deterministic rule-based extractor
        rule_extracted = self.rule_extractor.extract(ocr_result)

        # If LLM disabled or not configured, return rule extraction directly
        if not use_llm or not self.gemini_extractor.is_configured():
            return rule_extracted

        # 2. Run Gemini extraction
        gemini_extracted = self.gemini_extractor.extract(ocr_result)
        if not gemini_extracted:
            return rule_extracted

        # 3. Fuse extractions field by field
        fused_confidence: Dict[str, float] = {}
        needs_review: List[str] = []

        # Helper to fuse field
        def fuse_field(field_name: str, r_val: Any, g_val: Any, r_conf: float, g_conf: float):
            if r_val is not None and g_val is not None:
                if str(r_val).strip().lower() == str(g_val).strip().lower():
                    # Strong agreement
                    fused_confidence[field_name] = 0.99
                    return r_val
                else:
                    # Disagreement -> Pick higher confidence or arithmetic check
                    if r_conf >= g_conf:
                        fused_confidence[field_name] = r_conf
                        needs_review.append(field_name)
                        return r_val
                    else:
                        fused_confidence[field_name] = g_conf
                        needs_review.append(field_name)
                        return g_val
            elif r_val is not None:
                fused_confidence[field_name] = r_conf
                return r_val
            elif g_val is not None:
                fused_confidence[field_name] = g_conf
                return g_val
            else:
                fused_confidence[field_name] = 0.0
                needs_review.append(field_name)
                return None

        # Vendor Name
        vendor_name = fuse_field(
            "vendor_name",
            rule_extracted.vendor_name,
            gemini_extracted.vendor_name,
            rule_extracted.confidence_scores.get("vendor_name", 0.0),
            gemini_extracted.confidence_scores.get("vendor_name", 0.0)
        )

        # Vendor GSTIN
        vendor_gstin = fuse_field(
            "vendor_gstin",
            rule_extracted.vendor_gstin,
            gemini_extracted.vendor_gstin,
            rule_extracted.confidence_scores.get("vendor_gstin", 0.0),
            gemini_extracted.confidence_scores.get("vendor_gstin", 0.0)
        )

        # Invoice Number
        invoice_number = fuse_field(
            "invoice_number",
            rule_extracted.invoice_number,
            gemini_extracted.invoice_number,
            rule_extracted.confidence_scores.get("invoice_number", 0.0),
            gemini_extracted.confidence_scores.get("invoice_number", 0.0)
        )

        # Invoice Date
        invoice_date = fuse_field(
            "invoice_date",
            rule_extracted.invoice_date,
            gemini_extracted.invoice_date,
            rule_extracted.confidence_scores.get("invoice_date", 0.0),
            gemini_extracted.confidence_scores.get("invoice_date", 0.0)
        )

        # Due Date
        due_date = fuse_field(
            "due_date",
            rule_extracted.due_date,
            gemini_extracted.due_date,
            rule_extracted.confidence_scores.get("due_date", 0.0),
            gemini_extracted.confidence_scores.get("due_date", 0.0)
        )

        # PO Number
        po_number = fuse_field(
            "po_number",
            rule_extracted.po_number,
            gemini_extracted.po_number,
            rule_extracted.confidence_scores.get("po_number", 0.0),
            gemini_extracted.confidence_scores.get("po_number", 0.0)
        )

        # Grand Total
        grand_total = fuse_field(
            "grand_total",
            rule_extracted.grand_total,
            gemini_extracted.grand_total,
            rule_extracted.confidence_scores.get("grand_total", 0.0),
            gemini_extracted.confidence_scores.get("grand_total", 0.0)
        )

        # Subtotal
        subtotal = fuse_field(
            "subtotal",
            rule_extracted.subtotal,
            gemini_extracted.subtotal,
            rule_extracted.confidence_scores.get("subtotal", 0.0),
            gemini_extracted.confidence_scores.get("subtotal", 0.0)
        )

        # Taxes
        cgst = gemini_extracted.cgst if gemini_extracted.cgst else rule_extracted.cgst
        sgst = gemini_extracted.sgst if gemini_extracted.sgst else rule_extracted.sgst
        igst = gemini_extracted.igst if gemini_extracted.igst else rule_extracted.igst

        # Bank account
        bank_account_last4 = fuse_field(
            "bank_account_last4",
            rule_extracted.bank_account_last4,
            gemini_extracted.bank_account_last4,
            rule_extracted.confidence_scores.get("bank_account_last4", 0.0),
            gemini_extracted.confidence_scores.get("bank_account_last4", 0.0)
        )

        # Line items: Prefer Gemini if detailed items match subtotal, else fallback to rule line items
        if gemini_extracted.line_items and len(gemini_extracted.line_items) > len(rule_extracted.line_items):
            line_items = gemini_extracted.line_items
        else:
            line_items = rule_extracted.line_items

        return InvoiceExtraction(
            vendor_name=vendor_name,
            vendor_gstin=vendor_gstin,
            invoice_number=invoice_number,
            invoice_date=invoice_date,
            due_date=due_date,
            po_number=po_number,
            line_items=line_items,
            subtotal=subtotal,
            cgst=cgst,
            sgst=sgst,
            igst=igst,
            grand_total=grand_total,
            bank_account_last4=bank_account_last4,
            extraction_source="hybrid",
            confidence_scores=fused_confidence,
            raw_ocr_text=ocr_result.get("raw_text", ""),
            needs_review_fields=list(set(needs_review))
        )
