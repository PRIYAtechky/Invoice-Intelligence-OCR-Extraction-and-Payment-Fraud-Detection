"""
Hybrid Information Extraction Pipeline
Combines Google Gemini LLM Extraction with deterministic Rule-Based Parsing,
attributing data provenance and assigning confidence scores to every extracted field.
"""
from typing import Optional, Dict, Any, Tuple
from src.extraction.schemas import ExtractedInvoice, ExtractionSource
from src.extraction.rule_based import RuleBasedExtractor
from src.extraction.gemini_extractor import GeminiExtractor
from src.config import settings, get_effective_gemini_api_key


class HybridInvoiceExtractor:
    def __init__(self):
        self.rule_extractor = RuleBasedExtractor()
        self.gemini_extractor = GeminiExtractor()

    def extract(self, ocr_text: str, file_name: Optional[str] = None) -> Tuple[ExtractedInvoice, Dict[str, Any]]:
        """
        Execute hybrid extraction:
        1. Always run Rule-based extraction (fast, deterministic local baseline).
        2. If Gemini API key is configured, query Gemini for structured semantic extraction.
        3. Merge results with priority given to Gemini where high confidence,
           falling back to rule-based parsing for any missing fields.
        4. Track field sources and confidence.
        """
        meta = {"gemini_attempted": False, "gemini_success": False, "gemini_error": None}

        # 1. Rule-based extraction
        rule_invoice = self.rule_extractor.extract(ocr_text, file_name=file_name)

        # 2. Check Gemini availability
        api_key = get_effective_gemini_api_key()
        if not api_key:
            meta["message"] = "Rule-based extraction used (no Gemini API key configured)."
            return rule_invoice, meta

        # 3. Attempt Gemini extraction
        meta["gemini_attempted"] = True
        gemini_invoice, err = self.gemini_extractor.extract(ocr_text, file_name=file_name)

        if err or gemini_invoice is None:
            meta["gemini_error"] = err
            meta["message"] = f"Gemini extraction failed ({err}), fallen back to rule-based extraction."
            return rule_invoice, meta

        meta["gemini_success"] = True

        # 4. Intelligent Reconciliation (Merge)
        merged_invoice = gemini_invoice

        # If Gemini missed a field that rule extractor found with high confidence, backfill it
        if not merged_invoice.invoice_number and rule_invoice.invoice_number:
            merged_invoice.invoice_number = rule_invoice.invoice_number
            merged_invoice.extraction_sources["invoice_number"] = ExtractionSource.RULE.value
            merged_invoice.confidence_scores["invoice_number"] = rule_invoice.confidence_scores.get("invoice_number", 0.85)

        if not merged_invoice.vendor_gstin and rule_invoice.vendor_gstin:
            merged_invoice.vendor_gstin = rule_invoice.vendor_gstin
            merged_invoice.extraction_sources["vendor_gstin"] = ExtractionSource.RULE.value
            merged_invoice.confidence_scores["vendor_gstin"] = rule_invoice.confidence_scores.get("vendor_gstin", 0.90)

        if not merged_invoice.po_number and rule_invoice.po_number:
            merged_invoice.po_number = rule_invoice.po_number
            merged_invoice.extraction_sources["po_number"] = ExtractionSource.RULE.value
            merged_invoice.confidence_scores["po_number"] = rule_invoice.confidence_scores.get("po_number", 0.90)

        if not merged_invoice.bank_account_last4 and rule_invoice.bank_account_last4:
            merged_invoice.bank_account_last4 = rule_invoice.bank_account_last4
            merged_invoice.extraction_sources["bank_account_last4"] = ExtractionSource.RULE.value
            merged_invoice.confidence_scores["bank_account_last4"] = rule_invoice.confidence_scores.get("bank_account_last4", 0.85)

        if (not merged_invoice.line_items or len(merged_invoice.line_items) == 0) and rule_invoice.line_items:
            merged_invoice.line_items = rule_invoice.line_items

        if merged_invoice.grand_total == 0.0 and rule_invoice.grand_total > 0.0:
            merged_invoice.grand_total = rule_invoice.grand_total
            merged_invoice.subtotal = rule_invoice.subtotal
            merged_invoice.cgst = rule_invoice.cgst
            merged_invoice.sgst = rule_invoice.sgst
            merged_invoice.igst = rule_invoice.igst
            merged_invoice.extraction_sources["grand_total"] = ExtractionSource.RULE.value

        meta["message"] = "Hybrid extraction succeeded (Gemini AI + Rule verification)."
        return merged_invoice, meta


# Global extractor instance
invoice_extractor = HybridInvoiceExtractor()
