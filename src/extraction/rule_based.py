"""
Rule-Based Invoice Information Extractor
Implements deterministic regex, Indian currency normalization (1,23,456.00),
date normalization, vendor fuzzy matching, and line item parsing.
"""
import re
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
from difflib import SequenceMatcher

from src.extraction.schemas import ExtractedInvoice, LineItem, ExtractionSource
from src.config import settings


class RuleBasedExtractor:
    def __init__(self, vendor_master_path: Optional[str] = None, hsn_catalog_path: Optional[str] = None):
        self.vendor_master = self._load_vendor_master(vendor_master_path)
        self.hsn_catalog = self._load_hsn_catalog(hsn_catalog_path)

        # Regex patterns
        self.gstin_pattern = re.compile(r"\b([0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}[Z|2][0-9A-Z#]{1})\b")
        self.po_pattern = re.compile(r"\b(KIS/PO/\d{4}/\d{4,6})\b", re.IGNORECASE)
        self.inv_no_patterns = [
            re.compile(r"(?:invoice\s*(?:no|num|number|#)?|inv\s*(?:no|#)?|bill\s*no)\s*[:.\-]?\s*([A-Z0-9\/\-_]+)", re.IGNORECASE),
            re.compile(r"\b(TI/\d{4}/\d{3,6})\b", re.IGNORECASE),
            re.compile(r"\b(KVR/\d{2,4}-\d{2,4}/\d{3,6})\b", re.IGNORECASE),
            re.compile(r"\b(KVR/\d{4}/\d{3,6})\b", re.IGNORECASE),
            re.compile(r"\b(INV-\d{4,8})\b", re.IGNORECASE),
        ]
        self.date_patterns = [
            (re.compile(r"\b(\d{2})[-/.](\d{2})[-/.](\d{4})\b"), "%d-%m-%Y"),
            (re.compile(r"\b(\d{4})[-/.](\d{2})[-/.](\d{2})\b"), "%Y-%m-%d"),
            (re.compile(r"\b(\d{2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(\d{4})\b", re.IGNORECASE), "%d %b %Y"),
        ]

    def _load_vendor_master(self, path: Optional[str]) -> List[Dict[str, Any]]:
        p = path or settings.vendor_master_path
        try:
            df = pd.read_excel(p)
            return df.to_dict(orient="records")
        except Exception:
            return []

    def _load_hsn_catalog(self, path: Optional[str]) -> List[Dict[str, Any]]:
        p = path or settings.hsn_catalog_path
        try:
            df = pd.read_csv(p)
            return df.to_dict(orient="records")
        except Exception:
            return []

    def parse_indian_amount(self, text_val: str) -> float:
        """
        Parse Indian currency string: handles formats like '1,23,456.00', '2,914.00-', '₹ 45,000.50'.
        """
        if not text_val:
            return 0.0
        cleaned = str(text_val).replace("₹", "").replace("Rs.", "").replace("INR", "").strip()
        cleaned = cleaned.rstrip("-").strip()  # handle trailing negative/dash in OCR
        cleaned = cleaned.replace(",", "")
        try:
            return float(cleaned)
        except ValueError:
            # Fallback regex for numeric extraction
            match = re.search(r"\d+(?:\.\d{1,2})?", cleaned)
            return float(match.group(0)) if match else 0.0

    def normalize_date(self, text_val: str) -> Optional[str]:
        """Convert date to standard YYYY-MM-DD string."""
        if not text_val:
            return None
        text_val = str(text_val).strip()
        for pattern, fmt in self.date_patterns:
            match = pattern.search(text_val)
            if match:
                raw_str = match.group(0).replace("/", "-").replace(".", "-")
                try:
                    dt = datetime.strptime(raw_str, fmt.replace("/", "-").replace(".", "-"))
                    return dt.strftime("%Y-%m-%d")
                except ValueError:
                    continue
        return None

    def match_vendor_name(self, text: str) -> Tuple[Optional[str], float]:
        """Match vendor name from OCR text against vendor master using fuzzy string similarity."""
        best_name = None
        best_score = 0.0
        lines = [line.strip() for line in text.split("\n") if len(line.strip()) > 3]

        for vendor in self.vendor_master:
            v_name = vendor.get("vendor_name", "")
            # Direct check
            if v_name.lower() in text.lower():
                return v_name, 1.0
            
            # Fuzzy check against top lines
            for line in lines[:8]:
                sim = SequenceMatcher(None, v_name.lower(), line.lower()).ratio()
                if sim > best_score and sim >= 0.75:
                    best_score = sim
                    best_name = v_name

        return best_name, round(best_score, 2)

    def extract_gstin(self, text: str) -> Tuple[Optional[str], float]:
        """Extract GSTIN and score confidence."""
        match = self.gstin_pattern.search(text)
        if match:
            gst = match.group(1).upper()
            # If 14th char is '2', normalize to 'Z'
            if len(gst) == 15 and gst[13] == '2':
                gst = gst[:13] + 'Z' + gst[14]
            return gst, 0.95
        return None, 0.0

    def extract_invoice_number(self, text: str) -> Tuple[Optional[str], float]:
        """Extract invoice number using structured regex patterns."""
        for pat in self.inv_no_patterns:
            match = pat.search(text)
            if match:
                inv_no = match.group(1).strip().strip(":,.-")
                return inv_no, 0.90
        return None, 0.0

    def extract_po_number(self, text: str) -> Tuple[Optional[str], float]:
        """Extract Purchase Order reference number."""
        match = self.po_pattern.search(text)
        if match:
            return match.group(1).upper(), 0.95
        return None, 0.0

    def extract_dates(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract invoice date and due date."""
        inv_date = None
        due_date = None

        lines = text.split("\n")
        for i, line in enumerate(lines):
            l_lower = line.lower()
            if any(k in l_lower for k in ["invoice date", "date:", "inv date", "dated"]):
                inv_date = self.normalize_date(line)
            elif any(k in l_lower for k in ["due date", "payment due", "pay by"]):
                due_date = self.normalize_date(line)

        # Fallback if invoice date not found with keyword: search first 10 lines
        if not inv_date:
            for line in lines[:10]:
                d = self.normalize_date(line)
                if d:
                    inv_date = d
                    break

        return inv_date, due_date

    def extract_bank_account(self, text: str) -> Tuple[Optional[str], float]:
        """Extract last 4 digits of bank account."""
        bank_pat = re.compile(r"(?:bank|a\/c|account|acct).*?(\d{4,18})", re.IGNORECASE)
        match = bank_pat.search(text)
        if match:
            num = match.group(1)
            return num[-4:], 0.85
        
        # Check for masked format like XXXX5029 or ****5029
        masked_pat = re.compile(r"(?:[X*]{3,}|x{3,})(\d{4})")
        m_match = masked_pat.search(text)
        if m_match:
            return m_match.group(1), 0.90
        return None, 0.0

    def extract_totals(self, text: str) -> Dict[str, float]:
        """Extract Subtotal, CGST, SGST, IGST, and Grand Total."""
        totals = {"subtotal": 0.0, "cgst": 0.0, "sgst": 0.0, "igst": 0.0, "grand_total": 0.0}
        
        lines = text.split("\n")
        for line in lines:
            l_lower = line.lower()
            
            # Subtotal
            if any(k in l_lower for k in ["sub total", "subtotal", "sub-total"]):
                nums = re.findall(r"[\d,]+(?:\.\d{1,2})?", line)
                if nums:
                    totals["subtotal"] = self.parse_indian_amount(nums[-1])

            # CGST
            elif "cgst" in l_lower:
                nums = re.findall(r"[\d,]+(?:\.\d{1,2})?", line)
                if nums:
                    totals["cgst"] = self.parse_indian_amount(nums[-1])

            # SGST
            elif "sgst" in l_lower:
                nums = re.findall(r"[\d,]+(?:\.\d{1,2})?", line)
                if nums:
                    totals["sgst"] = self.parse_indian_amount(nums[-1])

            # IGST
            elif "igst" in l_lower:
                nums = re.findall(r"[\d,]+(?:\.\d{1,2})?", line)
                if nums:
                    totals["igst"] = self.parse_indian_amount(nums[-1])

            # Grand Total
            elif any(k in l_lower for k in ["grand total", "total amount", "net payable", "total payable", "total:"]):
                nums = re.findall(r"[\d,]+(?:\.\d{1,2})?", line)
                if nums:
                    totals["grand_total"] = self.parse_indian_amount(nums[-1])

        # If subtotal or grand total is missing, attempt arithmetic reconciliation
        if totals["subtotal"] > 0 and totals["grand_total"] == 0:
            totals["grand_total"] = round(totals["subtotal"] + totals["cgst"] + totals["sgst"] + totals["igst"], 2)

        return totals

    def extract_line_items(self, text: str) -> List[LineItem]:
        """Extract line items based on HSN catalog and tabular row patterns."""
        line_items = []
        lines = text.split("\n")
        
        for line in lines:
            # Check if line contains an HSN code (4 digits like 8482, 4016, 7318, etc.)
            hsn_match = re.search(r"\b(8482|4009|7318|4010|8311|6116|6804|8412|2710|8544|4016|8536|3403|8481|8483|8421|8708|3926)\b", line)
            if hsn_match:
                hsn_code = hsn_match.group(1)
                
                # Match quantities and amounts
                numbers = re.findall(r"[\d,]+(?:\.\d{1,2})?", line)
                # Remove HSN code from numbers list if matched
                clean_numbers = [n for n in numbers if n != hsn_code]
                
                if len(clean_numbers) >= 2:
                    qty = self.parse_indian_amount(clean_numbers[0])
                    amt = self.parse_indian_amount(clean_numbers[-1])
                    unit_price = self.parse_indian_amount(clean_numbers[1]) if len(clean_numbers) >= 3 else (round(amt / qty, 2) if qty > 0 else amt)
                    
                    # Extract description before HSN
                    parts = line.split(hsn_code)
                    desc = parts[0].strip(" -#|:")
                    if not desc or len(desc) < 3:
                        # Look up canonical description from catalog
                        cat_desc = [h["description"] for h in self.hsn_catalog if str(h.get("hsn")) == hsn_code]
                        desc = cat_desc[0] if cat_desc else f"Industrial Component (HSN {hsn_code})"

                    line_items.append(LineItem(
                        description=desc,
                        hsn=hsn_code,
                        quantity=qty if qty > 0 else 1.0,
                        unit_price=unit_price,
                        amount=amt
                    ))

        return line_items

    def extract(self, ocr_text: str, file_name: Optional[str] = None) -> ExtractedInvoice:
        """Run complete rule-based extraction pipeline on OCR text."""
        vendor_name, v_conf = self.match_vendor_name(ocr_text)
        gstin, g_conf = self.extract_gstin(ocr_text)
        inv_no, i_conf = self.extract_invoice_number(ocr_text)
        po_no, p_conf = self.extract_po_number(ocr_text)
        inv_date, due_date = self.extract_dates(ocr_text)
        bank_last4, b_conf = self.extract_bank_account(ocr_text)
        totals = self.extract_totals(ocr_text)
        lines = self.extract_line_items(ocr_text)

        confidences = {
            "vendor_name": v_conf,
            "vendor_gstin": g_conf,
            "invoice_number": i_conf,
            "po_number": p_conf,
            "invoice_date": 0.85 if inv_date else 0.0,
            "due_date": 0.85 if due_date else 0.0,
            "grand_total": 0.90 if totals["grand_total"] > 0 else 0.0,
            "bank_account_last4": b_conf
        }

        sources = {k: ExtractionSource.RULE.value for k in confidences.keys()}

        return ExtractedInvoice(
            file_name=file_name,
            invoice_number=inv_no,
            invoice_date=inv_date,
            due_date=due_date,
            vendor_name=vendor_name,
            vendor_gstin=gstin,
            po_number=po_no,
            subtotal=totals["subtotal"],
            cgst=totals["cgst"],
            sgst=totals["sgst"],
            igst=totals["igst"],
            grand_total=totals["grand_total"],
            bank_account_last4=bank_last4,
            line_items=lines,
            raw_ocr_text=ocr_text,
            confidence_scores=confidences,
            extraction_sources=sources,
            input_method="upload"
        )
