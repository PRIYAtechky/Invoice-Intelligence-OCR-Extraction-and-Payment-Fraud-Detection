"""Robust Rule-based deterministic extraction for Indian B2B Invoices."""

import re
from typing import Dict, Any, List, Optional, Tuple
from difflib import SequenceMatcher
from datetime import datetime
import pandas as pd

from src.extraction.schemas import InvoiceExtraction, LineItem
from config.settings import settings
from config.logging_config import logger

BUYER_GSTIN = "33AACCK4821M1Z5"
BUYER_NAME = "Kaveri Industrial Supplies"

MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12
}

class RuleExtractor:
    def __init__(self, vendor_master_path: Optional[str] = None):
        self.vendors = []
        vm_path = vendor_master_path or settings.VENDOR_MASTER_PATH
        try:
            df = pd.read_excel(vm_path)
            self.vendors = df.to_dict("records")
        except Exception as e:
            logger.warning(f"Could not load vendor master for rule extraction: {e}")

    def clean_text(self, text: str) -> str:
        """Normalize whitespace and common OCR artifacts."""
        text = text.replace("\r", "\n")
        text = text.replace("₹", " Rs ").replace("`", "")
        return text

    def parse_amount(self, text: str) -> Optional[float]:
        """Parse currency strings like '1,23,456.78' or '2,914.00.'."""
        if not text:
            return None
        clean = re.sub(r"[^\d.,]", "", text).strip().rstrip(".,").replace(",", "")
        clean = re.sub(r"\.+", ".", clean)
        try:
            val = float(clean)
            return round(val, 2)
        except ValueError:
            return None

    def normalize_date(self, date_str: str) -> Optional[str]:
        """Convert diverse date formats including month names to strict YYYY-MM-DD."""
        if not date_str:
            return None
        
        clean = date_str.strip().strip(".:;, ")
        
        # Check for month name strings like 'October 17, 2025' or '17 Oct 2025' or 'Octaber'
        clean_lower = clean.lower()
        for m_str, m_num in MONTHS.items():
            # Exact or fuzzy match for month
            if m_str in clean_lower or (m_str == "oct" and "octa" in clean_lower) or (m_str == "dec" and "decen" in clean_lower):
                # Search for day and year
                nums = re.findall(r"\b(\d{1,4})\b", clean)
                if len(nums) >= 2:
                    d = int(nums[0]) if len(nums[0]) <= 2 else int(nums[1])
                    y = int(nums[-1])
                    if y < 100:
                        y += 2000
                    if 1 <= d <= 31 and 2000 <= y <= 2035:
                        return f"{y:04d}-{m_num:02d}-{d:02d}"

        # Try standard numeric patterns
        clean_num = clean.replace("/", "-").replace(".", "-")
        # DD-MM-YYYY
        m = re.search(r"\b(\d{1,2})[-/](\d{1,2})[-/](\d{2,4})\b", clean_num)
        if m:
            d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if y < 100:
                y += 2000
            if d > 12 and mo <= 12:
                return f"{y:04d}-{mo:02d}-{d:02d}"
            elif mo > 12 and d <= 12:
                return f"{y:04d}-{d:02d}-{mo:02d}"
            else:
                return f"{y:04d}-{mo:02d}-{d:02d}"
                
        # YYYY-MM-DD
        m2 = re.search(r"\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b", clean_num)
        if m2:
            return f"{m2.group(1)}-{int(m2.group(2)):02d}-{int(m2.group(3)):02d}"
            
        return None

    def extract_gstins(self, lines: List[str]) -> Tuple[Optional[str], float]:
        """Extract vendor GSTIN, distinguishing it from buyer (Kaveri) GSTIN."""
        candidates = []
        for idx, line in enumerate(lines):
            clean_l = line.replace(" ", "").replace(".", "")
            matches = re.findall(r"([0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}[Z|2]{1}[0-9A-Z]{1})", clean_l, re.I)
            for m in matches:
                m_clean = m.upper()
                if len(m_clean) == 15 and m_clean[13] == '2':
                    m_clean = m_clean[:13] + 'Z' + m_clean[14:]
                candidates.append((m_clean, idx))
                
        vendor_candidates = [c for c in candidates if c[0] != BUYER_GSTIN]
        if vendor_candidates:
            return vendor_candidates[0][0], 0.98
            
        all_text = "".join(lines).upper().replace(".", "").replace(" ", "")
        for v in self.vendors:
            vgst = str(v.get("gstin", "")).upper()
            if vgst and vgst in all_text and vgst != BUYER_GSTIN:
                return vgst, 0.95
                
        return None, 0.0

    def extract_vendor_name(self, lines: List[str], vendor_gstin: Optional[str]) -> Tuple[Optional[str], float]:
        """Identify vendor name from header lines or vendor master join."""
        if vendor_gstin:
            for v in self.vendors:
                if str(v.get("gstin", "")).upper() == vendor_gstin.upper():
                    return v.get("vendor_name"), 0.99

        best_match = None
        best_score = 0.0
        
        for line in lines[:8]:
            l_clean = line.strip()
            if BUYER_NAME.lower() in l_clean.lower() or "invoice" in l_clean.lower():
                continue
            for v in self.vendors:
                vname = v.get("vendor_name", "")
                if not vname:
                    continue
                ratio = SequenceMatcher(None, vname.lower(), l_clean.lower()).ratio()
                if ratio > best_score:
                    best_score = ratio
                    best_match = vname
                    
        if best_score >= 0.70:
            return best_match, round(best_score, 2)

        if lines:
            first = lines[0].strip()
            if len(first) > 4 and not re.search(r"INVOICE|TAX|ORIGINAL|BILL", first, re.I):
                return first, 0.50
                
        return None, 0.0

    def extract_invoice_number(self, lines: List[str]) -> Tuple[Optional[str], float]:
        """Extract invoice number string, inspecting current, above, and below lines."""
        # 1. First priority: Known structured invoice patterns anywhere in header/body
        for l in lines[:20]:
            m_format = re.search(r"\b(TI\/\d{4}\/\d{4,5}|KVR\/[0-9\-]+\/\d{4,5}|INV-[0-9]{4,8})\b", l, re.I)
            if m_format:
                return m_format.group(1).strip(), 0.98

        # 2. Key-value prefix search
        for idx, line in enumerate(lines):
            l = line.strip()
            # Explicit prefix on same line (e.g. Inv #: TI/2025/8093)
            m = re.search(r"(?:Invoice\s*(?:No|Number|#)|Inv\s*(?:No|Number|#))\s*[:\s\.-]+\s*([A-Z0-9\/\-_]+)", l, re.I)
            if m:
                val = m.group(1).strip()
                if len(val) >= 4 and not re.match(r"^(OICE|DATE|DUE|PO|GST|INR|TOTAL)$", val, re.I):
                    return val, 0.98

            # Keyword on line: check next line OR previous line
            if re.match(r"^(?:Invoice\s*(?:No|Number|#)|Inv\s*#?|INVOICENO:?)$", l, re.I):
                # Check next line
                if idx + 1 < len(lines):
                    next_val = lines[idx + 1].strip()
                    if len(next_val) >= 4 and not re.match(r"^(OICE|DATE|DUE|PO|GST|INR|TOTAL)$", next_val, re.I):
                        return next_val, 0.95
                # Check previous line
                if idx - 1 >= 0:
                    prev_val = lines[idx - 1].strip()
                    if len(prev_val) >= 4 and not re.match(r"^(OICE|DATE|DUE|PO|GST|INR|TOTAL)$", prev_val, re.I):
                        return prev_val, 0.95
                    
        return None, 0.0

    def extract_dates(self, lines: List[str]) -> Tuple[Optional[str], Optional[str], float]:
        """Extract invoice date and due date checking adjacent lines (both above and below)."""
        inv_date = None
        due_date = None
        
        for idx, line in enumerate(lines):
            l = line.strip()
            
            # Invoice Date label
            if re.search(r"(?:Invoice\s*Date|Inv\s*Date|INVDATE|^\.?Date:?$)", l, re.I):
                # Same line
                m = self.normalize_date(l)
                if m:
                    inv_date = m
                else:
                    # Check next line
                    if idx + 1 < len(lines):
                        d_next = self.normalize_date(lines[idx + 1])
                        if d_next:
                            inv_date = d_next
                    # Check previous line
                    if not inv_date and idx - 1 >= 0:
                        d_prev = self.normalize_date(lines[idx - 1])
                        if d_prev:
                            inv_date = d_prev
                            
            # Due Date label
            if re.search(r"(?:Due\s*Date|Payment\s*Due|OUEDATE|DUEDATE)", l, re.I):
                m = self.normalize_date(l)
                if m:
                    due_date = m
                else:
                    if idx + 1 < len(lines):
                        d_next = self.normalize_date(lines[idx + 1])
                        if d_next:
                            due_date = d_next
                    if not due_date and idx - 1 >= 0:
                        d_prev = self.normalize_date(lines[idx - 1])
                        if d_prev:
                            due_date = d_prev

        # General date search fallback if inv_date missing
        if not inv_date:
            for line in lines[:15]:
                dt = self.normalize_date(line)
                if dt:
                    inv_date = dt
                    break

        conf = 0.95 if inv_date else 0.0
        return inv_date, due_date, conf

    def extract_po_number(self, lines: List[str]) -> Tuple[Optional[str], float]:
        """Extract PO number, normalizing OCR anomalies like KIS/P0/20.25/32914."""
        for idx, line in enumerate(lines):
            l = line.strip()
            m = re.search(r"(KIS\/[A-Z0-9\.\-_/]+)", l, re.I)
            if m:
                raw_po = m.group(1).upper()
                clean_po = re.sub(r"P0", "PO", raw_po)
                clean_po = re.sub(r"(\d+)\.(\d+)", r"\1\2", clean_po)
                m_std = re.search(r"\b(KIS\/PO\/\d{4}\/\d{4,6})\b", clean_po)
                if m_std:
                    return m_std.group(1), 0.98
                return clean_po, 0.85

            if re.search(r"(?:PO|Your\s*PO|Purchase\s*Order|PONO)[:\s\.-]*", l, re.I):
                # Check next line
                if idx + 1 < len(lines):
                    next_l = lines[idx + 1].strip()
                    if "KIS/" in next_l.upper():
                        clean_po = re.sub(r"P0", "PO", next_l.upper())
                        clean_po = re.sub(r"(\d+)\.(\d+)", r"\1\2", clean_po)
                        m_std = re.search(r"\b(KIS\/PO\/\d{4}\/\d{4,6})\b", clean_po)
                        if m_std:
                            return m_std.group(1), 0.98
                # Check previous line
                if idx - 1 >= 0:
                    prev_l = lines[idx - 1].strip()
                    if "KIS/" in prev_l.upper():
                        clean_po = re.sub(r"P0", "PO", prev_l.upper())
                        clean_po = re.sub(r"(\d+)\.(\d+)", r"\1\2", clean_po)
                        m_std = re.search(r"\b(KIS\/PO\/\d{4}\/\d{4,6})\b", clean_po)
                        if m_std:
                            return m_std.group(1), 0.98

        return None, 0.0

    def extract_bank_last4(self, lines: List[str]) -> Tuple[Optional[str], float]:
        """Extract last 4 digits of bank account even when glued to IFSC."""
        for line in lines:
            # Match 4 digits after Xs or asterisks (e.g. XXXXXXXXX2188IFSC)
            m = re.search(r"[Xx\*\-]{4,16}(\d{4})", line)
            if m:
                return m.group(1), 0.98
            m2 = re.search(r"(?:A\/c\s*No|Account\s*No|A\/C)[:\s\.-]*[Xx\*\-]*(\d{4})\b", line, re.I)
            if m2:
                return m2.group(1), 0.98
        return None, 0.0

    def extract_totals_and_taxes(self, lines: List[str]) -> Dict[str, Any]:
        """Extract Subtotal, CGST, SGST, IGST, and Grand Total across line pairs."""
        results = {
            "subtotal": None,
            "cgst": None,
            "sgst": None,
            "igst": None,
            "grand_total": None,
            "confidence": {}
        }
        
        for idx, line in enumerate(lines):
            l = line.strip()
            
            def get_amt_from_line_or_neighbors():
                vals = re.findall(r"[\d,]+\.\d{2}", l)
                if vals:
                    return self.parse_amount(vals[-1])
                # Check next line
                if idx + 1 < len(lines):
                    next_vals = re.findall(r"[\d,]+\.\d{2}", lines[idx + 1])
                    if next_vals:
                        return self.parse_amount(next_vals[-1])
                # Check previous line
                if idx - 1 >= 0:
                    prev_vals = re.findall(r"[\d,]+\.\d{2}", lines[idx - 1])
                    if prev_vals:
                        return self.parse_amount(prev_vals[-1])
                return None

            # Grand Total
            if re.search(r"(?:Grand\s*Tota[lt1i]|Invoice\s*Tota[lt1i]|Total\s*INR|Net\s*Payable|Total\s*Amount)", l, re.I):
                amt = get_amt_from_line_or_neighbors()
                if amt and amt > 0:
                    results["grand_total"] = amt
                    results["confidence"]["grand_total"] = 0.98
                    
            # Sub Total
            elif re.search(r"(?:Sub\s*Total|Taxable\s*Amount|Total\s*Before\s*Tax)", l, re.I):
                amt = get_amt_from_line_or_neighbors()
                if amt and amt > 0:
                    results["subtotal"] = amt
                    results["confidence"]["subtotal"] = 0.95
                    
            # CGST
            elif re.search(r"CGST", l, re.I):
                amt = get_amt_from_line_or_neighbors()
                if amt is not None:
                    results["cgst"] = amt
                    results["confidence"]["cgst"] = 0.95
                    
            # SGST
            elif re.search(r"SGST", l, re.I):
                amt = get_amt_from_line_or_neighbors()
                if amt is not None:
                    results["sgst"] = amt
                    results["confidence"]["sgst"] = 0.95
                    
            # IGST
            elif re.search(r"IGST", l, re.I):
                amt = get_amt_from_line_or_neighbors()
                if amt is not None:
                    results["igst"] = amt
                    results["confidence"]["igst"] = 0.95

        # Reconciliation: if Subtotal missing but Grand Total and Taxes present
        if results["subtotal"] is None and results["grand_total"] is not None:
            taxes = (results["cgst"] or 0.0) + (results["sgst"] or 0.0) + (results["igst"] or 0.0)
            if taxes > 0:
                results["subtotal"] = round(results["grand_total"] - taxes, 2)
                results["confidence"]["subtotal"] = 0.90
        elif results["grand_total"] is None and results["subtotal"] is not None:
            taxes = (results["cgst"] or 0.0) + (results["sgst"] or 0.0) + (results["igst"] or 0.0)
            results["grand_total"] = round(results["subtotal"] + taxes, 2)
            results["confidence"]["grand_total"] = 0.90

        return results

    def extract_line_items(self, lines: List[str]) -> List[LineItem]:
        """Extract line items from invoice table rows."""
        items: List[LineItem] = []
        i = 0
        while i < len(lines):
            l = lines[i].strip()
            if re.match(r"^\d{4}$", l):
                hsn = l
                desc = lines[i - 1].strip() if i > 0 else "Industrial Item"
                qty = 1.0
                rate = 0.0
                amt = 0.0
                
                if i + 1 < len(lines):
                    q_val = self.parse_amount(lines[i + 1])
                    if q_val:
                        qty = q_val
                if i + 2 < len(lines):
                    r_val = self.parse_amount(lines[i + 2])
                    if r_val:
                        rate = r_val
                if i + 3 < len(lines):
                    a_val = self.parse_amount(lines[i + 3])
                    if a_val:
                        amt = a_val
                        
                if amt > 0:
                    items.append(LineItem(
                        description=desc,
                        hsn=hsn,
                        quantity=qty,
                        unit_price=rate,
                        amount=amt
                    ))
                    i += 4
                    continue
            i += 1
            
        return items

    def extract(self, ocr_result: Dict[str, Any]) -> InvoiceExtraction:
        """Run complete rule extraction from OCR results."""
        lines = [l.strip() for l in ocr_result.get("lines", []) if l.strip()]
        raw_text = self.clean_text(ocr_result.get("raw_text", ""))
        
        confidence = {}
        
        # 1. GSTIN
        gstin, g_conf = self.extract_gstins(lines)
        confidence["vendor_gstin"] = g_conf
        
        # 2. Vendor Name
        vendor_name, v_conf = self.extract_vendor_name(lines, gstin)
        confidence["vendor_name"] = v_conf
        
        # 3. Invoice Number
        inv_no, inv_conf = self.extract_invoice_number(lines)
        confidence["invoice_number"] = inv_conf
        
        # 4. Dates
        inv_date, due_date, dt_conf = self.extract_dates(lines)
        confidence["invoice_date"] = dt_conf
        confidence["due_date"] = 0.90 if due_date else 0.0
        
        # 5. PO Number
        po_no, po_conf = self.extract_po_number(lines)
        confidence["po_number"] = po_conf
        
        # 6. Bank last 4
        bank_last4, b_conf = self.extract_bank_last4(lines)
        confidence["bank_account_last4"] = b_conf
        
        # 7. Totals & Taxes
        totals = self.extract_totals_and_taxes(lines)
        for k, v in totals["confidence"].items():
            confidence[k] = v
            
        # 8. Line items
        line_items = self.extract_line_items(lines)
        if not line_items and totals.get("subtotal"):
            line_items.append(LineItem(
                description="Industrial Supplies & Materials",
                quantity=1.0,
                unit_price=totals["subtotal"] or 0.0,
                amount=totals["subtotal"] or 0.0
            ))
            
        needs_review = []
        for field in ["vendor_name", "vendor_gstin", "invoice_number", "invoice_date", "grand_total"]:
            if confidence.get(field, 0.0) < 0.70:
                needs_review.append(field)
                
        return InvoiceExtraction(
            vendor_name=vendor_name,
            vendor_gstin=gstin,
            invoice_number=inv_no,
            invoice_date=inv_date,
            due_date=due_date,
            po_number=po_no,
            line_items=line_items,
            subtotal=totals.get("subtotal"),
            cgst=totals.get("cgst") or 0.0,
            sgst=totals.get("sgst") or 0.0,
            igst=totals.get("igst") or 0.0,
            grand_total=totals.get("grand_total"),
            bank_account_last4=bank_last4,
            extraction_source="rule",
            confidence_scores=confidence,
            raw_ocr_text=raw_text,
            needs_review_fields=needs_review
        )
