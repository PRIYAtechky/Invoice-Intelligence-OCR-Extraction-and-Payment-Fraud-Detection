"""
Payment Fraud Detection Patterns
Implements 7 evidence-based fraud detection rules for Accounts Payable audits.
"""
from typing import Dict, Any, List, Optional
from difflib import SequenceMatcher
from src.extraction.schemas import ExtractedInvoice, FraudAlertItem, FraudSeverity


def check_exact_duplicate(invoice: ExtractedInvoice, historical_invoices: List[Dict[str, Any]]) -> Optional[FraudAlertItem]:
    """Pattern 1: Exact duplicate invoice number for the same vendor."""
    inv_no = (invoice.invoice_number or "").strip().lower()
    v_name = (invoice.vendor_name or "").strip().lower()
    
    if not inv_no or not v_name:
        return None

    for prev in historical_invoices:
        prev_no = str(prev.get("vendor_invoice_no") or prev.get("invoice_number", "")).strip().lower()
        prev_vendor = str(prev.get("vendor_name") or prev.get("vendor_id", "")).strip().lower()
        
        if prev_no == inv_no and (prev_vendor in v_name or v_name in prev_vendor):
            return FraudAlertItem(
                pattern_id="FRAUD_001",
                pattern_name="Exact Duplicate Invoice",
                risk_score=95.0,
                severity=FraudSeverity.CRITICAL,
                evidence=f"Exact match on invoice number '{invoice.invoice_number}' with prior entry '{prev.get('ledger_id') or prev.get('id', 'N/A')}' for vendor '{invoice.vendor_name}'.",
                related_entity=str(prev.get("ledger_id") or prev.get("id")),
                amount_at_risk=invoice.grand_total,
                recommended_action="Block payment immediately. Audit prior payment history for duplicate disbursement."
            )
    return None


def check_fuzzy_duplicate(invoice: ExtractedInvoice, historical_invoices: List[Dict[str, Any]]) -> Optional[FraudAlertItem]:
    """Pattern 2: Fuzzy duplicate invoice number (transposed digits or character variations) with identical amount."""
    inv_no = (invoice.invoice_number or "").strip().lower()
    v_name = (invoice.vendor_name or "").strip().lower()
    
    if not inv_no or not v_name or invoice.grand_total <= 0:
        return None

    for prev in historical_invoices:
        prev_no = str(prev.get("vendor_invoice_no") or prev.get("invoice_number", "")).strip().lower()
        prev_vendor = str(prev.get("vendor_name") or prev.get("vendor_id", "")).strip().lower()
        prev_amt = float(str(prev.get("total_amount") or prev.get("grand_total", 0)).replace(",", ""))

        if (prev_vendor in v_name or v_name in prev_vendor) and prev_no != inv_no:
            # Check string similarity
            sim = SequenceMatcher(None, inv_no, prev_no).ratio()
            amt_match = abs(invoice.grand_total - prev_amt) <= 1.0

            if sim >= 0.85 and amt_match:
                return FraudAlertItem(
                    pattern_id="FRAUD_002",
                    pattern_name="Fuzzy Duplicate Invoice",
                    risk_score=85.0,
                    severity=FraudSeverity.HIGH,
                    evidence=f"Invoice number '{invoice.invoice_number}' is highly similar (sim: {sim:.0%}) to prior invoice '{prev_no}' with identical amount (₹{invoice.grand_total:,.2f}).",
                    related_entity=str(prev.get("ledger_id") or prev.get("id")),
                    amount_at_risk=invoice.grand_total,
                    recommended_action="Hold payment. Request clerk clarification on potential re-billing or typographical submission."
                )
    return None


def check_bank_tampering(invoice: ExtractedInvoice, vendor_master_rec: Optional[Dict[str, Any]]) -> Optional[FraudAlertItem]:
    """Pattern 3: Bank account diversion / tampering."""
    if not vendor_master_rec or not invoice.bank_account_last4:
        return None

    reg_bank = str(vendor_master_rec.get("bank_account_last4", "")).strip()
    inv_bank = str(invoice.bank_account_last4).strip()

    if reg_bank and inv_bank and reg_bank != inv_bank:
        return FraudAlertItem(
            pattern_id="FRAUD_003",
            pattern_name="Bank Account Diversion / Tampering",
            risk_score=95.0,
            severity=FraudSeverity.CRITICAL,
            evidence=f"Invoice directs disbursement to unverified bank account ending in '{inv_bank}', whereas vendor master mandates '...{reg_bank}'.",
            related_entity=vendor_master_rec.get("vendor_name"),
            amount_at_risk=invoice.grand_total,
            recommended_action="HALT DISBURSEMENT. Contact vendor via verified phone records before releasing funds."
        )
    return None


def check_po_overrun(invoice: ExtractedInvoice, po_rec: Optional[Dict[str, Any]]) -> Optional[FraudAlertItem]:
    """Pattern 4: Purchase order amount inflation."""
    if not po_rec:
        return None

    po_amt = float(po_rec.get("po_amount_excl_tax", 0.0))
    if invoice.subtotal > (po_amt * 1.05):
        overrun = invoice.subtotal - po_amt
        return FraudAlertItem(
            pattern_id="FRAUD_004",
            pattern_name="Purchase Order Cost Overrun",
            risk_score=75.0,
            severity=FraudSeverity.MEDIUM,
            evidence=f"Invoice subtotal (₹{invoice.subtotal:,.2f}) exceeds authorized PO '{po_rec.get('po_number')}' (₹{po_amt:,.2f}) by ₹{overrun:,.2f} (+{(overrun/po_amt):.1%}).",
            related_entity=po_rec.get("po_number"),
            amount_at_risk=overrun,
            recommended_action="Require procurement manager sign-off and PO amendment before processing."
        )
    return None


def check_blocked_vendor(invoice: ExtractedInvoice, vendor_master_rec: Optional[Dict[str, Any]]) -> Optional[FraudAlertItem]:
    """Pattern 5: Transaction with blocked/blacklisted vendor."""
    if not vendor_master_rec:
        return None

    status = str(vendor_master_rec.get("status", "Active")).strip()
    if status.lower() == "blocked":
        return FraudAlertItem(
            pattern_id="FRAUD_005",
            pattern_name="Blocked / Inactive Vendor Payment",
            risk_score=90.0,
            severity=FraudSeverity.HIGH,
            evidence=f"Vendor '{vendor_master_rec.get('vendor_name')}' ({vendor_master_rec.get('vendor_id')}) is flagged as BLOCKED in ERP master.",
            related_entity=vendor_master_rec.get("vendor_id"),
            amount_at_risk=invoice.grand_total,
            recommended_action="Reject invoice. Do not disburse to blocked entities without CFO approval."
        )
    return None


def check_approval_limit_split(invoice: ExtractedInvoice, approver_role: str, approval_limit: float) -> Optional[FraudAlertItem]:
    """Pattern 6: Approval limit circumvention (invoices hovering just beneath authorization limit)."""
    # E.g. invoice between 95% and 99.9% of clerk limit (190,000 to 199,999)
    threshold_low = approval_limit * 0.95
    if threshold_low <= invoice.grand_total < approval_limit:
        return FraudAlertItem(
            pattern_id="FRAUD_006",
            pattern_name="Threshold Smurfing / Split Invoice",
            risk_score=60.0,
            severity=FraudSeverity.MEDIUM,
            evidence=f"Invoice value ₹{invoice.grand_total:,.2f} sits just below the ₹{approval_limit:,.2f} threshold for role '{approver_role}' (smurfing pattern).",
            related_entity=approver_role,
            amount_at_risk=invoice.grand_total,
            recommended_action="Examine associated purchase orders for fragmented billing intentionally evading manager review."
        )
    return None
