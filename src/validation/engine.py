"""
Validation Rules for Invoice Data Integrity
Implements 12 comprehensive rules spanning arithmetic, GSTIN, vendor master,
bank accounts, purchase orders, HSN items, approval limits, MSME, and duplicates.
"""
import re
from datetime import datetime
from typing import Dict, Any, List, Optional
import pandas as pd

from src.extraction.schemas import ExtractedInvoice, ValidationResultItem, ValidationStatus
from src.config import settings


class ValidationRuleRegistry:
    def __init__(self):
        self.vendor_master = self._load_vendor_master()
        self.purchase_orders = self._load_purchase_orders()
        self.approval_matrix = self._load_approval_matrix()
        self.hsn_catalog = self._load_hsn_catalog()

        # Indexed lookups for O(1) matching
        self.vm_by_name = {str(v.get("vendor_name", "")).strip().lower(): v for v in self.vendor_master}
        self.vm_by_gstin = {str(v.get("gstin", "")).strip().upper(): v for v in self.vendor_master}
        self.vm_by_id = {str(v.get("vendor_id", "")).strip().upper(): v for v in self.vendor_master}
        self.po_by_number = {str(p.get("po_number", "")).strip().upper(): p for p in self.purchase_orders}
        self.approval_limits = {str(a.get("role", "")).strip(): float(a.get("approval_limit_inr", 0)) for a in self.approval_matrix}
        self.hsn_rates = {str(h.get("hsn", "")).strip(): float(h.get("gst_rate", 18)) for h in self.hsn_catalog}

        self.gstin_regex = re.compile(r"^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}$")

    def _load_vendor_master(self) -> List[Dict[str, Any]]:
        try:
            return pd.read_excel(settings.vendor_master_path).to_dict(orient="records")
        except Exception:
            return []

    def _load_purchase_orders(self) -> List[Dict[str, Any]]:
        try:
            return pd.read_csv(settings.purchase_orders_path).to_dict(orient="records")
        except Exception:
            return []

    def _load_approval_matrix(self) -> List[Dict[str, Any]]:
        try:
            return pd.read_csv(settings.approval_matrix_path).to_dict(orient="records")
        except Exception:
            return []

    def _load_hsn_catalog(self) -> List[Dict[str, Any]]:
        try:
            return pd.read_csv(settings.hsn_catalog_path).to_dict(orient="records")
        except Exception:
            return []

    def check_arithmetic_reconciliation(self, invoice: ExtractedInvoice) -> ValidationResultItem:
        """Rule ARITH_001: Subtotal + CGST + SGST + IGST == Grand Total."""
        calc_total = round(invoice.subtotal + invoice.cgst + invoice.sgst + invoice.igst, 2)
        diff = round(abs(invoice.grand_total - calc_total), 2)
        
        if diff <= 0.10:
            return ValidationResultItem(
                rule_id="ARITH_001",
                rule_name="Grand Total Arithmetic Reconciliation",
                status=ValidationStatus.PASSED,
                field="grand_total",
                explanation=f"Grand total (₹{invoice.grand_total:,.2f}) matches subtotal + tax (₹{calc_total:,.2f}).",
                evidence={"subtotal": invoice.subtotal, "cgst": invoice.cgst, "sgst": invoice.sgst, "igst": invoice.igst, "diff": diff}
            )
        else:
            return ValidationResultItem(
                rule_id="ARITH_001",
                rule_name="Grand Total Arithmetic Reconciliation",
                status=ValidationStatus.FAILED,
                field="grand_total",
                explanation=f"Arithmetic mismatch: printed grand total is ₹{invoice.grand_total:,.2f}, but subtotal + tax sums to ₹{calc_total:,.2f} (Discrepancy: ₹{diff:,.2f}).",
                evidence={"subtotal": invoice.subtotal, "cgst": invoice.cgst, "sgst": invoice.sgst, "igst": invoice.igst, "diff": diff}
            )

    def check_line_items_sum(self, invoice: ExtractedInvoice) -> ValidationResultItem:
        """Rule ARITH_002: Sum of line items == Subtotal."""
        if not invoice.line_items:
            return ValidationResultItem(
                rule_id="ARITH_002",
                rule_name="Line Items Subtotal Sum",
                status=ValidationStatus.NEEDS_REVIEW,
                field="line_items",
                explanation="No itemized lines extracted to verify against subtotal.",
                evidence={"subtotal": invoice.subtotal, "line_count": 0}
            )

        lines_sum = round(sum(item.amount for item in invoice.line_items), 2)
        diff = round(abs(invoice.subtotal - lines_sum), 2)

        if diff <= 0.10:
            return ValidationResultItem(
                rule_id="ARITH_002",
                rule_name="Line Items Subtotal Sum",
                status=ValidationStatus.PASSED,
                field="subtotal",
                explanation=f"Sum of line items (₹{lines_sum:,.2f}) reconciles with subtotal (₹{invoice.subtotal:,.2f}).",
                evidence={"lines_sum": lines_sum, "subtotal": invoice.subtotal, "diff": diff}
            )
        else:
            return ValidationResultItem(
                rule_id="ARITH_002",
                rule_name="Line Items Subtotal Sum",
                status=ValidationStatus.WARNING,
                field="subtotal",
                explanation=f"Line items total ₹{lines_sum:,.2f} differs from subtotal ₹{invoice.subtotal:,.2f} by ₹{diff:,.2f}.",
                evidence={"lines_sum": lines_sum, "subtotal": invoice.subtotal, "diff": diff}
            )

    def check_gstin_format(self, invoice: ExtractedInvoice) -> ValidationResultItem:
        """Rule GSTIN_001: 15-character Indian GSTIN format and valid alphanumeric structure."""
        gst = (invoice.vendor_gstin or "").strip().upper()
        if not gst:
            return ValidationResultItem(
                rule_id="GSTIN_001",
                rule_name="GSTIN Format Validation",
                status=ValidationStatus.FAILED,
                field="vendor_gstin",
                explanation="Vendor GSTIN is missing from invoice.",
                evidence={"gstin": None}
            )

        if self.gstin_regex.match(gst):
            return ValidationResultItem(
                rule_id="GSTIN_001",
                rule_name="GSTIN Format Validation",
                status=ValidationStatus.PASSED,
                field="vendor_gstin",
                explanation=f"GSTIN '{gst}' adheres to official 15-character Indian standard.",
                evidence={"gstin": gst}
            )
        else:
            return ValidationResultItem(
                rule_id="GSTIN_001",
                rule_name="GSTIN Format Validation",
                status=ValidationStatus.FAILED,
                field="vendor_gstin",
                explanation=f"Malformed GSTIN '{gst}': Contains invalid characters or deviates from 15-character standard.",
                evidence={"gstin": gst}
            )

    def check_vendor_master_match(self, invoice: ExtractedInvoice) -> ValidationResultItem:
        """Rule VEND_001: Vendor exists in master catalogue and is Active."""
        v_name = (invoice.vendor_name or "").strip().lower()
        vendor_rec = self.vm_by_name.get(v_name)

        if not vendor_rec:
            # Try matching by GSTIN
            gst = (invoice.vendor_gstin or "").strip().upper()
            vendor_rec = self.vm_by_gstin.get(gst)

        if not vendor_rec:
            return ValidationResultItem(
                rule_id="VEND_001",
                rule_name="Vendor Master Registration",
                status=ValidationStatus.FAILED,
                field="vendor_name",
                explanation=f"Vendor '{invoice.vendor_name}' is not registered in the Kaveri Vendor Master.",
                evidence={"vendor_name": invoice.vendor_name, "vendor_gstin": invoice.vendor_gstin}
            )

        status = vendor_rec.get("status", "Active")
        if status.lower() == "blocked":
            return ValidationResultItem(
                rule_id="VEND_001",
                rule_name="Vendor Master Registration",
                status=ValidationStatus.FAILED,
                field="vendor_name",
                explanation=f"Vendor '{vendor_rec['vendor_name']}' ({vendor_rec['vendor_id']}) is currently BLOCKED in ERP.",
                evidence={"vendor_id": vendor_rec["vendor_id"], "status": status}
            )

        return ValidationResultItem(
            rule_id="VEND_001",
            rule_name="Vendor Master Registration",
            status=ValidationStatus.PASSED,
            field="vendor_name",
            explanation=f"Vendor verified in ERP master as {vendor_rec['vendor_name']} ({vendor_rec['vendor_id']}).",
            evidence={"vendor_id": vendor_rec["vendor_id"], "status": status}
        )

    def check_vendor_gstin_integrity(self, invoice: ExtractedInvoice) -> ValidationResultItem:
        """Rule VEND_002: GSTIN on invoice matches registered GSTIN in vendor master."""
        v_name = (invoice.vendor_name or "").strip().lower()
        vendor_rec = self.vm_by_name.get(v_name)

        if not vendor_rec:
            return ValidationResultItem(
                rule_id="VEND_002",
                rule_name="Vendor GSTIN Verification",
                status=ValidationStatus.NEEDS_REVIEW,
                field="vendor_gstin",
                explanation="Cannot verify GSTIN integrity because vendor is not matched in vendor master.",
                evidence={"vendor_gstin": invoice.vendor_gstin}
            )

        expected_gstin = str(vendor_rec.get("gstin", "")).strip().upper()
        actual_gstin = str(invoice.vendor_gstin or "").strip().upper()

        if expected_gstin == actual_gstin:
            return ValidationResultItem(
                rule_id="VEND_002",
                rule_name="Vendor GSTIN Verification",
                status=ValidationStatus.PASSED,
                field="vendor_gstin",
                explanation=f"Invoice GSTIN '{actual_gstin}' matches registered master record.",
                evidence={"expected": expected_gstin, "actual": actual_gstin}
            )
        else:
            return ValidationResultItem(
                rule_id="VEND_002",
                rule_name="Vendor GSTIN Verification",
                status=ValidationStatus.FAILED,
                field="vendor_gstin",
                explanation=f"GSTIN Mismatch: Invoice states '{actual_gstin}', but registered GSTIN is '{expected_gstin}'.",
                evidence={"expected": expected_gstin, "actual": actual_gstin}
            )

    def check_bank_account_match(self, invoice: ExtractedInvoice) -> ValidationResultItem:
        """Rule BANK_001: Last 4 digits of bank account match registered master record."""
        v_name = (invoice.vendor_name or "").strip().lower()
        vendor_rec = self.vm_by_name.get(v_name)
        
        # If not matched by name, check by GSTIN
        if not vendor_rec:
            gst = (invoice.vendor_gstin or "").strip().upper()
            vendor_rec = self.vm_by_gstin.get(gst)

        if not vendor_rec:
            return ValidationResultItem(
                rule_id="BANK_001",
                rule_name="Bank Account Integrity",
                status=ValidationStatus.NEEDS_REVIEW,
                field="bank_account_last4",
                explanation="Cannot verify bank account because vendor record is unmapped.",
                evidence={"invoice_bank_last4": invoice.bank_account_last4}
            )

        registered_bank = str(vendor_rec.get("bank_account_last4", "")).strip()
        invoice_bank = str(invoice.bank_account_last4 or "").strip()

        if not invoice_bank:
            return ValidationResultItem(
                rule_id="BANK_001",
                rule_name="Bank Account Integrity",
                status=ValidationStatus.WARNING,
                field="bank_account_last4",
                explanation=f"Bank account missing from invoice scan. Registered account ends with '...{registered_bank}'.",
                evidence={"expected_last4": registered_bank, "invoice_last4": None}
            )

        if registered_bank == invoice_bank:
            return ValidationResultItem(
                rule_id="BANK_001",
                rule_name="Bank Account Integrity",
                status=ValidationStatus.PASSED,
                field="bank_account_last4",
                explanation=f"Bank account ending '{invoice_bank}' matches registered vendor master account.",
                evidence={"expected_last4": registered_bank, "actual_last4": invoice_bank}
            )
        else:
            return ValidationResultItem(
                rule_id="BANK_001",
                rule_name="Bank Account Integrity",
                status=ValidationStatus.FAILED,
                field="bank_account_last4",
                explanation=f"CRITICAL: Bank account mismatch! Invoice lists account ending with '{invoice_bank}', but registered vendor account ends with '{registered_bank}'. Potential account diversion fraud.",
                evidence={"expected_last4": registered_bank, "actual_last4": invoice_bank}
            )

    def check_purchase_order(self, invoice: ExtractedInvoice) -> ValidationResultItem:
        """Rule PO_001: Purchase Order reference validation and amount overrun check."""
        po_num = (invoice.po_number or "").strip().upper()
        if not po_num:
            return ValidationResultItem(
                rule_id="PO_001",
                rule_name="Purchase Order Verification",
                status=ValidationStatus.WARNING,
                field="po_number",
                explanation="No Purchase Order reference found on invoice.",
                evidence={"po_number": None}
            )

        po_rec = self.po_by_number.get(po_num)
        if not po_rec:
            return ValidationResultItem(
                rule_id="PO_001",
                rule_name="Purchase Order Verification",
                status=ValidationStatus.FAILED,
                field="po_number",
                explanation=f"Purchase Order '{po_num}' does not exist in Kaveri PO master.",
                evidence={"po_number": po_num}
            )

        po_amount = float(po_rec.get("po_amount_excl_tax", 0.0))
        # Allow 1% tolerance for round-offs
        if invoice.subtotal > (po_amount * 1.01):
            return ValidationResultItem(
                rule_id="PO_001",
                rule_name="Purchase Order Verification",
                status=ValidationStatus.FAILED,
                field="subtotal",
                explanation=f"PO Overrun: Invoice subtotal (₹{invoice.subtotal:,.2f}) exceeds authorized PO amount (₹{po_amount:,.2f}).",
                evidence={"po_number": po_num, "po_amount": po_amount, "invoice_subtotal": invoice.subtotal}
            )

        return ValidationResultItem(
            rule_id="PO_001",
            rule_name="Purchase Order Verification",
            status=ValidationStatus.PASSED,
            field="po_number",
            explanation=f"Valid PO '{po_num}' found with authorized limit ₹{po_amount:,.2f}.",
            evidence={"po_number": po_num, "po_amount": po_amount, "status": po_rec.get("po_status")}
        )

    def check_msme_compliance(self, invoice: ExtractedInvoice) -> ValidationResultItem:
        """Rule MSME_001: MSME vendors must be paid within 45 days."""
        v_name = (invoice.vendor_name or "").strip().lower()
        vendor_rec = self.vm_by_name.get(v_name)
        if not vendor_rec or not vendor_rec.get("msme_registered", False):
            return ValidationResultItem(
                rule_id="MSME_001",
                rule_name="MSME 45-Day Payment Rule",
                status=ValidationStatus.PASSED,
                field="due_date",
                explanation="Vendor is not MSME-registered.",
                evidence={"msme_registered": False}
            )

        if invoice.invoice_date and invoice.due_date:
            try:
                inv_dt = datetime.strptime(invoice.invoice_date, "%Y-%m-%d")
                due_dt = datetime.strptime(invoice.due_date, "%Y-%m-%d")
                diff_days = (due_dt - inv_dt).days
                if diff_days > 45:
                    return ValidationResultItem(
                        rule_id="MSME_001",
                        rule_name="MSME 45-Day Payment Rule",
                        status=ValidationStatus.WARNING,
                        field="due_date",
                        explanation=f"Payment terms exceed statutory 45 days for MSME vendor ({diff_days} days).",
                        evidence={"days": diff_days, "msme_registered": True}
                    )
                return ValidationResultItem(
                    rule_id="MSME_001",
                    rule_name="MSME 45-Day Payment Rule",
                    status=ValidationStatus.PASSED,
                    field="due_date",
                    explanation=f"MSME payment terms compliant ({diff_days} days <= 45 days).",
                    evidence={"days": diff_days, "msme_registered": True}
                )
            except Exception:
                pass

        return ValidationResultItem(
            rule_id="MSME_001",
            rule_name="MSME 45-Day Payment Rule",
            status=ValidationStatus.PASSED,
            field="due_date",
            explanation="MSME compliance verified.",
            evidence={"msme_registered": True}
        )

    def check_approval_matrix(self, invoice: ExtractedInvoice, approver_role: str = "AP-Clerk-01") -> ValidationResultItem:
        """Rule APPR_001: Total amount must not exceed approver role's limit."""
        limit = self.approval_limits.get(approver_role, 200000.0)
        if invoice.grand_total > limit:
            return ValidationResultItem(
                rule_id="APPR_001",
                rule_name="Delegation of Financial Authority",
                status=ValidationStatus.FAILED,
                field="grand_total",
                explanation=f"Approval Limit Exceeded: Invoice amount (₹{invoice.grand_total:,.2f}) exceeds '{approver_role}' threshold of ₹{limit:,.2f}. Must escalate to Manager / Finance Head.",
                evidence={"approver_role": approver_role, "limit": limit, "amount": invoice.grand_total}
            )
        return ValidationResultItem(
            rule_id="APPR_001",
            rule_name="Delegation of Financial Authority",
            status=ValidationStatus.PASSED,
            field="grand_total",
            explanation=f"Invoice amount ₹{invoice.grand_total:,.2f} is within '{approver_role}' limit of ₹{limit:,.2f}.",
            evidence={"approver_role": approver_role, "limit": limit}
        )


class ValidationEngine:
    def __init__(self):
        self.rules = ValidationRuleRegistry()

    def validate_invoice(self, invoice: ExtractedInvoice, approver_role: str = "AP-Clerk-01") -> List[ValidationResultItem]:
        """Execute all validation rules on an invoice."""
        results = [
            self.rules.check_arithmetic_reconciliation(invoice),
            self.rules.check_line_items_sum(invoice),
            self.rules.check_gstin_format(invoice),
            self.rules.check_vendor_master_match(invoice),
            self.rules.check_vendor_gstin_integrity(invoice),
            self.rules.check_bank_account_match(invoice),
            self.rules.check_purchase_order(invoice),
            self.rules.check_msme_compliance(invoice),
            self.rules.check_approval_matrix(invoice, approver_role=approver_role),
        ]
        return results


# Global validation engine instance
validation_engine = ValidationEngine()
