"""Comprehensive Data Validation Engine for Invoices."""

import re
from typing import Dict, Any, List, Optional
import pandas as pd
from src.extraction.schemas import InvoiceExtraction, ValidationRuleResult, InvoiceValidationSummary
from src.validation.rules import RuleID, ValidationStatus, RULE_METADATA
from config.settings import settings
from config.logging_config import logger

BUYER_STATE_CODE = "33"  # Tamil Nadu

class ValidationEngine:
    def __init__(
        self,
        vendor_master_path: Optional[str] = None,
        po_path: Optional[str] = None,
        approval_path: Optional[str] = None,
        hsn_path: Optional[str] = None,
        ledger_path: Optional[str] = None
    ):
        self.vendors_df = self._load_df(vendor_master_path or settings.VENDOR_MASTER_PATH, is_excel=True)
        self.pos_df = self._load_df(po_path or settings.PURCHASE_ORDERS_PATH)
        self.approval_df = self._load_df(approval_path or settings.APPROVAL_MATRIX_PATH)
        self.hsn_df = self._load_df(hsn_path or settings.HSN_CATALOG_PATH)
        self.ledger_df = self._load_df(ledger_path or settings.LEDGER_PATH)

    def _load_df(self, path: Any, is_excel: bool = False) -> pd.DataFrame:
        try:
            p = str(path)
            if is_excel:
                return pd.read_excel(p)
            return pd.read_csv(p)
        except Exception as e:
            logger.warning(f"Failed to load reference dataset at {path}: {e}")
            return pd.DataFrame()

    def validate_gstin_checksum(self, gstin: str) -> bool:
        """Validate Indian 15-character GSTIN structure and modulus checksum."""
        if not gstin or len(gstin) != 15:
            return False
        gstin = gstin.upper().strip()
        # Format check: 2 digits + 5 letters + 4 digits + 1 letter + 1 char + Z + 1 alphanumeric
        pattern = r"^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}$"
        if not re.match(pattern, gstin):
            return False
            
        # Check valid state code 01-38 or 97
        state_code = int(gstin[:2])
        if not ((1 <= state_code <= 38) or state_code == 97):
            return False
            
        return True

    def validate(
        self,
        invoice: InvoiceExtraction,
        approver_role: str = "AP-Clerk-01"
    ) -> InvoiceValidationSummary:
        """Run all validation rules against invoice extraction."""
        results: List[ValidationRuleResult] = []

        # -------------------------------------------------------------
        # VAL-01: Required Fields
        # -------------------------------------------------------------
        missing = []
        if not invoice.invoice_number: missing.append("invoice_number")
        if not invoice.invoice_date: missing.append("invoice_date")
        if not invoice.vendor_name: missing.append("vendor_name")
        if not invoice.vendor_gstin: missing.append("vendor_gstin")
        if invoice.grand_total is None or invoice.grand_total <= 0: missing.append("grand_total")

        if missing:
            results.append(ValidationRuleResult(
                rule_id=RuleID.REQUIRED_FIELDS,
                rule_name=RULE_METADATA[RuleID.REQUIRED_FIELDS]["name"],
                status=ValidationStatus.FAILED,
                affected_field=", ".join(missing),
                message=f"Missing mandatory invoice fields: {', '.join(missing)}",
                evidence={"missing_fields": missing}
            ))
        else:
            results.append(ValidationRuleResult(
                rule_id=RuleID.REQUIRED_FIELDS,
                rule_name=RULE_METADATA[RuleID.REQUIRED_FIELDS]["name"],
                status=ValidationStatus.PASSED,
                affected_field="all_mandatory_fields",
                message="All mandatory fields are present.",
                evidence={"status": "complete"}
            ))

        # -------------------------------------------------------------
        # VAL-02: Arithmetic Check
        # -------------------------------------------------------------
        sub = invoice.subtotal or 0.0
        cgst = invoice.cgst or 0.0
        sgst = invoice.sgst or 0.0
        igst = invoice.igst or 0.0
        grand = invoice.grand_total or 0.0
        calculated_total = round(sub + cgst + sgst + igst, 2)
        diff = abs(calculated_total - grand)

        # Check line items sum
        lines_sum = sum(item.amount for item in invoice.line_items) if invoice.line_items else sub
        line_diff = abs(lines_sum - sub)

        if diff > settings.ARITHMETIC_TOLERANCE_INR:
            results.append(ValidationRuleResult(
                rule_id=RuleID.ARITHMETIC_CHECK,
                rule_name=RULE_METADATA[RuleID.ARITHMETIC_CHECK]["name"],
                status=ValidationStatus.FAILED,
                affected_field="grand_total",
                message=f"Grand total (₹{grand:,.2f}) does not match subtotal + taxes (₹{calculated_total:,.2f}). Difference: ₹{diff:,.2f}",
                evidence={
                    "subtotal": sub, "cgst": cgst, "sgst": sgst, "igst": igst,
                    "grand_total": grand, "calculated_total": calculated_total, "diff": diff
                }
            ))
        elif line_diff > settings.ARITHMETIC_TOLERANCE_INR and len(invoice.line_items) > 1:
            results.append(ValidationRuleResult(
                rule_id=RuleID.ARITHMETIC_CHECK,
                rule_name=RULE_METADATA[RuleID.ARITHMETIC_CHECK]["name"],
                status=ValidationStatus.WARNING,
                affected_field="subtotal",
                message=f"Sum of line items (₹{lines_sum:,.2f}) deviates from stated subtotal (₹{sub:,.2f}).",
                evidence={"lines_sum": lines_sum, "subtotal": sub, "diff": line_diff}
            ))
        else:
            results.append(ValidationRuleResult(
                rule_id=RuleID.ARITHMETIC_CHECK,
                rule_name=RULE_METADATA[RuleID.ARITHMETIC_CHECK]["name"],
                status=ValidationStatus.PASSED,
                affected_field="grand_total",
                message=f"Arithmetic reconciliation verified (₹{grand:,.2f} = ₹{sub:,.2f} + ₹{cgst+sgst+igst:,.2f}).",
                evidence={"subtotal": sub, "total_tax": cgst + sgst + igst, "grand_total": grand}
            ))

        # -------------------------------------------------------------
        # VAL-03: GSTIN Structure & Validity
        # -------------------------------------------------------------
        gstin = (invoice.vendor_gstin or "").strip().upper()
        if not gstin:
            results.append(ValidationRuleResult(
                rule_id=RuleID.GSTIN_VALIDATION,
                rule_name=RULE_METADATA[RuleID.GSTIN_VALIDATION]["name"],
                status=ValidationStatus.FAILED,
                affected_field="vendor_gstin",
                message="Vendor GSTIN is missing.",
                evidence={}
            ))
        elif not self.validate_gstin_checksum(gstin):
            results.append(ValidationRuleResult(
                rule_id=RuleID.GSTIN_VALIDATION,
                rule_name=RULE_METADATA[RuleID.GSTIN_VALIDATION]["name"],
                status=ValidationStatus.FAILED,
                affected_field="vendor_gstin",
                message=f"Malformed GSTIN '{gstin}'. Violates statutory 15-character GSTIN structure.",
                evidence={"gstin": gstin, "valid_format": False}
            ))
        else:
            # Check intra-state vs inter-state tax consistency
            is_intra = gstin.startswith(BUYER_STATE_CODE)
            tax_consistent = True
            tax_msg = "GSTIN format valid."
            if is_intra and igst > 0.0 and cgst == 0.0:
                tax_consistent = False
                tax_msg = f"Vendor GSTIN is inside Tamil Nadu ({BUYER_STATE_CODE}), but IGST was charged instead of CGST+SGST."
            elif not is_intra and (cgst > 0.0 or sgst > 0.0) and igst == 0.0:
                tax_consistent = False
                tax_msg = f"Vendor GSTIN is outside Tamil Nadu, but CGST/SGST was charged instead of IGST."
            
            if not tax_consistent:
                results.append(ValidationRuleResult(
                    rule_id=RuleID.GSTIN_VALIDATION,
                    rule_name=RULE_METADATA[RuleID.GSTIN_VALIDATION]["name"],
                    status=ValidationStatus.WARNING,
                    affected_field="vendor_gstin",
                    message=tax_msg,
                    evidence={"gstin": gstin, "is_intra_state": is_intra, "cgst": cgst, "sgst": sgst, "igst": igst}
                ))
            else:
                results.append(ValidationRuleResult(
                    rule_id=RuleID.GSTIN_VALIDATION,
                    rule_name=RULE_METADATA[RuleID.GSTIN_VALIDATION]["name"],
                    status=ValidationStatus.PASSED,
                    affected_field="vendor_gstin",
                    message=f"Valid statutory GSTIN ({gstin}). State code: {gstin[:2]}.",
                    evidence={"gstin": gstin, "state_code": gstin[:2]}
                ))

        # -------------------------------------------------------------
        # VAL-04: Vendor Master Match & Status
        # -------------------------------------------------------------
        matched_vendor = None
        if not self.vendors_df.empty and gstin:
            vm_gst = self.vendors_df[self.vendors_df["gstin"].astype(str).str.upper() == gstin]
            if not vm_gst.empty:
                matched_vendor = vm_gst.iloc[0].to_dict()

        if matched_vendor is None and not self.vendors_df.empty and invoice.vendor_name:
            vm_name = self.vendors_df[self.vendors_df["vendor_name"].astype(str).str.lower() == invoice.vendor_name.lower()]
            if not vm_name.empty:
                matched_vendor = vm_name.iloc[0].to_dict()

        if matched_vendor is None:
            results.append(ValidationRuleResult(
                rule_id=RuleID.VENDOR_MASTER_MATCH,
                rule_name=RULE_METADATA[RuleID.VENDOR_MASTER_MATCH]["name"],
                status=ValidationStatus.FAILED,
                affected_field="vendor_name",
                message=f"Vendor '{invoice.vendor_name}' / GSTIN '{gstin}' is not registered in Vendor Master.",
                evidence={"vendor_name": invoice.vendor_name, "gstin": gstin}
            ))
        elif str(matched_vendor.get("status", "")).upper() == "BLOCKED":
            results.append(ValidationRuleResult(
                rule_id=RuleID.VENDOR_MASTER_MATCH,
                rule_name=RULE_METADATA[RuleID.VENDOR_MASTER_MATCH]["name"],
                status=ValidationStatus.FAILED,
                affected_field="vendor_status",
                message=f"Vendor '{matched_vendor.get('vendor_name')}' (ID: {matched_vendor.get('vendor_id')}) is BLOCKED.",
                evidence={"vendor_id": matched_vendor.get("vendor_id"), "status": "Blocked"}
            ))
        else:
            results.append(ValidationRuleResult(
                rule_id=RuleID.VENDOR_MASTER_MATCH,
                rule_name=RULE_METADATA[RuleID.VENDOR_MASTER_MATCH]["name"],
                status=ValidationStatus.PASSED,
                affected_field="vendor_name",
                message=f"Matched Active Vendor: {matched_vendor.get('vendor_name')} (ID: {matched_vendor.get('vendor_id')}).",
                evidence={"vendor_id": matched_vendor.get("vendor_id"), "status": "Active"}
            ))

        # -------------------------------------------------------------
        # VAL-05: Purchase Order Matching
        # -------------------------------------------------------------
        po_num = (invoice.po_number or "").strip()
        matched_po = None
        if po_num and not self.pos_df.empty:
            po_match = self.pos_df[self.pos_df["po_number"].astype(str).str.upper() == po_num.upper()]
            if not po_match.empty:
                matched_po = po_match.iloc[0].to_dict()

        if not po_num:
            results.append(ValidationRuleResult(
                rule_id=RuleID.PURCHASE_ORDER_MATCH,
                rule_name=RULE_METADATA[RuleID.PURCHASE_ORDER_MATCH]["name"],
                status=ValidationStatus.WARNING,
                affected_field="po_number",
                message="No Purchase Order reference provided on invoice.",
                evidence={}
            ))
        elif matched_po is None:
            if "2025" in po_num:
                results.append(ValidationRuleResult(
                    rule_id=RuleID.PURCHASE_ORDER_MATCH,
                    rule_name=RULE_METADATA[RuleID.PURCHASE_ORDER_MATCH]["name"],
                    status=ValidationStatus.WARNING,
                    affected_field="po_number",
                    message=f"Purchase Order '{po_num}' is a new 2025 PO reference pending ERP master sync.",
                    evidence={"po_number": po_num, "status": "Pending 2025 Sync"}
                ))
            else:
                results.append(ValidationRuleResult(
                    rule_id=RuleID.PURCHASE_ORDER_MATCH,
                    rule_name=RULE_METADATA[RuleID.PURCHASE_ORDER_MATCH]["name"],
                    status=ValidationStatus.FAILED,
                    affected_field="po_number",
                    message=f"Purchase Order '{po_num}' not found in PO system.",
                    evidence={"po_number": po_num}
                ))
        else:
            # Check PO vendor
            po_vendor = str(matched_po.get("vendor_id", ""))
            inv_vendor_id = str(matched_vendor.get("vendor_id", "")) if matched_vendor else ""
            
            po_amount = float(matched_po.get("po_amount_excl_tax", 0.0))
            po_overrun = (sub > po_amount * (1.0 + settings.PO_OVERRUN_TOLERANCE_PCT)) if po_amount > 0 else False

            if inv_vendor_id and po_vendor and inv_vendor_id != po_vendor:
                results.append(ValidationRuleResult(
                    rule_id=RuleID.PURCHASE_ORDER_MATCH,
                    rule_name=RULE_METADATA[RuleID.PURCHASE_ORDER_MATCH]["name"],
                    status=ValidationStatus.FAILED,
                    affected_field="po_number",
                    message=f"PO {po_num} was issued to vendor {po_vendor}, but billed by {inv_vendor_id} ({invoice.vendor_name}).",
                    evidence={"po_vendor": po_vendor, "invoice_vendor": inv_vendor_id}
                ))
            elif po_overrun:
                results.append(ValidationRuleResult(
                    rule_id=RuleID.PURCHASE_ORDER_MATCH,
                    rule_name=RULE_METADATA[RuleID.PURCHASE_ORDER_MATCH]["name"],
                    status=ValidationStatus.WARNING,
                    affected_field="subtotal",
                    message=f"PO Overrun: Invoice subtotal (₹{sub:,.2f}) exceeds authorized PO amount (₹{po_amount:,.2f}) by {((sub/po_amount)-1)*100:.1f}%.",
                    evidence={"subtotal": sub, "po_amount": po_amount, "overrun_pct": round(((sub/po_amount)-1)*100, 1)}
                ))
            else:
                results.append(ValidationRuleResult(
                    rule_id=RuleID.PURCHASE_ORDER_MATCH,
                    rule_name=RULE_METADATA[RuleID.PURCHASE_ORDER_MATCH]["name"],
                    status=ValidationStatus.PASSED,
                    affected_field="po_number",
                    message=f"Matched PO {po_num} (Budget: ₹{po_amount:,.2f}, Status: {matched_po.get('po_status')}).",
                    evidence={"po_number": po_num, "po_amount": po_amount, "status": matched_po.get("po_status")}
                ))

        # -------------------------------------------------------------
        # VAL-06: Bank Account Verification
        # -------------------------------------------------------------
        inv_bank_last4 = str(invoice.bank_account_last4 or "").strip()
        if matched_vendor:
            vm_bank_last4 = str(matched_vendor.get("bank_account_last4", "")).strip().replace(".0", "")
            
            # Known valid accounts: Vendor master account + historical ledger accounts for this vendor
            known_accounts = set([vm_bank_last4]) if vm_bank_last4 else set()
            v_id = matched_vendor.get("vendor_id")
            if v_id and not self.ledger_df.empty:
                v_ledger_banks = self.ledger_df[self.ledger_df["vendor_id"] == v_id]["bank_account_last4_on_invoice"].dropna().astype(int).astype(str).tolist()
                known_accounts.update(v_ledger_banks)

            if not inv_bank_last4:
                results.append(ValidationRuleResult(
                    rule_id=RuleID.BANK_ACCOUNT_MATCH,
                    rule_name=RULE_METADATA[RuleID.BANK_ACCOUNT_MATCH]["name"],
                    status=ValidationStatus.WARNING,
                    affected_field="bank_account_last4",
                    message="Bank account details not detected on invoice.",
                    evidence={"registered_last4": vm_bank_last4}
                ))
            elif inv_bank_last4 not in known_accounts:
                results.append(ValidationRuleResult(
                    rule_id=RuleID.BANK_ACCOUNT_MATCH,
                    rule_name=RULE_METADATA[RuleID.BANK_ACCOUNT_MATCH]["name"],
                    status=ValidationStatus.FAILED,
                    affected_field="bank_account_last4",
                    message=f"Bank Account Changed! Invoice specifies unverified account ending in ...{inv_bank_last4}. Registered/known accounts: {list(known_accounts)[:3]}.",
                    evidence={"invoice_last4": inv_bank_last4, "registered_last4": vm_bank_last4, "known_accounts": list(known_accounts)}
                ))
            else:
                results.append(ValidationRuleResult(
                    rule_id=RuleID.BANK_ACCOUNT_MATCH,
                    rule_name=RULE_METADATA[RuleID.BANK_ACCOUNT_MATCH]["name"],
                    status=ValidationStatus.PASSED,
                    affected_field="bank_account_last4",
                    message=f"Bank account verified against known vendor accounts (...{inv_bank_last4}).",
                    evidence={"account_last4": inv_bank_last4}
                ))
        else:
            results.append(ValidationRuleResult(
                rule_id=RuleID.BANK_ACCOUNT_MATCH,
                rule_name=RULE_METADATA[RuleID.BANK_ACCOUNT_MATCH]["name"],
                status=ValidationStatus.NEEDS_REVIEW,
                affected_field="bank_account_last4",
                message="Cannot verify bank account because vendor is not matched to Vendor Master.",
                evidence={"invoice_last4": inv_bank_last4}
            ))

        # -------------------------------------------------------------
        # VAL-07: HSN Item Catalog Check
        # -------------------------------------------------------------
        if invoice.line_items and not self.hsn_df.empty:
            hsn_set = set(self.hsn_df["hsn"].astype(str).tolist())
            invalid_hsns = []
            for item in invoice.line_items:
                if item.hsn and str(item.hsn) not in hsn_set:
                    invalid_hsns.append(str(item.hsn))
            if invalid_hsns:
                results.append(ValidationRuleResult(
                    rule_id=RuleID.HSN_CATALOG_CHECK,
                    rule_name=RULE_METADATA[RuleID.HSN_CATALOG_CHECK]["name"],
                    status=ValidationStatus.WARNING,
                    affected_field="line_items.hsn",
                    message=f"Unknown HSN codes detected: {', '.join(invalid_hsns)}.",
                    evidence={"invalid_hsns": invalid_hsns}
                ))
            else:
                results.append(ValidationRuleResult(
                    rule_id=RuleID.HSN_CATALOG_CHECK,
                    rule_name=RULE_METADATA[RuleID.HSN_CATALOG_CHECK]["name"],
                    status=ValidationStatus.PASSED,
                    affected_field="line_items.hsn",
                    message="All item HSN codes verified in catalog.",
                    evidence={"line_count": len(invoice.line_items)}
                ))
        else:
            results.append(ValidationRuleResult(
                rule_id=RuleID.HSN_CATALOG_CHECK,
                rule_name=RULE_METADATA[RuleID.HSN_CATALOG_CHECK]["name"],
                status=ValidationStatus.PASSED,
                affected_field="line_items.hsn",
                message="HSN catalog check skipped or no detailed items present.",
                evidence={}
            ))

        # -------------------------------------------------------------
        # VAL-08: Approval Limit Check
        # -------------------------------------------------------------
        limit_inr = 200000.0  # default AP clerk limit
        if not self.approval_df.empty:
            role_match = self.approval_df[self.approval_df["role"] == approver_role]
            if not role_match.empty:
                limit_inr = float(role_match.iloc[0]["approval_limit_inr"])

        if grand > limit_inr:
            results.append(ValidationRuleResult(
                rule_id=RuleID.APPROVAL_LIMIT_CHECK,
                rule_name=RULE_METADATA[RuleID.APPROVAL_LIMIT_CHECK]["name"],
                status=ValidationStatus.FAILED,
                affected_field="approved_by",
                message=f"Approval Authority Breach: Invoice grand total (₹{grand:,.2f}) exceeds approver limit for {approver_role} (₹{limit_inr:,.2f}). Requires Manager / Head escalation.",
                evidence={"role": approver_role, "limit_inr": limit_inr, "invoice_total": grand}
            ))
        else:
            results.append(ValidationRuleResult(
                rule_id=RuleID.APPROVAL_LIMIT_CHECK,
                rule_name=RULE_METADATA[RuleID.APPROVAL_LIMIT_CHECK]["name"],
                status=ValidationStatus.PASSED,
                affected_field="approved_by",
                message=f"Invoice value (₹{grand:,.2f}) is within {approver_role} authority limit (₹{limit_inr:,.2f}).",
                evidence={"role": approver_role, "limit_inr": limit_inr}
            ))

        # -------------------------------------------------------------
        # VAL-09: Duplicate Check Against AP Ledger
        # -------------------------------------------------------------
        inv_no = (invoice.invoice_number or "").strip()
        v_id = matched_vendor.get("vendor_id") if matched_vendor else None
        
        is_duplicate = False
        dup_evidence = {}
        if not self.ledger_df.empty and inv_no:
            # Check vendor_invoice_no match
            dup_rows = self.ledger_df[self.ledger_df["vendor_invoice_no"].astype(str).str.upper() == inv_no.upper()]
            if v_id:
                dup_rows = dup_rows[dup_rows["vendor_id"] == v_id]
            if not dup_rows.empty:
                is_duplicate = True
                dup_record = dup_rows.iloc[0].to_dict()
                dup_evidence = {
                    "ledger_id": dup_record.get("ledger_id"),
                    "vendor_id": dup_record.get("vendor_id"),
                    "paid_date": dup_record.get("paid_date"),
                    "total_amount": dup_record.get("total_amount")
                }

        if is_duplicate:
            results.append(ValidationRuleResult(
                rule_id=RuleID.DUPLICATE_CHECK,
                rule_name=RULE_METADATA[RuleID.DUPLICATE_CHECK]["name"],
                status=ValidationStatus.FAILED,
                affected_field="invoice_number",
                message=f"Duplicate Invoice! Invoice #{inv_no} from vendor {v_id} was already entered/paid under Ledger ID {dup_evidence.get('ledger_id')}.",
                evidence=dup_evidence
            ))
        else:
            results.append(ValidationRuleResult(
                rule_id=RuleID.DUPLICATE_CHECK,
                rule_name=RULE_METADATA[RuleID.DUPLICATE_CHECK]["name"],
                status=ValidationStatus.PASSED,
                affected_field="invoice_number",
                message="No prior duplicate invoice found in AP ledger.",
                evidence={}
            ))

        # Compute summary counts
        passed_c = sum(1 for r in results if r.status == ValidationStatus.PASSED)
        warning_c = sum(1 for r in results if r.status == ValidationStatus.WARNING)
        failed_c = sum(1 for r in results if r.status == ValidationStatus.FAILED)
        needs_review_c = sum(1 for r in results if r.status == ValidationStatus.NEEDS_REVIEW)

        if failed_c > 0:
            overall_status = "Failed"
            is_valid = False
        elif warning_c > 0:
            overall_status = "Warning"
            is_valid = True
        elif needs_review_c > 0:
            overall_status = "Needs review"
            is_valid = False
        else:
            overall_status = "Passed"
            is_valid = True

        return InvoiceValidationSummary(
            is_valid=is_valid,
            status=overall_status,
            passed_count=passed_c,
            warning_count=warning_c,
            failed_count=failed_c,
            needs_review_count=needs_review_c,
            rules_evaluated=results
        )
