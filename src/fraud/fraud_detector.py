"""Unified Evidence-based Payment Fraud Detection Engine."""

import re
from typing import Dict, Any, List, Optional
from difflib import SequenceMatcher
import pandas as pd

from src.extraction.schemas import InvoiceExtraction, FraudAlert, InvoiceFraudSummary
from src.fraud.patterns import FraudPatternID, FraudSeverity, FRAUD_METADATA
from src.fraud.anomaly_model import InvoiceAnomalyModel
from src.fraud.vendor_risk import VendorRiskScorer
from config.settings import settings
from config.logging_config import logger

class PaymentFraudDetector:
    def __init__(
        self,
        vendor_master_path: Optional[str] = None,
        ledger_path: Optional[str] = None,
        po_path: Optional[str] = None,
        approval_path: Optional[str] = None
    ):
        self.ledger_df = self._load_df(ledger_path or settings.LEDGER_PATH)
        self.vm_df = self._load_df(vendor_master_path or settings.VENDOR_MASTER_PATH, is_excel=True)
        self.po_df = self._load_df(po_path or settings.PURCHASE_ORDERS_PATH)
        self.approval_df = self._load_df(approval_path or settings.APPROVAL_MATRIX_PATH)
        
        self.anomaly_model = InvoiceAnomalyModel(ledger_path or settings.LEDGER_PATH)
        self.vendor_risk_scorer = VendorRiskScorer(
            vendor_master_path, ledger_path, po_path, approval_path
        )

    def _load_df(self, path: Any, is_excel: bool = False) -> pd.DataFrame:
        try:
            p = str(path)
            if is_excel:
                return pd.read_excel(p)
            return pd.read_csv(p)
        except Exception as e:
            logger.warning(f"Failed to load dataset at {path}: {e}")
            return pd.DataFrame()

    def detect_fraud(
        self,
        invoice: InvoiceExtraction,
        approver_role: str = "AP-Clerk-01"
    ) -> InvoiceFraudSummary:
        """Run all evidence-based fraud detection checks against an invoice."""
        alerts: List[FraudAlert] = []
        inv_no = (invoice.invoice_number or "").strip()
        grand = float(invoice.grand_total or 0.0)
        sub = float(invoice.subtotal or 0.0)
        total_tax = (invoice.cgst or 0.0) + (invoice.sgst or 0.0) + (invoice.igst or 0.0)

        # Match vendor ID
        v_id = None
        matched_vm = None
        if not self.vm_df.empty:
            if invoice.vendor_gstin:
                gst_m = self.vm_df[self.vm_df["gstin"].astype(str).str.upper() == invoice.vendor_gstin.upper()]
                if not gst_m.empty:
                    matched_vm = gst_m.iloc[0].to_dict()
                    v_id = str(matched_vm.get("vendor_id"))
            if not v_id and invoice.vendor_name:
                name_m = self.vm_df[self.vm_df["vendor_name"].astype(str).str.lower() == invoice.vendor_name.lower()]
                if not name_m.empty:
                    matched_vm = name_m.iloc[0].to_dict()
                    v_id = str(matched_vm.get("vendor_id"))

        # -----------------------------------------------------------------
        # Pattern 1: Exact Duplicate Invoice Check (FRD-01)
        # -----------------------------------------------------------------
        if not self.ledger_df.empty and inv_no:
            matches = self.ledger_df[self.ledger_df["vendor_invoice_no"].astype(str).str.upper() == inv_no.upper()]
            if v_id:
                matches = matches[matches["vendor_id"] == v_id]
                
            if not matches.empty:
                prior = matches.iloc[0].to_dict()
                meta = FRAUD_METADATA[FraudPatternID.EXACT_DUPLICATE]
                alerts.append(FraudAlert(
                    pattern_id=FraudPatternID.EXACT_DUPLICATE,
                    pattern_name=meta["name"],
                    severity=FraudSeverity.CRITICAL,
                    risk_score=meta["base_risk"],
                    explanation=f"Exact duplicate invoice detected! Invoice #{inv_no} from vendor {v_id or invoice.vendor_name} is already recorded in AP Ledger under ID {prior.get('ledger_id')} (Paid: {prior.get('paid_date')}, Total: ₹{prior.get('total_amount')}).",
                    evidence={
                        "prior_ledger_id": prior.get("ledger_id"),
                        "prior_paid_date": prior.get("paid_date"),
                        "prior_total": prior.get("total_amount"),
                        "vendor_id": v_id
                    },
                    recommended_action=meta["recommended_action"],
                    amount_at_risk=grand
                ))

        # -----------------------------------------------------------------
        # Pattern 2: Bank Account Mismatch / Diversion Fraud (FRD-02)
        # -----------------------------------------------------------------
        if matched_vm and invoice.bank_account_last4:
            vm_bank = str(matched_vm.get("bank_account_last4", "")).strip().replace(".0", "")
            inv_bank = str(invoice.bank_account_last4).strip()
            
            known_banks = set([vm_bank]) if vm_bank else set()
            if v_id and not self.ledger_df.empty:
                v_ledger_banks = self.ledger_df[self.ledger_df["vendor_id"] == v_id]["bank_account_last4_on_invoice"].dropna().astype(int).astype(str).tolist()
                known_banks.update(v_ledger_banks)

            if inv_bank and inv_bank not in known_banks:
                meta = FRAUD_METADATA[FraudPatternID.BANK_ACCOUNT_MISMATCH]
                alerts.append(FraudAlert(
                    pattern_id=FraudPatternID.BANK_ACCOUNT_MISMATCH,
                    pattern_name=meta["name"],
                    severity=FraudSeverity.CRITICAL,
                    risk_score=meta["base_risk"],
                    explanation=f"Potential payment diversion! Remittance account on invoice ends in ...{inv_bank}, which has never been used by vendor {v_id or invoice.vendor_name}. High risk of compromised vendor email or fraudulent invoice tampering.",
                    evidence={
                        "invoice_bank_last4": inv_bank,
                        "registered_bank_last4": vm_bank,
                        "vendor_name": matched_vm.get("vendor_name"),
                        "bank_ifsc": matched_vm.get("bank_ifsc")
                    },
                    recommended_action=meta["recommended_action"],
                    amount_at_risk=grand
                ))

        # -----------------------------------------------------------------
        # Pattern 3: Blocked / Deactivated Vendor Activity (FRD-05)
        # -----------------------------------------------------------------
        if matched_vm and str(matched_vm.get("status", "")).upper() == "BLOCKED":
            meta = FRAUD_METADATA[FraudPatternID.BLOCKED_VENDOR_ACTIVITY]
            alerts.append(FraudAlert(
                pattern_id=FraudPatternID.BLOCKED_VENDOR_ACTIVITY,
                pattern_name=meta["name"],
                severity=FraudSeverity.CRITICAL,
                risk_score=meta["base_risk"],
                explanation=f"Invoice submitted by BLOCKED vendor {v_id} ({matched_vm.get('vendor_name')}). This vendor has been debarred from commercial transactions.",
                evidence={
                    "vendor_id": v_id,
                    "vendor_status": matched_vm.get("status"),
                    "onboarded_on": matched_vm.get("onboarded_on")
                },
                recommended_action=meta["recommended_action"],
                amount_at_risk=grand
            ))

        # -----------------------------------------------------------------
        # Pattern 4: Approval Limit Circumvention (FRD-03)
        # -----------------------------------------------------------------
        limit_inr = 200000.0
        if not self.approval_df.empty:
            role_m = self.approval_df[self.approval_df["role"] == approver_role]
            if not role_m.empty:
                limit_inr = float(role_m.iloc[0]["approval_limit_inr"])

        if grand > limit_inr:
            meta = FRAUD_METADATA[FraudPatternID.APPROVAL_LIMIT_BREACH]
            alerts.append(FraudAlert(
                pattern_id=FraudPatternID.APPROVAL_LIMIT_BREACH,
                pattern_name=meta["name"],
                severity=FraudSeverity.HIGH,
                risk_score=meta["base_risk"],
                explanation=f"Approver limit exceeded: Invoice total (₹{grand:,.2f}) exceeds authorization ceiling for role '{approver_role}' (₹{limit_inr:,.2f}) by ₹{grand - limit_inr:,.2f}.",
                evidence={
                    "approver_role": approver_role,
                    "role_limit": limit_inr,
                    "invoice_total": grand,
                    "excess_amount": grand - limit_inr
                },
                recommended_action=meta["recommended_action"],
                amount_at_risk=grand - limit_inr
            ))

        # -----------------------------------------------------------------
        # Pattern 5: PO Overrun & Vendor Discrepancy (FRD-04)
        # -----------------------------------------------------------------
        po_num = (invoice.po_number or "").strip()
        if po_num and not self.po_df.empty:
            po_m = self.po_df[self.po_df["po_number"].astype(str).str.upper() == po_num.upper()]
            if not po_m.empty:
                po_rec = po_m.iloc[0].to_dict()
                po_budget = float(po_rec.get("po_amount_excl_tax", 0.0))
                po_v_id = str(po_rec.get("vendor_id", ""))
                
                # Check PO vendor mismatch
                if v_id and po_v_id and v_id != po_v_id:
                    meta = FRAUD_METADATA[FraudPatternID.PO_OVERRUN_OR_MISMATCH]
                    alerts.append(FraudAlert(
                        pattern_id=FraudPatternID.PO_OVERRUN_OR_MISMATCH,
                        pattern_name=meta["name"],
                        severity=FraudSeverity.HIGH,
                        risk_score=meta["base_risk"],
                        explanation=f"PO Vendor Mismatch! PO {po_num} was raised for vendor {po_v_id}, but invoice is submitted under vendor {v_id}.",
                        evidence={"po_vendor": po_v_id, "invoice_vendor": v_id, "po_number": po_num},
                        recommended_action="Cross-check purchase order authorization and contract assignments.",
                        amount_at_risk=grand
                    ))
                # Check PO overrun
                elif po_budget > 0 and sub > po_budget * 1.05:
                    overrun_amt = sub - po_budget
                    meta = FRAUD_METADATA[FraudPatternID.PO_OVERRUN_OR_MISMATCH]
                    alerts.append(FraudAlert(
                        pattern_id=FraudPatternID.PO_OVERRUN_OR_MISMATCH,
                        pattern_name=meta["name"],
                        severity=FraudSeverity.HIGH,
                        risk_score=meta["base_risk"],
                        explanation=f"Purchase order overrun! Invoice subtotal ₹{sub:,.2f} exceeds authorized PO budget of ₹{po_budget:,.2f} by ₹{overrun_amt:,.2f} ({((sub/po_budget)-1)*100:.1f}%).",
                        evidence={
                            "po_number": po_num,
                            "po_budget": po_budget,
                            "invoice_subtotal": sub,
                            "overrun_inr": overrun_amt
                        },
                        recommended_action=meta["recommended_action"],
                        amount_at_risk=overrun_amt
                    ))

        # -----------------------------------------------------------------
        # Pattern 6: Fuzzy / Near Duplicate Invoice (FRD-06)
        # -----------------------------------------------------------------
        if not self.ledger_df.empty and inv_no and v_id:
            vendor_invoices = self.ledger_df[self.ledger_df["vendor_id"] == v_id]
            for _, r in vendor_invoices.iterrows():
                r_inv_no = str(r["vendor_invoice_no"])
                # Ratio check on invoice string
                ratio = SequenceMatcher(None, inv_no.upper(), r_inv_no.upper()).ratio()
                r_total = float(str(r["total_amount"]).replace(",", "")) if pd.notnull(r["total_amount"]) else 0.0
                
                # If high string similarity (85%+) but not exact, with matching or very close total
                if 0.82 <= ratio < 1.0 and abs(grand - r_total) <= 1.0:
                    meta = FRAUD_METADATA[FraudPatternID.FUZZY_DUPLICATE]
                    alerts.append(FraudAlert(
                        pattern_id=FraudPatternID.FUZZY_DUPLICATE,
                        pattern_name=meta["name"],
                        severity=FraudSeverity.HIGH,
                        risk_score=meta["base_risk"],
                        explanation=f"Near-duplicate invoice detected! Current invoice #{inv_no} is {ratio*100:.0f}% similar to prior invoice #{r_inv_no} (Ledger ID {r['ledger_id']}) with identical amount ₹{grand:,.2f}.",
                        evidence={
                            "current_invoice": inv_no,
                            "prior_invoice": r_inv_no,
                            "prior_ledger_id": r["ledger_id"],
                            "similarity_ratio": round(ratio, 2),
                            "total_amount": grand
                        },
                        recommended_action=meta["recommended_action"],
                        amount_at_risk=grand
                    ))
                    break

        # -----------------------------------------------------------------
        # Pattern 7: Machine Learning Anomaly Detection (FRD-07)
        # -----------------------------------------------------------------
        if sub > 0:
            anomaly_res = self.anomaly_model.score(
                subtotal=sub,
                total_tax=total_tax,
                vendor_id=v_id,
                line_count=len(invoice.line_items) or 1
            )
            if anomaly_res.get("is_anomaly") or anomaly_res.get("anomaly_score", 0) >= settings.HIGH_RISK_SCORE_THRESHOLD:
                meta = FRAUD_METADATA[FraudPatternID.STATISTICAL_OUTLIER]
                reasons_str = "; ".join(anomaly_res.get("reasons", [])) or "Significant statistical deviation in amount or tax ratio."
                alerts.append(FraudAlert(
                    pattern_id=FraudPatternID.STATISTICAL_OUTLIER,
                    pattern_name=meta["name"],
                    severity=FraudSeverity.MEDIUM,
                    risk_score=anomaly_res.get("anomaly_score", 60.0),
                    explanation=f"Statistical anomaly detected by Isolation Forest: {reasons_str}",
                    evidence=anomaly_res,
                    recommended_action=meta["recommended_action"],
                    amount_at_risk=grand
                ))

        # Calculate composite summary
        has_fraud = len(alerts) > 0
        composite_score = max([a.risk_score for a in alerts], default=0.0)
        
        # Determine max severity
        severity_order = {FraudSeverity.CRITICAL: 4, FraudSeverity.HIGH: 3, FraudSeverity.MEDIUM: 2, FraudSeverity.LOW: 1}
        max_sev = "LOW"
        max_rank = 0
        for a in alerts:
            rank = severity_order.get(a.severity, 0)
            if rank > max_rank:
                max_rank = rank
                max_sev = str(a.severity)

        # Money at risk (sum unique maximums per pattern to avoid overcounting)
        total_risk_amt = max([a.amount_at_risk for a in alerts], default=0.0)

        return InvoiceFraudSummary(
            has_fraud_risk=has_fraud,
            composite_risk_score=round(composite_score, 1),
            max_severity=max_sev,
            total_amount_at_risk=round(total_risk_amt, 2),
            alerts=alerts
        )
