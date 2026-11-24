"""
AP Ledger Comprehensive Fraud Analytics and Money-at-Risk Engine
Scans 21 months of AP ledger history (6,007 transactions) to identify fraud patterns,
quantify total rupees at risk, and generate actionable auditor findings.
"""
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd

from src.config import settings
from src.fraud.patterns import check_fuzzy_duplicate
from src.fraud.vendor_risk import vendor_risk_scorer
from src.extraction.schemas import FraudSeverity


class LedgerFraudAnalyzer:
    def __init__(self, ledger_path: Optional[Path] = None):
        self.ledger_path = ledger_path or settings.ledger_path

    def analyze_ledger(self, output_report_path: Optional[Path] = None) -> Dict[str, Any]:
        """Execute full forensic audit of the historical AP ledger."""
        if not self.ledger_path.exists():
            return {"error": f"Ledger file not found at {self.ledger_path}"}

        df_led = pd.read_csv(self.ledger_path)
        df_vm = pd.read_excel(settings.vendor_master_path)
        df_po = pd.read_csv(settings.purchase_orders_path)
        df_app = pd.read_csv(settings.approval_matrix_path)

        # Normalize numeric columns
        df_led["total_amount_num"] = df_led["total_amount"].astype(str).str.replace(",", "").astype(float)
        df_led["subtotal_num"] = pd.to_numeric(df_led["subtotal"], errors="coerce").fillna(0.0)

        # Merge reference data
        merged_vm = df_led.merge(
            df_vm[["vendor_id", "vendor_name", "bank_account_last4", "status", "gstin"]],
            on="vendor_id",
            how="left"
        )
        merged_app = merged_vm.merge(
            df_app,
            left_on="approved_by",
            right_on="role",
            how="left"
        )
        merged_po = merged_app.merge(
            df_po[["po_number", "po_amount_excl_tax", "po_status"]],
            on="po_number",
            how="left"
        )

        findings = []
        total_money_at_risk = 0.0

        # Pattern 1: Exact Duplicate Invoices
        exact_dupes_mask = merged_po.duplicated(subset=["vendor_id", "vendor_invoice_no"], keep=False)
        exact_dupes_df = merged_po[exact_dupes_mask]
        # Money at risk is the duplicate portion (half of total duplicate rows)
        exact_dupes_first = merged_po[merged_po.duplicated(subset=["vendor_id", "vendor_invoice_no"], keep="first")]
        exact_dupe_risk = float(exact_dupes_first["total_amount_num"].sum())
        total_money_at_risk += exact_dupe_risk

        findings.append({
            "pattern_id": "FRAUD_001",
            "pattern_name": "Exact Duplicate Invoices",
            "incident_count": len(exact_dupes_df),
            "affected_invoice_pairs": len(exact_dupes_df) // 2,
            "money_at_risk_inr": round(exact_dupe_risk, 2),
            "severity": FraudSeverity.CRITICAL.value,
            "evidence_summary": f"Detected {len(exact_dupes_df)} rows ({len(exact_dupes_df)//2} pairs) sharing identical vendor ID and invoice number.",
            "sample_incidents": exact_dupes_df[["ledger_id", "vendor_id", "vendor_invoice_no", "total_amount_num", "paid_date"]].head(6).to_dict(orient="records")
        })

        # Pattern 2: Bank Account Mismatches / Diversion
        bank_mismatch_mask = merged_po["bank_account_last4_on_invoice"] != merged_po["bank_account_last4"]
        bank_mismatch_df = merged_po[bank_mismatch_mask]
        bank_risk = float(bank_mismatch_df["total_amount_num"].sum())
        total_money_at_risk += bank_risk

        findings.append({
            "pattern_id": "FRAUD_003",
            "pattern_name": "Bank Account Diversion / Tampering",
            "incident_count": len(bank_mismatch_df),
            "money_at_risk_inr": round(bank_risk, 2),
            "severity": FraudSeverity.CRITICAL.value,
            "evidence_summary": f"Found {len(bank_mismatch_df)} invoices directing payments to unregistered bank accounts.",
            "sample_incidents": bank_mismatch_df[["ledger_id", "vendor_id", "vendor_invoice_no", "bank_account_last4_on_invoice", "bank_account_last4", "total_amount_num"]].head(6).to_dict(orient="records")
        })

        # Pattern 3: Payments to Blocked / Inactive Vendors
        blocked_mask = merged_po["status"].str.lower() == "blocked"
        blocked_df = merged_po[blocked_mask]
        blocked_risk = float(blocked_df["total_amount_num"].sum())
        total_money_at_risk += blocked_risk

        findings.append({
            "pattern_id": "FRAUD_005",
            "pattern_name": "Payments to Blocked Vendors",
            "incident_count": len(blocked_df),
            "money_at_risk_inr": round(blocked_risk, 2),
            "severity": FraudSeverity.HIGH.value,
            "evidence_summary": f"Processed {len(blocked_df)} transactions totaling ₹{blocked_risk:,.2f} for vendors marked as BLOCKED in ERP master.",
            "sample_incidents": blocked_df[["ledger_id", "vendor_id", "vendor_name", "total_amount_num", "paid_date"]].head(6).to_dict(orient="records")
        })

        # Pattern 4: Approval Limit Violations
        app_violation_mask = merged_po["total_amount_num"] > merged_po["approval_limit_inr"]
        app_violation_df = merged_po[app_violation_mask]
        app_risk = float(app_violation_df["total_amount_num"].sum())
        total_money_at_risk += app_risk

        findings.append({
            "pattern_id": "FRAUD_006",
            "pattern_name": "Delegation of Financial Authority Violations",
            "incident_count": len(app_violation_df),
            "money_at_risk_inr": round(app_risk, 2),
            "severity": FraudSeverity.HIGH.value,
            "evidence_summary": f"{len(app_violation_df)} invoices exceeded the signatory authority threshold of the designated approver.",
            "sample_incidents": app_violation_df[["ledger_id", "approved_by", "approval_limit_inr", "total_amount_num", "approved_at"]].to_dict(orient="records")
        })

        # Pattern 5: PO Overruns
        po_overrun_mask = merged_po["subtotal_num"] > (merged_po["po_amount_excl_tax"] * 1.01)
        po_overrun_df = merged_po[po_overrun_mask]
        po_overrun_risk = float((po_overrun_df["subtotal_num"] - po_overrun_df["po_amount_excl_tax"]).sum())
        total_money_at_risk += po_overrun_risk

        findings.append({
            "pattern_id": "FRAUD_004",
            "pattern_name": "Purchase Order Cost Overruns",
            "incident_count": len(po_overrun_df),
            "money_at_risk_inr": round(po_overrun_risk, 2),
            "severity": FraudSeverity.MEDIUM.value,
            "evidence_summary": f"{len(po_overrun_df)} invoices billed in excess of the approved purchase order ceiling.",
            "sample_incidents": po_overrun_df[["ledger_id", "po_number", "subtotal_num", "po_amount_excl_tax"]].head(6).to_dict(orient="records")
        })

        # Pattern 6: GSTIN Mismatches
        gstin_mismatch_mask = merged_po["vendor_gstin_on_invoice"] != merged_po["gstin"]
        gstin_mismatch_df = merged_po[gstin_mismatch_mask]
        findings.append({
            "pattern_id": "FRAUD_007",
            "pattern_name": "GSTIN Discrepancy on Invoices",
            "incident_count": len(gstin_mismatch_df),
            "money_at_risk_inr": round(float(gstin_mismatch_df["total_amount_num"].sum()), 2),
            "severity": FraudSeverity.MEDIUM.value,
            "evidence_summary": f"{len(gstin_mismatch_df)} invoices printed a GSTIN differing from the master record.",
            "sample_incidents": gstin_mismatch_df[["ledger_id", "vendor_id", "vendor_gstin_on_invoice", "gstin"]].head(6).to_dict(orient="records")
        })

        vendor_scorecards = vendor_risk_scorer.compute_all_vendor_scores()

        report = {
            "audit_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "ledger_period": "2024-2025 (21 months)",
            "total_transactions_audited": len(df_led),
            "total_spend_audited_inr": round(float(df_led["total_amount_num"].sum()), 2),
            "total_money_at_risk_inr": round(total_money_at_risk, 2),
            "patterns_identified_count": len(findings),
            "findings": findings,
            "top_risk_vendors": vendor_scorecards[:10]
        }

        save_path = output_report_path or (settings.project_root / "reports" / "fraud_findings_report.json")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        return report


# Global ledger analyzer
ledger_fraud_analyzer = LedgerFraudAnalyzer()
