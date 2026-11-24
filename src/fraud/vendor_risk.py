"""Explainable Vendor Risk Scoring Engine."""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

from config.settings import settings
from config.logging_config import logger

class VendorRiskScorer:
    def __init__(
        self,
        vendor_master_path: Optional[str] = None,
        ledger_path: Optional[str] = None,
        po_path: Optional[str] = None,
        approval_path: Optional[str] = None
    ):
        self.vm_path = vendor_master_path or settings.VENDOR_MASTER_PATH
        self.ledger_path = ledger_path or settings.LEDGER_PATH
        self.po_path = po_path or settings.PURCHASE_ORDERS_PATH
        self.approval_path = approval_path or settings.APPROVAL_MATRIX_PATH
        self.vendor_profiles: Dict[str, Dict[str, Any]] = {}
        self.build_risk_profiles()

    def build_risk_profiles(self):
        """Analyze full 2-year AP ledger and compute explainable risk scores."""
        try:
            vm = pd.read_excel(self.vm_path)
            ledger = pd.read_csv(self.ledger_path)
            po = pd.read_csv(self.po_path)
            app = pd.read_csv(self.approval_path)

            ledger["total_amount_num"] = pd.to_numeric(
                ledger["total_amount"].astype(str).str.replace(",", ""), errors="coerce"
            ).fillna(0.0)

            # Join VM
            merged_vm = ledger.merge(vm, on="vendor_id", how="left", suffixes=("_ledger", "_vm"))

            # 1. Exact duplicates in ledger
            exact_dup_counts = ledger[ledger.duplicated(subset=["vendor_id", "vendor_invoice_no"], keep=False)].groupby("vendor_id")["ledger_id"].count()

            # 2. Bank mismatches
            bank_diff = merged_vm[merged_vm["bank_account_last4_on_invoice"] != merged_vm["bank_account_last4"]].groupby("vendor_id")["ledger_id"].count()

            # 3. GSTIN mismatches
            gstin_diff = merged_vm[merged_vm["vendor_gstin_on_invoice"] != merged_vm["gstin"]].groupby("vendor_id")["ledger_id"].count()

            # 4. Approval breaches
            merged_app = ledger.merge(app, left_on="approved_by", right_on="role", how="left")
            app_breaches = merged_app[merged_app["total_amount_num"] > merged_app["approval_limit_inr"]].groupby("vendor_id")["ledger_id"].count()

            # 5. PO Overruns
            merged_po = ledger.merge(po, on="po_number", how="left")
            po_overruns = merged_po[merged_po["subtotal"] > merged_po["po_amount_excl_tax"] * 1.05].groupby("vendor_id_x")["ledger_id"].count()

            # Build profile for every vendor in VM
            for _, row in vm.iterrows():
                v_id = str(row["vendor_id"])
                v_name = str(row["vendor_name"])
                status = str(row.get("status", "Active"))
                
                v_invoices = ledger[ledger["vendor_id"] == v_id]
                inv_count = len(v_invoices)
                total_spend = float(v_invoices["total_amount_num"].sum())

                # Risk factors
                factors = []
                score = 0.0

                # Factor 1: Status
                if status.upper() != "ACTIVE":
                    score += 45.0
                    factors.append(f"Vendor status is {status} in Vendor Master (+45)")

                # Factor 2: Bank Account Mismatches
                b_count = int(bank_diff.get(v_id, 0))
                if b_count > 0:
                    b_score = min(30.0, b_count * 5.0)
                    score += b_score
                    factors.append(f"{b_count} historical invoice(s) with unverified bank accounts (+{b_score:.0f})")

                # Factor 3: Exact Duplicates
                d_count = int(exact_dup_counts.get(v_id, 0))
                if d_count > 0:
                    d_score = min(25.0, d_count * 6.0)
                    score += d_score
                    factors.append(f"{d_count} duplicate billing instance(s) detected (+{d_score:.0f})")

                # Factor 4: GSTIN discrepancies
                g_count = int(gstin_diff.get(v_id, 0))
                if g_count > 0:
                    g_score = min(20.0, g_count * 4.0)
                    score += g_score
                    factors.append(f"{g_count} invoice(s) with mismatched GSTIN (+{g_score:.0f})")

                # Factor 5: PO Overruns
                po_count = int(po_overruns.get(v_id, 0))
                if po_count > 0:
                    po_score = min(15.0, po_count * 3.0)
                    score += po_score
                    factors.append(f"{po_count} invoice(s) with PO budget overruns (+{po_score:.0f})")

                # Factor 6: Approval breaches
                ap_count = int(app_breaches.get(v_id, 0))
                if ap_count > 0:
                    ap_score = min(15.0, ap_count * 5.0)
                    score += ap_score
                    factors.append(f"{ap_count} invoice(s) exceeding approver authorization (+{ap_score:.0f})")

                # Cap score at 100
                final_score = round(min(100.0, max(0.0, score)), 1)

                if final_score >= settings.HIGH_RISK_SCORE_THRESHOLD:
                    category = "CRITICAL" if final_score >= 85.0 else "HIGH"
                elif final_score >= settings.MEDIUM_RISK_SCORE_THRESHOLD:
                    category = "MEDIUM"
                else:
                    category = "LOW"

                # Calculate estimated money at risk
                money_at_risk = 0.0
                if d_count > 0:
                    dup_rows = v_invoices[v_invoices.duplicated(subset=["vendor_id", "vendor_invoice_no"], keep=False)]
                    money_at_risk += float(dup_rows["total_amount_num"].sum() / 2.0)
                if status.upper() != "ACTIVE":
                    money_at_risk += total_spend

                self.vendor_profiles[v_id] = {
                    "vendor_id": v_id,
                    "vendor_name": v_name,
                    "status": status,
                    "risk_score": final_score,
                    "risk_category": category,
                    "total_invoices": inv_count,
                    "total_spend": round(total_spend, 2),
                    "money_at_risk": round(money_at_risk, 2),
                    "duplicate_count": d_count,
                    "bank_mismatch_count": b_count,
                    "gstin_mismatch_count": g_count,
                    "po_overrun_count": po_count,
                    "approval_breach_count": ap_count,
                    "risk_factors": factors
                }

            logger.info(f"Built risk profiles for {len(self.vendor_profiles)} vendors.")

        except Exception as e:
            logger.error(f"Failed to build vendor risk profiles: {e}")

    def get_vendor_risk(self, vendor_id: Optional[str]) -> Dict[str, Any]:
        """Retrieve risk score and factors for a specific vendor."""
        if not vendor_id or str(vendor_id) not in self.vendor_profiles:
            return {
                "vendor_id": vendor_id or "UNKNOWN",
                "vendor_name": "Unregistered Vendor",
                "risk_score": 60.0,
                "risk_category": "MEDIUM",
                "risk_factors": ["Unregistered vendor not present in Vendor Master (+60)"],
                "money_at_risk": 0.0
            }
        return self.vendor_profiles[str(vendor_id)]

    def get_all_profiles_df(self) -> pd.DataFrame:
        """Return all vendor profiles as a sorted DataFrame."""
        if not self.vendor_profiles:
            return pd.DataFrame()
        df = pd.DataFrame(list(self.vendor_profiles.values()))
        return df.sort_values(by="risk_score", ascending=False)
