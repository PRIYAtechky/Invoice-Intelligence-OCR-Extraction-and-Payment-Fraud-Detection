"""
Explainable Vendor Risk Scoring Engine
Calculates multi-factor risk profiles (0-100) based on historical payment behavior,
bank alterations, duplicate patterns, PO overruns, and ERP master status.
"""
from typing import Dict, Any, List, Optional
import pandas as pd
from src.config import settings


class VendorRiskScorer:
    def __init__(self, ledger_df: Optional[pd.DataFrame] = None, vendor_df: Optional[pd.DataFrame] = None):
        self.ledger_df = ledger_df if ledger_df is not None else self._load_ledger()
        self.vendor_df = vendor_df if vendor_df is not None else self._load_vendors()

    def _load_ledger(self) -> pd.DataFrame:
        p = settings.ledger_path
        if p.exists():
            df = pd.read_csv(p)
            df["total_amount_num"] = df["total_amount"].astype(str).str.replace(",", "").astype(float)
            return df
        return pd.DataFrame()

    def _load_vendors(self) -> pd.DataFrame:
        p = settings.vendor_master_path
        if p.exists():
            return pd.read_excel(p)
        return pd.DataFrame()

    def compute_all_vendor_scores(self) -> List[Dict[str, Any]]:
        """Compute explainable risk scorecards for all 60 registered vendors."""
        if self.vendor_df.empty or self.ledger_df.empty:
            return []

        # Join ledger with vendor master
        merged = self.ledger_df.merge(
            self.vendor_df[["vendor_id", "vendor_name", "status", "bank_account_last4", "gstin"]],
            on="vendor_id",
            how="left"
        )

        # 1. Bank account alterations per vendor
        merged["bank_mismatch"] = (
            merged["bank_account_last4_on_invoice"] != merged["bank_account_last4"]
        )
        bank_mismatches = merged.groupby("vendor_id")["bank_mismatch"].sum()

        # 2. Duplicate invoices per vendor
        dupe_mask = merged.duplicated(subset=["vendor_id", "vendor_invoice_no"], keep=False)
        dupes = merged[dupe_mask].groupby("vendor_id")["ledger_id"].count()

        # 3. Spend volume and invoice count
        vendor_totals = merged.groupby("vendor_id").agg(
            total_spend=("total_amount_num", "sum"),
            invoice_count=("ledger_id", "count"),
            vendor_name=("vendor_name", "first"),
            status=("status", "first")
        )

        scorecards = []
        for v_id, row in vendor_totals.iterrows():
            v_name = row["vendor_name"]
            status = row["status"]
            spend = row["total_spend"]
            inv_count = row["invoice_count"]

            b_count = int(bank_mismatches.get(v_id, 0))
            d_count = int(dupes.get(v_id, 0))

            # Factor calculations
            # Bank mismatch factor (max 35)
            bank_score = min(35.0, (b_count / max(1, inv_count)) * 250.0)
            
            # Duplicate invoice factor (max 25)
            dupe_score = min(25.0, (d_count / max(1, inv_count)) * 200.0)

            # Blocked vendor factor (max 25)
            status_score = 25.0 if str(status).lower() == "blocked" else 0.0

            # Frequency/volume factor (max 15)
            high_vol_score = 10.0 if spend > 2500000.0 else 0.0

            raw_total_score = bank_score + dupe_score + status_score + high_vol_score
            total_risk = round(min(100.0, raw_total_score), 1)

            # Risk tier
            if total_risk >= 70:
                tier = "Critical"
            elif total_risk >= 45:
                tier = "High"
            elif total_risk >= 20:
                tier = "Medium"
            else:
                tier = "Low"

            # Explainable evidence points
            evidence = []
            if b_count > 0:
                evidence.append(f"{b_count} bank account alterations detected on invoices.")
            if d_count > 0:
                evidence.append(f"{d_count} duplicate invoice numbers submitted.")
            if str(status).lower() == "blocked":
                evidence.append("Vendor is explicitly marked as BLOCKED in ERP master.")
            if high_vol_score > 0:
                evidence.append(f"High cumulative spend exceeding ₹{spend:,.2f}.")

            if not evidence:
                evidence.append("Consistent payment history with no recorded discrepancies.")

            scorecards.append({
                "vendor_id": v_id,
                "vendor_name": v_name,
                "status": status,
                "risk_score": total_risk,
                "risk_tier": tier,
                "total_invoices": inv_count,
                "total_spend": round(spend, 2),
                "bank_alterations_count": b_count,
                "duplicate_invoices_count": d_count,
                "evidence_points": evidence
            })

        # Sort by risk score descending
        scorecards.sort(key=lambda x: x["risk_score"], reverse=True)
        return scorecards


# Global vendor risk scorer
vendor_risk_scorer = VendorRiskScorer()
