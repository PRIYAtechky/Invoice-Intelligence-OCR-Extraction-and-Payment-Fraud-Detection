"""
Statistical Anomaly Detection using Isolation Forest
Trains an unsupervised model on numerical ledger features to identify multi-dimensional outliers
without data leakage.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from src.extraction.schemas import ExtractedInvoice
from src.config import settings


class APAnomalyDetector:
    def __init__(self, contamination: float = 0.03):
        self.contamination = contamination
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            n_estimators=100,
            contamination=self.contamination,
            random_state=42,
            n_jobs=-1
        )
        self.is_fitted = False
        self._feature_means = {}
        self._feature_stds = {}

    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        # Extract features: subtotal, total_amount, line_count, tax_ratio
        df_feat = pd.DataFrame()
        
        # Numeric conversions
        df_feat["subtotal"] = pd.to_numeric(df["subtotal"], errors="coerce").fillna(0.0)
        
        if "total_amount_num" in df.columns:
            df_feat["total_amount"] = df["total_amount_num"]
        else:
            df_feat["total_amount"] = pd.to_numeric(
                df["total_amount"].astype(str).str.replace(",", ""), errors="coerce"
            ).fillna(0.0)

        df_feat["line_count"] = pd.to_numeric(df.get("line_count", 1), errors="coerce").fillna(1.0)
        
        # Derived: tax ratio
        df_feat["tax_ratio"] = np.where(
            df_feat["subtotal"] > 0,
            (df_feat["total_amount"] - df_feat["subtotal"]) / df_feat["subtotal"],
            0.18
        )
        # Cap outliers in tax_ratio for numerical stability
        df_feat["tax_ratio"] = df_feat["tax_ratio"].clip(0.0, 1.0)

        return df_feat[["subtotal", "total_amount", "line_count", "tax_ratio"]].values

    def fit_on_ledger(self, ledger_df: Optional[pd.DataFrame] = None) -> bool:
        """Train Isolation Forest on AP ledger historical baseline."""
        try:
            if ledger_df is None:
                p = settings.ledger_path
                if not p.exists():
                    return False
                ledger_df = pd.read_csv(p)

            X = self._prepare_features(ledger_df)
            X_scaled = self.scaler.fit_transform(X)
            self.model.fit(X_scaled)
            self.is_fitted = True

            # Cache means and stds for Z-score explanations
            cols = ["subtotal", "total_amount", "line_count", "tax_ratio"]
            for i, col in enumerate(cols):
                self._feature_means[col] = float(np.mean(X[:, i]))
                self._feature_stds[col] = float(np.std(X[:, i])) or 1.0

            return True
        except Exception:
            return False

    def predict_invoice(self, invoice: ExtractedInvoice) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Evaluate an invoice for statistical anomalies.
        Returns:
            Tuple of (is_anomaly, anomaly_score_0_to_100, explanation_dict)
        """
        if not self.is_fitted:
            self.fit_on_ledger()

        if not self.is_fitted:
            return False, 0.0, {"reason": "Model uncalibrated"}

        subtotal = float(invoice.subtotal or 0.0)
        total = float(invoice.grand_total or 0.0)
        line_count = float(len(invoice.line_items) or 1)
        tax_ratio = ((total - subtotal) / subtotal) if subtotal > 0 else 0.18
        tax_ratio = min(max(tax_ratio, 0.0), 1.0)

        vec = np.array([[subtotal, total, line_count, tax_ratio]])
        vec_scaled = self.scaler.transform(vec)
        
        # decision_function gives negative values for anomalies
        raw_score = float(self.model.decision_function(vec_scaled)[0])
        pred = int(self.model.predict(vec_scaled)[0])  # -1 = anomaly, 1 = normal

        # Convert to 0 - 100 risk score
        # raw_score typically ranges from -0.3 (highly anomalous) to +0.2 (very normal)
        anomaly_risk = max(0.0, min(100.0, (0.15 - raw_score) * 200.0))
        is_anomaly = (pred == -1)

        # Feature Z-score contributions for explainability
        explanations = []
        if self._feature_stds:
            z_amt = (total - self._feature_means["total_amount"]) / self._feature_stds["total_amount"]
            z_lines = (line_count - self._feature_means["line_count"]) / self._feature_stds["line_count"]
            z_tax = (tax_ratio - self._feature_means["tax_ratio"]) / self._feature_stds["tax_ratio"]

            if abs(z_amt) > 2.5:
                explanations.append(f"Invoice amount ₹{total:,.2f} is {z_amt:+.1f}σ from AP historical mean.")
            if abs(z_lines) > 2.5:
                explanations.append(f"Line count ({int(line_count)}) is {z_lines:+.1f}σ from mean.")
            if abs(z_tax) > 2.5:
                explanations.append(f"Effective tax rate ({tax_ratio:.1%}) deviates significantly from standard GST.")

        return is_anomaly, round(anomaly_risk, 1), {
            "is_outlier": is_anomaly,
            "raw_decision_score": round(raw_score, 4),
            "z_score_reasons": explanations
        }


# Global anomaly detector instance
anomaly_detector = APAnomalyDetector()
