"""Unsupervised Machine Learning Anomaly Detection using Isolation Forest."""

import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib

from config.settings import settings
from config.logging_config import logger

class InvoiceAnomalyModel:
    def __init__(self, ledger_path: Optional[str] = None):
        self.ledger_path = ledger_path or settings.LEDGER_PATH
        self.model_path = settings.PROCESSED_DIR / "isolation_forest.joblib"
        self.stats_path = settings.PROCESSED_DIR / "vendor_stats.joblib"
        self.model: Optional[IsolationForest] = None
        self.vendor_stats: Dict[str, Dict[str, float]] = {}
        self._initialize_or_train()

    def _initialize_or_train(self):
        """Load cached model or train new Isolation Forest on historical ledger."""
        if self.model_path.exists() and self.stats_path.exists():
            try:
                self.model = joblib.load(self.model_path)
                self.vendor_stats = joblib.load(self.stats_path)
                return
            except Exception as e:
                logger.warning(f"Could not load cached anomaly model: {e}")

        self.train()

    def train(self):
        """Train Isolation Forest on AP ledger features."""
        try:
            df = pd.read_csv(self.ledger_path)
            if df.empty:
                logger.warning("Empty ledger provided. Anomaly model training skipped.")
                return

            df["subtotal_num"] = pd.to_numeric(df["subtotal"], errors="coerce").fillna(0.0)
            df["total_tax"] = (
                pd.to_numeric(df["cgst"], errors="coerce").fillna(0.0) +
                pd.to_numeric(df["sgst"], errors="coerce").fillna(0.0) +
                pd.to_numeric(df["igst"], errors="coerce").fillna(0.0)
            )
            df["tax_ratio"] = df["total_tax"] / (df["subtotal_num"] + 1.0)
            df["line_count_num"] = pd.to_numeric(df.get("line_count", 1), errors="coerce").fillna(1.0)

            # Calculate historical mean and std per vendor
            grouped = df.groupby("vendor_id")["subtotal_num"]
            for v_id, grp in grouped:
                self.vendor_stats[str(v_id)] = {
                    "mean": float(grp.mean()),
                    "std": float(grp.std()) if len(grp) > 1 else 1000.0,
                    "count": int(len(grp))
                }

            # Global fallback stats
            global_mean = float(df["subtotal_num"].mean())
            global_std = float(df["subtotal_num"].std())
            self.vendor_stats["_GLOBAL_"] = {"mean": global_mean, "std": global_std, "count": len(df)}

            # Feature: Z-score deviation from vendor mean
            def get_z_score(row):
                v_id = str(row["vendor_id"])
                stats = self.vendor_stats.get(v_id, self.vendor_stats["_GLOBAL_"])
                std = stats["std"] if stats["std"] > 1.0 else 1.0
                return abs(row["subtotal_num"] - stats["mean"]) / std

            df["z_score"] = df.apply(get_z_score, axis=1)

            feature_cols = ["subtotal_num", "tax_ratio", "z_score", "line_count_num"]
            X = df[feature_cols].values

            # Train Isolation Forest
            self.model = IsolationForest(
                n_estimators=100,
                contamination=0.03,  # Top 3% anomalies
                random_state=42,
                n_jobs=-1
            )
            self.model.fit(X)

            # Persist artifacts
            settings.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.vendor_stats, self.stats_path)
            logger.info("Trained and persisted Isolation Forest model on AP ledger.")

        except Exception as e:
            logger.error(f"Failed to train Isolation Forest anomaly model: {e}")

    def score(
        self,
        subtotal: float,
        total_tax: float,
        vendor_id: Optional[str] = None,
        line_count: int = 1
    ) -> Dict[str, Any]:
        """Compute anomaly score and classification for an invoice."""
        if not self.model:
            return {"is_anomaly": False, "anomaly_score": 0.0, "reason": "Model not fitted"}

        sub = float(subtotal or 0.0)
        tax = float(total_tax or 0.0)
        tax_ratio = tax / (sub + 1.0)
        lines = float(line_count or 1)

        v_id = str(vendor_id) if vendor_id else "_GLOBAL_"
        stats = self.vendor_stats.get(v_id, self.vendor_stats.get("_GLOBAL_", {"mean": 25000.0, "std": 15000.0}))
        std = stats["std"] if stats.get("std", 0) > 1.0 else 1.0
        z_score = abs(sub - stats.get("mean", sub)) / std

        features = np.array([[sub, tax_ratio, z_score, lines]])
        
        # Decision function: lower values mean more anomalous
        raw_score = float(self.model.decision_function(features)[0])
        pred = int(self.model.predict(features)[0])  # -1 for anomaly, 1 for normal
        
        is_anomaly = (pred == -1)
        
        # Scale anomaly score into 0 - 100 risk score (lower decision -> higher anomaly risk)
        # Decision function is typically between -0.3 and +0.3
        normalized_risk = float(np.clip((0.15 - raw_score) * 200.0, 0.0, 100.0))

        reasons = []
        if z_score > 3.0:
            reasons.append(f"Subtotal ₹{sub:,.2f} deviates significantly (>3σ) from vendor average ₹{stats.get('mean', 0):,.2f}")
        if tax_ratio < 0.05 or tax_ratio > 0.25:
            reasons.append(f"Unusual effective tax ratio of {tax_ratio*100:.1f}% (standard is ~18%)")
        if lines > 15:
            reasons.append(f"Unusually high line item count ({int(lines)} items)")

        return {
            "is_anomaly": is_anomaly,
            "anomaly_score": round(normalized_risk, 1),
            "raw_decision_score": round(raw_score, 4),
            "z_score": round(z_score, 2),
            "vendor_mean": round(stats.get("mean", 0), 2),
            "reasons": reasons
        }
