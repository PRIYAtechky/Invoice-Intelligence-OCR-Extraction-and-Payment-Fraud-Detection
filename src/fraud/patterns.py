"""Fraud Patterns and Anomaly Detection Rules."""

from enum import Enum
from typing import Dict, Any

class FraudPatternID(str, Enum):
    EXACT_DUPLICATE = "FRD-01"
    BANK_ACCOUNT_MISMATCH = "FRD-02"
    APPROVAL_LIMIT_BREACH = "FRD-03"
    PO_OVERRUN_OR_MISMATCH = "FRD-04"
    BLOCKED_VENDOR_ACTIVITY = "FRD-05"
    FUZZY_DUPLICATE = "FRD-06"
    STATISTICAL_OUTLIER = "FRD-07"

class FraudSeverity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

FRAUD_METADATA: Dict[str, Dict[str, Any]] = {
    FraudPatternID.EXACT_DUPLICATE: {
        "name": "Exact Duplicate Invoice / Double Billing",
        "description": "Invoice with identical vendor ID and invoice number already exists or was paid.",
        "default_severity": FraudSeverity.CRITICAL,
        "base_risk": 95.0,
        "recommended_action": "Reject invoice immediately. Verify AP payment logs to prevent duplicate disbursement."
    },
    FraudPatternID.BANK_ACCOUNT_MISMATCH: {
        "name": "Unverified Bank Account / Potential Diversion",
        "description": "Remittance bank account on invoice differs from registered bank master on file.",
        "default_severity": FraudSeverity.CRITICAL,
        "base_risk": 90.0,
        "recommended_action": "Put invoice on strict payment hold. Perform out-of-band vendor call back verification."
    },
    FraudPatternID.APPROVAL_LIMIT_BREACH: {
        "name": "Approval Authority Threshold Exceeded",
        "description": "Invoice value exceeds the statutory financial authorization threshold of the assigned approver.",
        "default_severity": FraudSeverity.HIGH,
        "base_risk": 75.0,
        "recommended_action": "Escalate to Finance Head or Department Director for formal re-authorization."
    },
    FraudPatternID.PO_OVERRUN_OR_MISMATCH: {
        "name": "Purchase Order Overrun / Vendor Discrepancy",
        "description": "Billed subtotal exceeds authorized PO ceiling or billed by an unauthorized third party.",
        "default_severity": FraudSeverity.HIGH,
        "base_risk": 70.0,
        "recommended_action": "Route to Procurement team for purchase order revision or change order justification."
    },
    FraudPatternID.BLOCKED_VENDOR_ACTIVITY: {
        "name": "Blacklisted / Deactivated Vendor Invoicing",
        "description": "Invoice submitted for or by a vendor flagged as Blocked or Inactive in Vendor Master.",
        "default_severity": FraudSeverity.CRITICAL,
        "base_risk": 95.0,
        "recommended_action": "Block invoice processing. Review compliance and audit register for reason of debarment."
    },
    FraudPatternID.FUZZY_DUPLICATE: {
        "name": "Fuzzy / Near-Duplicate Invoice Number",
        "description": "Invoice number closely resembles a prior invoice from the same vendor with identical amount.",
        "default_severity": FraudSeverity.HIGH,
        "base_risk": 80.0,
        "recommended_action": "Inspect previous invoice copy to ensure this is not an altered duplicate resubmission."
    },
    FraudPatternID.STATISTICAL_OUTLIER: {
        "name": "Unusual Amount / High Statistical Anomaly",
        "description": "Isolation Forest model detected abnormal billing pattern relative to vendor historical distribution.",
        "default_severity": FraudSeverity.MEDIUM,
        "base_risk": 55.0,
        "recommended_action": "Review item quantities, line item prices, and compare with contract agreements."
    }
}
