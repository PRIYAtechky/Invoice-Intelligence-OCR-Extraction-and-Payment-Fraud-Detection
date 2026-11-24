"""Tests for Evidence-based Payment Fraud Detection."""

import pytest
from src.extraction.schemas import InvoiceExtraction
from src.fraud.fraud_detector import PaymentFraudDetector
from src.fraud.patterns import FraudPatternID, FraudSeverity

@pytest.fixture
def fraud_detector():
    return PaymentFraudDetector()

def test_detect_bank_account_mismatch(fraud_detector, sample_clean_extraction):
    sample_clean_extraction.bank_account_last4 = "9999"  # Unknown account
    res = fraud_detector.detect_fraud(sample_clean_extraction)
    assert res.has_fraud_risk is True
    bank_alert = next((a for a in res.alerts if a.pattern_id == FraudPatternID.BANK_ACCOUNT_MISMATCH), None)
    assert bank_alert is not None
    assert bank_alert.severity == FraudSeverity.CRITICAL

def test_detect_exact_duplicate(fraud_detector, sample_clean_extraction):
    sample_clean_extraction.invoice_number = "INV-244618"  # Known ledger invoice from V1014
    sample_clean_extraction.vendor_gstin = "36DSQAK9331E1ZF"
    res = fraud_detector.detect_fraud(sample_clean_extraction)
    assert res.has_fraud_risk is True
    dup_alert = next((a for a in res.alerts if a.pattern_id == FraudPatternID.EXACT_DUPLICATE), None)
    assert dup_alert is not None
    assert dup_alert.risk_score >= 90.0

def test_detect_approval_limit_abuse(fraud_detector, sample_clean_extraction):
    sample_clean_extraction.grand_total = 275000.0  # Above 200k clerk limit
    res = fraud_detector.detect_fraud(sample_clean_extraction, approver_role="AP-Clerk-01")
    app_alert = next((a for a in res.alerts if a.pattern_id == FraudPatternID.APPROVAL_LIMIT_BREACH), None)
    assert app_alert is not None

def test_detect_blocked_vendor(fraud_detector, sample_clean_extraction):
    # V1060 is Vel Tools Pvt Ltd (status: Blocked)
    sample_clean_extraction.vendor_gstin = "33KXNFC4950Y1Z1"
    sample_clean_extraction.vendor_name = "Vel Tools Pvt Ltd"
    res = fraud_detector.detect_fraud(sample_clean_extraction)
    blocked_alert = next((a for a in res.alerts if a.pattern_id == FraudPatternID.BLOCKED_VENDOR_ACTIVITY), None)
    assert blocked_alert is not None
    assert blocked_alert.severity == FraudSeverity.CRITICAL
