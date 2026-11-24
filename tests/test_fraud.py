"""
Tests for Payment Fraud Detection Patterns and Anomaly Model
"""
from src.extraction.schemas import ExtractedInvoice, FraudSeverity
from src.fraud.patterns import (
    check_exact_duplicate, check_fuzzy_duplicate,
    check_bank_tampering, check_blocked_vendor
)
from src.fraud.anomaly_detector import anomaly_detector


def test_exact_duplicate_detection():
    inv = ExtractedInvoice(
        invoice_number="INV-2025-001",
        vendor_name="Venkateswara Bearings & Co",
        grand_total=15000.0
    )
    history = [
        {"vendor_invoice_no": "INV-2025-001", "vendor_name": "Venkateswara Bearings & Co", "id": 101}
    ]
    alert = check_exact_duplicate(inv, history)
    assert alert is not None
    assert alert.pattern_id == "FRAUD_001"
    assert alert.severity == FraudSeverity.CRITICAL


def test_fuzzy_duplicate_detection():
    inv = ExtractedInvoice(
        invoice_number="INV-2025-001A",
        vendor_name="Venkateswara Bearings & Co",
        grand_total=15000.0
    )
    history = [
        {"vendor_invoice_no": "INV-2025-001B", "vendor_name": "Venkateswara Bearings & Co", "total_amount": 15000.0, "id": 102}
    ]
    alert = check_fuzzy_duplicate(inv, history)
    assert alert is not None
    assert alert.pattern_id == "FRAUD_002"


def test_bank_tampering_detection():
    inv = ExtractedInvoice(
        vendor_name="Venkateswara Bearings & Co",
        bank_account_last4="9999",
        grand_total=50000.0
    )
    vm_rec = {"vendor_name": "Venkateswara Bearings & Co", "bank_account_last4": 4479}
    alert = check_bank_tampering(inv, vm_rec)
    assert alert is not None
    assert alert.pattern_id == "FRAUD_003"
    assert alert.severity == FraudSeverity.CRITICAL


def test_anomaly_detector_scoring():
    inv = ExtractedInvoice(
        subtotal=10000.0,
        grand_total=11800.0,
        line_items=[]
    )
    is_anomaly, score, details = anomaly_detector.predict_invoice(inv)
    assert isinstance(is_anomaly, bool)
    assert 0.0 <= score <= 100.0
