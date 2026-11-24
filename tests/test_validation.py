"""
Tests for Data Validation Engine
"""
from src.extraction.schemas import ExtractedInvoice, ValidationStatus
from src.validation.engine import validation_engine


def test_arithmetic_reconciliation_pass():
    inv = ExtractedInvoice(
        subtotal=1000.0,
        cgst=90.0,
        sgst=90.0,
        igst=0.0,
        grand_total=1180.0
    )
    res = validation_engine.rules.check_arithmetic_reconciliation(inv)
    assert res.status == ValidationStatus.PASSED


def test_arithmetic_reconciliation_fail():
    inv = ExtractedInvoice(
        subtotal=1000.0,
        cgst=90.0,
        sgst=90.0,
        igst=0.0,
        grand_total=1280.0  # Discrepancy of 100
    )
    res = validation_engine.rules.check_arithmetic_reconciliation(inv)
    assert res.status == ValidationStatus.FAILED


def test_gstin_format_valid():
    inv = ExtractedInvoice(vendor_gstin="33XXYEN7492X1Z2")
    res = validation_engine.rules.check_gstin_format(inv)
    assert res.status == ValidationStatus.PASSED


def test_gstin_format_malformed():
    inv = ExtractedInvoice(vendor_gstin="27BFOPK9906J1Z#")  # Has # char from ground truth
    res = validation_engine.rules.check_gstin_format(inv)
    assert res.status == ValidationStatus.FAILED


def test_bank_account_match():
    # Valid matching vendor and bank
    inv = ExtractedInvoice(
        vendor_name="Delta Tools Pvt Ltd",
        bank_account_last4="4335"
    )
    res = validation_engine.rules.check_bank_account_match(inv)
    assert res.status == ValidationStatus.PASSED


def test_bank_account_tampering():
    # Tampered bank account
    inv = ExtractedInvoice(
        vendor_name="Delta Tools Pvt Ltd",
        bank_account_last4="9999"  # Differs from registered 4335
    )
    res = validation_engine.rules.check_bank_account_match(inv)
    assert res.status == ValidationStatus.FAILED
