"""Tests for Validation Engine (Statutory Rules & Consistency)."""

import pytest
from src.extraction.schemas import InvoiceExtraction, LineItem
from src.validation.validator import ValidationEngine
from src.validation.rules import ValidationStatus, RuleID

@pytest.fixture
def validator():
    return ValidationEngine()

def test_validation_arithmetic_success(validator, sample_clean_extraction):
    res = validator.validate(sample_clean_extraction)
    # Check VAL-02
    arith_rule = next(r for r in res.rules_evaluated if r.rule_id == RuleID.ARITHMETIC_CHECK)
    assert arith_rule.status == ValidationStatus.PASSED

def test_validation_arithmetic_failure(validator, sample_clean_extraction):
    sample_clean_extraction.grand_total = 99999.00  # Deliberate mismatch
    res = validator.validate(sample_clean_extraction)
    arith_rule = next(r for r in res.rules_evaluated if r.rule_id == RuleID.ARITHMETIC_CHECK)
    assert arith_rule.status == ValidationStatus.FAILED
    assert res.is_valid is False

def test_validation_malformed_gstin(validator, sample_clean_extraction):
    sample_clean_extraction.vendor_gstin = "INVALID_GST_123"
    res = validator.validate(sample_clean_extraction)
    gst_rule = next(r for r in res.rules_evaluated if r.rule_id == RuleID.GSTIN_VALIDATION)
    assert gst_rule.status == ValidationStatus.FAILED

def test_validation_missing_required_fields(validator):
    empty_inv = InvoiceExtraction()
    res = validator.validate(empty_inv)
    req_rule = next(r for r in res.rules_evaluated if r.rule_id == RuleID.REQUIRED_FIELDS)
    assert req_rule.status == ValidationStatus.FAILED

def test_validation_approval_limit_breach(validator, sample_clean_extraction):
    sample_clean_extraction.grand_total = 250000.0  # Above 200,000 INR clerk limit
    res = validator.validate(sample_clean_extraction, approver_role="AP-Clerk-01")
    app_rule = next(r for r in res.rules_evaluated if r.rule_id == RuleID.APPROVAL_LIMIT_CHECK)
    assert app_rule.status == ValidationStatus.FAILED
