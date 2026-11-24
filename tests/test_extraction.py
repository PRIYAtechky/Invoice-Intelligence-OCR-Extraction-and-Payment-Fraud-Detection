"""Tests for Information Extraction (Rules, Currency Normalization, Regex)."""

import pytest
from src.extraction.rule_extractor import RuleExtractor

@pytest.fixture
def extractor():
    return RuleExtractor()

def test_parse_indian_currency_formats(extractor):
    assert extractor.parse_amount("1,23,456.00") == 123456.00
    assert extractor.parse_amount("4,164.28") == 4164.28
    assert extractor.parse_amount("₹ 3,438.52") == 3438.52
    assert extractor.parse_amount("2,914.00.") == 2914.00
    assert extractor.parse_amount("invalid_text") is None

def test_normalize_date_formats(extractor):
    assert extractor.normalize_date("10-10-2025") == "2025-10-10"
    assert extractor.normalize_date("19/10/2025") == "2025-10-19"
    assert extractor.normalize_date("October 17, 2025") == "2025-10-17"
    assert extractor.normalize_date("2025-11-24") == "2025-11-24"

def test_extract_gstin(extractor):
    lines = ["Vel Engineering Works", "GST No. 33XXYEN7492X1Z2", "INVOICE"]
    gstin, conf = extractor.extract_gstins(lines)
    assert gstin == "33XXYEN7492X1Z2"
    assert conf >= 0.90

def test_buyer_gstin_filter(extractor):
    # Should ignore buyer GSTIN and find vendor GSTIN
    lines = [
        "Vendor Line",
        "GSTIN: 33XXYEN7492X1Z2",
        "Bill To: Kaveri Industrial Supplies",
        "GSTIN: 33AACCK4821M1Z5"
    ]
    gstin, _ = extractor.extract_gstins(lines)
    assert gstin == "33XXYEN7492X1Z2"

def test_extract_invoice_number(extractor):
    lines = ["INVOICE", "Inv #: TI/2025/8093", "Date: 10-10-2025"]
    inv_no, conf = extractor.extract_invoice_number(lines)
    assert inv_no == "TI/2025/8093"
    assert conf >= 0.90

def test_extract_bank_account_last4(extractor):
    lines = ["Bank Details:", "A/c No: XXXXXXXX5029", "IFSC: ICIC0466138"]
    last4, conf = extractor.extract_bank_last4(lines)
    assert last4 == "5029"
    assert conf >= 0.90
