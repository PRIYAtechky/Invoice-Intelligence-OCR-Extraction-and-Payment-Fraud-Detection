"""
Tests for Information Extraction (Rule-Based & Hybrid)
"""
from src.extraction.rule_based import RuleBasedExtractor
from src.extraction.schemas import ExtractedInvoice


def test_rule_based_currency_normalization():
    extractor = RuleBasedExtractor()
    assert extractor.parse_indian_amount("1,23,456.00") == 123456.00
    assert extractor.parse_indian_amount("₹ 2,914.00-") == 2914.00
    assert extractor.parse_indian_amount("Rs. 45,000.50") == 45000.50
    assert extractor.parse_indian_amount("3,438.52") == 3438.52


def test_rule_based_date_normalization():
    extractor = RuleBasedExtractor()
    assert extractor.normalize_date("Date: 10-10-2025") == "2025-10-10"
    assert extractor.normalize_date("2025/11/24") == "2025-11-24"
    assert extractor.normalize_date("15 Oct 2025") == "2025-10-15"


def test_rule_based_text_extraction():
    sample_ocr = """
    Vel Engineering Works
    GST No: 33XXYEN7492X1Z2
    Invoice #: TI/2025/8093
    Date: 10-10-2025
    Payment Due: 24-11-2025
    Your PO: KIS/PO/2025/30592
    
    Coupling Spider L-100 4016 20 145.70 2,914.00
    Sub Total 2,914.00
    CGST @9% 262.26
    SGST @9% 262.26
    Total 3,438.52
    Bank Details: A/C XXXX5029
    """

    extractor = RuleBasedExtractor()
    inv = extractor.extract(sample_ocr)

    assert inv.vendor_name == "Vel Engineering Works"
    assert inv.vendor_gstin == "33XXYEN7492X1Z2"
    assert inv.invoice_number == "TI/2025/8093"
    assert inv.po_number == "KIS/PO/2025/30592"
    assert inv.invoice_date == "2025-10-10"
    assert inv.subtotal == 2914.00
    assert inv.cgst == 262.26
    assert inv.sgst == 262.26
    assert inv.grand_total == 3438.52
    assert inv.bank_account_last4 == "5029"
