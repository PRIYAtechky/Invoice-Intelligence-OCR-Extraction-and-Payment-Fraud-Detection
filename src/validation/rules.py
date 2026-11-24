"""Validation Rules Definition and Enumerations."""

from enum import Enum
from typing import Dict, Any

class ValidationStatus(str, Enum):
    PASSED = "Passed"
    WARNING = "Warning"
    FAILED = "Failed"
    NEEDS_REVIEW = "Needs review"

class RuleID(str, Enum):
    REQUIRED_FIELDS = "VAL-01"
    ARITHMETIC_CHECK = "VAL-02"
    GSTIN_VALIDATION = "VAL-03"
    VENDOR_MASTER_MATCH = "VAL-04"
    PURCHASE_ORDER_MATCH = "VAL-05"
    BANK_ACCOUNT_MATCH = "VAL-06"
    HSN_CATALOG_CHECK = "VAL-07"
    APPROVAL_LIMIT_CHECK = "VAL-08"
    DUPLICATE_CHECK = "VAL-09"

RULE_METADATA: Dict[str, Dict[str, Any]] = {
    RuleID.REQUIRED_FIELDS: {
        "name": "Mandatory Fields Completeness",
        "description": "Verifies presence of invoice number, date, vendor, GSTIN, and grand total.",
        "severity": "CRITICAL"
    },
    RuleID.ARITHMETIC_CHECK: {
        "name": "Arithmetic & Total Reconciliation",
        "description": "Verifies subtotal + taxes == grand total and line item sums match subtotal.",
        "severity": "CRITICAL"
    },
    RuleID.GSTIN_VALIDATION: {
        "name": "GSTIN Structure & Validity",
        "description": "Validates 15-character GSTIN format, state code, PAN structure, and check character.",
        "severity": "HIGH"
    },
    RuleID.VENDOR_MASTER_MATCH: {
        "name": "Vendor Master Registration & Status",
        "description": "Matches vendor legal name and GSTIN against Vendor Master; checks Active status.",
        "severity": "CRITICAL"
    },
    RuleID.PURCHASE_ORDER_MATCH: {
        "name": "3-Way PO Matching & Budget",
        "description": "Matches PO existence, vendor consistency, and checks for budget overruns.",
        "severity": "HIGH"
    },
    RuleID.BANK_ACCOUNT_MATCH: {
        "name": "Bank Account Verification",
        "description": "Ensures invoice bank account matches registered bank details on file.",
        "severity": "CRITICAL"
    },
    RuleID.HSN_CATALOG_CHECK: {
        "name": "HSN & GST Rate Compliance",
        "description": "Validates HSN code in master catalog and verifies compliant GST tax rates.",
        "severity": "MEDIUM"
    },
    RuleID.APPROVAL_LIMIT_CHECK: {
        "name": "Approval Matrix Authority Limit",
        "description": "Ensures approver role has adequate financial authority for invoice value.",
        "severity": "HIGH"
    },
    RuleID.DUPLICATE_CHECK: {
        "name": "Duplicate Invoice Detection",
        "description": "Checks against historical AP ledger and active records for identical invoice numbers.",
        "severity": "CRITICAL"
    }
}
