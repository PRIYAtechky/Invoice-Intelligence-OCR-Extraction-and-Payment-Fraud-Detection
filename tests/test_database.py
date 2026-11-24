"""Tests for Database Repository and Audit Trail."""

import pytest
from src.database.repository import InvoiceRepository
from src.validation.validator import ValidationEngine
from src.fraud.fraud_detector import PaymentFraudDetector

def test_create_and_get_invoice(db_session, sample_clean_extraction):
    validator = ValidationEngine()
    fraud_detector = PaymentFraudDetector()

    val_res = validator.validate(sample_clean_extraction)
    frd_res = fraud_detector.detect_fraud(sample_clean_extraction)

    repo = InvoiceRepository(db_session)
    inv = repo.create_invoice(
        extraction=sample_clean_extraction,
        validation=val_res,
        fraud=frd_res,
        input_method="upload",
        file_name="test_inv.jpg"
    )

    assert inv.id is not None
    fetched = repo.get_invoice(inv.id)
    assert fetched is not None
    assert fetched.vendor_name == "Vel Engineering Works"
    assert len(fetched.line_items) == 1
    assert len(fetched.validations) > 0

def test_invoice_review_and_audit(db_session, sample_clean_extraction):
    validator = ValidationEngine()
    fraud_detector = PaymentFraudDetector()
    val_res = validator.validate(sample_clean_extraction)
    frd_res = fraud_detector.detect_fraud(sample_clean_extraction)

    repo = InvoiceRepository(db_session)
    inv = repo.create_invoice(sample_clean_extraction, val_res, frd_res)

    # Perform review
    updated = repo.update_invoice_review(
        invoice_id=inv.id,
        decision="APPROVED",
        reviewer_name="Finance-Head",
        comments="Approved after 3-way match verification."
    )

    assert updated.status == "APPROVED"
    assert updated.reviewer_decision == "APPROVED"
    
    # Verify audit event logged
    audits = repo.get_audit_trail_for_invoice(inv.id)
    assert len(audits) >= 2  # CREATED and REVIEW_APPROVED
    assert audits[-1].action == "REVIEW_APPROVED"
    assert audits[-1].user == "Finance-Head"
