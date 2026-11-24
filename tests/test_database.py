"""
Tests for Database Operations and Audit Trail
"""
from src.database.session import SessionLocal, init_db
from src.database.repository import InvoiceRepository
from src.extraction.schemas import (
    ExtractedInvoice, ValidationResultItem, FraudAlertItem,
    InvoiceReviewAction, ValidationStatus, FraudSeverity, ReviewStatus
)


def test_database_crud_and_audit():
    init_db()
    db = SessionLocal()
    try:
        inv_data = ExtractedInvoice(
            invoice_number="INV-TEST-999",
            vendor_name="Venkateswara Bearings & Co",
            vendor_gstin="37RLOBV9249G1Z7",
            subtotal=5000.0,
            grand_total=5900.0,
            input_method="manual"
        )
        vals = [
            ValidationResultItem(
                rule_id="ARITH_001",
                rule_name="Arithmetic",
                status=ValidationStatus.PASSED,
                field="grand_total",
                explanation="Valid"
            )
        ]
        frauds = []

        # Create
        created = InvoiceRepository.create_invoice(db, inv_data, vals, frauds)
        assert created.id is not None
        assert created.invoice_number == "INV-TEST-999"
        assert created.status == "Pending"

        # Review action
        action = InvoiceReviewAction(
            action=ReviewStatus.APPROVED,
            reviewer_role="AP-Manager-01",
            reviewer_name="Audit Manager",
            comments="Verified and approved."
        )
        updated = InvoiceRepository.review_and_update_invoice(db, created.id, action)
        assert updated.status == "Approved"
        assert len(updated.audit_logs) >= 2  # Creation log + Approval log

    finally:
        db.close()
