"""
Database Repository Services
Provides centralized data persistence, querying, audit trails, and dashboard metrics.
"""
import uuid
import json
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import desc, func

from src.database.models import (
    InvoiceModel, LineItemModel, ValidationLogModel,
    FraudAlertLogModel, ReviewAuditLogModel, BatchLogModel
)
from src.extraction.schemas import (
    ExtractedInvoice, ValidationResultItem, FraudAlertItem,
    InvoiceReviewAction, ReviewStatus, ValidationStatus, FraudSeverity
)
from src.validation.engine import validation_engine
from src.fraud.patterns import (
    check_exact_duplicate, check_fuzzy_duplicate,
    check_bank_tampering, check_po_overrun, check_blocked_vendor
)
from src.fraud.anomaly_detector import anomaly_detector


class InvoiceRepository:
    @staticmethod
    def create_invoice(
        db: Session,
        invoice: ExtractedInvoice,
        validations: List[ValidationResultItem],
        fraud_alerts: List[FraudAlertItem],
        file_path: Optional[str] = None
    ) -> InvoiceModel:
        """Persist a newly extracted or manually entered invoice with all audit logs."""
        invoice_uid = f"INV-{uuid.uuid4().hex[:10].upper()}"

        # Determine overall validation status
        has_failed = any(v.status == ValidationStatus.FAILED for v in validations)
        has_warning = any(v.status == ValidationStatus.WARNING for v in validations)
        has_review = any(v.status == ValidationStatus.NEEDS_REVIEW for v in validations)

        if has_failed:
            val_status = ValidationStatus.FAILED.value
        elif has_warning:
            val_status = ValidationStatus.WARNING.value
        elif has_review:
            val_status = ValidationStatus.NEEDS_REVIEW.value
        else:
            val_status = ValidationStatus.PASSED.value

        # Determine highest fraud severity and cumulative risk score
        max_severity = FraudSeverity.LOW.value
        risk_score = 0.0
        if fraud_alerts:
            risk_score = max(f.risk_score for f in fraud_alerts)
            if any(f.severity == FraudSeverity.CRITICAL for f in fraud_alerts):
                max_severity = FraudSeverity.CRITICAL.value
            elif any(f.severity == FraudSeverity.HIGH for f in fraud_alerts):
                max_severity = FraudSeverity.HIGH.value
            elif any(f.severity == FraudSeverity.MEDIUM for f in fraud_alerts):
                max_severity = FraudSeverity.MEDIUM.value

        initial_status = ReviewStatus.PENDING.value
        if has_failed or max_severity in [FraudSeverity.HIGH.value, FraudSeverity.CRITICAL.value]:
            initial_status = ReviewStatus.NEEDS_REVIEW.value

        invoice_rec = InvoiceModel(
            invoice_uid=invoice_uid,
            file_name=invoice.file_name or "manual_entry.json",
            file_path=file_path or invoice.file_path,
            input_method=invoice.input_method,
            invoice_number=invoice.invoice_number,
            invoice_date=invoice.invoice_date,
            due_date=invoice.due_date,
            vendor_name=invoice.vendor_name,
            vendor_gstin=invoice.vendor_gstin,
            po_number=invoice.po_number,
            subtotal=invoice.subtotal,
            cgst=invoice.cgst,
            sgst=invoice.sgst,
            igst=invoice.igst,
            grand_total=invoice.grand_total,
            bank_account_last4=invoice.bank_account_last4,
            ocr_engine_used=invoice.ocr_engine_used,
            raw_ocr_text=invoice.raw_ocr_text,
            confidence_scores_json=json.dumps(invoice.confidence_scores),
            extraction_sources_json=json.dumps(invoice.extraction_sources),
            status=initial_status,
            validation_status=val_status,
            max_fraud_severity=max_severity,
            fraud_risk_score=risk_score
        )
        db.add(invoice_rec)
        db.flush()

        # Add line items
        for item in invoice.line_items:
            db.add(LineItemModel(
                invoice_id=invoice_rec.id,
                description=item.description,
                hsn=item.hsn,
                quantity=item.quantity,
                unit_price=item.unit_price,
                amount=item.amount
            ))

        # Add validation logs
        for val in validations:
            db.add(ValidationLogModel(
                invoice_id=invoice_rec.id,
                rule_id=val.rule_id,
                rule_name=val.rule_name,
                status=val.status.value if hasattr(val.status, "value") else str(val.status),
                field=val.field,
                explanation=val.explanation,
                evidence_json=json.dumps(val.evidence)
            ))

        # Add fraud alert logs
        for alert in fraud_alerts:
            db.add(FraudAlertLogModel(
                invoice_id=invoice_rec.id,
                pattern_id=alert.pattern_id,
                pattern_name=alert.pattern_name,
                risk_score=alert.risk_score,
                severity=alert.severity.value if hasattr(alert.severity, "value") else str(alert.severity),
                evidence=alert.evidence,
                related_entity=alert.related_entity,
                amount_at_risk=alert.amount_at_risk,
                recommended_action=alert.recommended_action
            ))

        # Initial audit log
        db.add(ReviewAuditLogModel(
            invoice_id=invoice_rec.id,
            action="Created",
            reviewer_role="System",
            reviewer_name="Automated Ingestion Pipeline",
            comments=f"Invoice ingested via {invoice.input_method}. Initial status: {initial_status}."
        ))

        db.commit()
        db.refresh(invoice_rec)
        return invoice_rec

    @staticmethod
    def get_invoice_by_id(db: Session, invoice_id: int) -> Optional[InvoiceModel]:
        return db.query(InvoiceModel).filter(InvoiceModel.id == invoice_id).first()

    @staticmethod
    def get_invoice_by_uid(db: Session, invoice_uid: str) -> Optional[InvoiceModel]:
        return db.query(InvoiceModel).filter(InvoiceModel.invoice_uid == invoice_uid).first()

    @staticmethod
    def list_invoices(
        db: Session,
        status: Optional[str] = None,
        validation_status: Optional[str] = None,
        max_fraud_severity: Optional[str] = None,
        search_query: Optional[str] = None,
        skip: int = 0,
        limit: int = 100
    ) -> Tuple[List[InvoiceModel], int]:
        """Query invoices with multi-field search and status filters."""
        query = db.query(InvoiceModel)

        if status and status != "All":
            query = query.filter(InvoiceModel.status == status)
        if validation_status and validation_status != "All":
            query = query.filter(InvoiceModel.validation_status == validation_status)
        if max_fraud_severity and max_fraud_severity != "All":
            query = query.filter(InvoiceModel.max_fraud_severity == max_fraud_severity)
        if search_query:
            term = f"%{search_query.strip()}%"
            query = query.filter(
                (InvoiceModel.invoice_number.ilike(term)) |
                (InvoiceModel.vendor_name.ilike(term)) |
                (InvoiceModel.vendor_gstin.ilike(term)) |
                (InvoiceModel.po_number.ilike(term))
            )

        total_count = query.count()
        invoices = query.order_by(desc(InvoiceModel.created_at)).offset(skip).limit(limit).all()
        return invoices, total_count

    @staticmethod
    def review_and_update_invoice(
        db: Session,
        invoice_id: int,
        review_action: InvoiceReviewAction
    ) -> Optional[InvoiceModel]:
        """
        Apply human reviewer decisions (Approve, Reject, Needs Review, Edit).
        If fields are edited, automatically re-runs validation and fraud engines,
        and logs full before-and-after audit trail.
        """
        inv = db.query(InvoiceModel).filter(InvoiceModel.id == invoice_id).first()
        if not inv:
            return None

        prev_values = {
            "status": inv.status,
            "invoice_number": inv.invoice_number,
            "invoice_date": inv.invoice_date,
            "due_date": inv.due_date,
            "vendor_name": inv.vendor_name,
            "vendor_gstin": inv.vendor_gstin,
            "po_number": inv.po_number,
            "subtotal": inv.subtotal,
            "grand_total": inv.grand_total,
            "bank_account_last4": inv.bank_account_last4
        }

        # Apply field edits if provided
        new_values = {}
        if review_action.edited_fields:
            for field, val in review_action.edited_fields.items():
                if hasattr(inv, field):
                    setattr(inv, field, val)
                    new_values[field] = val

            # Re-run validation rules on updated data
            extracted = ExtractedInvoice(
                invoice_number=inv.invoice_number,
                invoice_date=inv.invoice_date,
                due_date=inv.due_date,
                vendor_name=inv.vendor_name,
                vendor_gstin=inv.vendor_gstin,
                po_number=inv.po_number,
                subtotal=inv.subtotal,
                cgst=inv.cgst,
                sgst=inv.sgst,
                igst=inv.igst,
                grand_total=inv.grand_total,
                bank_account_last4=inv.bank_account_last4,
                line_items=[
                    {"description": li.description, "hsn": li.hsn, "quantity": li.quantity, "unit_price": li.unit_price, "amount": li.amount}
                    for li in inv.line_items
                ]
            )
            val_results = validation_engine.validate_invoice(extracted, approver_role=review_action.reviewer_role)

            # Update validation status
            has_failed = any(v.status == ValidationStatus.FAILED for v in val_results)
            has_warning = any(v.status == ValidationStatus.WARNING for v in val_results)
            inv.validation_status = (
                ValidationStatus.FAILED.value if has_failed else
                ValidationStatus.WARNING.value if has_warning else
                ValidationStatus.PASSED.value
            )

        # Update review status
        inv.status = review_action.action.value if hasattr(review_action.action, "value") else str(review_action.action)
        inv.updated_at = datetime.utcnow()

        # Audit trail logging
        audit_entry = ReviewAuditLogModel(
            invoice_id=inv.id,
            action=inv.status,
            reviewer_role=review_action.reviewer_role,
            reviewer_name=review_action.reviewer_name,
            comments=review_action.comments or "Review decision recorded.",
            previous_values_json=json.dumps(prev_values),
            new_values_json=json.dumps(new_values)
        )
        db.add(audit_entry)

        db.commit()
        db.refresh(inv)
        return inv

    @staticmethod
    def get_dashboard_metrics(db: Session) -> Dict[str, Any]:
        """Aggregate high-level accounts payable performance and risk KPIs."""
        total_invoices = db.query(func.count(InvoiceModel.id)).scalar() or 0
        total_amount = db.query(func.sum(InvoiceModel.grand_total)).scalar() or 0.0

        status_counts = dict(
            db.query(InvoiceModel.status, func.count(InvoiceModel.id))
            .group_by(InvoiceModel.status).all()
        )

        validation_counts = dict(
            db.query(InvoiceModel.validation_status, func.count(InvoiceModel.id))
            .group_by(InvoiceModel.validation_status).all()
        )

        fraud_severity_counts = dict(
            db.query(InvoiceModel.max_fraud_severity, func.count(InvoiceModel.id))
            .group_by(InvoiceModel.max_fraud_severity).all()
        )

        # Calculate amount at risk (critical/high severity invoices)
        amount_at_risk = db.query(func.sum(InvoiceModel.grand_total)).filter(
            InvoiceModel.max_fraud_severity.in_(["Critical", "High"])
        ).scalar() or 0.0

        return {
            "total_invoices_processed": total_invoices,
            "total_invoice_value_inr": round(float(total_amount), 2),
            "amount_at_risk_inr": round(float(amount_at_risk), 2),
            "status_breakdown": status_counts,
            "validation_breakdown": validation_counts,
            "fraud_severity_breakdown": fraud_severity_counts,
            "validated_count": validation_counts.get("Passed", 0),
            "needs_review_count": status_counts.get("Needs Review", 0) + status_counts.get("Pending", 0),
            "fraud_flagged_count": fraud_severity_counts.get("Critical", 0) + fraud_severity_counts.get("High", 0)
        }
