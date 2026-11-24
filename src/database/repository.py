"""Database Repository for Invoices, Validations, Fraud, and Audit Trail."""

import json
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import desc, func

from src.database.models import (
    InvoiceModel,
    LineItemModel,
    ValidationResultModel,
    FraudAlertModel,
    AuditTrailModel,
    BatchModel
)
from src.extraction.schemas import InvoiceExtraction, InvoiceValidationSummary, InvoiceFraudSummary
from config.logging_config import logger

class InvoiceRepository:
    def __init__(self, session: Session):
        self.session = session

    def create_invoice(
        self,
        extraction: InvoiceExtraction,
        validation: InvoiceValidationSummary,
        fraud: InvoiceFraudSummary,
        input_method: str = "upload",
        file_name: Optional[str] = None,
        original_image_path: Optional[str] = None,
        processed_image_path: Optional[str] = None,
        custom_id: Optional[str] = None
    ) -> InvoiceModel:
        """Create and persist a complete invoice record with child relations."""
        inv_id = custom_id or f"INV-{uuid.uuid4().hex[:8].upper()}"

        # Determine initial review status
        if not validation.is_valid or fraud.has_fraud_risk:
            status = "FLAGGED" if fraud.max_severity in ("CRITICAL", "HIGH") else "PENDING_REVIEW"
        else:
            status = "PENDING_REVIEW"

        invoice = InvoiceModel(
            id=inv_id,
            file_name=file_name or extraction.file_name,
            original_image_path=original_image_path,
            processed_image_path=processed_image_path,
            input_method=input_method,
            status=status,
            extraction_source=extraction.extraction_source,
            vendor_name=extraction.vendor_name,
            vendor_gstin=extraction.vendor_gstin,
            invoice_number=extraction.invoice_number,
            invoice_date=extraction.invoice_date,
            due_date=extraction.due_date,
            po_number=extraction.po_number,
            subtotal=extraction.subtotal,
            cgst=extraction.cgst or 0.0,
            sgst=extraction.sgst or 0.0,
            igst=extraction.igst or 0.0,
            grand_total=extraction.grand_total,
            bank_account_last4=extraction.bank_account_last4,
            validation_status=validation.status,
            is_valid=validation.is_valid,
            has_fraud_risk=fraud.has_fraud_risk,
            fraud_risk_score=fraud.composite_risk_score,
            fraud_severity=fraud.max_severity,
            amount_at_risk=fraud.total_amount_at_risk,
            confidence_scores=json.dumps(extraction.confidence_scores),
            raw_ocr_text=extraction.raw_ocr_text
        )
        self.session.add(invoice)

        # Line items
        for item in extraction.line_items:
            li = LineItemModel(
                invoice_id=inv_id,
                description=item.description,
                hsn=item.hsn,
                quantity=item.quantity,
                unit_price=item.unit_price,
                amount=item.amount
            )
            self.session.add(li)

        # Validation results
        for val in validation.rules_evaluated:
            vr = ValidationResultModel(
                invoice_id=inv_id,
                rule_id=val.rule_id,
                rule_name=val.rule_name,
                status=val.status,
                affected_field=val.affected_field,
                message=val.message,
                evidence=json.dumps(val.evidence)
            )
            self.session.add(vr)

        # Fraud alerts
        for fa in fraud.alerts:
            fam = FraudAlertModel(
                invoice_id=inv_id,
                pattern_id=fa.pattern_id,
                pattern_name=fa.pattern_name,
                severity=fa.severity,
                risk_score=fa.risk_score,
                explanation=fa.explanation,
                evidence=json.dumps(fa.evidence),
                recommended_action=fa.recommended_action,
                amount_at_risk=fa.amount_at_risk
            )
            self.session.add(fam)

        # Audit event
        audit = AuditTrailModel(
            invoice_id=inv_id,
            action="CREATED",
            user="System (OCR Pipeline)",
            new_values=json.dumps({
                "vendor_name": extraction.vendor_name,
                "invoice_number": extraction.invoice_number,
                "grand_total": extraction.grand_total,
                "status": status,
                "validation_status": validation.status,
                "fraud_score": fraud.composite_risk_score
            }),
            comments=f"Invoice ingested via {input_method}."
        )
        self.session.add(audit)
        self.session.flush()
        return invoice

    def get_invoice(self, invoice_id: str) -> Optional[InvoiceModel]:
        """Fetch invoice by primary ID."""
        return self.session.query(InvoiceModel).filter(InvoiceModel.id == invoice_id).first()

    def list_invoices(
        self,
        status: Optional[str] = None,
        vendor: Optional[str] = None,
        has_fraud: Optional[bool] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[InvoiceModel]:
        """Query invoices with optional filters."""
        query = self.session.query(InvoiceModel)
        if status:
            query = query.filter(InvoiceModel.status == status)
        if vendor:
            query = query.filter(InvoiceModel.vendor_name.ilike(f"%{vendor}%"))
        if has_fraud is not None:
            query = query.filter(InvoiceModel.has_fraud_risk == has_fraud)
        return query.order_by(desc(InvoiceModel.created_at)).offset(offset).limit(limit).all()

    def update_invoice_review(
        self,
        invoice_id: str,
        decision: str,  # 'APPROVED', 'REJECTED', 'FLAGGED_FOR_REVIEW'
        reviewer_name: str,
        comments: Optional[str] = None,
        edited_fields: Optional[Dict[str, Any]] = None
    ) -> Optional[InvoiceModel]:
        """Update reviewer decision and maintain audit trail."""
        inv = self.get_invoice(invoice_id)
        if not inv:
            return None

        old_state = {
            "status": inv.status,
            "grand_total": inv.grand_total,
            "vendor_name": inv.vendor_name,
            "invoice_number": inv.invoice_number,
            "invoice_date": inv.invoice_date,
            "bank_account_last4": inv.bank_account_last4
        }

        # Apply edits if any
        if edited_fields:
            for field, val in edited_fields.items():
                if hasattr(inv, field):
                    setattr(inv, field, val)

        inv.status = decision
        inv.reviewer_decision = decision
        inv.reviewer_name = reviewer_name
        inv.reviewer_comments = comments
        inv.reviewed_at = datetime.utcnow()
        inv.updated_at = datetime.utcnow()

        new_state = {
            "status": inv.status,
            "grand_total": inv.grand_total,
            "vendor_name": inv.vendor_name,
            "invoice_number": inv.invoice_number,
            "invoice_date": inv.invoice_date,
            "bank_account_last4": inv.bank_account_last4
        }

        # Add audit record
        audit = AuditTrailModel(
            invoice_id=invoice_id,
            action=f"REVIEW_{decision}",
            user=reviewer_name,
            old_values=json.dumps(old_state),
            new_values=json.dumps(new_state),
            comments=comments
        )
        self.session.add(audit)
        self.session.flush()
        return inv

    def get_dashboard_metrics(self) -> Dict[str, Any]:
        """Compute real stored metrics for the Overview Dashboard."""
        total_invoices = self.session.query(func.count(InvoiceModel.id)).scalar() or 0
        total_value = self.session.query(func.sum(InvoiceModel.grand_total)).scalar() or 0.0
        validated_invoices = self.session.query(func.count(InvoiceModel.id)).filter(InvoiceModel.is_valid == True).scalar() or 0
        pending_review = self.session.query(func.count(InvoiceModel.id)).filter(InvoiceModel.status.in_(["PENDING_REVIEW", "FLAGGED"])).scalar() or 0
        fraud_flagged = self.session.query(func.count(InvoiceModel.id)).filter(InvoiceModel.has_fraud_risk == True).scalar() or 0
        money_at_risk = self.session.query(func.sum(InvoiceModel.amount_at_risk)).scalar() or 0.0
        approved_invoices = self.session.query(func.count(InvoiceModel.id)).filter(InvoiceModel.status == "APPROVED").scalar() or 0
        rejected_invoices = self.session.query(func.count(InvoiceModel.id)).filter(InvoiceModel.status == "REJECTED").scalar() or 0

        return {
            "total_invoices_processed": total_invoices,
            "total_invoice_value_inr": round(float(total_value), 2),
            "validated_invoices": validated_invoices,
            "invoices_requiring_review": pending_review,
            "potentially_fraudulent_invoices": fraud_flagged,
            "total_amount_at_risk_inr": round(float(money_at_risk), 2),
            "approved_invoices": approved_invoices,
            "rejected_invoices": rejected_invoices
        }

    def get_audit_trail_for_invoice(self, invoice_id: str) -> List[AuditTrailModel]:
        """Fetch chronological audit events for an invoice."""
        return self.session.query(AuditTrailModel).filter(
            AuditTrailModel.invoice_id == invoice_id
        ).order_by(AuditTrailModel.timestamp.asc()).all()

    def create_batch(self, batch_name: str, total_files: int) -> BatchModel:
        """Create new batch processing tracker."""
        b_id = f"BATCH-{uuid.uuid4().hex[:6].upper()}"
        b = BatchModel(id=b_id, batch_name=batch_name, total_files=total_files)
        self.session.add(b)
        self.session.flush()
        return b

    def update_batch(
        self,
        batch_id: str,
        processed_count: int,
        success_count: int,
        failed_count: int,
        status: str = "IN_PROGRESS",
        error_summary: Optional[Dict[str, Any]] = None
    ) -> Optional[BatchModel]:
        """Update batch progress and stats."""
        b = self.session.query(BatchModel).filter(BatchModel.id == batch_id).first()
        if b:
            b.processed_count = processed_count
            b.success_count = success_count
            b.failed_count = failed_count
            b.status = status
            if error_summary:
                b.error_summary = json.dumps(error_summary)
            if status in ("COMPLETED", "FAILED"):
                b.completed_at = datetime.utcnow()
            self.session.flush()
        return b
