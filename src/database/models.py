"""
SQLAlchemy Relational Database Models
Implements normalized tables for Invoices, Line Items, Validation Logs,
Fraud Alerts, Review Decisions, Audit Trails, and Batch Processing Logs.
"""
from datetime import datetime
import json
from sqlalchemy import Column, Integer, String, Float, Text, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import relationship, declarative_base

Base = declarative_base()


class InvoiceModel(Base):
    __tablename__ = "invoices"

    id = Column(Integer, primary_key=True, autoincrement=True)
    invoice_uid = Column(String(64), unique=True, index=True, nullable=False)
    file_name = Column(String(255), nullable=True)
    file_path = Column(String(512), nullable=True)
    input_method = Column(String(32), default="upload")  # "upload" or "manual"
    
    invoice_number = Column(String(128), index=True, nullable=True)
    invoice_date = Column(String(32), nullable=True)
    due_date = Column(String(32), nullable=True)
    vendor_name = Column(String(255), index=True, nullable=True)
    vendor_gstin = Column(String(64), index=True, nullable=True)
    po_number = Column(String(128), index=True, nullable=True)

    subtotal = Column(Float, default=0.0)
    cgst = Column(Float, default=0.0)
    sgst = Column(Float, default=0.0)
    igst = Column(Float, default=0.0)
    grand_total = Column(Float, default=0.0)
    bank_account_last4 = Column(String(16), nullable=True)

    ocr_engine_used = Column(String(64), nullable=True)
    raw_ocr_text = Column(Text, nullable=True)
    confidence_scores_json = Column(Text, nullable=True)
    extraction_sources_json = Column(Text, nullable=True)

    status = Column(String(32), default="Pending", index=True)  # Pending, Approved, Rejected, Needs Review, Edited
    validation_status = Column(String(32), default="Passed", index=True)  # Passed, Warning, Failed, Needs Review
    max_fraud_severity = Column(String(32), default="Low")  # Low, Medium, High, Critical
    fraud_risk_score = Column(Float, default=0.0)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    line_items = relationship("LineItemModel", back_populates="invoice", cascade="all, delete-orphan")
    validation_logs = relationship("ValidationLogModel", back_populates="invoice", cascade="all, delete-orphan")
    fraud_alerts = relationship("FraudAlertLogModel", back_populates="invoice", cascade="all, delete-orphan")
    audit_logs = relationship("ReviewAuditLogModel", back_populates="invoice", cascade="all, delete-orphan")

    def to_dict(self):
        return {
            "id": self.id,
            "invoice_uid": self.invoice_uid,
            "file_name": self.file_name,
            "input_method": self.input_method,
            "invoice_number": self.invoice_number,
            "invoice_date": self.invoice_date,
            "due_date": self.due_date,
            "vendor_name": self.vendor_name,
            "vendor_gstin": self.vendor_gstin,
            "po_number": self.po_number,
            "subtotal": self.subtotal,
            "cgst": self.cgst,
            "sgst": self.sgst,
            "igst": self.igst,
            "grand_total": self.grand_total,
            "bank_account_last4": self.bank_account_last4,
            "ocr_engine_used": self.ocr_engine_used,
            "status": self.status,
            "validation_status": self.validation_status,
            "max_fraud_severity": self.max_fraud_severity,
            "fraud_risk_score": self.fraud_risk_score,
            "created_at": self.created_at.strftime("%Y-%m-%d %H:%M:%S") if self.created_at else None,
            "line_items": [item.to_dict() for item in self.line_items],
            "validation_logs": [log.to_dict() for log in self.validation_logs],
            "fraud_alerts": [alert.to_dict() for alert in self.fraud_alerts],
            "audit_logs": [audit.to_dict() for audit in self.audit_logs]
        }


class LineItemModel(Base):
    __tablename__ = "line_items"

    id = Column(Integer, primary_key=True, autoincrement=True)
    invoice_id = Column(Integer, ForeignKey("invoices.id"), nullable=False)
    description = Column(String(255), nullable=False)
    hsn = Column(String(32), nullable=True)
    quantity = Column(Float, default=1.0)
    unit_price = Column(Float, default=0.0)
    amount = Column(Float, default=0.0)

    invoice = relationship("InvoiceModel", back_populates="line_items")

    def to_dict(self):
        return {
            "id": self.id,
            "description": self.description,
            "hsn": self.hsn,
            "quantity": self.quantity,
            "unit_price": self.unit_price,
            "amount": self.amount
        }


class ValidationLogModel(Base):
    __tablename__ = "validation_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    invoice_id = Column(Integer, ForeignKey("invoices.id"), nullable=False)
    rule_id = Column(String(64), nullable=False)
    rule_name = Column(String(128), nullable=False)
    status = Column(String(32), nullable=False)
    field = Column(String(64), nullable=False)
    explanation = Column(Text, nullable=False)
    evidence_json = Column(Text, nullable=True)

    invoice = relationship("InvoiceModel", back_populates="validation_logs")

    def to_dict(self):
        return {
            "rule_id": self.rule_id,
            "rule_name": self.rule_name,
            "status": self.status,
            "field": self.field,
            "explanation": self.explanation,
            "evidence": json.loads(self.evidence_json) if self.evidence_json else {}
        }


class FraudAlertLogModel(Base):
    __tablename__ = "fraud_alerts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    invoice_id = Column(Integer, ForeignKey("invoices.id"), nullable=False)
    pattern_id = Column(String(64), nullable=False)
    pattern_name = Column(String(128), nullable=False)
    risk_score = Column(Float, default=0.0)
    severity = Column(String(32), nullable=False)
    evidence = Column(Text, nullable=False)
    related_entity = Column(String(128), nullable=True)
    amount_at_risk = Column(Float, default=0.0)
    recommended_action = Column(Text, nullable=False)

    invoice = relationship("InvoiceModel", back_populates="fraud_alerts")

    def to_dict(self):
        return {
            "pattern_id": self.pattern_id,
            "pattern_name": self.pattern_name,
            "risk_score": self.risk_score,
            "severity": self.severity,
            "evidence": self.evidence,
            "related_entity": self.related_entity,
            "amount_at_risk": self.amount_at_risk,
            "recommended_action": self.recommended_action
        }


class ReviewAuditLogModel(Base):
    __tablename__ = "review_audit_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    invoice_id = Column(Integer, ForeignKey("invoices.id"), nullable=False)
    action = Column(String(32), nullable=False)
    reviewer_role = Column(String(64), default="AP-Clerk-01")
    reviewer_name = Column(String(128), default="AP Reviewer")
    comments = Column(Text, nullable=True)
    previous_values_json = Column(Text, nullable=True)
    new_values_json = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)

    invoice = relationship("InvoiceModel", back_populates="audit_logs")

    def to_dict(self):
        return {
            "id": self.id,
            "action": self.action,
            "reviewer_role": self.reviewer_role,
            "reviewer_name": self.reviewer_name,
            "comments": self.comments,
            "previous_values": json.loads(self.previous_values_json) if self.previous_values_json else {},
            "new_values": json.loads(self.new_values_json) if self.new_values_json else {},
            "timestamp": self.timestamp.strftime("%Y-%m-%d %H:%M:%S") if self.timestamp else None
        }


class BatchLogModel(Base):
    __tablename__ = "batch_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    batch_id = Column(String(64), unique=True, index=True, nullable=False)
    folder_path = Column(String(512), nullable=False)
    total_files = Column(Integer, default=0)
    successful_count = Column(Integer, default=0)
    failed_count = Column(Integer, default=0)
    total_amount = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "batch_id": self.batch_id,
            "folder_path": self.folder_path,
            "total_files": self.total_files,
            "successful_count": self.successful_count,
            "failed_count": self.failed_count,
            "total_amount": self.total_amount,
            "created_at": self.created_at.strftime("%Y-%m-%d %H:%M:%S") if self.created_at else None
        }
