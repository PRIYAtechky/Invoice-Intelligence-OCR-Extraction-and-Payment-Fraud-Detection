"""SQLAlchemy Database Models for Invoice Intelligence."""

from datetime import datetime
from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, Text, ForeignKey
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

class InvoiceModel(Base):
    __tablename__ = "invoices"

    id = Column(String(64), primary_key=True, index=True)
    file_name = Column(String(255), nullable=True)
    original_image_path = Column(String(512), nullable=True)
    processed_image_path = Column(String(512), nullable=True)
    input_method = Column(String(32), default="upload")  # 'upload', 'manual', 'batch'
    status = Column(String(32), default="PENDING_REVIEW")  # 'PENDING_REVIEW', 'APPROVED', 'REJECTED', 'FLAGGED'
    extraction_source = Column(String(32), default="rule")  # 'rule', 'gemini', 'hybrid', 'manual'
    
    # Financial and Entity Fields
    vendor_name = Column(String(255), nullable=True, index=True)
    vendor_gstin = Column(String(32), nullable=True, index=True)
    invoice_number = Column(String(128), nullable=True, index=True)
    invoice_date = Column(String(32), nullable=True, index=True)
    due_date = Column(String(32), nullable=True)
    po_number = Column(String(64), nullable=True, index=True)
    subtotal = Column(Float, nullable=True)
    cgst = Column(Float, default=0.0)
    sgst = Column(Float, default=0.0)
    igst = Column(Float, default=0.0)
    grand_total = Column(Float, nullable=True)
    bank_account_last4 = Column(String(16), nullable=True)

    # Validation and Fraud Flags
    validation_status = Column(String(32), default="Passed")  # 'Passed', 'Warning', 'Failed', 'Needs review'
    is_valid = Column(Boolean, default=True)
    has_fraud_risk = Column(Boolean, default=False)
    fraud_risk_score = Column(Float, default=0.0)
    fraud_severity = Column(String(32), default="LOW")
    amount_at_risk = Column(Float, default=0.0)

    # Metadata & Text
    confidence_scores = Column(Text, nullable=True)  # JSON
    raw_ocr_text = Column(Text, nullable=True)
    
    # Reviewer decision
    reviewer_decision = Column(String(32), nullable=True)
    reviewer_name = Column(String(128), nullable=True)
    reviewer_comments = Column(Text, nullable=True)
    reviewed_at = Column(DateTime, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    line_items = relationship("LineItemModel", back_populates="invoice", cascade="all, delete-orphan", lazy="selectin")
    validations = relationship("ValidationResultModel", back_populates="invoice", cascade="all, delete-orphan", lazy="selectin")
    fraud_alerts = relationship("FraudAlertModel", back_populates="invoice", cascade="all, delete-orphan", lazy="selectin")
    audit_events = relationship("AuditTrailModel", back_populates="invoice", cascade="all, delete-orphan", lazy="selectin")

class LineItemModel(Base):
    __tablename__ = "line_items"

    id = Column(Integer, primary_key=True, autoincrement=True)
    invoice_id = Column(String(64), ForeignKey("invoices.id"), index=True)
    description = Column(String(512), nullable=False)
    hsn = Column(String(32), nullable=True)
    quantity = Column(Float, default=1.0)
    unit_price = Column(Float, default=0.0)
    amount = Column(Float, default=0.0)

    invoice = relationship("InvoiceModel", back_populates="line_items", lazy="joined")

class ValidationResultModel(Base):
    __tablename__ = "validation_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    invoice_id = Column(String(64), ForeignKey("invoices.id"), index=True)
    rule_id = Column(String(32), nullable=False)
    rule_name = Column(String(128), nullable=False)
    status = Column(String(32), nullable=False)
    affected_field = Column(String(64), nullable=False)
    message = Column(Text, nullable=False)
    evidence = Column(Text, nullable=True)  # JSON

    invoice = relationship("InvoiceModel", back_populates="validations", lazy="joined")

class FraudAlertModel(Base):
    __tablename__ = "fraud_alerts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    invoice_id = Column(String(64), ForeignKey("invoices.id"), index=True)
    pattern_id = Column(String(32), nullable=False)
    pattern_name = Column(String(128), nullable=False)
    severity = Column(String(32), nullable=False)
    risk_score = Column(Float, default=0.0)
    explanation = Column(Text, nullable=False)
    evidence = Column(Text, nullable=True)  # JSON
    recommended_action = Column(Text, nullable=False)
    amount_at_risk = Column(Float, default=0.0)

    invoice = relationship("InvoiceModel", back_populates="fraud_alerts", lazy="joined")

class AuditTrailModel(Base):
    __tablename__ = "audit_trail"

    id = Column(Integer, primary_key=True, autoincrement=True)
    invoice_id = Column(String(64), ForeignKey("invoices.id"), index=True)
    action = Column(String(64), nullable=False)
    user = Column(String(128), default="System")
    old_values = Column(Text, nullable=True)  # JSON
    new_values = Column(Text, nullable=True)  # JSON
    comments = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

    invoice = relationship("InvoiceModel", back_populates="audit_events", lazy="joined")

class BatchModel(Base):
    __tablename__ = "batches"

    id = Column(String(64), primary_key=True, index=True)
    batch_name = Column(String(255), nullable=False)
    total_files = Column(Integer, default=0)
    processed_count = Column(Integer, default=0)
    success_count = Column(Integer, default=0)
    failed_count = Column(Integer, default=0)
    status = Column(String(32), default="IN_PROGRESS")  # 'IN_PROGRESS', 'COMPLETED', 'FAILED'
    error_summary = Column(Text, nullable=True)  # JSON
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
