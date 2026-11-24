"""Pydantic Data Models and Schemas."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime

class LineItem(BaseModel):
    description: str = Field(..., description="Item or service description")
    hsn: Optional[str] = Field(None, description="HSN/SAC code")
    quantity: float = Field(default=1.0, description="Quantity delivered/billed")
    unit_price: float = Field(default=0.0, description="Unit rate in INR")
    amount: float = Field(default=0.0, description="Total line amount before tax")

class InvoiceExtraction(BaseModel):
    invoice_id: Optional[str] = None
    file_name: Optional[str] = None
    vendor_name: Optional[str] = None
    vendor_gstin: Optional[str] = None
    invoice_number: Optional[str] = None
    invoice_date: Optional[str] = None
    due_date: Optional[str] = None
    po_number: Optional[str] = None
    line_items: List[LineItem] = Field(default_factory=list)
    subtotal: Optional[float] = None
    cgst: Optional[float] = None
    sgst: Optional[float] = None
    igst: Optional[float] = None
    grand_total: Optional[float] = None
    bank_account_last4: Optional[str] = None
    extraction_source: str = "rule"  # 'rule', 'gemini', 'hybrid', 'manual'
    confidence_scores: Dict[str, float] = Field(default_factory=dict)
    raw_ocr_text: Optional[str] = None
    needs_review_fields: List[str] = Field(default_factory=list)

class ValidationRuleResult(BaseModel):
    rule_id: str
    rule_name: str
    status: str  # 'PASSED', 'WARNING', 'FAILED', 'NEEDS_REVIEW'
    affected_field: str
    message: str
    evidence: Dict[str, Any] = Field(default_factory=dict)

class FraudAlert(BaseModel):
    pattern_id: str
    pattern_name: str
    severity: str  # 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
    risk_score: float  # 0.0 - 100.0
    explanation: str
    evidence: Dict[str, Any] = Field(default_factory=dict)
    recommended_action: str
    amount_at_risk: float = 0.0

class InvoiceValidationSummary(BaseModel):
    is_valid: bool
    status: str  # 'Passed', 'Warning', 'Failed', 'Needs review'
    passed_count: int
    warning_count: int
    failed_count: int
    needs_review_count: int
    rules_evaluated: List[ValidationRuleResult] = Field(default_factory=list)

class InvoiceFraudSummary(BaseModel):
    has_fraud_risk: bool
    composite_risk_score: float
    max_severity: str
    total_amount_at_risk: float
    alerts: List[FraudAlert] = Field(default_factory=list)

class InvoiceRecord(BaseModel):
    id: str
    file_name: Optional[str]
    created_at: datetime
    updated_at: datetime
    input_method: str  # 'upload', 'manual', 'batch'
    status: str  # 'PENDING_REVIEW', 'APPROVED', 'REJECTED', 'FLAGGED'
    extraction: InvoiceExtraction
    validation: InvoiceValidationSummary
    fraud: InvoiceFraudSummary
    reviewer_decision: Optional[str] = None
    reviewer_name: Optional[str] = None
    reviewer_comments: Optional[str] = None
    reviewed_at: Optional[datetime] = None

class ManualInvoiceInput(BaseModel):
    vendor_name: str
    vendor_gstin: str
    invoice_number: str
    invoice_date: str
    due_date: Optional[str] = None
    po_number: Optional[str] = None
    line_items: List[LineItem] = Field(default_factory=list)
    subtotal: float
    cgst: float = 0.0
    sgst: float = 0.0
    igst: float = 0.0
    grand_total: float
    bank_account_last4: Optional[str] = None
