"""
Pydantic Schemas for Standardized Invoice Intelligence Data Structures
Used across Preprocessing, OCR, Extraction, Validation, Fraud Detection, API, and UI.
"""
from typing import List, Optional, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field, field_validator


class ExtractionSource(str, Enum):
    RULE = "rule"
    GEMINI = "gemini"
    HYBRID = "hybrid"
    MANUAL = "manual"


class ValidationStatus(str, Enum):
    PASSED = "Passed"
    WARNING = "Warning"
    FAILED = "Failed"
    NEEDS_REVIEW = "Needs Review"


class FraudSeverity(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"


class ReviewStatus(str, Enum):
    PENDING = "Pending"
    APPROVED = "Approved"
    REJECTED = "Rejected"
    NEEDS_REVIEW = "Needs Review"
    EDITED = "Edited"


class LineItem(BaseModel):
    description: str = Field(..., description="Item or service description")
    hsn: Optional[str] = Field(None, description="HSN/SAC Code")
    quantity: float = Field(default=1.0, description="Quantity supplied")
    unit_price: float = Field(default=0.0, description="Unit price before tax")
    amount: float = Field(default=0.0, description="Line total amount before tax")

    @field_validator("quantity", "unit_price", "amount", mode="before")
    @classmethod
    def parse_float(cls, v):
        if v is None:
            return 0.0
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            clean = v.replace(",", "").replace("₹", "").strip()
            try:
                return float(clean)
            except ValueError:
                return 0.0
        return 0.0


class ExtractedInvoice(BaseModel):
    """Unified invoice representation produced by OCR/extraction and manual entry."""
    invoice_number: Optional[str] = None
    invoice_date: Optional[str] = None  # Expected format YYYY-MM-DD
    due_date: Optional[str] = None      # Expected format YYYY-MM-DD
    vendor_name: Optional[str] = None
    vendor_gstin: Optional[str] = None
    po_number: Optional[str] = None
    
    subtotal: float = 0.0
    cgst: float = 0.0
    sgst: float = 0.0
    igst: float = 0.0
    grand_total: float = 0.0
    
    bank_account_last4: Optional[str] = None
    line_items: List[LineItem] = Field(default_factory=list)
    
    raw_ocr_text: Optional[str] = None
    ocr_engine_used: Optional[str] = None
    confidence_scores: Dict[str, float] = Field(default_factory=dict)
    extraction_sources: Dict[str, str] = Field(default_factory=dict)
    
    # Metadata
    file_name: Optional[str] = None
    file_path: Optional[str] = None
    degradation_level: Optional[float] = None
    input_method: str = "upload"  # "upload" or "manual"


class ManualInvoiceInput(BaseModel):
    """Schema for manual invoice entry form."""
    vendor_name: str
    vendor_gstin: str
    invoice_number: str
    invoice_date: str
    due_date: Optional[str] = None
    po_number: Optional[str] = None
    subtotal: float
    cgst: float = 0.0
    sgst: float = 0.0
    igst: float = 0.0
    grand_total: float
    bank_account_last4: Optional[str] = None
    line_items: List[LineItem] = Field(default_factory=list)


class ValidationResultItem(BaseModel):
    rule_id: str
    rule_name: str
    status: ValidationStatus
    field: str
    explanation: str
    evidence: Dict[str, Any] = Field(default_factory=dict)


class FraudAlertItem(BaseModel):
    pattern_id: str
    pattern_name: str
    risk_score: float = Field(..., ge=0.0, le=100.0)
    severity: FraudSeverity
    evidence: str
    related_entity: Optional[str] = None
    amount_at_risk: float = 0.0
    recommended_action: str


class InvoiceReviewAction(BaseModel):
    action: ReviewStatus
    reviewer_role: str = "AP-Clerk-01"
    reviewer_name: str = "AP Reviewer"
    comments: Optional[str] = None
    edited_fields: Optional[Dict[str, Any]] = None


class BatchProcessSummary(BaseModel):
    batch_id: str
    total_files: int
    successful_count: int
    failed_count: int
    flagged_fraud_count: int
    flagged_validation_count: int
    total_amount_processed: float
    files_processed: List[Dict[str, Any]] = Field(default_factory=list)
