"""FastAPI Routes and REST Endpoints."""

import os
import shutil
import uuid
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status
from sqlalchemy.orm import Session

from src.database.session import get_db
from src.database.repository import InvoiceRepository
from src.extraction.schemas import ManualInvoiceInput, InvoiceRecord
from src.pipeline.invoice_pipeline import InvoiceProcessingPipeline
from src.pipeline.batch_processor import BatchProcessor
from src.fraud.vendor_risk import VendorRiskScorer
from src.extraction.gemini_extractor import GeminiExtractor
from config.settings import settings
from config.logging_config import logger

router = APIRouter(prefix="/api/v1", tags=["Invoice Intelligence API"])

# -------------------------------------------------------------
# Health Check
# -------------------------------------------------------------
@router.get("/health", summary="Service Health & Diagnostic Status")
def health_check():
    return {
        "status": "healthy",
        "app_name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT,
        "default_ocr": settings.DEFAULT_OCR_ENGINE,
        "gemini_configured": bool(settings.GEMINI_API_KEY)
    }

# -------------------------------------------------------------
# Invoice Upload & Processing
# -------------------------------------------------------------
@router.post("/invoices/upload", summary="Upload and Process Invoice Image")
async def upload_invoice(
    file: UploadFile = File(...),
    ocr_engine: Optional[str] = Form(None),
    use_llm: bool = Form(True),
    approver_role: str = Form("AP-Clerk-01"),
    db: Session = Depends(get_db)
):
    if not file.filename.lower().endswith((".jpg", ".jpeg", ".png", ".pdf")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported file format. Please upload JPG, PNG, or PDF."
        )

    # Save to processed upload directory
    save_dir = settings.DATA_DIR / "uploads"
    save_dir.mkdir(parents=True, exist_ok=True)
    temp_filename = f"{uuid.uuid4().hex[:8]}_{file.filename}"
    save_path = save_dir / temp_filename

    try:
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        pipeline = InvoiceProcessingPipeline(session=db)
        result = pipeline.process_image(
            image_path=str(save_path),
            engine_name=ocr_engine,
            use_llm=use_llm,
            input_method="upload",
            approver_role=approver_role
        )
        return {
            "success": True,
            "invoice_id": result["invoice_id"],
            "file_name": file.filename,
            "extraction": result["extraction"],
            "validation": result["validation"],
            "fraud": result["fraud"],
            "ocr_meta": result["ocr_meta"]
        }
    except Exception as e:
        logger.error(f"Upload processing failed: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

# -------------------------------------------------------------
# Manual Invoice Submission
# -------------------------------------------------------------
@router.post("/invoices/manual", summary="Submit Manual Invoice Data")
def submit_manual_invoice(
    data: ManualInvoiceInput,
    approver_role: str = "AP-Clerk-01",
    db: Session = Depends(get_db)
):
    try:
        pipeline = InvoiceProcessingPipeline(session=db)
        result = pipeline.process_manual(data, approver_role=approver_role)
        return {
            "success": True,
            "invoice_id": result["invoice_id"],
            "validation": result["validation"],
            "fraud": result["fraud"]
        }
    except Exception as e:
        logger.error(f"Manual invoice submission failed: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

# -------------------------------------------------------------
# Invoice Retrieval
# -------------------------------------------------------------
@router.get("/invoices/{invoice_id}", summary="Get Stored Invoice by ID")
def get_invoice(invoice_id: str, db: Session = Depends(get_db)):
    repo = InvoiceRepository(db)
    inv = repo.get_invoice(invoice_id)
    if not inv:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Invoice not found.")
    
    return {
        "id": inv.id,
        "file_name": inv.file_name,
        "status": inv.status,
        "vendor_name": inv.vendor_name,
        "vendor_gstin": inv.vendor_gstin,
        "invoice_number": inv.invoice_number,
        "invoice_date": inv.invoice_date,
        "due_date": inv.due_date,
        "po_number": inv.po_number,
        "subtotal": inv.subtotal,
        "cgst": inv.cgst,
        "sgst": inv.sgst,
        "igst": inv.igst,
        "grand_total": inv.grand_total,
        "bank_account_last4": inv.bank_account_last4,
        "validation_status": inv.validation_status,
        "is_valid": inv.is_valid,
        "has_fraud_risk": inv.has_fraud_risk,
        "fraud_risk_score": inv.fraud_risk_score,
        "fraud_severity": inv.fraud_severity,
        "amount_at_risk": inv.amount_at_risk,
        "reviewer_decision": inv.reviewer_decision,
        "reviewer_name": inv.reviewer_name,
        "reviewer_comments": inv.reviewer_comments,
        "created_at": inv.created_at,
        "line_items": [
            {
                "description": li.description,
                "hsn": li.hsn,
                "quantity": li.quantity,
                "unit_price": li.unit_price,
                "amount": li.amount
            } for li in inv.line_items
        ]
    }

# -------------------------------------------------------------
# Invoice Review & Approval
# -------------------------------------------------------------
@router.post("/invoices/{invoice_id}/review", summary="Review, Edit, and Decide on Invoice")
def review_invoice(
    invoice_id: str,
    decision: str = Form(..., description="APPROVED, REJECTED, or FLAGGED_FOR_REVIEW"),
    reviewer_name: str = Form("AP-Reviewer"),
    comments: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    repo = InvoiceRepository(db)
    updated = repo.update_invoice_review(
        invoice_id=invoice_id,
        decision=decision.upper(),
        reviewer_name=reviewer_name,
        comments=comments
    )
    if not updated:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Invoice not found.")
    return {
        "success": True,
        "invoice_id": invoice_id,
        "status": updated.status,
        "reviewer_decision": updated.reviewer_decision,
        "reviewer_name": updated.reviewer_name
    }

# -------------------------------------------------------------
# Validation Results for Invoice
# -------------------------------------------------------------
@router.get("/invoices/{invoice_id}/validation", summary="Get Validation Rule Results")
def get_invoice_validation(invoice_id: str, db: Session = Depends(get_db)):
    repo = InvoiceRepository(db)
    inv = repo.get_invoice(invoice_id)
    if not inv:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Invoice not found.")
    return {
        "invoice_id": invoice_id,
        "validation_status": inv.validation_status,
        "is_valid": inv.is_valid,
        "rules": [
            {
                "rule_id": vr.rule_id,
                "rule_name": vr.rule_name,
                "status": vr.status,
                "affected_field": vr.affected_field,
                "message": vr.message
            } for vr in inv.validations
        ]
    }

# -------------------------------------------------------------
# Fraud Findings for Invoice
# -------------------------------------------------------------
@router.get("/invoices/{invoice_id}/fraud", summary="Get Payment Fraud Alerts")
def get_invoice_fraud(invoice_id: str, db: Session = Depends(get_db)):
    repo = InvoiceRepository(db)
    inv = repo.get_invoice(invoice_id)
    if not inv:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Invoice not found.")
    return {
        "invoice_id": invoice_id,
        "has_fraud_risk": inv.has_fraud_risk,
        "composite_risk_score": inv.fraud_risk_score,
        "max_severity": inv.fraud_severity,
        "amount_at_risk": inv.amount_at_risk,
        "alerts": [
            {
                "pattern_id": fa.pattern_id,
                "pattern_name": fa.pattern_name,
                "severity": fa.severity,
                "risk_score": fa.risk_score,
                "explanation": fa.explanation,
                "recommended_action": fa.recommended_action,
                "amount_at_risk": fa.amount_at_risk
            } for fa in inv.fraud_alerts
        ]
    }

# -------------------------------------------------------------
# Dashboard Metrics
# -------------------------------------------------------------
@router.get("/dashboard/metrics", summary="Get Real-time AP Dashboard Metrics")
def get_dashboard_metrics(db: Session = Depends(get_db)):
    repo = InvoiceRepository(db)
    return repo.get_dashboard_metrics()

# -------------------------------------------------------------
# Vendor Risk Analysis
# -------------------------------------------------------------
@router.get("/vendors/risk", summary="Get Vendor Risk Scores and Rankings")
def get_vendor_risk_analysis():
    scorer = VendorRiskScorer()
    df = scorer.get_all_profiles_df()
    return {
        "total_vendors": len(df),
        "high_risk_vendors": int((df["risk_score"] >= settings.HIGH_RISK_SCORE_THRESHOLD).sum()) if not df.empty else 0,
        "vendors": df.to_dict("records") if not df.empty else []
    }

# -------------------------------------------------------------
# Batch Processing
# -------------------------------------------------------------
@router.post("/batches/process", summary="Process Incoming Invoice Batch Folder")
def process_batch(
    folder_path: Optional[str] = None,
    batch_name: Optional[str] = None,
    db: Session = Depends(get_db)
):
    processor = BatchProcessor(session=db)
    res = processor.process_directory(folder_path=folder_path, batch_name=batch_name)
    return res

# -------------------------------------------------------------
# Settings & Gemini API Key Management
# -------------------------------------------------------------
@router.get("/settings/gemini", summary="Get Gemini AI Provider Status")
def get_gemini_status():
    extractor = GeminiExtractor()
    test_res = extractor.test_connection() if extractor.is_configured() else {"success": False, "message": "Not configured"}
    return {
        "configured": extractor.is_configured(),
        "masked_key": settings.get_masked_api_key(),
        "model": settings.GEMINI_MODEL,
        "connection_test": test_res
    }

@router.post("/settings/gemini", summary="Update or Set Gemini API Key")
def update_gemini_key(api_key: str = Form(...)):
    clean = api_key.strip()
    settings.update_gemini_key(clean)
    extractor = GeminiExtractor(api_key=clean)
    test_res = extractor.test_connection()
    return {
        "success": True,
        "masked_key": settings.get_masked_api_key(clean),
        "connection_test": test_res
    }
