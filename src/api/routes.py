"""
FastAPI REST API Endpoints for Invoice Intelligence
Provides programmatic integration for Ingestion, Manual Submission, Review Actions,
Fraud Auditing, Vendor Risk, and System Health.
"""
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query, status
from sqlalchemy.orm import Session

from src.database.session import get_db
from src.database.models import InvoiceModel
from src.database.repository import InvoiceRepository
from src.pipeline import pipeline
from src.extraction.schemas import (
    ManualInvoiceInput, InvoiceReviewAction, LineItem
)
from src.fraud.analytics import ledger_fraud_analyzer
from src.fraud.vendor_risk import vendor_risk_scorer
from src.config import (
    settings, get_effective_gemini_api_key, mask_api_key,
    save_user_api_key, remove_user_api_key
)
from src.extraction.gemini_extractor import GeminiExtractor

router = APIRouter(prefix="/api/v1")


@router.get("/health", tags=["System"])
def health_check(db: Session = Depends(get_db)):
    """Health and diagnostic endpoint verifying database, OCR engines, and AI status."""
    db_ok = True
    try:
        db.execute("SELECT 1")
    except Exception:
        db_ok = False

    gemini_key = get_effective_gemini_api_key()
    gemini_configured = bool(gemini_key)

    return {
        "status": "healthy" if db_ok else "degraded",
        "database_connected": db_ok,
        "database_url": settings.database_url.split("///")[-1],
        "default_ocr_engine": settings.default_ocr_engine,
        "gemini_api_configured": gemini_configured,
        "gemini_model": settings.gemini_model,
        "active_api_key_masked": mask_api_key(gemini_key)
    }


@router.post("/invoices/upload", tags=["Invoices"], status_code=status.HTTP_201_CREATED)
async def upload_invoice(
    file: UploadFile = File(..., description="Invoice image (JPG, PNG) or PDF"),
    approver_role: str = Form("AP-Clerk-01"),
    ocr_engine: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """Upload and process an invoice through the complete end-to-end pipeline."""
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # Storage destination
    uploads_dir = settings.project_root / "data" / "uploads"
    save_path = uploads_dir / file.filename

    rec, meta = pipeline.process_image_file(
        db=db,
        file_path_or_bytes=contents,
        file_name=file.filename,
        save_destination=str(save_path),
        approver_role=approver_role,
        ocr_engine_name=ocr_engine
    )

    if not rec:
        raise HTTPException(status_code=422, detail=meta.get("error", "Processing pipeline failed."))

    return {
        "message": "Invoice processed successfully.",
        "invoice_id": rec.id,
        "invoice_uid": rec.invoice_uid,
        "invoice_number": rec.invoice_number,
        "vendor_name": rec.vendor_name,
        "grand_total": rec.grand_total,
        "status": rec.status,
        "validation_status": rec.validation_status,
        "max_fraud_severity": rec.max_fraud_severity,
        "fraud_risk_score": rec.fraud_risk_score,
        "pipeline_metadata": meta
    }


@router.post("/invoices/manual", tags=["Invoices"], status_code=status.HTTP_201_CREATED)
def submit_manual_invoice(
    payload: ManualInvoiceInput,
    approver_role: str = Query("AP-Clerk-01"),
    db: Session = Depends(get_db)
):
    """Submit invoice details manually; bypasses OCR but runs validation and fraud checks."""
    rec, meta = pipeline.process_manual_entry(
        db=db,
        manual_data=payload,
        approver_role=approver_role
    )
    return {
        "message": "Manual invoice created successfully.",
        "invoice": rec.to_dict(),
        "metadata": meta
    }


@router.get("/invoices", tags=["Invoices"])
def list_invoices(
    status: Optional[str] = Query("All"),
    validation_status: Optional[str] = Query("All"),
    fraud_severity: Optional[str] = Query("All"),
    search: Optional[str] = Query(None),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db)
):
    """Retrieve invoices with multi-field search and status filtering."""
    invoices, total = InvoiceRepository.list_invoices(
        db=db,
        status=status,
        validation_status=validation_status,
        max_fraud_severity=fraud_severity,
        search_query=search,
        skip=skip,
        limit=limit
    )
    return {
        "total_count": total,
        "skip": skip,
        "limit": limit,
        "invoices": [inv.to_dict() for inv in invoices]
    }


@router.get("/invoices/{invoice_id}", tags=["Invoices"])
def get_invoice(invoice_id: int, db: Session = Depends(get_db)):
    """Fetch complete details, line items, validation findings, and audit log for an invoice."""
    inv = InvoiceRepository.get_invoice_by_id(db, invoice_id)
    if not inv:
        raise HTTPException(status_code=404, detail=f"Invoice ID {invoice_id} not found.")
    return inv.to_dict()


@router.post("/invoices/{invoice_id}/review", tags=["Review Workflow"])
def review_invoice(
    invoice_id: int,
    action: InvoiceReviewAction,
    db: Session = Depends(get_db)
):
    """
    Submit human review decision (Approve, Reject, Needs Review, Edit fields).
    Re-runs validation checks and records complete before/after audit log.
    """
    updated = InvoiceRepository.review_and_update_invoice(
        db=db,
        invoice_id=invoice_id,
        review_action=action
    )
    if not updated:
        raise HTTPException(status_code=404, detail=f"Invoice ID {invoice_id} not found.")
    return {
        "message": f"Review action '{action.action.value}' applied successfully.",
        "invoice": updated.to_dict()
    }


@router.get("/fraud/ledger-findings", tags=["Fraud Analytics"])
def get_ledger_fraud_findings():
    """Execute comprehensive audit on the AP historical ledger (6,007 rows) and return findings."""
    return ledger_fraud_analyzer.analyze_ledger()


@router.get("/vendor-risk", tags=["Fraud Analytics"])
def get_vendor_risk_scorecards():
    """Fetch explainable risk profiles for all registered vendors."""
    return vendor_risk_scorer.compute_all_vendor_scores()


@router.post("/batches/process", tags=["Batch Processing"])
def trigger_batch_processing(
    approver_role: str = Query("AP-Clerk-01"),
    db: Session = Depends(get_db)
):
    """Process all uningested invoices from 02_incoming_batches/ folder."""
    return pipeline.process_incoming_batch_directory(db=db, approver_role=approver_role)


@router.get("/metrics", tags=["Dashboard"])
def get_dashboard_metrics(db: Session = Depends(get_db)):
    """Get aggregate Accounts Payable and Fraud KPIs calculated from stored data."""
    return InvoiceRepository.get_dashboard_metrics(db)


@router.get("/settings/gemini", tags=["Settings"])
def get_gemini_settings():
    """Inspect AI Provider status and masked API key."""
    key = get_effective_gemini_api_key()
    return {
        "is_configured": bool(key),
        "masked_key": mask_api_key(key),
        "model": settings.gemini_model,
        "default_ocr_engine": settings.default_ocr_engine
    }


@router.post("/settings/gemini", tags=["Settings"])
def update_gemini_settings(
    api_key: str = Form(..., description="Google AI Studio Gemini API Key"),
    model_name: Optional[str] = Form("gemini-3.5-flash-lite")
):
    """Update and persist Gemini API key at runtime without code changes or restarts."""
    cleaned = api_key.strip()
    if not cleaned:
        raise HTTPException(status_code=400, detail="API key cannot be empty.")

    # Validate with test connection
    extractor = GeminiExtractor(api_key=cleaned, model_name=model_name)
    ok, msg = extractor.test_connection()
    if not ok:
        raise HTTPException(status_code=400, detail=f"API Key validation failed: {msg}")

    save_user_api_key(cleaned)
    settings.gemini_model = model_name
    return {
        "message": "Gemini API key verified and saved successfully.",
        "model": model_name,
        "masked_key": mask_api_key(cleaned)
    }


@router.delete("/settings/gemini", tags=["Settings"])
def delete_gemini_settings():
    """Remove user-configured Gemini API key."""
    remove_user_api_key()
    return {"message": "Gemini API key removed. Rule-based extraction will now be used."}
