"""
Unified End-to-End Invoice Processing Pipeline Orchestrator
Coordinates: Preprocessing -> OCR -> Extraction (Rules + Gemini) -> Validation -> Fraud Detection -> Storage
Supports both File Uploads (JPG/PNG/PDF) and Manual Submissions.
"""
import os
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from sqlalchemy.orm import Session

from src.extraction.schemas import ExtractedInvoice, ManualInvoiceInput, ValidationResultItem, FraudAlertItem
from src.preprocessing.image_pipeline import preprocessor
from src.ocr.ocr_comparison import get_ocr_engine
from src.extraction.hybrid_extractor import invoice_extractor
from src.validation.engine import validation_engine
from src.fraud.patterns import (
    check_exact_duplicate, check_fuzzy_duplicate,
    check_bank_tampering, check_po_overrun, check_blocked_vendor, check_approval_limit_split
)
from src.fraud.anomaly_detector import anomaly_detector
from src.database.repository import InvoiceRepository
from src.database.models import InvoiceModel
from src.config import settings


class InvoiceProcessingPipeline:
    def __init__(self):
        self.preprocessor = preprocessor
        self.validation_engine = validation_engine
        self.extractor = invoice_extractor
        self.anomaly_detector = anomaly_detector

    def _get_historical_invoices_sample(self, db: Session) -> List[Dict[str, Any]]:
        """Retrieve recent invoices for duplicate and collision checks."""
        recent, _ = InvoiceRepository.list_invoices(db, limit=300)
        return [r.to_dict() for r in recent]

    def _run_fraud_checks(
        self,
        invoice: ExtractedInvoice,
        db: Session,
        approver_role: str = "AP-Clerk-01"
    ) -> List[FraudAlertItem]:
        """Run all fraud rules and Isolation Forest anomaly detector on an extracted invoice."""
        alerts: List[FraudAlertItem] = []
        historical = self._get_historical_invoices_sample(db)

        # 1. Exact duplicate check
        alert1 = check_exact_duplicate(invoice, historical)
        if alert1:
            alerts.append(alert1)

        # 2. Fuzzy duplicate check
        alert2 = check_fuzzy_duplicate(invoice, historical)
        if alert2:
            alerts.append(alert2)

        # 3. Bank tampering check
        v_name = (invoice.vendor_name or "").strip().lower()
        vm_rec = self.validation_engine.rules.vm_by_name.get(v_name)
        if not vm_rec:
            gst = (invoice.vendor_gstin or "").strip().upper()
            vm_rec = self.validation_engine.rules.vm_by_gstin.get(gst)

        alert3 = check_bank_tampering(invoice, vm_rec)
        if alert3:
            alerts.append(alert3)

        # 4. Blocked vendor check
        alert4 = check_blocked_vendor(invoice, vm_rec)
        if alert4:
            alerts.append(alert4)

        # 5. PO overrun check
        po_num = (invoice.po_number or "").strip().upper()
        po_rec = self.validation_engine.rules.po_by_number.get(po_num)
        alert5 = check_po_overrun(invoice, po_rec)
        if alert5:
            alerts.append(alert5)

        # 6. Approval threshold split check
        limit = self.validation_engine.rules.approval_limits.get(approver_role, 200000.0)
        alert6 = check_approval_limit_split(invoice, approver_role, limit)
        if alert6:
            alerts.append(alert6)

        # 7. Statistical Anomaly Detection (Isolation Forest)
        is_anomaly, risk_score, details = self.anomaly_detector.predict_invoice(invoice)
        if is_anomaly:
            reasons = "; ".join(details.get("z_score_reasons", [])) or "Multi-dimensional statistical outlier in AP distribution."
            alerts.append(FraudAlertItem(
                pattern_id="ANOMALY_001",
                pattern_name="Statistical Anomaly (Isolation Forest)",
                risk_score=risk_score,
                severity="High" if risk_score >= 70 else "Medium",
                evidence=f"Anomaly detector flagged unusual invoice characteristics: {reasons}",
                related_entity="AP Statistical Baseline",
                amount_at_risk=invoice.grand_total,
                recommended_action="Conduct peer review before scheduling disbursement."
            ))

        return alerts

    def process_image_file(
        self,
        db: Session,
        file_path_or_bytes,
        file_name: str,
        save_destination: Optional[str] = None,
        approver_role: str = "AP-Clerk-01",
        ocr_engine_name: Optional[str] = None
    ) -> Tuple[Optional[InvoiceModel], Dict[str, Any]]:
        """
        Complete processing workflow for uploaded images (JPG, PNG) and PDFs.
        """
        result_meta = {"file_name": file_name, "stage": "start", "success": False}

        # 1. Load image
        img, err = self.preprocessor.load_image(file_path_or_bytes, file_name=file_name)
        if img is None:
            result_meta["error"] = f"Image loading failed: {err}"
            return None, result_meta

        # Save copy of uploaded file if path provided
        saved_file_path = None
        if save_destination:
            dest_path = Path(save_destination)
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(file_path_or_bytes, (str, Path)):
                shutil.copy(str(file_path_or_bytes), str(dest_path))
            elif isinstance(file_path_or_bytes, bytes):
                with open(dest_path, "wb") as f:
                    f.write(file_path_or_bytes)
            saved_file_path = str(dest_path)

        # 2. Preprocess image
        result_meta["stage"] = "preprocessing"
        prep_img, prep_meta = self.preprocessor.preprocess(img)

        # 3. OCR text extraction
        result_meta["stage"] = "ocr"
        ocr_engine = get_ocr_engine(ocr_engine_name)
        ocr_res = ocr_engine.extract_text(prep_img)
        result_meta["ocr_engine"] = ocr_res.engine_name
        result_meta["ocr_latency"] = ocr_res.latency_seconds

        # 4. Structured information extraction (Hybrid: Rule + Gemini)
        result_meta["stage"] = "extraction"
        extracted_invoice, ext_meta = self.extractor.extract(ocr_res.full_text, file_name=file_name)
        extracted_invoice.file_name = file_name
        extracted_invoice.file_path = saved_file_path
        extracted_invoice.ocr_engine_used = ocr_res.engine_name
        result_meta["extraction_details"] = ext_meta

        # 5. Data validation engine
        result_meta["stage"] = "validation"
        val_results = self.validation_engine.validate_invoice(extracted_invoice, approver_role=approver_role)

        # 6. Payment fraud detection & anomaly scoring
        result_meta["stage"] = "fraud_detection"
        fraud_alerts = self._run_fraud_checks(extracted_invoice, db, approver_role=approver_role)

        # 7. Database persistence and audit trail
        result_meta["stage"] = "database_storage"
        saved_record = InvoiceRepository.create_invoice(
            db=db,
            invoice=extracted_invoice,
            validations=val_results,
            fraud_alerts=fraud_alerts,
            file_path=saved_file_path
        )

        result_meta["stage"] = "complete"
        result_meta["success"] = True
        result_meta["invoice_id"] = saved_record.id
        result_meta["invoice_uid"] = saved_record.invoice_uid

        return saved_record, result_meta

    def process_manual_entry(
        self,
        db: Session,
        manual_data: ManualInvoiceInput,
        approver_role: str = "AP-Clerk-01"
    ) -> Tuple[InvoiceModel, Dict[str, Any]]:
        """
        Processing workflow for manually entered invoices:
        Bypasses OCR and image extraction, but executes identical validation,
        fraud detection, review, and database storage.
        """
        # Convert to standardized ExtractedInvoice
        extracted = ExtractedInvoice(
            invoice_number=manual_data.invoice_number,
            invoice_date=manual_data.invoice_date,
            due_date=manual_data.due_date,
            vendor_name=manual_data.vendor_name,
            vendor_gstin=manual_data.vendor_gstin,
            po_number=manual_data.po_number,
            subtotal=manual_data.subtotal,
            cgst=manual_data.cgst,
            sgst=manual_data.sgst,
            igst=manual_data.igst,
            grand_total=manual_data.grand_total,
            bank_account_last4=manual_data.bank_account_last4,
            line_items=manual_data.line_items,
            confidence_scores={k: 1.0 for k in ["invoice_number", "vendor_name", "vendor_gstin", "grand_total"]},
            extraction_sources={k: "manual" for k in ["invoice_number", "vendor_name", "vendor_gstin", "grand_total"]},
            input_method="manual",
            file_name="manual_submission.json"
        )

        # Run validation engine
        val_results = self.validation_engine.validate_invoice(extracted, approver_role=approver_role)

        # Run fraud detection & anomaly checks
        fraud_alerts = self._run_fraud_checks(extracted, db, approver_role=approver_role)

        # Persist to database
        saved_record = InvoiceRepository.create_invoice(
            db=db,
            invoice=extracted,
            validations=val_results,
            fraud_alerts=fraud_alerts
        )

        return saved_record, {"success": True, "invoice_id": saved_record.id, "invoice_uid": saved_record.invoice_uid}

    def process_incoming_batch_directory(
        self,
        db: Session,
        batch_folder_path: Optional[Path] = None,
        approver_role: str = "AP-Clerk-01"
    ) -> Dict[str, Any]:
        """
        Scan a batch directory (e.g. 02_incoming_batches/),
        ingesting all newly discovered invoices without reprocessing existing ones.
        """
        target_dir = batch_folder_path or settings.incoming_batches_dir
        if not target_dir.exists():
            return {"error": f"Batch folder not found: {target_dir}"}

        # Supported file types
        valid_extensions = {".jpg", ".jpeg", ".png", ".pdf"}
        files = [p for p in target_dir.iterdir() if p.is_file() and p.suffix.lower() in valid_extensions]

        # Prevent duplicate re-processing: check existing file names in database
        existing_names = set(
            name[0] for name in db.query(InvoiceModel.file_name).filter(InvoiceModel.file_name.isnot(None)).all()
        )

        successful = 0
        failed = 0
        total_amt = 0.0
        details = []

        for f in files:
            if f.name in existing_names:
                details.append({"file": f.name, "status": "skipped", "reason": "Already processed in database"})
                continue

            try:
                rec, meta = self.process_image_file(db, f, file_name=f.name, approver_role=approver_role)
                if rec:
                    successful += 1
                    total_amt += rec.grand_total
                    details.append({"file": f.name, "status": "success", "invoice_id": rec.id, "grand_total": rec.grand_total})
                else:
                    failed += 1
                    details.append({"file": f.name, "status": "failed", "error": meta.get("error")})
            except Exception as e:
                failed += 1
                details.append({"file": f.name, "status": "failed", "error": str(e)})

        return {
            "batch_directory": str(target_dir),
            "total_files_found": len(files),
            "successful_processed": successful,
            "failed_count": failed,
            "skipped_existing": len(files) - (successful + failed),
            "total_value_processed_inr": round(total_amt, 2),
            "file_details": details
        }


# Global pipeline instance
pipeline = InvoiceProcessingPipeline()
