"""End-to-End Invoice Processing Pipeline."""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from sqlalchemy.orm import Session

from src.preprocessing.image_processor import ImageProcessor
from src.ocr.ocr_manager import ocr_manager
from src.extraction.hybrid_extractor import HybridExtractor
from src.extraction.schemas import InvoiceExtraction, ManualInvoiceInput
from src.validation.validator import ValidationEngine
from src.fraud.fraud_detector import PaymentFraudDetector
from src.database.repository import InvoiceRepository
from config.settings import settings
from config.logging_config import logger

class InvoiceProcessingPipeline:
    def __init__(self, session: Optional[Session] = None):
        self.session = session
        self.preprocessor = ImageProcessor()
        self.hybrid_extractor = HybridExtractor()
        self.validator = ValidationEngine()
        self.fraud_detector = PaymentFraudDetector()

    def process_image(
        self,
        image_path: str,
        engine_name: Optional[str] = None,
        use_llm: bool = True,
        input_method: str = "upload",
        approver_role: str = "AP-Clerk-01"
    ) -> Dict[str, Any]:
        """
        Execute complete pipeline:
        Image -> Preprocessing -> OCR -> Extraction -> Validation -> Fraud Detection -> Database.
        """
        path_obj = Path(image_path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Invoice file not found: {image_path}")

        file_name = path_obj.name
        logger.info(f"Starting pipeline processing for: {file_name}")

        # Stage 1: Image Preprocessing
        try:
            prep_res = self.preprocessor.process(str(path_obj))
            proc_img_path = prep_res.get("processed_path") or str(path_obj)
        except Exception as e:
            logger.warning(f"Preprocessing fallback to raw image for {file_name}: {e}")
            proc_img_path = str(path_obj)

        # Stage 2: OCR Extraction
        ocr_res = ocr_manager.extract(proc_img_path, engine_name=engine_name)

        # Stage 3: Information Extraction (Rules + Gemini LLM)
        extraction = self.hybrid_extractor.extract(ocr_res, use_llm=use_llm)
        extraction.file_name = file_name

        # Stage 4: Validation Engine
        validation = self.validator.validate(extraction, approver_role=approver_role)

        # Stage 5: Payment Fraud Detection
        fraud = self.fraud_detector.detect_fraud(extraction, approver_role=approver_role)

        # Stage 6: Database Storage (if session provided)
        db_record = None
        if self.session:
            repo = InvoiceRepository(self.session)
            db_record = repo.create_invoice(
                extraction=extraction,
                validation=validation,
                fraud=fraud,
                input_method=input_method,
                file_name=file_name,
                original_image_path=str(path_obj),
                processed_image_path=proc_img_path
            )

        return {
            "invoice_id": db_record.id if db_record else None,
            "file_name": file_name,
            "original_image_path": str(path_obj),
            "processed_image_path": proc_img_path,
            "extraction": extraction,
            "validation": validation,
            "fraud": fraud,
            "ocr_meta": {
                "engine": ocr_res.get("engine"),
                "mean_confidence": ocr_res.get("mean_confidence"),
                "latency_ms": ocr_res.get("latency_ms")
            }
        }

    def process_manual(
        self,
        manual_data: ManualInvoiceInput,
        approver_role: str = "AP-Clerk-01"
    ) -> Dict[str, Any]:
        """Process manual invoice input bypassing OCR but running full validation and fraud checks."""
        extraction = InvoiceExtraction(
            file_name="MANUAL_ENTRY",
            vendor_name=manual_data.vendor_name,
            vendor_gstin=manual_data.vendor_gstin,
            invoice_number=manual_data.invoice_number,
            invoice_date=manual_data.invoice_date,
            due_date=manual_data.due_date,
            po_number=manual_data.po_number,
            line_items=manual_data.line_items,
            subtotal=manual_data.subtotal,
            cgst=manual_data.cgst,
            sgst=manual_data.sgst,
            igst=manual_data.igst,
            grand_total=manual_data.grand_total,
            bank_account_last4=manual_data.bank_account_last4,
            extraction_source="manual",
            confidence_scores={"manual": 1.0}
        )

        validation = self.validator.validate(extraction, approver_role=approver_role)
        fraud = self.fraud_detector.detect_fraud(extraction, approver_role=approver_role)

        db_record = None
        if self.session:
            repo = InvoiceRepository(self.session)
            db_record = repo.create_invoice(
                extraction=extraction,
                validation=validation,
                fraud=fraud,
                input_method="manual",
                file_name="MANUAL_ENTRY"
            )

        return {
            "invoice_id": db_record.id if db_record else None,
            "file_name": "MANUAL_ENTRY",
            "extraction": extraction,
            "validation": validation,
            "fraud": fraud
        }
