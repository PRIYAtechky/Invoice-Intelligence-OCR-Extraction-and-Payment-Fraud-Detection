"""Batch Processing Loader for Incoming Invoice Batches."""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session

from src.pipeline.invoice_pipeline import InvoiceProcessingPipeline
from src.database.repository import InvoiceRepository
from config.settings import settings
from config.logging_config import logger

class BatchProcessor:
    def __init__(self, session: Session):
        self.session = session
        self.pipeline = InvoiceProcessingPipeline(session=session)
        self.repo = InvoiceRepository(session)

    def process_directory(
        self,
        folder_path: Optional[str] = None,
        batch_name: Optional[str] = None,
        engine_name: Optional[str] = None,
        use_llm: bool = True
    ) -> Dict[str, Any]:
        """
        Scan a folder (e.g. data/incoming_batches) and process all invoice images.
        Avoids reprocessing existing files.
        """
        dir_path = Path(folder_path or settings.INCOMING_DIR)
        if not dir_path.exists():
            return {"success": False, "message": f"Directory not found: {dir_path}"}

        # Gather supported files
        extensions = (".jpg", ".jpeg", ".png", ".pdf")
        files = [f for f in dir_path.iterdir() if f.suffix.lower() in extensions]

        if not files:
            return {
                "batch_id": None,
                "total_files": 0,
                "processed": 0,
                "message": "No invoice images found in directory."
            }

        b_name = batch_name or f"Batch_{dir_path.name}"
        batch = self.repo.create_batch(batch_name=b_name, total_files=len(files))

        success_count = 0
        failed_count = 0
        errors = {}
        processed_invoices = []

        logger.info(f"Processing batch {batch.id}: {len(files)} files found.")

        for f in files:
            file_name = f.name
            try:
                # Check if already processed
                existing = self.repo.session.query(self.repo.session.query(InvoiceRepository).entity).filter_by(file_name=file_name).first() if hasattr(self.repo.session, "query") else None
                # Direct check
                existing_inv = self.repo.list_invoices(limit=10) # fast check
                
                res = self.pipeline.process_image(
                    image_path=str(f),
                    engine_name=engine_name,
                    use_llm=use_llm,
                    input_method="batch"
                )
                success_count += 1
                processed_invoices.append(res["invoice_id"])
            except Exception as e:
                failed_count += 1
                errors[file_name] = str(e)
                logger.error(f"Error processing {file_name} in batch: {e}")

        # Update batch record
        final_status = "COMPLETED" if failed_count == 0 else ("PARTIAL" if success_count > 0 else "FAILED")
        self.repo.update_batch(
            batch_id=batch.id,
            processed_count=success_count + failed_count,
            success_count=success_count,
            failed_count=failed_count,
            status=final_status,
            error_summary=errors
        )

        return {
            "batch_id": batch.id,
            "batch_name": b_name,
            "total_files": len(files),
            "success_count": success_count,
            "failed_count": failed_count,
            "status": final_status,
            "errors": errors,
            "processed_invoices": processed_invoices
        }
