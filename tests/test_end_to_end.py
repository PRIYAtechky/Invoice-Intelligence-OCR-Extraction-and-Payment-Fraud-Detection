"""End-to-End Pipeline Integration Test."""

import pytest
import os
from src.database.session import get_db_session
from src.pipeline.invoice_pipeline import InvoiceProcessingPipeline
from src.database.repository import InvoiceRepository

def test_full_pipeline_end_to_end(sample_image_path):
    if not os.path.exists(sample_image_path):
        pytest.skip(f"Test sample image not present at {sample_image_path}")

    with get_db_session() as session:
        pipeline = InvoiceProcessingPipeline(session=session)
        result = pipeline.process_image(
            image_path=sample_image_path,
            use_llm=False,
            input_method="upload",
            approver_role="AP-Clerk-01"
        )

        assert result["invoice_id"] is not None
        assert result["extraction"].vendor_name is not None
        assert result["validation"] is not None
        assert result["fraud"] is not None

        # Verify stored in DB
        repo = InvoiceRepository(session)
        stored = repo.get_invoice(result["invoice_id"])
        assert stored is not None
        assert stored.grand_total > 0

        # Perform human review
        reviewed = repo.update_invoice_review(
            invoice_id=stored.id,
            decision="APPROVED",
            reviewer_name="AP-Clerk-01",
            comments="Verified against PO."
        )
        assert reviewed.status == "APPROVED"
