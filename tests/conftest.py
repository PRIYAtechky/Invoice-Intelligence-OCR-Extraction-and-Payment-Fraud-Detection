"""Pytest Configuration and Shared Test Fixtures."""

import os
import sys
import pytest
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Ensure project root is in path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.database.models import Base
from src.extraction.schemas import InvoiceExtraction, LineItem

TEST_DB_URL = "sqlite:///:memory:"

@pytest.fixture(scope="session")
def db_engine():
    engine = create_engine(TEST_DB_URL, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)

@pytest.fixture(scope="function")
def db_session(db_engine):
    connection = db_engine.connect()
    transaction = connection.begin()
    SessionTest = sessionmaker(bind=connection)
    session = SessionTest()

    yield session

    session.close()
    transaction.rollback()
    connection.close()

@pytest.fixture
def sample_clean_extraction():
    return InvoiceExtraction(
        vendor_name="Vel Engineering Works",
        vendor_gstin="33XXYEN7492X1Z2",
        invoice_number="TI/2025/8093",
        invoice_date="2025-10-10",
        due_date="2025-11-24",
        po_number="KIS/PO/2025/30592",
        line_items=[
            LineItem(
                description="Coupling Spider L-100",
                hsn="4016",
                quantity=20.0,
                unit_price=145.7,
                amount=2914.0
            )
        ],
        subtotal=2914.0,
        cgst=262.26,
        sgst=262.26,
        igst=0.0,
        grand_total=3438.52,
        bank_account_last4="5029",
        extraction_source="rule",
        confidence_scores={"grand_total": 0.98}
    )

@pytest.fixture
def sample_image_path():
    path = r"c:\Users\Badmapriya S\Downloads\T01_Invoice_Intelligence_Starter_Pack\T01_Invoice_Intelligence_Starter_Pack\01_initial_data\invoice_images\inv_0001.jpg"
    if os.path.exists(path):
        return path
    # Fallback to local test image if needed
    return str(BASE_DIR / "data" / "invoice_images" / "inv_0001.jpg")
