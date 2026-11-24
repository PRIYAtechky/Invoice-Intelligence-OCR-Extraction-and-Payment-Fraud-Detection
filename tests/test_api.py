"""
Tests for FastAPI REST API Endpoints
"""
import pytest
from fastapi.testclient import TestClient
from src.database.session import init_db
from src.api.main import app


@pytest.fixture(scope="session", autouse=True)
def setup_database():
    init_db()


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["healthy", "degraded"]
    assert "gemini_api_configured" in data


def test_manual_invoice_submission():
    payload = {
        "vendor_name": "Venkateswara Bearings & Co",
        "vendor_gstin": "37RLOBV9249G1Z7",
        "invoice_number": "API-TEST-001",
        "invoice_date": "2025-10-10",
        "due_date": "2025-11-20",
        "po_number": "KIS/PO/2024/10001",
        "subtotal": 2000.0,
        "cgst": 180.0,
        "sgst": 180.0,
        "igst": 0.0,
        "grand_total": 2360.0,
        "bank_account_last4": "4479",
        "line_items": [
            {
                "description": "Ball Bearing 6205",
                "hsn": "8482",
                "quantity": 2,
                "unit_price": 1000.0,
                "amount": 2000.0
            }
        ]
    }
    response = client.post("/api/v1/invoices/manual", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert "invoice" in data
    assert data["invoice"]["invoice_number"] == "API-TEST-001"
    assert data["invoice"]["validation_status"] == "Passed"


def test_list_invoices():
    response = client.get("/api/v1/invoices")
    assert response.status_code == 200
    data = response.json()
    assert "invoices" in data
    assert "total_count" in data


def test_metrics_endpoint():
    response = client.get("/api/v1/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "total_invoices_processed" in data
    assert "total_invoice_value_inr" in data
