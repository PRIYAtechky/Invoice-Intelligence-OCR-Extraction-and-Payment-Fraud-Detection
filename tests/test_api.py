"""Tests for FastAPI REST Endpoints."""

import pytest
from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app)

def test_health_endpoint():
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "version" in data

def test_manual_invoice_submission():
    payload = {
        "vendor_name": "Vel Engineering Works",
        "vendor_gstin": "33XXYEN7492X1Z2",
        "invoice_number": "API-TEST-001",
        "invoice_date": "2025-10-10",
        "due_date": "2025-11-24",
        "subtotal": 1000.0,
        "cgst": 90.0,
        "sgst": 90.0,
        "igst": 0.0,
        "grand_total": 1180.0,
        "bank_account_last4": "5029",
        "line_items": [
            {
                "description": "Contactor 32A",
                "hsn": "8536",
                "quantity": 1.0,
                "unit_price": 1000.0,
                "amount": 1000.0
            }
        ]
    }
    resp = client.post("/api/v1/invoices/manual", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert "invoice_id" in data

def test_dashboard_metrics_endpoint():
    resp = client.get("/api/v1/dashboard/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert "total_invoices_processed" in data
    assert "total_invoice_value_inr" in data

def test_vendor_risk_endpoint():
    resp = client.get("/api/v1/vendors/risk")
    assert resp.status_code == 200
    data = resp.json()
    assert "total_vendors" in data
    assert len(data["vendors"]) > 0

def test_gemini_settings_endpoint():
    resp = client.get("/api/v1/settings/gemini")
    assert resp.status_code == 200
    data = resp.json()
    assert "configured" in data
    assert "masked_key" in data
