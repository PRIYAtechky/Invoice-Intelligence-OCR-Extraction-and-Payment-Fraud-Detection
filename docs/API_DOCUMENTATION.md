# FastAPI Service Documentation

## 1. Overview
The Invoice Intelligence FastAPI backend exposes standardized REST endpoints for invoice OCR processing, data validation, fraud detection, review workflows, and dashboard telemetry.

- **Base URL**: `http://localhost:8000/api/v1`
- **Interactive Swagger UI**: `http://localhost:8000/docs`
- **ReDoc Documentation**: `http://localhost:8000/redoc`

---

## 2. Endpoints Reference

### 1. Health & Diagnostics
- `GET /health`
  - Response:
    ```json
    {
      "status": "healthy",
      "app_name": "Invoice Intelligence — OCR & Fraud Detection",
      "version": "1.0.0",
      "environment": "production",
      "default_ocr": "rapidocr",
      "gemini_configured": true
    }
    ```

### 2. Ingest & Process Scanned Invoice
- `POST /invoices/upload`
  - Content-Type: `multipart/form-data`
  - Parameters:
    - `file`: Scanned invoice image (JPG, PNG, PDF)
    - `ocr_engine`: Optional ("rapidocr" | "easyocr")
    - `use_llm`: Boolean (default `true`)
    - `approver_role`: Approver string (default `AP-Clerk-01`)
  - Response:
    ```json
    {
      "success": true,
      "invoice_id": "INV-62F779D0",
      "file_name": "inv_0001.jpg",
      "extraction": {
        "vendor_name": "Vel Engineering Works",
        "vendor_gstin": "33XXYEN7492X1Z2",
        "invoice_number": "TI/2025/8093",
        "grand_total": 3438.52
      },
      "validation": { "status": "Passed", "is_valid": true },
      "fraud": { "has_fraud_risk": false, "composite_risk_score": 0.0 }
    }
    ```

### 3. Manual Invoice Submission
- `POST /invoices/manual`
  - Content-Type: `application/json`
  - Payload:
    ```json
    {
      "vendor_name": "Vel Engineering Works",
      "vendor_gstin": "33XXYEN7492X1Z2",
      "invoice_number": "MANUAL-001",
      "invoice_date": "2025-10-10",
      "subtotal": 1000.0,
      "cgst": 90.0,
      "sgst": 90.0,
      "grand_total": 1180.0,
      "bank_account_last4": "5029",
      "line_items": [
        { "description": "Contactor 32A", "hsn": "8536", "quantity": 1, "unit_price": 1000.0, "amount": 1000.0 }
      ]
    }
    ```

### 4. Fetch Stored Invoice
- `GET /invoices/{invoice_id}`
  - Returns complete invoice entity with child line items, validation records, and reviewer decisions.

### 5. Review & Authorize Invoice
- `POST /invoices/{invoice_id}/review`
  - Parameters:
    - `decision`: "APPROVED" | "REJECTED" | "FLAGGED_FOR_REVIEW"
    - `reviewer_name`: String
    - `comments`: Optional audit note

### 6. Dashboard Telemetry
- `GET /dashboard/metrics`
  - Returns real aggregated metrics: total invoices, total value, validation rate, review queue, and money at risk.

### 7. Vendor Risk Ratings
- `GET /vendors/risk`
  - Returns explainable risk score (0-100) and risk factors for all registered suppliers.

### 8. AI Provider Configuration
- `GET /settings/gemini`: Return masked API key and connection test.
- `POST /settings/gemini`: Update API key dynamically at runtime without restarting service.
