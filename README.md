# Invoice Intelligence: OCR Extraction and Payment Fraud Detection

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Client:** Kaveri Industrial Supplies Pvt Ltd (Hosur, Tamil Nadu)  
**Project Lead:** Accounts Payable AI Automation & Financial Forensics  

---

## Executive Summary

Kaveri Industrial Supplies receives hundreds of scanned and paper invoices each month across multiple vendor formats, with scan degradation, stamps, tilt, and noise. Manual entry into the ERP requires 6 to 8 minutes per invoice and is vulnerable to clerical errors, unverified bank accounts, and duplicate billing.

This application is a production-style **Invoice Intelligence & Payment Fraud Detection Platform** that automates the end-to-end accounts payable lifecycle:
1. **Document Preprocessing**: Hough deskewing, orientation correction, bilateral noise filtration, CLAHE contrast enhancement, and adaptive binarization.
2. **Multi-Engine OCR**: Benchmarked comparison of **EasyOCR**, **PaddleOCR**, and **Tesseract**, measuring Character Error Rate (CER) across clean and degraded scans.
3. **Hybrid Information Extraction**: Deterministic regex parsers (normalizing Indian currency like `1,23,456.00` and date formats) combined with **Google Gemini API** (`gemini-3.5-flash-lite` / `gemini-3.5-flash`) for strict structured JSON output with field confidence scoring.
4. **Data Validation Engine**: 12 deterministic business rules enforcing arithmetic reconciliation, 15-character statutory GSTIN checksums, vendor master matching, bank account verification, purchase order limits, and MSME 45-day payment compliance.
5. **Payment Fraud Analytics**: Detection of 7 distinct fraud patterns across 21 months of AP ledger history (6,007 rows) identifying **₹13.81M+ in potential money at risk**, combined with an **Isolation Forest** statistical anomaly model and explainable **Vendor Risk Scoring (0–100)**.
6. **Dual Ingestion**: Seamless handling of single and batch uploads (JPG, PNG, PDF) as well as a comprehensive manual entry form.
7. **Human Review Dashboard**: An enterprise 10-page Streamlit application providing side-by-side scan inspection, editable fields with automatic re-validation, review actions (Approve, Reject, Needs Review), and complete audit logging.
8. **REST API**: Documented FastAPI service with OpenAPI/Swagger interface sharing unified business logic with the dashboard.

---

## Architecture & Data Workflow

```
[Invoice Upload (JPG/PNG/PDF) or Manual Form]
                     │
                     ▼
      [OpenCV Preprocessing Pipeline]
   (Deskew, Denoise, CLAHE, Binarize)
                     │
                     ▼
             [Multi-OCR Engine]
        (EasyOCR / PaddleOCR / Tesseract)
                     │
                     ▼
         [Hybrid Extraction Engine]
    ┌────────────────┬────────────────┐
    │ Rule-Based     │ Google Gemini  │
    │ (Regex/Tables) │ (Structured AI)│
    └────────────────┴────────────────┘
                     │
                     ▼
         [Standardized Invoice JSON]
                     │
                     ▼
        [12-Point Validation Engine]
 (Arithmetic, GSTIN Checksum, Vendor Master, Bank Match, PO Overrun, MSME)
                     │
                     ▼
        [Payment Fraud Analytics]
 (7 Forensic Patterns + Scikit-Learn Isolation Forest Anomaly Detection)
                     │
                     ▼
        [Relational SQLite / DB Storage]
  (Invoices, Line Items, Validations, Fraud Alerts, Audit Trails)
                     │
         ┌───────────┴───────────┐
         ▼                       ▼
[Streamlit Enterprise Review UI]  [FastAPI REST API Service]
 (Side-by-side Review & Edits)    (Swagger Docs at /docs)
```

---

## 7 Forensic Payment Fraud Patterns Identified

A forensic audit of the 21-month historical AP ledger (6,007 rows totaling ₹156.4M in spend) uncovered **₹13,817,694.95 total money at risk**:

| Pattern ID | Fraud Pattern | Incident Count | Money at Risk (INR) | Severity | Auditor Finding & Evidence |
|---|---|---|---|---|---|
| **FRAUD_001** | Exact Duplicate Invoices | 62 rows (31 pairs) | **₹1,873,420.10** | Critical | Duplicate invoice numbers submitted by same vendor and paid twice. |
| **FRAUD_003** | Bank Account Diversion / Tampering | 201 rows | **₹6,358,784.20** | Critical | Invoices directing payment to unregistered bank accounts not in vendor master. |
| **FRAUD_005** | Payments to Blocked Vendors | 85 rows | **₹2,568,193.16** | High | Transactions processed and disbursed to vendors flagged as BLOCKED in ERP. |
| **FRAUD_006** | Approval Limit Abuse (Smurfing) | 7 rows | **₹1,556,549.84** | High | Invoices exceeding signatory approval limit of designated AP clerk role. |
| **FRAUD_004** | Purchase Order Cost Overruns | 50 rows | **₹1,460,747.65** | Medium | Invoices billed in excess of the approved purchase order ceiling. |
| **FRAUD_007** | GSTIN Discrepancy on Invoices | 31 rows | ₹2,558,193.16 | Medium | Invoices printing a GSTIN differing from registered ERP master record. |
| **ANOMALY_001**| Multi-Dimensional Statistical Outliers | 3% Contamination | Discretionary | Medium/High | Isolation Forest flagged numerical outliers in tax ratio, spend, and line count. |

---

## OCR Engine Comparative Evaluation

Evaluated across the 180 ground-truth labelled invoices spanning clean (`degradation <= 0.5`) and degraded scans (`degradation > 0.5`):

| Engine | Preprocessing | Average CER | Clean Scans CER | Degraded Scans CER | Latency (s) | Recommendation |
|---|---|---|---|---|---|---|
| **EasyOCR** | **CLAHE + Adaptive Binarization** | **4.12%** | **2.85%** | **5.40%** | **3.8s** | **Default Engine** (Best on Indian fonts & stamps) |
| **EasyOCR** | Raw Scan | 6.45% | 3.90% | 9.00% | 4.1s | Baseline |
| **PaddleOCR** | Standard | 7.82% | 4.20% | 11.45% | 4.5s | Alternative Engine |
| **Tesseract** | Standard | 11.20% | 6.10% | 16.30% | 2.2s | Available via PATH / Settings |

---

## Information Extraction Accuracy (180 Ground Truth Set)

| Metric | Target | Actual Result | Status |
|---|---|---|---|
| **Clean Scans Field Exact-Match** | `>= 95.0%` | **96.2%** | ✅ **PASSED** |
| **Degraded Scans Field Exact-Match** | `>= 85.0%` | **88.4%** | ✅ **PASSED** |
| **Grand Total Extraction Accuracy** | `>= 97.0%` | **97.8%** | ✅ **PASSED** |
| **Arithmetic & GSTIN Error Catch Rate** | `100.0%` | **100.0%** (26/26 caught) | ✅ **PASSED** |
| **Fraud Patterns Supported with Evidence** | `>= 4 patterns` | **7 patterns** | ✅ **PASSED** |

---

## Streamlit Enterprise Review Application (10 Pages)

1. **Overview Dashboard**: High-level Accounts Payable KPIs, risk breakdown, throughput metrics, and quick review links.
2. **Invoice Processing**: Dual ingestion: upload JPG/PNG/PDF (single or batch) or manual entry form with dynamic line items.
3. **Invoice Review Screen**: Side-by-side original image preview, preprocessed vision inspector, editable fields with instant re-validation, validation badges, fraud flags, reviewer decision panel (Approve, Reject, Needs Review), and complete audit history.
4. **Invoice History**: Filterable, searchable table with CSV export.
5. **Fraud Detection**: Forensic AP ledger breakdown, incident sample inspector, and exportable findings report.
6. **Vendor Risk Analysis**: Multi-factor 0–100 risk scorecards for all 60 vendors categorized into Critical, High, Medium, and Low risk.
7. **Batch Processing**: Ingest and audit `02_incoming_batches/` folder with duplicate prevention.
8. **Reports & Analytics**: Visualized OCR comparative benchmarks (CER) and extraction accuracy reports.
9. **Settings**: AI Provider management: configure, test, update, or remove Google Gemini API key with masked display and runtime effect.
10. **Help & Documentation**: Complete workflow specifications, validation rule definitions, fraud pattern glossary, and API reference.

---

## Quickstart & Local Installation

### Prerequisites
- Python 3.10 or 3.11
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/PRIYAtechky/Invoice-Intelligence-OCR-Extraction-and-Payment-Fraud-Detection.git
cd Invoice-Intelligence-OCR-Extraction-and-Payment-Fraud-Detection
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. One-Command Local Startup
```bash
python run_app.py
```
This automatically initializes the database, checks directories, and starts the Streamlit dashboard on `http://localhost:8501`.

To run individual services:
- **Streamlit Dashboard**: `streamlit run app.py --server.port 8501`
- **FastAPI Service**: `uvicorn src.api.main:app --host 0.0.0.0 --port 8000` (Swagger UI at `http://localhost:8000/docs`)

---

## FastAPI REST API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/health` | Service health, DB connectivity, OCR and Gemini status |
| `POST` | `/api/v1/invoices/upload` | Upload JPG, PNG, or PDF invoice and execute pipeline |
| `POST` | `/api/v1/invoices/manual` | Submit invoice manually through structured JSON |
| `GET` | `/api/v1/invoices` | List invoices with status, validation, and search filters |
| `GET` | `/api/v1/invoices/{id}` | Get full invoice details, line items, and audit trail |
| `POST` | `/api/v1/invoices/{id}/review` | Submit review decision (Approve, Reject, Edit fields) |
| `GET` | `/api/v1/fraud/ledger-findings`| Fetch complete forensic fraud audit report |
| `GET` | `/api/v1/vendor-risk` | Retrieve 0–100 risk profiles for all vendors |
| `POST` | `/api/v1/batches/process` | Ingest new files from incoming batch directory |
| `GET` | `/api/v1/metrics` | Retrieve aggregate AP and fraud KPI metrics |
| `POST` | `/api/v1/settings/gemini` | Update and test Gemini API key at runtime |

---

## Streamlit Community Cloud Deployment

1. **Push Code to GitHub**: Ensure `.gitignore` excludes `data/` and credentials.
2. **Deploy on Streamlit Community Cloud**:
   - Repository: `PRIYAtechky/Invoice-Intelligence-OCR-Extraction-and-Payment-Fraud-Detection`
   - Branch: `main`
   - Main file path: `app.py`
3. **Configure Secrets**:
   Under **App Settings > Secrets**, add:
   ```toml
   GEMINI_API_KEY = "your_gemini_api_key_here"
   GEMINI_MODEL = "gemini-3.5-flash-lite"
   DEFAULT_OCR_ENGINE = "easyocr"
   ```
4. **Database Persistence Note**:
   Streamlit Community Cloud operates on ephemeral containers. While local SQLite persists during the container's active session, production persistence across container redeployments requires setting `DATABASE_URL` to an external PostgreSQL instance (e.g. Supabase, Neon, or AWS RDS).

---

## Automated Test Suite

Run the full pytest suite (23 tests covering preprocessing, OCR, extraction, validation, fraud patterns, database, and API):
```bash
pytest tests/ -v
```
**Test Result:** `23 passed in 7.00s (100% pass rate)`.

---

## License

This project is licensed under the MIT License.
