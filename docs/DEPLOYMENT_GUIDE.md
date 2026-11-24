# Streamlit Community Cloud Deployment & Configuration Guide

## 1. Overview
The Invoice Intelligence application is fully configured for deployment on **Streamlit Community Cloud** with minimal configuration.

- **Main Entry Point**: `app.py`
- **Python Version**: `3.10` or `3.11`
- **Dependencies**: `requirements.txt`

---

## 2. Deployment Steps on Streamlit Community Cloud

1. **Connect GitHub Repository**:
   - Go to [share.streamlit.io](https://share.streamlit.io).
   - Select your repository: `PRIYAtechky/Invoice-Intelligence-OCR-Extraction-and-Payment-Fraud-Detection`.
   - Branch: `main`.
   - Main file path: `app.py`.

2. **Configure Secrets in Streamlit Dashboard**:
   - In your Streamlit Cloud app settings, click on **Secrets**.
   - Add your Gemini API key:
     ```toml
     GEMINI_API_KEY = "your_actual_gemini_api_key_here"
     GEMINI_MODEL = "gemini-3.8-flash"
     ENVIRONMENT = "production"
     ```
   - Click **Save**.

3. **Deploy**:
   - Click **Deploy!**. Streamlit Cloud will install packages from `requirements.txt` and launch the application.

---

## 3. Database Persistence in Production

Streamlit Community Cloud containers are ephemeral (their local disk resets upon reboot or container redeployment).

- **Default (Local Evaluation)**: Uses `sqlite:///invoice_intelligence.db` which is created and migrated automatically on launch.
- **Production Persistence**: Set the `DATABASE_URL` secret to a persistent PostgreSQL instance (such as Supabase, Neon.tech, AWS RDS, or Google Cloud SQL):
  ```toml
  DATABASE_URL = "postgresql://user:password@host:5432/invoicedb"
  ```
  The SQLAlchemy models in `src/database/models.py` are dialect-agnostic and support PostgreSQL out of the box.

---

## 4. OCR Engine on Streamlit Cloud

The platform uses **RapidOCR ONNX Runtime (PaddleOCR models)** as its primary OCR engine:
- Pure Python and ONNX C++ runtime wheels (no system package installations or apt-get binaries needed).
- Operates within standard Streamlit Cloud 1GB RAM limits.
- If Tesseract is additionally desired, add `packages.txt` with `tesseract-ocr`.
