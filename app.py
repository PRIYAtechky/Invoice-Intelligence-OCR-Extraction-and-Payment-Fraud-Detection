"""Invoice Intelligence — Enterprise AP Automation & Fraud Detection Platform
Clean, minimal, enterprise invoice processing and forensic audit application.
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import streamlit as st
import cv2
from PIL import Image

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config.settings import settings
from config.logging_config import logger
from src.database.session import init_db, get_db_session
from src.database.repository import InvoiceRepository
from src.pipeline.invoice_pipeline import InvoiceProcessingPipeline
from src.pipeline.batch_processor import BatchProcessor
from src.extraction.schemas import ManualInvoiceInput, LineItem
from src.ocr.ocr_manager import ocr_manager
from src.ocr.evaluator import evaluate_ocr_engines
from src.extraction.evaluator import evaluate_extraction_accuracy
from src.fraud.vendor_risk import VendorRiskScorer
from src.extraction.gemini_extractor import GeminiExtractor

# Initialize database on application start
init_db()

# Page configuration
st.set_page_config(
    page_title="Invoice Intelligence — Enterprise AP & Fraud Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for navigation and review selection
if "active_nav" not in st.session_state:
    st.session_state["active_nav"] = "Process Invoice"
if "view_invoice_id" not in st.session_state:
    st.session_state["view_invoice_id"] = None
if "show_manual_entry" not in st.session_state:
    st.session_state["show_manual_entry"] = False
if "show_batch_upload" not in st.session_state:
    st.session_state["show_batch_upload"] = False

# -----------------------------------------------------------------------------
# Enterprise Design System: Dark Navy Sidebar + Light Gray & Blue Content
# Strict rule: No emojis anywhere in the interface.
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        color: #0F172A;
    }

    /* Main background */
    .stApp {
        background-color: #F8FAFC;
    }

    /* Dark Navy Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0F172A !important;
        border-right: 1px solid #1E293B;
    }
    section[data-testid="stSidebar"] div, 
    section[data-testid="stSidebar"] p, 
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label {
        color: #E2E8F0 !important;
    }
    section[data-testid="stSidebar"] hr {
        border-color: #1E293B !important;
    }

    /* Sidebar Radio Navigation */
    section[data-testid="stSidebar"] .stRadio > div {
        gap: 6px;
    }
    section[data-testid="stSidebar"] .stRadio label {
        padding: 10px 14px;
        border-radius: 8px;
        font-weight: 500;
        font-size: 0.92rem;
        transition: all 0.15s ease;
        background: transparent;
        cursor: pointer;
    }
    section[data-testid="stSidebar"] .stRadio label:hover {
        background-color: #1E293B;
        color: #FFFFFF !important;
    }

    /* Content Cards */
    .content-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 24px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        margin-bottom: 20px;
    }

    /* Headers */
    .view-title {
        font-size: 1.55rem;
        font-weight: 700;
        color: #0F172A;
        letter-spacing: -0.02em;
        margin-bottom: 4px;
    }
    .view-subtitle {
        font-size: 0.92rem;
        color: #64748B;
        margin-bottom: 24px;
    }

    /* Metric Cards */
    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px 20px;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
    }
    .kpi-title {
        font-size: 0.78rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .kpi-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #0F172A;
        margin: 4px 0 2px 0;
        letter-spacing: -0.02em;
    }
    .kpi-sub {
        font-size: 0.8rem;
        color: #94A3B8;
    }

    /* Status Pills */
    .pill {
        display: inline-flex;
        align-items: center;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        line-height: 1.2;
    }
    .pill-passed { background: #ECFDF5; color: #065F46; border: 1px solid #A7F3D0; }
    .pill-warning { background: #FFFBEB; color: #92400E; border: 1px solid #FDE68A; }
    .pill-failed { background: #FEF2F2; color: #991B1B; border: 1px solid #FECACA; }
    .pill-info { background: #EFF6FF; color: #1E40AF; border: 1px solid #BFDBFE; }
    .pill-neutral { background: #F1F5F9; color: #475569; border: 1px solid #E2E8F0; }

    /* Button Customization */
    .stButton>button {
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.88rem;
        padding: 8px 18px;
        transition: background-color 0.15s ease, border-color 0.15s ease;
    }
    .stButton>button[kind="primary"] {
        background-color: #2563EB;
        color: #FFFFFF;
        border: 1px solid #2563EB;
    }
    .stButton>button[kind="primary"]:hover {
        background-color: #1D4ED8;
        border-color: #1D4ED8;
    }
    .stButton>button[kind="secondary"] {
        background-color: #FFFFFF;
        color: #334155;
        border: 1px solid #CBD5E1;
    }
    .stButton>button[kind="secondary"]:hover {
        background-color: #F8FAFC;
        border-color: #94A3B8;
    }

    /* Pipeline Step Tracker */
    .step-track {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: #F1F5F9;
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        padding: 10px 16px;
        margin: 16px 0;
        font-size: 0.82rem;
        font-weight: 600;
        color: #475569;
    }
    .step-node {
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .step-complete {
        color: #166534;
    }
    .step-active {
        color: #2563EB;
    }

    /* Clean file upload zone styling */
    div[data-testid="stFileUploader"] {
        border: 1px dashed #CBD5E1;
        border-radius: 8px;
        padding: 12px;
        background: #FAFAFA;
        transition: border-color 0.15s;
    }
    div[data-testid="stFileUploader"]:hover {
        border-color: #2563EB;
    }

    /* Form Inputs */
    .stTextInput input, .stNumberInput input, .stDateInput input, .stSelectbox select {
        border-radius: 6px;
        border: 1px solid #CBD5E1;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Clean Sidebar Navigation (3 Main Views + Settings)
# -----------------------------------------------------------------------------
st.sidebar.markdown("""
<div style="padding: 12px 6px 18px 6px;">
    <div style="font-size: 1.15rem; font-weight: 800; color: #FFFFFF; letter-spacing: -0.02em;">INVOICE INTELLIGENCE</div>
    <div style="font-size: 0.72rem; color: #94A3B8; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 2px;">Enterprise Accounts Payable</div>
</div>
""", unsafe_allow_html=True)

# 3 Primary Navigation Items
NAV_OPTIONS = [
    "Process Invoice",
    "Invoice History",
    "Fraud Detection"
]

# Check if an explicit navigation request was made
current_nav_index = 0
if st.session_state["active_nav"] in NAV_OPTIONS:
    current_nav_index = NAV_OPTIONS.index(st.session_state["active_nav"])

selected_nav = st.sidebar.radio(
    "Navigation Menu",
    NAV_OPTIONS,
    index=current_nav_index,
    label_visibility="collapsed"
)

# Update session state if changed via sidebar
if selected_nav != st.session_state["active_nav"]:
    st.session_state["active_nav"] = selected_nav
    st.session_state["view_invoice_id"] = None

st.sidebar.markdown("<br>", unsafe_allow_html=True)
st.sidebar.markdown("---")

# Active Approver Role Profile
ROLES_CONFIG = {
    "AP-Clerk-01": {"limit": 200000.0, "title": "Accounts Payable Clerk"},
    "AP-Clerk-02": {"limit": 200000.0, "title": "Accounts Payable Clerk"},
    "AP-Senior-01": {"limit": 500000.0, "title": "Senior AP Specialist"},
    "AP-Manager-01": {"limit": 1500000.0, "title": "Finance Manager"},
    "Finance-Head": {"limit": 50000000.0, "title": "VP Finance / CFO"}
}

approver_role = st.sidebar.selectbox(
    "Approver Profile",
    list(ROLES_CONFIG.keys()),
    index=0
)
current_role_info = ROLES_CONFIG[approver_role]
formatted_limit = f"INR {current_role_info['limit']:,.0f}" if current_role_info['limit'] < 10000000 else "Unlimited"

st.sidebar.markdown(f"""
<div style="background: #1E293B; border-radius: 6px; padding: 10px 12px; margin-top: -6px; border: 1px solid #334155;">
    <div style="font-size: 0.72rem; color: #94A3B8; font-weight: 600; text-transform: uppercase;">Approval Authority</div>
    <div style="font-size: 0.92rem; font-weight: 700; color: #FFFFFF; margin-top: 2px;">{formatted_limit}</div>
    <div style="font-size: 0.75rem; color: #CBD5E1; margin-top: 2px;">{current_role_info['title']}</div>
</div>
""", unsafe_allow_html=True)

# Engine Status Check
gemini_ext = GeminiExtractor()
gemini_active = gemini_ext.is_configured()

st.sidebar.markdown("---")

# Settings Trigger at the Bottom of Sidebar
settings_active = (st.session_state["active_nav"] == "Settings")
if st.sidebar.button("Settings", use_container_width=True, type="secondary" if not settings_active else "primary"):
    st.session_state["active_nav"] = "Settings"
    st.session_state["view_invoice_id"] = None
    st.rerun()

st.sidebar.markdown(f"""
<div style="padding: 6px 2px; font-size: 0.75rem; color: #64748B; line-height: 1.6;">
    <div>OCR: <b>{settings.DEFAULT_OCR_ENGINE.upper()}</b></div>
    <div>AI Extraction: <b>{'Online (Gemini)' if gemini_active else 'Rule-Based (Offline)'}</b></div>
    <div>Database: <b>SQLite Active</b></div>
</div>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Reusable Component: Detailed Invoice Review & Authorization
# -----------------------------------------------------------------------------
def render_invoice_review(invoice_id: str, origin_view: str = "Process Invoice"):
    """Render two-column document inspection, editable extracted data, and review actions."""
    with get_db_session() as session:
        repo = InvoiceRepository(session)
        inv = repo.get_invoice(invoice_id)

    if not inv:
        st.error(f"Invoice record '{invoice_id}' not found in repository.")
        if st.button("Return"):
            st.session_state["view_invoice_id"] = None
            st.rerun()
        return

    # Back Header
    b_col1, b_col2 = st.columns([1, 5])
    with b_col1:
        if st.button("Back to Invoices", key=f"back_{invoice_id}"):
            st.session_state["view_invoice_id"] = None
            st.rerun()

    # Top Status Banner
    st.markdown(f"""
    <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 14px 20px; margin-bottom: 20px; display: flex; justify-content: space-between; align-items: center;">
        <div>
            <span style="font-size: 1.25rem; font-weight: 700; color: #0F172A;">Invoice {inv.invoice_number or inv.id}</span>
            <span style="font-size: 0.9rem; color: #64748B; margin-left: 12px;">ID: {inv.id} &bull; Vendor: <b>{inv.vendor_name or 'Unknown'}</b></span>
        </div>
        <div style="display: flex; gap: 10px; align-items: center;">
            <span class="pill {'pill-passed' if inv.status == 'APPROVED' else ('pill-failed' if inv.status == 'REJECTED' else 'pill-warning')}">
                {inv.status}
            </span>
            <span class="pill {'pill-passed' if inv.validation_status == 'Passed' else 'pill-failed'}">
                Validation: {inv.validation_status}
            </span>
            <span class="pill {'pill-passed' if not inv.has_fraud_risk else 'pill-failed'}">
                Risk: {inv.fraud_risk_score}/100 [{inv.fraud_severity}]
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Authority Limit Check
    invoice_val = float(inv.grand_total or 0.0)
    user_limit = current_role_info["limit"]
    is_over_limit = invoice_val > user_limit

    if is_over_limit:
        st.warning(f"Authorization Notice: Invoice total (INR {invoice_val:,.2f}) exceeds your active role authority limit ({formatted_limit}). Approval requires escalation to a Senior Specialist or Finance Manager.")

    # Two-Column Layout: Left = Document, Right = Extracted Fields Form
    col_doc, col_data = st.columns([1, 1], gap="large")

    with col_doc:
        st.markdown("#### Document Scan")
        view_mode = st.radio("Image View", ["Original Scan", "Preprocessed (Deskewed & Binarized)"], horizontal=True, label_visibility="collapsed")
        
        target_path = inv.original_image_path
        if view_mode.startswith("Preprocessed") and inv.processed_image_path and os.path.exists(inv.processed_image_path):
            target_path = inv.processed_image_path

        if target_path and os.path.exists(target_path):
            st.image(target_path, use_container_width=True)
        else:
            st.info("No physical image file on record (e.g. manual entry or missing upload path).")

    with col_data:
        st.markdown("#### Extracted Information & Review")
        with st.form(f"form_review_{inv.id}"):
            f_vendor = st.text_input("Vendor Legal Name", value=inv.vendor_name or "")
            f_gstin = st.text_input("Vendor GSTIN", value=inv.vendor_gstin or "")

            c_n1, c_n2 = st.columns(2)
            f_inv_no = c_n1.text_input("Invoice Number", value=inv.invoice_number or "")
            f_date = c_n2.text_input("Invoice Date (YYYY-MM-DD)", value=inv.invoice_date or "")

            c_p1, c_p2 = st.columns(2)
            f_po = c_p1.text_input("Purchase Order #", value=inv.po_number or "")
            f_bank = c_p2.text_input("Bank Account Last 4 Digits", value=inv.bank_account_last4 or "")

            c_t1, c_t2 = st.columns(2)
            f_subtotal = c_t1.number_input("Subtotal (INR)", value=float(inv.subtotal or 0.0), step=100.0)
            f_grand = c_t2.number_input("Grand Total (INR)", value=float(inv.grand_total or 0.0), step=100.0)

            st.markdown("##### Reviewer Decision")
            decision = st.selectbox(
                "Reviewer Action",
                ["APPROVED", "REJECTED", "FLAGGED_FOR_REVIEW"],
                index=0 if inv.status == "APPROVED" else (1 if inv.status == "REJECTED" else 2)
            )
            reviewer_notes = st.text_area("Reviewer Audit Notes / Justification", value=inv.reviewer_comments or "")

            btn_save = st.form_submit_button("Record Review Decision & Save", type="primary")

        if btn_save:
            if decision == "APPROVED" and (inv.has_fraud_risk and inv.fraud_severity == "CRITICAL"):
                st.error("Approval Blocked: Invoices with CRITICAL fraud flags cannot be approved. Reject or escalate.")
            elif decision == "APPROVED" and is_over_limit:
                st.error(f"Approval Blocked: Amount exceeds your role threshold of {formatted_limit}.")
            else:
                with get_db_session() as save_session:
                    save_repo = InvoiceRepository(save_session)
                    save_repo.update_invoice_review(
                        invoice_id=inv.id,
                        decision=decision,
                        reviewer_name=f"{approver_role} ({current_role_info['title']})",
                        comments=reviewer_notes,
                        edited_fields={
                            "vendor_name": f_vendor,
                            "vendor_gstin": f_gstin,
                            "invoice_number": f_inv_no,
                            "invoice_date": f_date,
                            "po_number": f_po,
                            "bank_account_last4": f_bank,
                            "subtotal": f_subtotal,
                            "grand_total": f_grand
                        }
                    )
                st.success(f"Review decision '{decision}' recorded with immutable audit trail.")
                st.rerun()

    # Bottom Sections: Validation, Fraud Findings, Line Items, and Audit History
    st.markdown("---")
    t_val, t_fraud, t_items, t_audit = st.tabs(["Validation Engine Results", "Fraud Detection & Risk Findings", "Line Items", "Audit Trail"])

    with t_val:
        st.markdown("##### Statutory Validation Checks (9 Rules)")
        for vr in inv.validations:
            pill_class = "pill-passed" if vr.status == "Passed" else ("pill-warning" if vr.status == "Warning" else "pill-failed")
            st.markdown(f"""
            <div style="padding: 8px 0; border-bottom: 1px solid #F1F5F9; display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <b>{vr.rule_name}</b> <span style="font-size: 0.8rem; color: #64748B;">({vr.rule_id})</span>
                    <div style="font-size: 0.85rem; color: #334155; margin-top: 2px;">{vr.message}</div>
                </div>
                <div><span class="pill {pill_class}">{vr.status}</span></div>
            </div>
            """, unsafe_allow_html=True)

    with t_fraud:
        st.markdown("##### Payment Fraud Analysis & Evidence")
        if not inv.fraud_alerts:
            st.info("No active fraud patterns or abnormal payment indicators detected.")
        else:
            for fa in inv.fraud_alerts:
                st.markdown(f"""
                <div style="background: #FEF2F2; border: 1px solid #FECACA; border-radius: 6px; padding: 12px 16px; margin-bottom: 12px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <b style="color: #991B1B;">{fa.pattern_name}</b>
                        <span class="pill pill-failed">Severity: {fa.severity} &bull; Score: {fa.risk_score}</span>
                    </div>
                    <div style="font-size: 0.85rem; color: #450A0A; margin-top: 6px;"><b>Evidence:</b> {fa.explanation}</div>
                    <div style="font-size: 0.82rem; color: #7F1D1D; margin-top: 4px;"><b>Recommended Action:</b> {fa.recommended_action}</div>
                </div>
                """, unsafe_allow_html=True)

    with t_items:
        if inv.line_items:
            items_list = []
            for item in inv.line_items:
                items_list.append({
                    "Description": item.description,
                    "HSN Code": item.hsn or "N/A",
                    "Quantity": item.quantity,
                    "Unit Price (INR)": f"{item.unit_price:,.2f}",
                    "Amount (INR)": f"{item.amount:,.2f}"
                })
            st.table(items_list)
        else:
            st.info("No line items extracted for this invoice.")

    with t_audit:
        with get_db_session() as session:
            repo = InvoiceRepository(session)
            audits = repo.get_audit_trail_for_invoice(inv.id)
        if audits:
            for a in audits:
                st.markdown(f"""
                <div style="font-size: 0.84rem; padding: 6px 0; border-bottom: 1px solid #F1F5F9;">
                    <span style="color: #64748B;">{a.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</span> &bull; 
                    <b>{a.action}</b> by <code>{a.user}</code>
                    <div style="color: #334155; margin-top: 2px;">{a.comments or 'No notes provided.'}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No audit entries recorded.")


# -----------------------------------------------------------------------------
# 1. Page 1 — Process Invoice (Main Landing Page)
# -----------------------------------------------------------------------------
if st.session_state["active_nav"] == "Process Invoice":
    # If currently inspecting an invoice, show review screen
    if st.session_state["view_invoice_id"]:
        render_invoice_review(st.session_state["view_invoice_id"], origin_view="Process Invoice")
    else:
        st.markdown('<div class="view-title">Process an Invoice</div>', unsafe_allow_html=True)
        st.markdown('<div class="view-subtitle">Upload supplier invoices for automated multi-engine OCR extraction, statutory compliance verification, and fraud detection.</div>', unsafe_allow_html=True)

        col_left, col_right = st.columns([1, 1], gap="large")

        # Left Column: Upload and Ingestion
        with col_left:
            st.markdown("""
            <div style="margin-bottom: 12px; font-weight: 600; font-size: 1rem; color: #0F172A;">
                Upload Invoice Document
            </div>
            """, unsafe_allow_html=True)

            uploaded_file = st.file_uploader(
                "Select invoice file",
                type=["pdf", "png", "jpg", "jpeg"],
                help="Supported: PDF, JPG, PNG up to 25MB.",
                label_visibility="collapsed"
            )

            # Processing Options (Clean & Minimal)
            c_opt1, c_opt2 = st.columns(2)
            with c_opt1:
                ocr_choice = st.selectbox("OCR Engine", ["rapidocr", "easyocr"], index=0, help="RapidOCR uses PaddleOCR ONNX models.")
            with c_opt2:
                use_llm = st.checkbox("Gemini AI Extraction", value=gemini_active, disabled=not gemini_active)

            if uploaded_file is not None:
                # Save uploaded file
                temp_dir = settings.DATA_DIR / "uploads"
                temp_dir.mkdir(parents=True, exist_ok=True)
                temp_file_path = temp_dir / uploaded_file.name

                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                file_size_kb = len(uploaded_file.getbuffer()) / 1024.0
                st.markdown(f"""
                <div style="background: #F1F5F9; border: 1px solid #E2E8F0; border-radius: 6px; padding: 10px 14px; margin: 12px 0; font-size: 0.85rem; display: flex; justify-content: space-between;">
                    <div><b>{uploaded_file.name}</b> ({file_size_kb:.1f} KB)</div>
                    <div style="color: #166534; font-weight: 600;">Ready to Process</div>
                </div>
                """, unsafe_allow_html=True)

                if st.button("Process Invoice", type="primary", use_container_width=True):
                    # Step Tracker Visualizer
                    step_placeholder = st.empty()
                    step_placeholder.markdown("""
                    <div class="step-track">
                        <span class="step-node step-active">1. Preprocessing</span>
                        <span>&rarr;</span>
                        <span class="step-node">2. Multi-Engine OCR</span>
                        <span>&rarr;</span>
                        <span class="step-node">3. Extraction</span>
                        <span>&rarr;</span>
                        <span class="step-node">4. Validation</span>
                        <span>&rarr;</span>
                        <span class="step-node">5. Fraud Scan</span>
                    </div>
                    """, unsafe_allow_html=True)

                    with get_db_session() as session:
                        pipeline = InvoiceProcessingPipeline(session=session)
                        res = pipeline.process_image(
                            image_path=str(temp_file_path),
                            engine_name=ocr_choice,
                            use_llm=use_llm,
                            input_method="upload",
                            approver_role=approver_role
                        )

                    step_placeholder.markdown("""
                    <div class="step-track">
                        <span class="step-node step-complete">1. Preprocessed</span>
                        <span>&rarr;</span>
                        <span class="step-node step-complete">2. OCR Completed</span>
                        <span>&rarr;</span>
                        <span class="step-node step-complete">3. Data Extracted</span>
                        <span>&rarr;</span>
                        <span class="step-node step-complete">4. Validated</span>
                        <span>&rarr;</span>
                        <span class="step-node step-complete">5. Fraud Checked</span>
                    </div>
                    """, unsafe_allow_html=True)

                    st.session_state["view_invoice_id"] = res["invoice_id"]
                    st.rerun()

            st.markdown("<br>", unsafe_allow_html=True)

            # Secondary Option: Manual Invoice Entry Toggle
            t_col1, t_col2 = st.columns(2)
            with t_col1:
                if st.button("Manual Invoice Entry", type="secondary", use_container_width=True):
                    st.session_state["show_manual_entry"] = not st.session_state["show_manual_entry"]
            with t_col2:
                if st.button("Batch Folder Ingestion", type="secondary", use_container_width=True):
                    st.session_state["show_batch_upload"] = not st.session_state["show_batch_upload"]

            # Collapsible Manual Entry Form
            if st.session_state["show_manual_entry"]:
                st.markdown("---")
                st.markdown("##### Manual Invoice Form")
                with st.form("manual_entry_form"):
                    m_v = st.text_input("Vendor Legal Name*", placeholder="e.g. Vel Engineering Works")
                    m_g = st.text_input("Vendor GSTIN*", placeholder="15-character statutory GSTIN")
                    c_m1, c_m2 = st.columns(2)
                    m_inv = c_m1.text_input("Invoice Number*", placeholder="e.g. TI/2025/8093")
                    m_date = c_m2.date_input("Invoice Date*", value=datetime.today())
                    m_po = c_m1.text_input("Purchase Order #", placeholder="e.g. KIS/PO/2024/10001")
                    m_bank = c_m2.text_input("Bank Account Last 4 Digits", max_chars=4, placeholder="e.g. 5029")

                    c_amt1, c_amt2 = st.columns(2)
                    m_sub = c_amt1.number_input("Subtotal (INR)*", min_value=0.0, step=100.0)
                    m_grand = c_amt2.number_input("Grand Total (INR)*", min_value=0.0, step=100.0)

                    # Dynamic Line Item
                    st.markdown("Line Item Details")
                    li_desc = st.text_input("Item Description", value="Industrial Ball Bearing 6205-2RS")
                    c_li1, c_li2, c_li3 = st.columns(3)
                    li_hsn = c_li1.text_input("HSN Code", value="8482")
                    li_qty = c_li2.number_input("Quantity", min_value=1.0, value=5.0)
                    li_rate = c_li3.number_input("Unit Rate (INR)", min_value=0.0, value=m_sub/5.0 if m_sub else 100.0)

                    btn_manual = st.form_submit_button("Submit & Validate Invoice", type="primary")

                if btn_manual:
                    if not m_v or not m_g or not m_inv or m_grand <= 0:
                        st.error("Please fill in all mandatory fields marked with an asterisk (*).")
                    else:
                        line_item = LineItem(
                            description=li_desc,
                            hsn=li_hsn,
                            quantity=li_qty,
                            unit_price=li_rate,
                            amount=li_qty * li_rate
                        )
                        data_input = ManualInvoiceInput(
                            vendor_name=m_v,
                            vendor_gstin=m_g,
                            invoice_number=m_inv,
                            invoice_date=str(m_date),
                            due_date=str(m_date),
                            po_number=m_po,
                            line_items=[line_item],
                            subtotal=m_sub,
                            cgst=0.0,
                            sgst=0.0,
                            igst=0.0,
                            grand_total=m_grand,
                            bank_account_last4=m_bank
                        )
                        with get_db_session() as session:
                            pipeline = InvoiceProcessingPipeline(session=session)
                            res = pipeline.process_manual(data_input, approver_role=approver_role)

                        st.session_state["show_manual_entry"] = False
                        st.session_state["view_invoice_id"] = res["invoice_id"]
                        st.rerun()

            # Collapsible Batch Upload Section
            if st.session_state["show_batch_upload"]:
                st.markdown("---")
                st.markdown("##### Batch Processing")
                st.markdown(f"Incoming Batch Directory: `02_incoming_batches/`")
                if st.button("Process Incoming Directory"):
                    with st.spinner("Processing directory..."):
                        with get_db_session() as session:
                            batch_proc = BatchProcessor(session=session)
                            b_res = batch_proc.process_directory(str(settings.INCOMING_DIR), use_llm=gemini_active)
                    if b_res.get("total_files", 0) > 0:
                        st.success(f"Batch completed: {b_res['success_count']} successful, {b_res['failed_count']} failed.")
                    else:
                        st.info("No unhandled files in incoming directory.")

        # Right Column: Professional Invoice Scanning Visual (Navy, White, Light Blue)
        with col_right:
            st.markdown("""
            <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 10px; padding: 24px; box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04); height: 100%;">
                <div style="font-size: 0.85rem; font-weight: 700; color: #1E40AF; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 12px; display: flex; justify-content: space-between;">
                    <span>Document Processing Engine</span>
                    <span>Multi-Engine OCR Active</span>
                </div>
                
                <!-- Realistic Enterprise Invoice Scanning SVG Graphic -->
                <svg viewBox="0 0 460 520" width="100%" height="auto" style="border-radius: 6px; background: #F8FAFC; border: 1px solid #E2E8F0;">
                    <!-- Invoice Document Paper -->
                    <rect x="25" y="20" width="410" height="480" rx="4" fill="#FFFFFF" stroke="#CBD5E1" stroke-width="1.5" />
                    
                    <!-- Document Header -->
                    <rect x="50" y="45" width="140" height="12" rx="2" fill="#0F172A" />
                    <rect x="50" y="65" width="90" height="7" rx="1.5" fill="#64748B" />
                    <rect x="50" y="78" width="110" height="7" rx="1.5" fill="#94A3B8" />
                    
                    <!-- TAX INVOICE Title -->
                    <rect x="290" y="45" width="120" height="16" rx="2" fill="#2563EB" opacity="0.15" />
                    <text x="350" y="57" font-family="Inter, sans-serif" font-size="10" font-weight="700" fill="#2563EB" text-anchor="middle">TAX INVOICE</text>
                    <rect x="310" y="68" width="100" height="7" rx="1.5" fill="#64748B" />
                    <rect x="330" y="80" width="80" height="7" rx="1.5" fill="#94A3B8" />

                    <!-- Divider -->
                    <line x1="50" y1="102" x2="410" y2="102" stroke="#E2E8F0" stroke-width="1.5" />

                    <!-- OCR Bounding Box 1: Vendor -->
                    <rect x="46" y="115" width="180" height="60" rx="3" fill="#EFF6FF" stroke="#3B82F6" stroke-width="1.2" stroke-dasharray="3,2" />
                    <text x="52" y="126" font-family="Inter, sans-serif" font-size="7.5" font-weight="700" fill="#1D4ED8">[OCR: VENDOR LEGAL NAME]</text>
                    <rect x="52" y="134" width="130" height="8" rx="2" fill="#1E293B" />
                    <rect x="52" y="148" width="150" height="6" rx="1.5" fill="#64748B" />
                    <rect x="52" y="159" width="110" height="6" rx="1.5" fill="#64748B" />

                    <!-- OCR Bounding Box 2: Metadata (Invoice No, PO, Date) -->
                    <rect x="236" y="115" width="176" height="60" rx="3" fill="#EFF6FF" stroke="#3B82F6" stroke-width="1.2" stroke-dasharray="3,2" />
                    <text x="242" y="126" font-family="Inter, sans-serif" font-size="7.5" font-weight="700" fill="#1D4ED8">[OCR: INVOICE & PO #]</text>
                    <rect x="242" y="134" width="120" height="8" rx="2" fill="#1E293B" />
                    <rect x="242" y="148" width="100" height="6" rx="1.5" fill="#64748B" />
                    <rect x="242" y="159" width="90" height="6" rx="1.5" fill="#64748B" />

                    <!-- Table Header -->
                    <rect x="50" y="195" width="360" height="20" rx="2" fill="#F1F5F9" />
                    <text x="60" y="209" font-family="Inter, sans-serif" font-size="8" font-weight="600" fill="#475569">ITEM DESCRIPTION</text>
                    <text x="225" y="209" font-family="Inter, sans-serif" font-size="8" font-weight="600" fill="#475569">HSN</text>
                    <text x="280" y="209" font-family="Inter, sans-serif" font-size="8" font-weight="600" fill="#475569">QTY</text>
                    <text x="330" y="209" font-family="Inter, sans-serif" font-size="8" font-weight="600" fill="#475569">RATE</text>
                    <text x="385" y="209" font-family="Inter, sans-serif" font-size="8" font-weight="600" fill="#475569">AMOUNT</text>

                    <!-- Table Rows -->
                    <line x1="50" y1="230" x2="410" y2="230" stroke="#F1F5F9" stroke-width="1" />
                    <rect x="60" y="222" width="120" height="7" rx="1.5" fill="#334155" />
                    <rect x="225" y="222" width="30" height="7" rx="1.5" fill="#64748B" />
                    <rect x="280" y="222" width="20" height="7" rx="1.5" fill="#64748B" />
                    <rect x="330" y="222" width="30" height="7" rx="1.5" fill="#64748B" />
                    <rect x="380" y="222" width="25" height="7" rx="1.5" fill="#334155" />

                    <line x1="50" y1="248" x2="410" y2="248" stroke="#F1F5F9" stroke-width="1" />
                    <rect x="60" y="240" width="100" height="7" rx="1.5" fill="#334155" />
                    <rect x="225" y="240" width="30" height="7" rx="1.5" fill="#64748B" />
                    <rect x="280" y="240" width="20" height="7" rx="1.5" fill="#64748B" />
                    <rect x="330" y="240" width="30" height="7" rx="1.5" fill="#64748B" />
                    <rect x="380" y="240" width="25" height="7" rx="1.5" fill="#334155" />

                    <line x1="50" y1="266" x2="410" y2="266" stroke="#F1F5F9" stroke-width="1" />
                    <rect x="60" y="258" width="110" height="7" rx="1.5" fill="#334155" />
                    <rect x="225" y="258" width="30" height="7" rx="1.5" fill="#64748B" />
                    <rect x="280" y="258" width="20" height="7" rx="1.5" fill="#64748B" />
                    <rect x="330" y="258" width="30" height="7" rx="1.5" fill="#64748B" />
                    <rect x="380" y="258" width="25" height="7" rx="1.5" fill="#334155" />

                    <!-- OCR Bounding Box 3: Grand Total -->
                    <rect x="245" y="295" width="165" height="75" rx="3" fill="#EFF6FF" stroke="#2563EB" stroke-width="1.5" />
                    <text x="253" y="310" font-family="Inter, sans-serif" font-size="8" font-weight="700" fill="#1E40AF">[STATUTORY RECONCILIATION]</text>
                    <text x="253" y="326" font-family="Inter, sans-serif" font-size="8" fill="#475569">Taxable Subtotal:</text>
                    <rect x="345" y="320" width="55" height="7" rx="1.5" fill="#334155" />
                    <text x="253" y="341" font-family="Inter, sans-serif" font-size="8" fill="#475569">CGST (9%) + SGST (9%):</text>
                    <rect x="360" y="335" width="40" height="7" rx="1.5" fill="#334155" />
                    <text x="253" y="358" font-family="Inter, sans-serif" font-size="9" font-weight="700" fill="#0F172A">GRAND TOTAL:</text>
                    <rect x="335" y="350" width="65" height="11" rx="2" fill="#2563EB" />

                    <!-- Bank Details Box -->
                    <rect x="50" y="300" width="180" height="65" rx="3" fill="#F8FAFC" stroke="#E2E8F0" stroke-width="1" />
                    <text x="58" y="315" font-family="Inter, sans-serif" font-size="8" font-weight="700" fill="#0F172A">BANK REMITTANCE DETAILS</text>
                    <text x="58" y="330" font-family="Inter, sans-serif" font-size="7.5" fill="#64748B">A/C: **** **** **** 5029</text>
                    <text x="58" y="344" font-family="Inter, sans-serif" font-size="7.5" fill="#64748B">IFSC: HDFC0001824</text>
                    <rect x="58" y="352" width="60" height="5" rx="1" fill="#16A34A" />

                    <!-- Scanning Laser Beam -->
                    <defs>
                        <linearGradient id="scanGradient" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="0%" stop-color="#2563EB" stop-opacity="0" />
                            <stop offset="50%" stop-color="#3B82F6" stop-opacity="0.3" />
                            <stop offset="100%" stop-color="#2563EB" stop-opacity="0.9" />
                        </linearGradient>
                    </defs>
                    <rect x="25" y="210" width="410" height="28" fill="url(#scanGradient)" />
                    <line x1="25" y1="238" x2="435" y2="238" stroke="#2563EB" stroke-width="2.5" />
                    
                    <!-- Scan Pulse Indicator -->
                    <circle cx="410" cy="238" r="4" fill="#2563EB" />
                    <text x="418" y="241" font-family="Inter, sans-serif" font-size="8" font-weight="700" fill="#2563EB">OCR SCANNING</text>

                    <!-- Bottom Verification Badge -->
                    <rect x="50" y="415" width="360" height="40" rx="4" fill="#F8FAFC" stroke="#E2E8F0" stroke-width="1" />
                    <text x="64" y="432" font-family="Inter, sans-serif" font-size="9" font-weight="700" fill="#0F172A">Intelligent Document Processing (IDP)</text>
                    <text x="64" y="445" font-family="Inter, sans-serif" font-size="8" fill="#64748B">Deskewing &bull; Noise Suppression &bull; 9 Statutory Rules &bull; Fraud Shield</text>
                </svg>
                
                <div style="font-size: 0.8rem; color: #64748B; margin-top: 14px; text-align: center;">
                    Compliant with Indian statutory GST standards &bull; 3-Way PO matching &bull; Immutable audit logging
                </div>
            </div>
            """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. Page 2 — Invoice History
# -----------------------------------------------------------------------------
elif st.session_state["active_nav"] == "Invoice History":
    if st.session_state["view_invoice_id"]:
        render_invoice_review(st.session_state["view_invoice_id"], origin_view="Invoice History")
    else:
        st.markdown('<div class="view-title">Invoice History</div>', unsafe_allow_html=True)
        st.markdown('<div class="view-subtitle">Search, filter, and inspect processed invoices, statutory validation records, and audit events.</div>', unsafe_allow_html=True)

        with get_db_session() as session:
            repo = InvoiceRepository(session)
            
            c_s1, c_s2, c_s3 = st.columns([2, 1, 1])
            search_query = c_s1.text_input("Search Invoices", placeholder="Search by Vendor, Invoice #, or PO #")
            filter_status = c_s2.selectbox("Status", ["ALL", "PENDING_REVIEW", "APPROVED", "REJECTED", "FLAGGED"])
            show_fraud_only = c_s3.checkbox("Flagged Fraud Only", value=False)

            invoices = repo.list_invoices(
                status=None if filter_status == "ALL" else filter_status,
                vendor=search_query if search_query else None,
                has_fraud=True if show_fraud_only else None,
                limit=250
            )

        if not invoices:
            st.info("No invoice records match the selected query or filter criteria.")
        else:
            # Summary Metrics Bar
            total_count = len(invoices)
            approved_count = sum(1 for i in invoices if i.status == "APPROVED")
            pending_count = sum(1 for i in invoices if i.status == "PENDING_REVIEW")
            flagged_count = sum(1 for i in invoices if i.has_fraud_risk)

            m1, m2, m3, m4 = st.columns(4)
            m1.markdown(f"""<div class="kpi-card"><div class="kpi-title">Total Filtered</div><div class="kpi-value">{total_count}</div></div>""", unsafe_allow_html=True)
            m2.markdown(f"""<div class="kpi-card"><div class="kpi-title">Approved</div><div class="kpi-value" style="color: #166534;">{approved_count}</div></div>""", unsafe_allow_html=True)
            m3.markdown(f"""<div class="kpi-card"><div class="kpi-title">Pending Review</div><div class="kpi-value" style="color: #D97706;">{pending_count}</div></div>""", unsafe_allow_html=True)
            m4.markdown(f"""<div class="kpi-card"><div class="kpi-title">Fraud Risk</div><div class="kpi-value" style="color: #DC2626;">{flagged_count}</div></div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Clean Table
            table_rows = []
            for inv in invoices:
                table_rows.append({
                    "Invoice ID": inv.id,
                    "Vendor": inv.vendor_name or "Unknown",
                    "Invoice No": inv.invoice_number or "N/A",
                    "Date": inv.invoice_date or "N/A",
                    "Amount (INR)": float(inv.grand_total or 0.0),
                    "Validation": inv.validation_status or "Needs Review",
                    "Fraud Severity": inv.fraud_severity or "LOW",
                    "Review Status": inv.status or "PENDING"
                })

            df_history = pd.DataFrame(table_rows)
            st.dataframe(
                df_history,
                column_config={
                    "Amount (INR)": st.column_config.NumberColumn(format="INR %.2f"),
                    "Validation": st.column_config.TextColumn(),
                    "Review Status": st.column_config.TextColumn()
                },
                use_container_width=True,
                hide_index=True
            )

            # Selector to inspect specific invoice
            st.markdown("##### Inspect Invoice Details")
            c_sel1, c_sel2 = st.columns([3, 1])
            selected_inv_choice = c_sel1.selectbox(
                "Select invoice to inspect",
                [f"{i.id} &bull; {i.vendor_name} ({i.invoice_number}) &bull; INR {i.grand_total or 0:,.2f}" for i in invoices],
                label_visibility="collapsed"
            )
            if c_sel2.button("Open Review", type="primary", use_container_width=True):
                chosen_id = selected_inv_choice.split(" &bull; ")[0]
                st.session_state["view_invoice_id"] = chosen_id
                st.rerun()

            # CSV Export
            csv_bytes = df_history.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Export Filtered Records (CSV)",
                data=csv_bytes,
                file_name=f"invoice_history_{datetime.today().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )


# -----------------------------------------------------------------------------
# 3. Page 3 — Fraud Detection
# -----------------------------------------------------------------------------
elif st.session_state["active_nav"] == "Fraud Detection":
    if st.session_state["view_invoice_id"]:
        render_invoice_review(st.session_state["view_invoice_id"], origin_view="Fraud Detection")
    else:
        st.markdown('<div class="view-title">Payment Fraud Detection</div>', unsafe_allow_html=True)
        st.markdown('<div class="view-subtitle">Evidence-based anomaly detection, unverified bank remittance checks, duplicate detection, and vendor risk intelligence.</div>', unsafe_allow_html=True)

        with get_db_session() as session:
            repo = InvoiceRepository(session)
            fraud_invoices = repo.list_invoices(has_fraud=True, limit=150)

        scorer = VendorRiskScorer()
        df_vendors = scorer.get_all_profiles_df()

        # Summary KPI Cards
        f1, f2, f3 = st.columns(3)
        total_flagged = len(fraud_invoices)
        total_risk_amt = sum(float(i.amount_at_risk or 0.0) for i in fraud_invoices)
        high_risk_vendors = int((df_vendors["risk_score"] >= settings.HIGH_RISK_SCORE_THRESHOLD).sum()) if not df_vendors.empty else 0

        with f1:
            st.markdown(f"""<div class="kpi-card"><div class="kpi-title">Flagged Invoices</div><div class="kpi-value" style="color: #DC2626;">{total_flagged}</div><div class="kpi-sub">Active invoices with alert triggers</div></div>""", unsafe_allow_html=True)
        with f2:
            st.markdown(f"""<div class="kpi-card"><div class="kpi-title">Money at Risk</div><div class="kpi-value" style="color: #DC2626;">INR {total_risk_amt:,.2f}</div><div class="kpi-sub">Total value across flagged invoices</div></div>""", unsafe_allow_html=True)
        with f3:
            st.markdown(f"""<div class="kpi-card"><div class="kpi-title">High-Risk Vendors</div><div class="kpi-value">{high_risk_vendors}</div><div class="kpi-sub">Suppliers exceeding risk threshold</div></div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Tabbed Views: Active Flagged Invoices, Vendor Risk Profiles, and Historical AP Ledger Forensics
        tab_active, tab_vendors, tab_ledger = st.tabs(["Active Invoices with Fraud Alerts", "Vendor Risk Profiles", "Historical AP Ledger Forensics (2 Years)"])

        with tab_active:
            if not fraud_invoices:
                st.info("No active invoices currently flagged for payment fraud.")
            else:
                for inv in fraud_invoices:
                    with st.expander(f"Invoice {inv.id} &bull; {inv.vendor_name} ({inv.invoice_number}) &bull; Risk Score: {inv.fraud_risk_score}/100 [{inv.fraud_severity}]"):
                        st.markdown(f"**Amount at Risk:** INR {inv.amount_at_risk:,.2f}")
                        for fa in inv.fraud_alerts:
                            st.markdown(f"- **Pattern:** {fa.pattern_name} ({fa.severity})")
                            st.markdown(f"  - *Evidence:* {fa.explanation}")
                            st.markdown(f"  - *Recommended Action:* {fa.recommended_action}")
                        if st.button("Inspect Full Review Details", key=f"f_inspect_{inv.id}"):
                            st.session_state["view_invoice_id"] = inv.id
                            st.rerun()

        with tab_vendors:
            if df_vendors.empty:
                st.info("Vendor master data not available.")
            else:
                st.markdown("##### Supplier Risk Matrix")
                st.dataframe(
                    df_vendors[[
                        "vendor_id", "vendor_name", "status", "risk_score", "risk_category",
                        "total_invoices", "total_spend", "money_at_risk", "duplicate_count", "bank_mismatch_count"
                    ]],
                    column_config={
                        "risk_score": st.column_config.ProgressColumn("Risk Score", min_value=0, max_value=100, format="%.0f"),
                        "total_spend": st.column_config.NumberColumn("Total Spend", format="INR %.2f"),
                        "money_at_risk": st.column_config.NumberColumn("Money at Risk", format="INR %.2f")
                    },
                    use_container_width=True,
                    hide_index=True
                )

                st.markdown("##### Inspect Specific Supplier Profile")
                sel_v_id = st.selectbox("Select Vendor ID", df_vendors["vendor_id"].tolist(), label_visibility="collapsed")
                v_prof = scorer.get_vendor_risk(sel_v_id)
                st.markdown(f"**Legal Name:** {v_prof.get('vendor_name')} (ID: `{v_prof.get('vendor_id')}`) &bull; **Risk Score:** {v_prof.get('risk_score')}/100 [{v_prof.get('risk_category')}]")
                for f_item in v_prof.get("risk_factors", []):
                    st.markdown(f"- {f_item}")

        with tab_ledger:
            st.markdown("##### 2-Year Historical AP Ledger Forensic Analysis")
            st.markdown("""
            Forensic analysis of the 6,007 historical AP transactions across 60 registered suppliers identified 7 critical fraud and anomaly patterns:
            1. **Exact Duplicate Invoices**: 62 transactions (31 duplicate invoice pairs) resulting in potential double payments.
            2. **Bank Account Remittance Divergence**: 201 invoices with unverified bank accounts across 14 suppliers.
            3. **GSTIN Spoofing / Mismatches**: 31 invoices with modified GST numbers.
            4. **PO Budget Overruns**: 50 invoices exceeding purchase order authorization by >5%.
            5. **Approver Limit Abuse**: 7 invoices approved in breach of statutory clerk authorization limits.
            6. **Blacklisted Vendor Invoicing**: 85 invoices processed for deactivated vendor V1060.
            """)
            try:
                ledger_df = pd.read_csv(settings.LEDGER_PATH)
                dups = ledger_df[ledger_df.duplicated(subset=['vendor_id', 'vendor_invoice_no'], keep=False)]
                st.markdown(f"**Identified Duplicate Invoices in Historical Ledger ({len(dups)} rows):**")
                st.dataframe(
                    dups[['ledger_id', 'vendor_id', 'vendor_invoice_no', 'invoice_date', 'total_amount', 'paid_date']].head(25),
                    use_container_width=True,
                    hide_index=True
                )
            except Exception as e:
                st.error(f"Could not load historical AP ledger file: {e}")


# -----------------------------------------------------------------------------
# 4. Settings (Small Gear / Bottom Section)
# -----------------------------------------------------------------------------
elif st.session_state["active_nav"] == "Settings":
    st.markdown('<div class="view-title">System Settings</div>', unsafe_allow_html=True)
    st.markdown('<div class="view-subtitle">Configure Google Gemini API key, test connectivity, customize OCR preferences, and run accuracy benchmarks.</div>', unsafe_allow_html=True)

    # API Configuration Section
    st.markdown("#### Google AI Studio — Gemini Configuration")
    c_s1, c_s2 = st.columns(2)
    c_s1.markdown(f"**Status:** {'Configured' if gemini_active else 'Not Configured'}")
    c_s1.caption(f"Masked Key: `{settings.get_masked_api_key()}`")
    c_s2.markdown(f"**Active Model:** `{settings.GEMINI_MODEL}`")

    with st.form("form_settings_key"):
        new_key = st.text_input(
            "Enter Gemini API Key",
            type="password",
            placeholder="Paste your API key here (AIza...)",
            help="Your key is stored securely in your local environment file and never displayed in plaintext."
        )
        save_btn = st.form_submit_button("Save & Test Connection", type="primary")

    if save_btn:
        if not new_key.strip():
            st.error("Please enter a valid API key string.")
        else:
            settings.update_gemini_key(new_key.strip())
            test_ext = GeminiExtractor(api_key=new_key.strip())
            with st.spinner("Testing API connection..."):
                t_res = test_ext.test_connection()
            if t_res.get("success"):
                st.success(f"Connection test PASSED! {t_res.get('message')}")
            else:
                st.warning(f"Key saved, but test failed: {t_res.get('message')}. Rule-based extraction will remain active.")
            st.rerun()

    c_b1, c_b2 = st.columns([1, 4])
    with c_b1:
        if st.button("Test Connection"):
            with st.spinner("Testing connection..."):
                t_res = gemini_ext.test_connection()
            if t_res.get("success"):
                st.success(f"Connection PASSED! {t_res.get('message')}")
            else:
                st.error(f"Connection FAILED: {t_res.get('message')}")
    with c_b2:
        if st.button("Remove API Key"):
            settings.update_gemini_key("")
            st.warning("API key removed. System will operate 100% offline using deterministic rule extraction.")
            st.rerun()

    st.markdown("---")

    # Benchmarks & Evaluations (Preserves required project evaluation capabilities)
    st.markdown("#### System Benchmarks & Accuracy Reports")
    tab_bench1, tab_bench2 = st.tabs(["Extraction Accuracy (180 Dataset)", "OCR Engine CER Comparison"])

    with tab_bench1:
        if st.button("Run Accuracy Evaluation on Benchmark Dataset"):
            with st.spinner("Evaluating extraction across test invoices..."):
                acc_res = evaluate_extraction_accuracy(num_samples=20)
            st.markdown(f"**Invoices Evaluated:** {acc_res['invoices_evaluated']} (Clean: {acc_res['clean_count']}, Degraded: {acc_res['degraded_count']})")
            b1, b2, b3 = st.columns(3)
            b1.metric("Clean Scan Accuracy", f"{acc_res['clean_avg_exact_match']}%", "Target: >=95%")
            b2.metric("Degraded Scan Accuracy", f"{acc_res['degraded_avg_exact_match']}%", "Target: >=85%")
            b3.metric("Grand Total Accuracy", f"{acc_res['grand_total_accuracy']}%", "Target: >=97%")

            fields_df = pd.DataFrame({
                "Clean Scans (%)": acc_res["clean_field_accuracy"],
                "Degraded Scans (%)": acc_res["degraded_field_accuracy"],
                "Overall (%)": acc_res["overall_field_accuracy"]
            })
            st.dataframe(fields_df, use_container_width=True)

    with tab_bench2:
        if st.button("Run OCR Comparison (PaddleOCR vs EasyOCR)"):
            with st.spinner("Running CER and latency benchmarks..."):
                ocr_eval = evaluate_ocr_engines(sample_size=10)
            st.dataframe(pd.DataFrame(ocr_eval).T, use_container_width=True)
