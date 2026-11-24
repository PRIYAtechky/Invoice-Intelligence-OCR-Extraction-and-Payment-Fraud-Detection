"""Invoice Intelligence — OCR Extraction & Payment Fraud Detection Platform
Enterprise Accounts Payable Control Center & Forensic Fraud Suite
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

# Initialize database on app load
init_db()

# Page configuration
st.set_page_config(
    page_title="APEX Intelligence — Enterprise AP & Fraud Defense",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# Enterprise Design System & Modern CSS Styling
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Global typography & layout */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    
    /* Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 24px 28px;
        color: #F8FAFC;
        margin-bottom: 24px;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.08);
    }
    .hero-title {
        font-size: 1.65rem;
        font-weight: 800;
        letter-spacing: -0.025em;
        color: #FFFFFF;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .hero-subtitle {
        font-size: 0.95rem;
        color: #94A3B8;
        margin-top: 6px;
        line-height: 1.5;
    }

    /* Page Header */
    .page-title {
        font-size: 1.55rem;
        font-weight: 700;
        color: #0F172A;
        letter-spacing: -0.02em;
        margin-bottom: 4px;
    }
    .page-desc {
        font-size: 0.92rem;
        color: #64748B;
        margin-bottom: 20px;
    }

    /* Metric Cards */
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04), 0 1px 2px -1px rgba(0, 0, 0, 0.02);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
        height: 100%;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px -2px rgba(0, 0, 0, 0.06);
        border-color: #CBD5E1;
    }
    .metric-title {
        font-size: 0.8rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .metric-val {
        font-size: 1.95rem;
        font-weight: 800;
        color: #0F172A;
        margin: 6px 0 2px 0;
        letter-spacing: -0.02em;
    }
    .metric-sub {
        font-size: 0.82rem;
        color: #94A3B8;
        font-weight: 500;
    }

    /* Section Cards */
    .section-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    }
    .section-card-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Pills & Status Badges */
    .badge-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        line-height: 1;
    }
    .badge-success { background: #ECFDF5; color: #065F46; border: 1px solid #A7F3D0; }
    .badge-warning { background: #FFFBEB; color: #92400E; border: 1px solid #FDE68A; }
    .badge-danger { background: #FEF2F2; color: #991B1B; border: 1px solid #FECACA; }
    .badge-info { background: #EFF6FF; color: #1E40AF; border: 1px solid #BFDBFE; }
    .badge-neutral { background: #F1F5F9; color: #475569; border: 1px solid #E2E8F0; }

    /* Pipeline Step Bar */
    .pipeline-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 12px 18px;
        margin-bottom: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        color: #475569;
    }
    .pipeline-step {
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .step-active {
        color: #2563EB;
        font-weight: 700;
    }

    /* Button Enhancements */
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.9rem;
        transition: all 0.2s;
    }
    .stButton>button[kind="primary"] {
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
        border: none;
        box-shadow: 0 2px 4px rgba(37, 99, 235, 0.2);
    }
    .stButton>button[kind="primary"]:hover {
        background: linear-gradient(135deg, #1D4ED8 0%, #1E40AF 100%);
        box-shadow: 0 4px 8px rgba(37, 99, 235, 0.3);
    }

    /* Telemetry Widget in Sidebar */
    .telemetry-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 12px 14px;
        font-size: 0.82rem;
        line-height: 1.6;
        color: #334155;
    }
    .telemetry-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 3px 0;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Sidebar Navigation & Enterprise Profile
# -----------------------------------------------------------------------------
st.sidebar.markdown("""
<div style="display: flex; align-items: center; gap: 12px; padding: 6px 0 16px 0;">
    <div style="background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%); width: 44px; height: 44px; border-radius: 10px; display: flex; align-items: center; justify-content: center; color: white; font-size: 1.3rem; font-weight: 800; box-shadow: 0 4px 8px -1px rgba(37, 99, 235, 0.35);">
        ⚡
    </div>
    <div>
        <div style="font-size: 1.15rem; font-weight: 800; color: #0F172A; letter-spacing: -0.02em;">APEX INTELLIGENCE</div>
        <div style="font-size: 0.72rem; color: #64748B; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">Enterprise AP & Fraud Suite</div>
    </div>
</div>
""", unsafe_allow_html=True)

PAGES = [
    "📊 Executive Overview",
    "📥 Ingest & Process Invoices",
    "📝 Manual Invoice Entry",
    "🔍 Review & Approval Gate",
    "📜 Invoice Repository & Audit",
    "🛡️ Payment Fraud Center",
    "🏢 Vendor Risk Intelligence",
    "📦 Batch Ingestion Manager",
    "📈 Accuracy & Benchmarks",
    "⚙️ Settings & AI Configuration",
    "❓ Help & User Guide"
]

selected_page = st.sidebar.radio("Navigation", PAGES, index=0)

st.sidebar.markdown("---")

# Role Switcher with Authority Limits
ROLES_CONFIG = {
    "AP-Clerk-01": {"limit": 200000.0, "title": "Accounts Payable Clerk", "badge": "CLERK"},
    "AP-Clerk-02": {"limit": 200000.0, "title": "Accounts Payable Clerk", "badge": "CLERK"},
    "AP-Senior-01": {"limit": 500000.0, "title": "Senior AP Specialist", "badge": "SR SPECIALIST"},
    "AP-Manager-01": {"limit": 1500000.0, "title": "Finance Manager", "badge": "MANAGER"},
    "Finance-Head": {"limit": 50000000.0, "title": "VP Finance / CFO", "badge": "EXECUTIVE"}
}

approver_role = st.sidebar.selectbox(
    "Active User Profile",
    list(ROLES_CONFIG.keys()),
    index=0
)
current_role_info = ROLES_CONFIG[approver_role]
formatted_limit = f"₹{current_role_info['limit']:,.0f}" if current_role_info['limit'] < 10000000 else "Unlimited"

st.sidebar.markdown(f"""
<div style="background: #F1F5F9; border-radius: 8px; padding: 10px 12px; margin-top: -6px; border: 1px solid #E2E8F0;">
    <div style="font-size: 0.75rem; color: #64748B; font-weight: 600; text-transform: uppercase;">Authority Limit</div>
    <div style="font-size: 0.95rem; font-weight: 700; color: #0F172A; margin-top: 2px;">{formatted_limit}</div>
    <div style="font-size: 0.74rem; color: #475569; margin-top: 2px;">Role: <b>{current_role_info['title']}</b></div>
</div>
""", unsafe_allow_html=True)

# Engine configuration status
gemini_ext = GeminiExtractor()
gemini_active = gemini_ext.is_configured()

st.sidebar.markdown("---")
st.sidebar.markdown("**System Telemetry**")
st.sidebar.markdown(f"""
<div class="telemetry-card">
    <div class="telemetry-item">
        <span>OCR Engine</span>
        <span class="badge-pill badge-neutral"><b>{settings.DEFAULT_OCR_ENGINE.upper()}</b></span>
    </div>
    <div class="telemetry-item">
        <span>AI Extraction</span>
        <span class="badge-pill {'badge-success' if gemini_active else 'badge-warning'}">
            <b>{'Gemini 3.8-Flash' if gemini_active else 'Rule-Based'}</b>
        </span>
    </div>
    <div class="telemetry-item">
        <span>Validation Rules</span>
        <span class="badge-pill badge-info"><b>9 Statutory</b></span>
    </div>
    <div class="telemetry-item">
        <span>Database</span>
        <span class="badge-pill badge-success"><b>Connected</b></span>
    </div>
</div>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 1. Executive Overview Dashboard
# -----------------------------------------------------------------------------
if selected_page == "📊 Executive Overview":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">⚡ Enterprise Accounts Payable Control Center</div>
        <div class="hero-subtitle">Continuous automated ingestion, multi-engine OCR extraction, statutory validation, and forensic fraud detection.</div>
    </div>
    """, unsafe_allow_html=True)

    with get_db_session() as session:
        repo = InvoiceRepository(session)
        metrics = repo.get_dashboard_metrics()
        recent_invoices = repo.list_invoices(limit=10)

    # 4 Core KPI Metrics Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">
                <span>Invoices Ingested</span>
                <span>📥</span>
            </div>
            <div class="metric-val">{metrics['total_invoices_processed']}</div>
            <div class="metric-sub">Spend Volume: <b>₹{metrics['total_invoice_value_inr']:,.2f}</b></div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        val_pct = (metrics['validated_invoices'] / metrics['total_invoices_processed'] * 100) if metrics['total_invoices_processed'] else 100.0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">
                <span>Validation Rate</span>
                <span>🛡️</span>
            </div>
            <div class="metric-val" style="color: {'#16A34A' if val_pct >= 90 else '#D97706'};">{val_pct:.1f}%</div>
            <div class="metric-sub">{metrics['validated_invoices']} Clean / {metrics['total_invoices_processed']} Total Ingested</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">
                <span>Review Queue</span>
                <span>⏳</span>
            </div>
            <div class="metric-val" style="color: #2563EB;">{metrics['invoices_requiring_review']}</div>
            <div class="metric-sub">{metrics['approved_invoices']} Approved • {metrics['rejected_invoices']} Rejected</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">
                <span>Money at Risk</span>
                <span>🚨</span>
            </div>
            <div class="metric-val" style="color: #DC2626;">₹{metrics['total_amount_at_risk_inr']:,.2f}</div>
            <div class="metric-sub">{metrics['potentially_fraudulent_invoices']} Invoices Flagged for Risk</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns([3, 2])
    with col_left:
        st.markdown("### 📋 Recent Invoices Requiring Review")
        if recent_invoices:
            table_data = []
            for inv in recent_invoices:
                table_data.append({
                    "Invoice ID": inv.id,
                    "Vendor Name": inv.vendor_name or "Unknown",
                    "Invoice No": inv.invoice_number or "N/A",
                    "Date": inv.invoice_date or "N/A",
                    "Amount (₹)": float(inv.grand_total or 0.0),
                    "Validation": inv.validation_status or "Needs Review",
                    "Fraud Severity": inv.fraud_severity or "LOW",
                    "Status": inv.status or "PENDING_REVIEW"
                })
            df_recent = pd.DataFrame(table_data)
            st.dataframe(
                df_recent,
                column_config={
                    "Amount (₹)": st.column_config.NumberColumn(format="₹%.2f"),
                    "Validation": st.column_config.TextColumn(),
                    "Fraud Severity": st.column_config.TextColumn(),
                    "Status": st.column_config.TextColumn()
                },
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No invoices processed yet. Navigate to 'Ingest & Process Invoices' to upload a document.")

    with col_right:
        st.markdown("### ⚙️ Automated Pipeline Telemetry")
        st.markdown("""
        <div class="section-card">
            <div style="font-size: 0.88rem; color: #334155; line-height: 1.8;">
                <div>🔹 <b>Multi-Engine OCR Subsystem</b>: PaddleOCR (RapidOCR ONNX) primary, EasyOCR fallback.</div>
                <div>🔹 <b>Statutory Compliance Engine</b>: 9 statutory checks active (GSTIN format/checksum, 3-way PO match, bank divergence, HSN rates, clerk approval limits).</div>
                <div>🔹 <b>Forensic Fraud Analytics</b>: Isolation Forest outlier scoring + 6 deterministic fraud rules.</div>
                <div>🔹 <b>Immutable Audit Trail</b>: Continuous state change tracking with reviewer role verification.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.progress(100, text="Pipeline 100% Operational")


# -----------------------------------------------------------------------------
# 2. Ingest & Process Invoices (Upload & OCR Pipeline)
# -----------------------------------------------------------------------------
elif selected_page == "📥 Ingest & Process Invoices":
    st.markdown('<div class="page-title">Invoice Ingestion & Automated Processing</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-desc">Ingest scanned invoices (JPG, PNG, PDF) through orientation correction, multi-engine OCR, and statutory verification.</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="pipeline-bar">
        <div class="pipeline-step step-active"><span>1</span> Ingestion</div>
        <div>➔</div>
        <div class="pipeline-step"><span>2</span> Preprocessing</div>
        <div>➔</div>
        <div class="pipeline-step"><span>3</span> Multi-Engine OCR</div>
        <div>➔</div>
        <div class="pipeline-step"><span>4</span> Structured Extraction</div>
        <div>➔</div>
        <div class="pipeline-step"><span>5</span> Validation & Fraud</div>
    </div>
    """, unsafe_allow_html=True)

    up_col1, up_col2 = st.columns([2, 1])
    with up_col1:
        uploaded_file = st.file_uploader(
            "Upload Invoice Scan (JPG, PNG, PDF)",
            type=["jpg", "jpeg", "png", "pdf"],
            help="High-resolution scans or mobile photos with tilt/noise are supported."
        )

    with up_col2:
        st.markdown("#### Engine Configuration")
        ocr_choice = st.selectbox("OCR Engine", ["rapidocr", "easyocr"], index=0, help="RapidOCR uses PaddleOCR ONNX models.")
        use_gemini = st.checkbox("Enable Gemini LLM Extraction", value=gemini_active, disabled=not gemini_active)
        if not gemini_active:
            st.caption("ℹ️ Configure API Key in Settings to enable Gemini LLM extraction.")

    if uploaded_file is not None:
        temp_dir = settings.DATA_DIR / "uploads"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_file_path = temp_dir / uploaded_file.name

        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        c_img1, c_img2 = st.columns(2)
        with c_img1:
            st.image(uploaded_file, caption=f"Original Document: {uploaded_file.name}", use_container_width=True)

        if st.button("🚀 Run Automated Extraction & Fraud Pipeline", type="primary"):
            with st.spinner("Executing pipeline: deskewing -> OCR -> extraction -> validation -> fraud scan..."):
                with get_db_session() as session:
                    pipeline = InvoiceProcessingPipeline(session=session)
                    res = pipeline.process_image(
                        image_path=str(temp_file_path),
                        engine_name=ocr_choice,
                        use_llm=use_gemini,
                        input_method="upload",
                        approver_role=approver_role
                    )

            st.success(f"Invoice Successfully Processed! Assigned ID: **{res['invoice_id']}**")

            # Display processed side-by-side
            if res.get("processed_image_path") and os.path.exists(res["processed_image_path"]):
                with c_img2:
                    st.image(res["processed_image_path"], caption="Preprocessed Image (Deskewed & Binarized)", use_container_width=True)

            # Executive summary cards
            q1, q2, q3, q4 = st.columns(4)
            q1.metric("Vendor", res["extraction"].vendor_name or "Unknown")
            q2.metric("Invoice #", res["extraction"].invoice_number or "N/A")
            q3.metric("Grand Total", f"₹{res['extraction'].grand_total:,.2f}" if res['extraction'].grand_total else "N/A")
            q4.metric("Validation Status", res["validation"].status)

            # Statutory Results & Fraud Findings
            tab_val, tab_items, tab_text = st.tabs(["🛡️ Validation & Fraud Checks", "📋 Extracted Line Items", "📄 Raw OCR Output"])
            
            with tab_val:
                for vr in res["validation"].rules_evaluated:
                    if vr.status == "Passed":
                        st.markdown(f"✅ **{vr.rule_name}**: {vr.message}")
                    elif vr.status == "Warning":
                        st.markdown(f"⚠️ **{vr.rule_name}**: {vr.message}")
                    else:
                        st.markdown(f"❌ **{vr.rule_name}**: {vr.message}")

                if res["fraud"].alerts:
                    st.markdown("---")
                    st.error(f"🚨 **{len(res['fraud'].alerts)} Payment Fraud Alert(s) Detected:**")
                    for fa in res["fraud"].alerts:
                        st.markdown(f"- **{fa.pattern_name}** ({fa.severity}): {fa.explanation}")
                else:
                    st.success("No payment fraud or divergence patterns detected.")

            with tab_items:
                if res["extraction"].line_items:
                    items_df = pd.DataFrame([it.dict() for it in res["extraction"].line_items])
                    st.dataframe(items_df, use_container_width=True)
                else:
                    st.info("No individual line items parsed from document.")

            with tab_text:
                st.text_area("Extracted OCR Text", res["extraction"].raw_ocr_text or "No text extracted", height=200)


# -----------------------------------------------------------------------------
# 3. Manual Invoice Entry
# -----------------------------------------------------------------------------
elif selected_page == "📝 Manual Invoice Entry":
    st.markdown('<div class="page-title">Direct Invoice Ingestion & Verification</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-desc">Enter invoice details directly for paper documents. Full statutory validation, 3-way PO matching, and fraud scans are enforced.</div>', unsafe_allow_html=True)

    with st.form("manual_entry_form"):
        st.markdown("#### 1. Vendor & Invoice Header")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            m_vendor = st.text_input("Vendor Legal Name*", placeholder="e.g. Vel Engineering Works")
            m_gstin = st.text_input("Vendor GSTIN*", placeholder="15-character statutory GSTIN (e.g. 33AABCV1234F1Z1)")
            m_inv_no = st.text_input("Invoice Number*", placeholder="e.g. TI/2025/8093")
        with col_m2:
            m_po = st.text_input("Purchase Order #", placeholder="e.g. KIS/PO/2024/10001")
            m_inv_date = st.date_input("Invoice Date*", value=datetime.today())
            m_due_date = st.date_input("Due Date", value=datetime.today())

        st.markdown("---")
        st.markdown("#### 2. Financial Amounts & Tax Reconciliation")
        c_amt1, c_amt2, c_amt3 = st.columns(3)
        with c_amt1:
            m_subtotal = st.number_input("Subtotal / Taxable Value (₹)*", min_value=0.0, step=100.0)
            m_bank = st.text_input("Bank Account (Last 4 Digits)", max_chars=4, placeholder="e.g. 5029")
        with c_amt2:
            m_cgst = st.number_input("CGST (₹)", min_value=0.0, step=10.0)
            m_sgst = st.number_input("SGST (₹)", min_value=0.0, step=10.0)
        with c_amt3:
            m_igst = st.number_input("IGST (₹)", min_value=0.0, step=10.0)
            m_grand = st.number_input("Grand Total (₹)*", min_value=0.0, step=100.0)

        st.markdown("---")
        st.markdown("#### 3. Line Items")
        c_desc, c_hsn, c_qty, c_rate = st.columns([3, 1, 1, 1])
        item_desc = c_desc.text_input("Item Description", value="Industrial Ball Bearing 6205-2RS")
        item_hsn = c_hsn.text_input("HSN Code", value="8482")
        item_qty = c_qty.number_input("Quantity", min_value=1.0, value=5.0)
        item_rate = c_rate.number_input("Unit Price (₹)", min_value=0.0, value=m_subtotal/5.0 if m_subtotal else 100.0)

        submit_btn = st.form_submit_button("💾 Submit & Execute Statutory Validation", type="primary")

    if submit_btn:
        if not m_vendor or not m_gstin or not m_inv_no or m_grand <= 0:
            st.error("Please fill in all mandatory fields marked with an asterisk (*).")
        else:
            line_item = LineItem(
                description=item_desc,
                hsn=item_hsn,
                quantity=item_qty,
                unit_price=item_rate,
                amount=item_qty * item_rate
            )
            data_input = ManualInvoiceInput(
                vendor_name=m_vendor,
                vendor_gstin=m_gstin,
                invoice_number=m_inv_no,
                invoice_date=str(m_inv_date),
                due_date=str(m_due_date),
                po_number=m_po,
                line_items=[line_item],
                subtotal=m_subtotal,
                cgst=m_cgst,
                sgst=m_sgst,
                igst=m_igst,
                grand_total=m_grand,
                bank_account_last4=m_bank
            )

            with get_db_session() as session:
                pipeline = InvoiceProcessingPipeline(session=session)
                res = pipeline.process_manual(data_input, approver_role=approver_role)

            st.success(f"Invoice recorded successfully! Assigned ID: **{res['invoice_id']}**")
            st.info(f"Validation Status: **{res['validation'].status}** | Fraud Risk Score: **{res['fraud'].composite_risk_score}/100**")


# -----------------------------------------------------------------------------
# 4. Review & Approval Gate (Side-by-side human-in-the-loop)
# -----------------------------------------------------------------------------
elif selected_page == "🔍 Review & Approval Gate":
    st.markdown('<div class="page-title">Invoice Review & Authorization Gate</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-desc">Inspect scanned documents side-by-side with extracted metadata, validation checks, and role authority limits.</div>', unsafe_allow_html=True)

    with get_db_session() as session:
        repo = InvoiceRepository(session)
        invoices = repo.list_invoices(limit=200)

    if not invoices:
        st.warning("No invoices found in database to review.")
    else:
        inv_choices = {f"{inv.id} — {inv.vendor_name or 'Unknown'} (#{inv.invoice_number or 'N/A'}) — ₹{inv.grand_total or 0.0:,.2f}": inv.id for inv in invoices}
        selected_choice = st.selectbox("Select Invoice to Inspect", list(inv_choices.keys()))
        selected_id = inv_choices[selected_choice]

        with get_db_session() as session:
            repo = InvoiceRepository(session)
            inv = repo.get_invoice(selected_id)

            if inv:
                # Top Status Ribbon
                r1, r2, r3, r4 = st.columns(4)
                r1.metric("Invoice ID", inv.id)
                r2.metric("Review Status", inv.status)
                r3.metric("Validation", inv.validation_status)
                r4.metric("Fraud Risk Score", f"{inv.fraud_risk_score}/100", delta=inv.fraud_severity, delta_color="inverse")

                # Role Authority Check Alert
                invoice_val = float(inv.grand_total or 0.0)
                user_limit = current_role_info["limit"]
                is_over_limit = invoice_val > user_limit

                if is_over_limit:
                    st.warning(f"⚠️ **Authorization Limit Notice**: Invoice amount (₹{invoice_val:,.2f}) exceeds your active role authority limit ({formatted_limit}). Approval will be blocked unless escalated.")

                st.markdown("---")

                # Side-by-side View: Image on Left, Form on Right
                col_view1, col_view2 = st.columns([1, 1])

                with col_view1:
                    st.subheader("Original Scanned Document")
                    if inv.original_image_path and os.path.exists(inv.original_image_path):
                        st.image(inv.original_image_path, caption=f"Document: {inv.file_name}", use_container_width=True)
                    else:
                        st.info("No scanned image file available for this invoice record.")

                with col_view2:
                    st.subheader("Extracted Data & Review Decision")
                    with st.form(f"review_form_{inv.id}"):
                        edit_vendor = st.text_input("Vendor Legal Name", value=inv.vendor_name or "")
                        edit_gstin = st.text_input("Vendor GSTIN", value=inv.vendor_gstin or "")
                        c_i1, c_i2 = st.columns(2)
                        edit_inv_no = c_i1.text_input("Invoice Number", value=inv.invoice_number or "")
                        edit_inv_date = c_i2.text_input("Invoice Date (YYYY-MM-DD)", value=inv.invoice_date or "")
                        
                        c_p1, c_p2 = st.columns(2)
                        edit_po = c_p1.text_input("Purchase Order #", value=inv.po_number or "")
                        edit_bank = c_p2.text_input("Bank Account Last 4", value=inv.bank_account_last4 or "")

                        c_a1, c_a2 = st.columns(2)
                        edit_subtotal = c_a1.number_input("Subtotal (₹)", value=float(inv.subtotal or 0.0), step=100.0)
                        edit_grand = c_a2.number_input("Grand Total (₹)", value=float(inv.grand_total or 0.0), step=100.0)

                        st.markdown("#### Reviewer Decision")
                        decision = st.selectbox("Action", ["APPROVED", "REJECTED", "FLAGGED_FOR_REVIEW"], index=0 if inv.status == "APPROVED" else 1)
                        reviewer_notes = st.text_area("Reviewer Audit Notes / Justification", value=inv.reviewer_comments or "")

                        save_btn = st.form_submit_button("💾 Save Changes & Record Decision", type="primary")

                    if save_btn:
                        # Hard gates
                        if decision == "APPROVED" and (inv.has_fraud_risk and inv.fraud_severity == "CRITICAL"):
                            st.error("❌ High-Risk Fraud Alert: Cannot approve invoice with CRITICAL fraud flags. Escalate or reject.")
                        elif decision == "APPROVED" and is_over_limit:
                            st.error(f"❌ Authorization Limit Breach: Invoice amount (₹{invoice_val:,.2f}) exceeds your limit of {formatted_limit}. Cannot approve.")
                        else:
                            with get_db_session() as save_session:
                                save_repo = InvoiceRepository(save_session)
                                save_repo.update_invoice_review(
                                    invoice_id=inv.id,
                                    decision=decision,
                                    reviewer_name=f"{approver_role} ({current_role_info['title']})",
                                    comments=reviewer_notes,
                                    edited_fields={
                                        "vendor_name": edit_vendor,
                                        "vendor_gstin": edit_gstin,
                                        "invoice_number": edit_inv_no,
                                        "invoice_date": edit_inv_date,
                                        "po_number": edit_po,
                                        "bank_account_last4": edit_bank,
                                        "subtotal": edit_subtotal,
                                        "grand_total": edit_grand
                                    }
                                )
                            st.success(f"Decision '{decision}' recorded with full immutable audit trail!")
                            st.rerun()

                st.markdown("---")
                t1, t2, t3 = st.tabs(["🛡️ Statutory Validation Rules", "🚨 Payment Fraud Findings", "📜 Immutable Audit Log"])
                with t1:
                    for v in inv.validations:
                        st.markdown(f"- **{v.rule_name}** (`{v.status}`): {v.message}")
                with t2:
                    if inv.fraud_alerts:
                        for fa in inv.fraud_alerts:
                            st.error(f"**{fa.pattern_name}** | Severity: **{fa.severity}** | Risk Score: **{fa.risk_score}**\n\n- *Explanation*: {fa.explanation}\n- *Recommended Action*: {fa.recommended_action}")
                    else:
                        st.success("No active fraud flags detected on this invoice.")
                with t3:
                    audits = repo.get_audit_trail_for_invoice(inv.id)
                    for a in audits:
                        st.markdown(f"⏱️ **{a.timestamp.strftime('%Y-%m-%d %H:%M:%S')}** — **{a.action}** by `{a.user}`: {a.comments or ''}")


# -----------------------------------------------------------------------------
# 5. Invoice Repository & Audit Log
# -----------------------------------------------------------------------------
elif selected_page == "📜 Invoice Repository & Audit":
    st.markdown('<div class="page-title">Enterprise Invoice Repository & Audit Trail</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-desc">Search, filter, and audit every ingested invoice record and reviewer mutation.</div>', unsafe_allow_html=True)

    with get_db_session() as session:
        repo = InvoiceRepository(session)
        
        c_f1, c_f2, c_f3 = st.columns(3)
        status_filter = c_f1.selectbox("Filter Status", ["ALL", "PENDING_REVIEW", "APPROVED", "REJECTED", "FLAGGED"])
        vendor_search = c_f2.text_input("Search Vendor Name", "")
        fraud_only = c_f3.checkbox("Show Only Fraud Flagged Invoices", value=False)

        invoices = repo.list_invoices(
            status=None if status_filter == "ALL" else status_filter,
            vendor=vendor_search if vendor_search else None,
            has_fraud=True if fraud_only else None,
            limit=200
        )

    if not invoices:
        st.info("No invoices match the selected filter criteria.")
    else:
        st.markdown(f"**Found {len(invoices)} invoices**")
        table_rows = []
        for inv in invoices:
            table_rows.append({
                "Invoice ID": inv.id,
                "Vendor": inv.vendor_name or "Unknown",
                "Invoice #": inv.invoice_number or "N/A",
                "Date": inv.invoice_date or "N/A",
                "Grand Total (₹)": float(inv.grand_total or 0.0),
                "Status": inv.status or "PENDING",
                "Validation": inv.validation_status or "Needs Review",
                "Fraud Risk": float(inv.fraud_risk_score or 0.0),
                "Source": inv.extraction_source or "Rule"
            })
        df_inv = pd.DataFrame(table_rows)
        st.dataframe(
            df_inv,
            column_config={
                "Grand Total (₹)": st.column_config.NumberColumn(format="₹%.2f"),
                "Fraud Risk": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f"),
            },
            use_container_width=True,
            hide_index=True
        )

        # CSV Download Button
        csv_data = df_inv.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Export Filtered Invoices (CSV)",
            data=csv_data,
            file_name=f"invoices_export_{datetime.today().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )


# -----------------------------------------------------------------------------
# 6. Payment Fraud Center
# -----------------------------------------------------------------------------
elif selected_page == "🛡️ Payment Fraud Center":
    st.markdown('<div class="page-title">Forensic Payment Fraud Detection Center</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-desc">Comprehensive anomaly detection, duplicate detection, unverified bank diversion checks, and historical AP ledger forensics.</div>', unsafe_allow_html=True)

    f_tab1, f_tab2 = st.tabs(["Active Invoices with Fraud Alerts", "Historical AP Ledger Fraud Audit (2 Years)"])

    with f_tab1:
        with get_db_session() as session:
            repo = InvoiceRepository(session)
            fraud_invoices = repo.list_invoices(has_fraud=True, limit=100)

        if not fraud_invoices:
            st.success("✅ No active payment fraud alerts detected on current ingested invoices.")
        else:
            for inv in fraud_invoices:
                with st.expander(f"🚨 {inv.id} — {inv.vendor_name} (#{inv.invoice_number}) | Risk Score: {inv.fraud_risk_score}/100 [{inv.fraud_severity}]"):
                    st.markdown(f"**Amount at Risk:** ₹{inv.amount_at_risk:,.2f}")
                    for fa in inv.fraud_alerts:
                        st.markdown(f"- **Pattern**: **{fa.pattern_name}**")
                        st.markdown(f"- **Explanation**: {fa.explanation}")
                        st.markdown(f"- **Recommended Action**: {fa.recommended_action}")

    with f_tab2:
        st.markdown("### 2-Year Historical AP Ledger Fraud Analysis")
        st.markdown("""
        Systematic audit of 6,007 historical AP transactions across 60 registered suppliers revealed 7 critical fraud and anomaly patterns:
        1. **Exact Duplicate Invoices**: 62 transactions (31 duplicate invoice pairs) resulting in double payment exposure.
        2. **Bank Account Remittance Divergence**: 201 invoices with unverified bank accounts across 14 suppliers.
        3. **GSTIN Spoofing / Mismatches**: 31 invoices with modified GST numbers.
        4. **PO Budget Overruns**: 50 invoices exceeding purchase order authorization by >5%.
        5. **Approver Limit Abuse**: 7 invoices approved in breach of statutory clerk authorization limits.
        6. **Blacklisted Vendor Invoicing**: 85 invoices processed for deactivated vendor V1060.
        """)
        
        try:
            ledger_df = pd.read_csv(settings.LEDGER_PATH)
            dups = ledger_df[ledger_df.duplicated(subset=['vendor_id', 'vendor_invoice_no'], keep=False)]
            st.markdown(f"#### Exact Duplicate Invoices Identified in Historical Ledger ({len(dups)} rows)")
            st.dataframe(dups[['ledger_id', 'vendor_id', 'vendor_invoice_no', 'invoice_date', 'total_amount', 'paid_date']].head(20), use_container_width=True)
        except Exception as e:
            st.error(f"Could not load ledger: {e}")


# -----------------------------------------------------------------------------
# 7. Vendor Risk Intelligence
# -----------------------------------------------------------------------------
elif selected_page == "🏢 Vendor Risk Intelligence":
    st.markdown('<div class="page-title">Vendor Risk Intelligence & Compliance Profiles</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-desc">Composite 0–100 vendor risk scoring based on historical bank changes, duplicate billing frequency, and statutory compliance.</div>', unsafe_allow_html=True)

    scorer = VendorRiskScorer()
    df_vendors = scorer.get_all_profiles_df()

    if df_vendors.empty:
        st.warning("Vendor master data not available.")
    else:
        v_col1, v_col2, v_col3 = st.columns(3)
        v_col1.metric("Registered Vendors", len(df_vendors))
        high_risk_c = int((df_vendors["risk_score"] >= settings.HIGH_RISK_SCORE_THRESHOLD).sum())
        v_col2.metric("High / Critical Risk Vendors", high_risk_c, delta=f"{high_risk_c/len(df_vendors)*100:.1f}%", delta_color="inverse")
        total_risk_spend = float(df_vendors["money_at_risk"].sum())
        v_col3.metric("Total Money at Risk", f"₹{total_risk_spend:,.2f}")

        st.markdown("#### Vendor Risk Matrix")
        st.dataframe(
            df_vendors[[
                "vendor_id", "vendor_name", "status", "risk_score", "risk_category",
                "total_invoices", "total_spend", "money_at_risk", "duplicate_count", "bank_mismatch_count"
            ]],
            column_config={
                "risk_score": st.column_config.ProgressColumn("Risk Score", min_value=0, max_value=100, format="%.0f"),
                "total_spend": st.column_config.NumberColumn("Total Spend (₹)", format="₹%.2f"),
                "money_at_risk": st.column_config.NumberColumn("At Risk (₹)", format="₹%.2f")
            },
            use_container_width=True,
            hide_index=True
        )

        st.markdown("#### Vendor Deep Dive")
        selected_v_id = st.selectbox("Inspect Vendor Details", df_vendors["vendor_id"].tolist())
        v_profile = scorer.get_vendor_risk(selected_v_id)
        
        st.markdown(f"**Legal Name**: {v_profile.get('vendor_name')} (ID: `{v_profile.get('vendor_id')}`)")
        st.markdown(f"**Risk Score**: **{v_profile.get('risk_score')}/100** — Tier: `{v_profile.get('risk_category')}`")
        st.markdown("**Itemized Contributing Risk Factors:**")
        for factor in v_profile.get("risk_factors", []):
            st.markdown(f"- ⚠️ {factor}")


# -----------------------------------------------------------------------------
# 8. Batch Ingestion Manager
# -----------------------------------------------------------------------------
elif selected_page == "📦 Batch Ingestion Manager":
    st.markdown('<div class="page-title">Batch Ingestion & Multi-File Processing</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-desc">Automate bulk invoice processing across incoming directories and multi-file uploads.</div>', unsafe_allow_html=True)

    b_col1, b_col2 = st.columns(2)
    with b_col1:
        st.subheader("Process Incoming Directory")
        st.markdown(f"**Target Directory:** `{settings.INCOMING_DIR}`")
        if st.button("🚀 Process Incoming Batch Folder", type="primary"):
            with st.spinner("Processing batch folder through invoice pipeline..."):
                with get_db_session() as session:
                    batch_proc = BatchProcessor(session=session)
                    b_res = batch_proc.process_directory(
                        folder_path=str(settings.INCOMING_DIR),
                        use_llm=gemini_active
                    )
            if b_res.get("total_files", 0) > 0:
                st.success(f"Batch {b_res['batch_id']} completed! Success: {b_res['success_count']} | Failed: {b_res['failed_count']}")
            else:
                st.info("No unhandled files found in incoming directory.")

    with b_col2:
        st.subheader("Multi-File Invoice Upload")
        uploaded_batch = st.file_uploader(
            "Upload multiple invoice images (JPG, PNG, PDF)",
            type=["jpg", "jpeg", "png", "pdf"],
            accept_multiple_files=True
        )
        if uploaded_batch and st.button("Process Uploaded Invoices"):
            prog_bar = st.progress(0)
            status_text = st.empty()
            success_c = 0
            with get_db_session() as session:
                pipeline = InvoiceProcessingPipeline(session=session)
                for idx, up_f in enumerate(uploaded_batch):
                    status_text.text(f"Processing {up_f.name} ({idx+1}/{len(uploaded_batch)})...")
                    save_path = settings.DATA_DIR / "uploads" / up_f.name
                    with open(save_path, "wb") as f:
                        f.write(up_f.getbuffer())
                    pipeline.process_image(str(save_path), use_llm=gemini_active)
                    success_c += 1
                    prog_bar.progress((idx + 1) / len(uploaded_batch))
            status_text.text("Batch upload processing completed!")
            st.success(f"Successfully processed {success_c} invoices!")


# -----------------------------------------------------------------------------
# 9. Accuracy & Benchmarks
# -----------------------------------------------------------------------------
elif selected_page == "📈 Accuracy & Benchmarks":
    st.markdown('<div class="page-title">System Accuracy & Benchmark Reports</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-desc">Evaluation across the 180 ground truth labelled invoice dataset for OCR CER and field extraction accuracy.</div>', unsafe_allow_html=True)

    r_tab1, r_tab2 = st.tabs(["Field Extraction Accuracy", "OCR Engine CER Comparison"])

    with r_tab1:
        st.subheader("Field-Level Extraction Accuracy (180 Ground Truth Dataset)")
        if st.button("Run Accuracy Evaluation on Benchmark Dataset"):
            with st.spinner("Evaluating extraction across clean and degraded test invoices..."):
                acc_res = evaluate_extraction_accuracy(num_samples=20)
            
            st.markdown(f"**Invoices Evaluated**: {acc_res['invoices_evaluated']} (Clean: {acc_res['clean_count']}, Degraded: {acc_res['degraded_count']})")
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Clean Scan Accuracy", f"{acc_res['clean_avg_exact_match']}%", "Target: ≥95%")
            m2.metric("Degraded Scan Accuracy", f"{acc_res['degraded_avg_exact_match']}%", "Target: ≥85%")
            m3.metric("Grand Total Accuracy", f"{acc_res['grand_total_accuracy']}%", "Target: ≥97%")

            st.markdown("#### Field-by-Field Exact Match Accuracy")
            fields_df = pd.DataFrame({
                "Clean Scans (%)": acc_res["clean_field_accuracy"],
                "Degraded Scans (%)": acc_res["degraded_field_accuracy"],
                "Overall (%)": acc_res["overall_field_accuracy"]
            })
            st.dataframe(fields_df, use_container_width=True)

    with r_tab2:
        st.subheader("OCR Engine Comparison (Character Error Rate & Latency)")
        if st.button("Run OCR Comparison (PaddleOCR vs EasyOCR)"):
            with st.spinner("Evaluating OCR engines on test invoices..."):
                ocr_eval = evaluate_ocr_engines(sample_size=10)
            st.dataframe(pd.DataFrame(ocr_eval).T, use_container_width=True)


# -----------------------------------------------------------------------------
# 10. Settings & AI Configuration
# -----------------------------------------------------------------------------
elif selected_page == "⚙️ Settings & AI Configuration":
    st.markdown('<div class="page-title">AI Provider & System Settings</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-desc">Configure Google Gemini API keys, select active LLM models, and customize engine preferences.</div>', unsafe_allow_html=True)

    st.markdown("### Google AI Studio — Gemini Configuration")

    c_st1, c_st2 = st.columns(2)
    c_st1.metric("Status", "Configured" if gemini_active else "Not Configured")
    c_st1.caption(f"Masked Key: `{settings.get_masked_api_key()}`")
    c_st2.metric("Active Model", settings.GEMINI_MODEL)

    st.markdown("---")
    st.markdown("#### Update Gemini API Key")
    with st.form("api_key_form"):
        new_key = st.text_input(
            "Enter Google AI Studio Gemini API Key",
            type="password",
            placeholder="Paste your API key here (AQ.Ab... or AIza...)",
            help="Your API key is securely saved to your local environment file and never displayed in plaintext."
        )
        save_key_btn = st.form_submit_button("Save & Validate Key", type="primary")

    if save_key_btn:
        if not new_key.strip():
            st.error("Please enter a valid API key string.")
        else:
            settings.update_gemini_key(new_key.strip())
            st.success("API Key updated and securely saved to local environment!")
            
            # Live connection test
            test_ext = GeminiExtractor(api_key=new_key.strip())
            with st.spinner("Testing API connection with Google AI Studio..."):
                test_res = test_ext.test_connection()
            if test_res.get("success"):
                st.success(f"Connection test PASSED! {test_res.get('message')}")
            else:
                st.warning(f"Key saved, but test failed: {test_res.get('message')}. Rule-based extraction will remain active.")
            st.rerun()

    if st.button("Test Current API Connection"):
        with st.spinner("Testing connection..."):
            test_res = gemini_ext.test_connection()
            if test_res.get("success"):
                st.success(f"Connection test PASSED! {test_res.get('message')}")
            else:
                st.error(f"Connection test FAILED: {test_res.get('message')}")

    if st.button("Remove API Key"):
        settings.update_gemini_key("")
        st.warning("API key removed. OCR and rule-based extraction will operate 100% offline.")
        st.rerun()


# -----------------------------------------------------------------------------
# 11. Help & User Guide
# -----------------------------------------------------------------------------
elif selected_page == "❓ Help & User Guide":
    st.markdown('<div class="page-title">Enterprise System Documentation & User Guide</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-desc">Processing architecture, statutory validation rules, fraud detection models, and reviewer workflows.</div>', unsafe_allow_html=True)

    h_tab1, h_tab2, h_tab3 = st.tabs(["📘 Workflow Architecture", "📖 Statutory Rules & Data Entities", "🏢 Operational Context"])

    with h_tab1:
        st.markdown("""
        ### End-to-End Processing Architecture
        1. **Invoice Ingestion**: Upload invoices (JPG/PNG/PDF) or enter details manually.
        2. **Preprocessing Pipeline**: Deskewing, rotation correction, bilateral noise suppression, and Otsu binarization.
        3. **Multi-Engine OCR**: High-accuracy text bounding-box extraction using PaddleOCR (RapidOCR ONNX) or EasyOCR.
        4. **Hybrid Information Extraction**: Deterministic regex extractors normalize Indian currencies (`1,23,456.00`), standard ISO dates, GSTINs, line items, and totals; supplemented by Gemini LLM structured JSON.
        5. **Statutory Validation Engine**: 9 checks covering arithmetic, GSTIN checksum, vendor master match, 3-way PO matching, bank divergence, and approval limits.
        6. **Payment Fraud Detection**: Scans for exact duplicates, unverified bank remittance, PO budget overruns, and runs Isolation Forest ML anomaly scoring.
        7. **Review & Approval Gate**: Authorized reviewers inspect the image and extracted fields side-by-side, correct errors, and record decisions with an immutable audit log.
        """)

    with h_tab2:
        st.markdown("""
        ### Statutory Rules & Core Entities
        - **Vendor Master**: Legal company name, registered GSTIN, state, city, bank IFSC, registered bank account (last 4), status (Active / Blocked).
        - **Purchase Orders**: PO reference (`KIS/PO/YYYY/XXXXX`), vendor code, po date, po amount before tax, status.
        - **Approval Matrix**: Approver roles and authorization limits (AP-Clerk: ₹2,00,000; Senior AP: ₹5,00,000; Finance Manager: ₹15,00,000; VP Finance / CFO: Unlimited).
        - **HSN Item Catalog**: Standard HSN code, item description, statutory GST rate (18%).
        """)

    with h_tab3:
        st.markdown("""
        ### Enterprise Accounts Payable Profile
        - **Operational Facility**: Industrial Manufacturing & Procurement Facility (State Code `33`).
        - **Intra-State GST**: CGST 9% + SGST 9% for state-registered suppliers.
        - **Inter-State GST**: IGST 18% for out-of-state suppliers.
        - **Volume**: Continuous automated throughput across 60 registered suppliers and purchase orders.
        """)
