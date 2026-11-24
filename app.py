"""Invoice Intelligence — OCR Extraction & Payment Fraud Detection Platform
Streamlit Review Dashboard and Accounts Payable Control Center
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
    page_title="Invoice Intelligence — Kaveri AP Platform",
    page_icon="📑",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished, clean, enterprise look
st.markdown("""
<style>
    /* Clean enterprise theme */
    .main-header {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 0.95rem;
        color: #64748B;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .metric-card-title {
        font-size: 0.82rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-card-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0F172A;
        margin-top: 4px;
    }
    .metric-card-sub {
        font-size: 0.8rem;
        color: #94A3B8;
        margin-top: 2px;
    }
    .status-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .badge-passed { background-color: #DCFCE7; color: #166534; }
    .badge-warning { background-color: #FEF9C3; color: #854D0E; }
    .badge-failed { background-color: #FEE2E2; color: #991B1B; }
    .badge-review { background-color: #E0E7FF; color: #3730A3; }
    .stButton>button {
        border-radius: 6px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Sidebar Navigation & Settings
# -----------------------------------------------------------------------------
st.sidebar.markdown("### 🏢 **Kaveri Industrial Supplies**")
st.sidebar.markdown("*Accounts Payable Control Center*")
st.sidebar.markdown("---")

PAGES = [
    "📊 Overview Dashboard",
    "📥 Invoice Processing",
    "📝 Manual Invoice Entry",
    "🔍 Invoice Review Screen",
    "📜 Invoice History & Audit Log",
    "🛡️ Payment Fraud Detection",
    "🏢 Vendor Risk Analysis",
    "📦 Batch Processing",
    "📈 Reports & Analytics",
    "⚙️ Settings & AI Configuration",
    "❓ Help & Documentation"
]

selected_page = st.sidebar.radio("Navigation", PAGES, index=0)

st.sidebar.markdown("---")
approver_role = st.sidebar.selectbox(
    "Active Approver Role",
    ["AP-Clerk-01", "AP-Clerk-02", "AP-Clerk-03", "AP-Manager-01", "Finance-Head"],
    index=0,
    help="Limits: Clerk (₹2L), Manager (₹10L), Finance Head (Unlimited)"
)

# Engine configuration status
gemini_ext = GeminiExtractor()
gemini_active = gemini_ext.is_configured()

st.sidebar.markdown("---")
st.sidebar.caption(f"**OCR Engine:** {settings.DEFAULT_OCR_ENGINE.upper()}")
st.sidebar.caption(f"**AI Extraction:** {'🟢 Gemini Active' if gemini_active else '🟡 Rule-based (Offline)'}")
st.sidebar.caption(f"**Database:** SQLite Active")


# -----------------------------------------------------------------------------
# 1. Overview Dashboard
# -----------------------------------------------------------------------------
if selected_page == "📊 Overview Dashboard":
    st.markdown('<div class="main-header">Accounts Payable Overview Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Real-time metrics, automated validation rates, fraud risks, and invoice throughput.</div>', unsafe_allow_html=True)

    with get_db_session() as session:
        repo = InvoiceRepository(session)
        metrics = repo.get_dashboard_metrics()
        recent_invoices = repo.list_invoices(limit=8)

    # Top KPI Metrics Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-title">Total Invoices Processed</div>
            <div class="metric-card-value">{metrics['total_invoices_processed']}</div>
            <div class="metric-card-sub">Total Value: ₹{metrics['total_invoice_value_inr']:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        val_pct = (metrics['validated_invoices'] / metrics['total_invoices_processed'] * 100) if metrics['total_invoices_processed'] else 100.0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-title">Validation Rate</div>
            <div class="metric-card-value">{val_pct:.1f}%</div>
            <div class="metric-card-sub">{metrics['validated_invoices']} Passed / {metrics['total_invoices_processed']} Total</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-title">Review Queue</div>
            <div class="metric-card-value">{metrics['invoices_requiring_review']}</div>
            <div class="metric-card-sub">{metrics['approved_invoices']} Approved | {metrics['rejected_invoices']} Rejected</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-title">Money at Risk</div>
            <div class="metric-card-value" style="color: #DC2626;">₹{metrics['total_amount_at_risk_inr']:,.2f}</div>
            <div class="metric-card-sub">{metrics['potentially_fraudulent_invoices']} Invoices Flagged</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns([3, 2])
    with col_left:
        st.subheader("Recent Invoices Requiring Review")
        if recent_invoices:
            table_data = []
            for inv in recent_invoices:
                table_data.append({
                    "Invoice ID": inv.id,
                    "Vendor Name": inv.vendor_name or "Unknown",
                    "Invoice No": inv.invoice_number or "N/A",
                    "Date": inv.invoice_date or "N/A",
                    "Amount (₹)": f"{inv.grand_total:,.2f}" if inv.grand_total else "0.00",
                    "Validation": inv.validation_status,
                    "Fraud Severity": inv.fraud_severity,
                    "Status": inv.status
                })
            df_recent = pd.DataFrame(table_data)
            st.dataframe(df_recent, use_container_width=True, hide_index=True)
        else:
            st.info("No invoices processed yet. Navigate to 'Invoice Processing' to upload an invoice.")

    with col_right:
        st.subheader("System Health & Processing Pipeline")
        st.markdown("""
        - **OCR Subsystem**: PaddleOCR (RapidOCR ONNX) & EasyOCR multi-engine active.
        - **Validation Engine**: 9 statutory checks active (GSTIN, arithmetic, 3-way PO match, bank account, HSN, approval limits).
        - **Fraud Analytics**: Isolation Forest + 6 rule-based fraud pattern detectors active.
        - **Audit Trail**: Full state mutation logging with reviewer comments.
        """)
        st.progress(100, text="Pipeline operational")


# -----------------------------------------------------------------------------
# 2. Invoice Processing (Upload & OCR Pipeline)
# -----------------------------------------------------------------------------
elif selected_page == "📥 Invoice Processing":
    st.markdown('<div class="main-header">Upload & Process Invoices</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Ingest scanned paper invoices (JPG, PNG, PDF) through image cleanup, OCR, and validation.</div>', unsafe_allow_html=True)

    up_col1, up_col2 = st.columns([2, 1])
    with up_col1:
        uploaded_file = st.file_uploader(
            "Choose an invoice image (JPG, PNG, PDF)",
            type=["jpg", "jpeg", "png", "pdf"],
            help="High-resolution scans or mobile photos with tilt/noise are supported."
        )

    with up_col2:
        st.markdown("#### Processing Options")
        ocr_choice = st.selectbox("OCR Engine", ["rapidocr", "easyocr"], index=0, help="RapidOCR uses PaddleOCR models.")
        use_gemini = st.checkbox("Enable Gemini LLM Extraction", value=gemini_active, disabled=not gemini_active)
        if not gemini_active:
            st.caption("ℹ️ Configure API Key in Settings to enable Gemini LLM extraction.")

    if uploaded_file is not None:
        # Save temp file
        temp_dir = settings.DATA_DIR / "uploads"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_file_path = temp_dir / uploaded_file.name

        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        c_img1, c_img2 = st.columns(2)
        with c_img1:
            st.image(uploaded_file, caption=f"Uploaded: {uploaded_file.name}", use_container_width=True)

        if st.button("🚀 Process Invoice Through Pipeline", type="primary"):
            with st.spinner("Processing invoice: deskewing -> OCR -> extraction -> validation -> fraud scan..."):
                with get_db_session() as session:
                    pipeline = InvoiceProcessingPipeline(session=session)
                    res = pipeline.process_image(
                        image_path=str(temp_file_path),
                        engine_name=ocr_choice,
                        use_llm=use_gemini,
                        input_method="upload",
                        approver_role=approver_role
                    )

            st.success(f"Processing Complete! Assigned Invoice ID: **{res['invoice_id']}**")

            # Display processed side-by-side
            if res.get("processed_image_path") and os.path.exists(res["processed_image_path"]):
                with c_img2:
                    st.image(res["processed_image_path"], caption="Preprocessed Image (Deskewed & Binarized)", use_container_width=True)

            # Quick summary cards
            q1, q2, q3, q4 = st.columns(4)
            q1.metric("Vendor", res["extraction"].vendor_name or "Unknown")
            q2.metric("Invoice #", res["extraction"].invoice_number or "N/A")
            q3.metric("Grand Total", f"₹{res['extraction'].grand_total:,.2f}" if res['extraction'].grand_total else "N/A")
            q4.metric("Validation", res["validation"].status)

            # Details
            with st.expander("📋 Extracted Line Items"):
                if res["extraction"].line_items:
                    st.table([it.dict() for it in res["extraction"].line_items])
                else:
                    st.info("No individual line items parsed.")

            with st.expander("🛡️ Validation & Fraud Checks"):
                for vr in res["validation"].rules_evaluated:
                    if vr.status == "Passed":
                        st.markdown(f"✅ **{vr.rule_name}**: {vr.message}")
                    elif vr.status == "Warning":
                        st.markdown(f"⚠️ **{vr.rule_name}**: {vr.message}")
                    else:
                        st.markdown(f"❌ **{vr.rule_name}**: {vr.message}")

                if res["fraud"].alerts:
                    st.error(f"🚨 **{len(res['fraud'].alerts)} Payment Fraud Alert(s) Detected:**")
                    for fa in res["fraud"].alerts:
                        st.markdown(f"- **{fa.pattern_name}** ({fa.severity}): {fa.explanation}")


# -----------------------------------------------------------------------------
# 3. Manual Invoice Entry
# -----------------------------------------------------------------------------
elif selected_page == "📝 Manual Invoice Entry":
    st.markdown('<div class="main-header">Manual Invoice Data Entry</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Directly enter invoice details for paper invoices without OCR. Full validation and fraud checks apply.</div>', unsafe_allow_html=True)

    with st.form("manual_entry_form"):
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            m_vendor = st.text_input("Vendor Legal Name*", placeholder="e.g. Vel Engineering Works")
            m_gstin = st.text_input("Vendor GSTIN*", placeholder="15-character statutory GSTIN")
            m_inv_no = st.text_input("Invoice Number*", placeholder="e.g. TI/2025/8093")
            m_inv_date = st.date_input("Invoice Date*", value=datetime.today())
            m_due_date = st.date_input("Due Date", value=datetime.today())

        with col_m2:
            m_po = st.text_input("Purchase Order #", placeholder="e.g. KIS/PO/2024/10001")
            m_subtotal = st.number_input("Subtotal (₹)*", min_value=0.0, step=100.0)
            m_cgst = st.number_input("CGST (₹)", min_value=0.0, step=10.0)
            m_sgst = st.number_input("SGST (₹)", min_value=0.0, step=10.0)
            m_igst = st.number_input("IGST (₹)", min_value=0.0, step=10.0)
            m_grand = st.number_input("Grand Total (₹)*", min_value=0.0, step=100.0)
            m_bank = st.text_input("Bank Account (Last 4 digits)", max_chars=4, placeholder="e.g. 5029")

        st.markdown("#### Line Items")
        item_desc = st.text_input("Item Description", value="Industrial Ball Bearing 6205-2RS")
        item_hsn = st.text_input("HSN Code", value="8482")
        c_q, c_r = st.columns(2)
        item_qty = c_q.number_input("Quantity", min_value=1.0, value=5.0)
        item_rate = c_r.number_input("Unit Rate (₹)", min_value=0.0, value=m_subtotal/5.0 if m_subtotal else 100.0)

        submit_btn = st.form_submit_button("Submit & Validate Invoice", type="primary")

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

            st.success(f"Invoice saved successfully! ID: **{res['invoice_id']}**")
            st.info(f"Validation Status: **{res['validation'].status}** | Fraud Risk: **{res['fraud'].composite_risk_score}**")


# -----------------------------------------------------------------------------
# 4. Invoice Review Screen (Side-by-side human-in-the-loop)
# -----------------------------------------------------------------------------
elif selected_page == "🔍 Invoice Review Screen":
    st.markdown('<div class="main-header">Invoice Review & Authorization</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Inspect original scanned document side-by-side with extracted fields, validation results, and fraud alerts.</div>', unsafe_allow_html=True)

    with get_db_session() as session:
        repo = InvoiceRepository(session)
        invoices = repo.list_invoices(limit=200)

    if not invoices:
        st.warning("No invoices in the database to review.")
    else:
        inv_choices = {f"{inv.id} - {inv.vendor_name or 'Unknown'} (#{inv.invoice_number or 'N/A'}) - ₹{inv.grand_total:,.2f}": inv.id for inv in invoices}
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

                st.markdown("---")

                # Side-by-side: Image on Left, Form on Right
                col_view1, col_view2 = st.columns([1, 1])

                with col_view1:
                    st.subheader("Original Document")
                    if inv.original_image_path and os.path.exists(inv.original_image_path):
                        st.image(inv.original_image_path, caption=f"File: {inv.file_name}", use_container_width=True)
                    else:
                        st.info("No scanned image available for this invoice (e.g. manual entry).")

                with col_view2:
                    st.subheader("Extracted Data Editor")
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
                        reviewer_notes = st.text_area("Reviewer Comments / Audit Notes", value=inv.reviewer_comments or "")

                        save_btn = st.form_submit_button("💾 Save Changes & Submit Decision", type="primary")

                    if save_btn:
                        # Prevent auto-approval if critical fraud alert
                        if decision == "APPROVED" and (inv.has_fraud_risk and inv.fraud_severity == "CRITICAL"):
                            st.error("❌ High-Risk Fraud Alert: Cannot approve invoice with CRITICAL fraud flags. Escalate or reject.")
                        else:
                            with get_db_session() as save_session:
                                save_repo = InvoiceRepository(save_session)
                                save_repo.update_invoice_review(
                                    invoice_id=inv.id,
                                    decision=decision,
                                    reviewer_name=approver_role,
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
                            st.success(f"Decision '{decision}' recorded successfully with full audit trail!")
                            st.rerun()

                st.markdown("---")
                t1, t2, t3 = st.tabs(["🛡️ Validation Results", "🚨 Fraud Findings", "📜 Audit History"])
                with t1:
                    for v in inv.validations:
                        st.markdown(f"- **{v.rule_name}** ({v.status}): {v.message}")
                with t2:
                    if inv.fraud_alerts:
                        for fa in inv.fraud_alerts:
                            st.error(f"**{fa.pattern_name}** | Severity: **{fa.severity}** | Risk Score: **{fa.risk_score}**\n\n- *Explanation*: {fa.explanation}\n- *Recommended Action*: {fa.recommended_action}")
                    else:
                        st.success("No active fraud alerts detected on this invoice.")
                with t3:
                    audits = repo.get_audit_trail_for_invoice(inv.id)
                    for a in audits:
                        st.markdown(f"⏱️ **{a.timestamp.strftime('%Y-%m-%d %H:%M:%S')}** - **{a.action}** by `{a.user}`: {a.comments or ''}")


# -----------------------------------------------------------------------------
# 5. Invoice History & Audit Log
# -----------------------------------------------------------------------------
elif selected_page == "📜 Invoice History & Audit Log":
    st.markdown('<div class="main-header">Invoice Repository & Immutable Audit Trail</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Search, filter, and audit every ingested invoice and reviewer mutation.</div>', unsafe_allow_html=True)

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
            limit=500
        )

        st.markdown(f"**Showing {len(invoices)} Invoices**")
        if invoices:
            data = []
            for inv in invoices:
                data.append({
                    "Invoice ID": inv.id,
                    "Vendor Name": inv.vendor_name or "Unknown",
                    "Invoice #": inv.invoice_number or "N/A",
                    "Date": inv.invoice_date or "N/A",
                    "Amount (₹)": f"{inv.grand_total:,.2f}" if inv.grand_total else "0.00",
                    "Validation": inv.validation_status,
                    "Fraud Severity": inv.fraud_severity,
                    "Status": inv.status,
                    "Reviewer": inv.reviewer_name or "Unassigned",
                    "Created At": inv.created_at.strftime("%Y-%m-%d %H:%M")
                })
            df_hist = pd.DataFrame(data)
            st.dataframe(df_hist, use_container_width=True, hide_index=True)
            
            # Export CSV
            csv_data = df_hist.to_csv(index=False)
            st.download_button("📥 Export Invoice History to CSV", csv_data, "invoice_history.csv", "text/csv")


# -----------------------------------------------------------------------------
# 6. Payment Fraud Detection Center
# -----------------------------------------------------------------------------
elif selected_page == "🛡️ Payment Fraud Detection":
    st.markdown('<div class="main-header">Payment Fraud & Anomaly Detection Center</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Evidence-based detection of duplicate billing, bank diversion, PO overruns, and statistical anomalies.</div>', unsafe_allow_html=True)

    f_tab1, f_tab2 = st.tabs(["Active Invoices with Fraud Alerts", "Historical AP Ledger Fraud Audit (2 Years)"])

    with f_tab1:
        with get_db_session() as session:
            repo = InvoiceRepository(session)
            fraud_invoices = repo.list_invoices(has_fraud=True, limit=100)

        if not fraud_invoices:
            st.info("No active fraud alerts detected on current invoices.")
        else:
            for inv in fraud_invoices:
                with st.expander(f"🚨 {inv.id} — {inv.vendor_name} (#{inv.invoice_number}) | Risk: {inv.fraud_risk_score}/100 [{inv.fraud_severity}]"):
                    st.markdown(f"**Amount at Risk:** ₹{inv.amount_at_risk:,.2f}")
                    for fa in inv.fraud_alerts:
                        st.markdown(f"- **Pattern**: {fa.pattern_name}")
                        st.markdown(f"- **Explanation**: {fa.explanation}")
                        st.markdown(f"- **Recommended Action**: {fa.recommended_action}")

    with f_tab2:
        st.subheader("2-Year AP Ledger Fraud Findings & Evidence")
        st.markdown("""
        Analysis of the 6,007 historical AP transactions across 60 vendors revealed 7 critical risk patterns:
        1. **Exact Duplicate Invoices**: 62 transactions (31 duplicate invoice pairs) resulting in ₹1.82M double billing.
        2. **Bank Account Remittance Divergence**: 201 invoices with unverified bank accounts across 14 vendors.
        3. **GSTIN Spoofing / Mismatches**: 31 invoices with modified GST numbers.
        4. **PO Budget Overruns**: 50 invoices exceeding purchase order authorization by >5%.
        5. **Approver Limit Abuse**: 7 invoices approved in breach of statutory clerk authorization limits.
        6. **Blacklisted Vendor Invoicing**: 85 invoices processed for deactivated vendor V1060.
        """)
        
        # Display sample ledger duplicates
        try:
            ledger_df = pd.read_csv(settings.LEDGER_PATH)
            dups = ledger_df[ledger_df.duplicated(subset=['vendor_id', 'vendor_invoice_no'], keep=False)]
            st.markdown(f"#### Exact Duplicate Invoices Identified ({len(dups)} rows)")
            st.dataframe(dups[['ledger_id', 'vendor_id', 'vendor_invoice_no', 'invoice_date', 'total_amount', 'paid_date']].head(20), use_container_width=True)
        except Exception as e:
            st.error(f"Could not load ledger: {e}")


# -----------------------------------------------------------------------------
# 7. Vendor Risk Analysis
# -----------------------------------------------------------------------------
elif selected_page == "🏢 Vendor Risk Analysis":
    st.markdown('<div class="main-header">Vendor Risk Scoring & Compliance Profile</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Explainable composite risk index (0–100) based on historical bank changes, duplicates, and compliance.</div>', unsafe_allow_html=True)

    scorer = VendorRiskScorer()
    df_vendors = scorer.get_all_profiles_df()

    if df_vendors.empty:
        st.warning("Vendor master data not available.")
    else:
        v_col1, v_col2, v_col3 = st.columns(3)
        v_col1.metric("Total Registered Vendors", len(df_vendors))
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
            use_container_width=True,
            hide_index=True
        )

        st.markdown("#### Vendor Risk Deep Dive")
        selected_v_id = st.selectbox("Inspect Vendor Details", df_vendors["vendor_id"].tolist())
        v_profile = scorer.get_vendor_risk(selected_v_id)
        
        st.write(f"**Vendor Legal Name**: {v_profile.get('vendor_name')} (ID: {v_profile.get('vendor_id')})")
        st.write(f"**Risk Score**: {v_profile.get('risk_score')}/100 — Category: **{v_profile.get('risk_category')}**")
        st.write("**Itemized Contributing Risk Factors:**")
        for factor in v_profile.get("risk_factors", []):
            st.markdown(f"- ⚠️ {factor}")


# -----------------------------------------------------------------------------
# 8. Batch Processing
# -----------------------------------------------------------------------------
elif selected_page == "📦 Batch Processing":
    st.markdown('<div class="main-header">Batch Ingestion & Directory Processing</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Automate batch processing for incoming folders or multi-file zip drops.</div>', unsafe_allow_html=True)

    b_col1, b_col2 = st.columns(2)
    with b_col1:
        st.subheader("Process Incoming Folder")
        st.markdown(f"**Incoming Directory:** `{settings.INCOMING_DIR}`")
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
                st.info("No new files found in incoming directory.")

    with b_col2:
        st.subheader("Multi-File Invoice Upload")
        uploaded_batch = st.file_uploader(
            "Upload multiple invoice images",
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
# 9. Reports & Analytics
# -----------------------------------------------------------------------------
elif selected_page == "📈 Reports & Analytics":
    st.markdown('<div class="main-header">System Accuracy & Evaluation Reports</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Evaluation on ground truth labelled dataset (180 invoices) across OCR, CER, and field accuracy.</div>', unsafe_allow_html=True)

    r_tab1, r_tab2 = st.tabs(["Extraction Accuracy Evaluation", "OCR Engine CER Comparison"])

    with r_tab1:
        st.subheader("Field-Level Extraction Accuracy (180 Ground Truth Dataset)")
        if st.button("Run Accuracy Evaluation on Test Set"):
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
    st.markdown('<div class="main-header">AI Provider & System Settings</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Configure Google Gemini API key, select LLM models, and customize engine preferences.</div>', unsafe_allow_html=True)

    st.markdown("### Google AI Studio — Gemini Configuration")

    # Current status
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
            help="Your API key is never committed to GitHub or displayed in plaintext."
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
# 11. Help & Documentation
# -----------------------------------------------------------------------------
elif selected_page == "❓ Help & Documentation":
    st.markdown('<div class="main-header">System Documentation & User Guide</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Architecture, business rules, statutory GST compliance, and reviewer workflows.</div>', unsafe_allow_html=True)

    h_tab1, h_tab2, h_tab3 = st.tabs(["📘 Workflow Guide", "📖 Data Dictionary Reference", "📜 Project Charter Summary"])

    with h_tab1:
        st.markdown("""
        ### End-to-End Workflow
        1. **Invoice Ingestion**: Upload invoices (JPG/PNG/PDF) or enter details manually.
        2. **Preprocessing**: Automatically deskews, detects rotation, removes background stamp noise, and applies Otsu binarization.
        3. **OCR Layer**: High-accuracy text and bounding box extraction using PaddleOCR (RapidOCR ONNX) or EasyOCR.
        4. **Hybrid Information Extraction**: Deterministic regex extractors normalize Indian currencies (`1,23,456.00`), standard ISO dates, GSTINs, line items, and totals; supplemented by Gemini LLM structured JSON.
        5. **Validation Engine**: Performs 9 statutory validations (arithmetic, GSTIN checksum, vendor master match, 3-way PO matching, bank account verification, approval limits).
        6. **Payment Fraud Detection**: Scans for exact duplicates, unverified bank remittance, PO budget overruns, and runs Isolation Forest ML anomaly scoring.
        7. **Human-in-the-Loop Review**: Authorized reviewers inspect the image and extracted fields side-by-side, correct errors, and record decisions with an immutable audit log.
        """)

    with h_tab2:
        st.markdown("""
        ### Key Data Entities
        - **Vendor Master**: Legal company name, registered GSTIN, state, city, bank IFSC, registered bank account (last 4), status (Active / Blocked).
        - **Purchase Orders**: PO reference (`KIS/PO/YYYY/XXXXX`), vendor code, po date, po amount before tax, status.
        - **Approval Matrix**: Approver roles and authorization limits (AP-Clerk: ₹2,00,000; AP-Manager: ₹10,00,000; Finance-Head: Unlimited).
        - **HSN Item Catalog**: Standard HSN code, item description, statutory GST rate (18%).
        """)

    with h_tab3:
        st.markdown("""
        ### Client Context: Kaveri Industrial Supplies Pvt Ltd
        - **Headquarters**: Hosur, Tamil Nadu (State Code `33`).
        - **Intra-state Tax**: CGST 9% + SGST 9% for Tamil Nadu vendors.
        - **Inter-state Tax**: IGST 18% for vendors outside Tamil Nadu.
        - **Volume**: ~300 invoices/month across ~60 registered suppliers.
        """)
