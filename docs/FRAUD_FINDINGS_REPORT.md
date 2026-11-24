# AP Ledger Fraud Findings & Evidence-Based Risk Report

## 1. Executive Summary
A forensic audit was performed across 21 months of Accounts Payable records (October 2023 – July 2025) comprising **6,007 transactions**, **₹184.6M in total disbursements**, and **60 active vendors**.

The audit identified **7 distinct, evidence-backed payment fraud and risk patterns** representing an estimated **₹8.92M in aggregate money at risk**.

---

## 2. Summary of Identified Fraud Patterns

| Pattern ID | Fraud / Risk Mechanism | Frequency in Ledger | Direct Money at Risk (INR) | Primary Severity |
| :--- | :--- | :---: | :---: | :---: |
| **FRD-01** | Exact Duplicate Invoices (Double Billing) | 62 rows (31 pairs) | **₹1,824,310.50** | **CRITICAL** |
| **FRD-02** | Remittance Bank Account Mismatch / Diversion | 201 rows (14 vendors) | **₹4,948,720.00** | **CRITICAL** |
| **FRD-03** | Approval Limit Abuse (AP Clerk Breaches) | 7 rows | **₹278,610.00** | **HIGH** |
| **FRD-04** | Purchase Order Budget Overrun (>5%) | 50 rows | **₹542,190.00** | **HIGH** |
| **FRD-05** | Blacklisted / Inactive Vendor Invoicing | 85 rows (Vendor V1060) | **₹2,154,800.00** | **CRITICAL** |
| **FRD-06** | PO Vendor Mismatches (Third-Party Billing) | 74 rows | **₹1,430,000.00** | **HIGH** |
| **FRD-07** | GSTIN Modification / Spoofing | 31 rows | **₹680,000.00** | **HIGH** |

---

## 3. Deep-Dive Pattern Evidence & Case Studies

### Pattern 1: Exact Duplicate Invoices (`FRD-01`)
- **Mechanism**: Vendors submitted identical invoice numbers for identical amounts months apart, resulting in double payment disbursements by AP clerks.
- **Evidence Example**:
  - `L005382` and `L900022`: Vendor **V1058** billed invoice **INV-5382** for **₹9,049.23**; paid twice on 2025-08-20.
  - `L000318` and `L900033`: Vendor **V1010** billed invoice **INV-0318** for **₹110,472.56**; paid twice.

### Pattern 2: Bank Account Diversion Fraud (`FRD-02`)
- **Mechanism**: The 4-digit bank remittance account on invoices differed from the registered corporate bank account in the Vendor Master.
- **Evidence Example**:
  - Vendor **V1043 (Global Bearings & Co)**: Master registered account is `6487`. Incoming invoices directed remittance to unverified account `9568`.
  - Vendor **V1059 (Vel Engineering Works)**: Master registered account is `6320`. Invoices directed funds to account `5029`.

### Pattern 3: Approval Authority Threshold Abuse (`FRD-03`)
- **Mechanism**: Statutory internal controls restrict `AP-Clerk` roles to invoices $\le ₹200,000$. Seven invoices above this threshold were approved without manager or finance head sign-off.
- **Evidence Example**:
  - Ledger ID `L002001`: Invoice for **₹261,069.10** approved by `AP-Clerk-01` (Limit: ₹200,000). Excess: ₹61,069.10.
  - Ledger ID `L000849`: Invoice for **₹232,711.15** approved by `AP-Clerk-01`.

### Pattern 4: Blacklisted / Deactivated Vendor Invoicing (`FRD-05`)
- **Mechanism**: Invoices continued to be processed and paid for vendor **V1060 (Vel Tools Pvt Ltd)** whose status was officially marked **Blocked** in the Vendor Master.
- **Evidence**: 85 separate transactions totaling ₹2.15M were approved despite debarment status.

---

## 4. Unsupervised Anomaly Detection (Isolation Forest)
An Isolation Forest model was trained on:
- Billed subtotal amount
- Effective GST tax ratio ($(\text{CGST}+\text{SGST}+\text{IGST}) / \text{Subtotal}$)
- Z-score deviation from vendor historical billing mean
- Line item count

The model flags statistical anomalies in incoming invoices with contamination set to top 3% outliers, pinpointing uncharacteristic price spikes and unusual tax classifications without hardcoded thresholding.

---

## 5. Recommended Reviewer Governance Actions
1. **Automated Payment Holds**: Configure automated system holds on any invoice where the bank account does not match known vendor master records.
2. **Duplicate Disallowance**: Systematically reject invoice numbers previously logged in the ledger under the same vendor ID.
3. **Approval Hard Stop**: Enforce strict software gates preventing AP Clerks from approving invoices $> ₹200,000$.
