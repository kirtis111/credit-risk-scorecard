"""
Credit Risk Scorecard Dashboard
Canadian BFSI — Internal Model Monitoring Tool

Run:  streamlit run app.py
"""

import streamlit as st

st.set_page_config(
    page_title="Credit Risk Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📊 Credit Risk Dashboard")
    st.caption("Canadian BFSI | OSFI E-23 / IFRS 9")
    st.divider()

    st.markdown("**Model Status**")
    st.success("Champion (LR Scorecard) v1.0 — Active")
    st.warning("Challenger (GBM) v1.0 — In Validation")

    st.divider()
    st.markdown("**Regulatory Framework**")
    st.markdown("""
- OSFI E-23 (Model Risk)
- IFRS 9 (ECL Provisioning)
- Basel III AIRB
- OSFI B-20 (Mortgages)
- PIPEDA
    """)
    st.divider()
    st.caption("Last refreshed: March 2025")


# ── Main page ──────────────────────────────────────────────────────────────
st.title("Credit Risk Scorecard — Home")
st.markdown("Internal model monitoring and loan decisioning tool. Navigate using the sidebar pages.")

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📋 Loan Origination")
    st.markdown("""
Real-time credit scoring for loan officers.

- Enter applicant details
- Get score (300–850 scale)
- View PD, LGD, EAD estimates
- Adverse action reason codes
- IFRS 9 stage assignment
    """)

with col2:
    st.subheader("📈 Portfolio Monitoring")
    st.markdown("""
Monthly model performance tracking.

- AUC / Gini / KS over time
- PSI score distribution stability
- Score shift — dev vs current
- RAG status per OSFI E-23
- Champion vs challenger comparison
    """)

with col3:
    st.subheader("🏛️ IFRS 9 Provisioning")
    st.markdown("""
Expected Credit Loss engine.

- Stage 1 / 2 / 3 breakdown
- ECL = PD × LGD × EAD × DF
- FLI macro scenario overlay
- BoC optimistic / base / adverse
- Coverage ratio by stage
    """)

st.divider()
st.info(
    "**Dataset:** Kaggle HMEQ (Home Equity Loan Default) — "
    "https://www.kaggle.com/datasets/ajay1735/hmeq-data — "
    "n = 5,960 rows | Default rate ≈ 20%"
)
