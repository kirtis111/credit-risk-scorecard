"""
app.py — Streamlit Credit Risk Dashboard
Canadian BFSI — Loan Officer Simulation Tool

Main entry point. Mirrors loan origination tools used at
RBC, TD Banknote, BMO, and Meridian Credit Union.

Run: streamlit run app.py
"""

import streamlit as st

# ─────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CreditIQ — Canadian Risk Platform",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "BFSI Credit Risk Scorecard — Canadian Banking Edition\nOSFI E-23 / IFRS 9 Compliant"
    },
)

# ─────────────────────────────────────────────
# Custom CSS — Corporate Banking Aesthetic
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Import clean corporate font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * { font-family: 'Inter', sans-serif; }

    /* Main container */
    .main { background-color: #f8f9fa; }
    .block-container { padding-top: 1rem; padding-bottom: 1rem; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #003366 0%, #1a3a5c 60%, #003366 100%);
    }
    [data-testid="stSidebar"] * { color: #ffffff !important; }
    [data-testid="stSidebar"] .stSelectbox label { color: #a8c8e8 !important; }
    [data-testid="stSidebar"] h1, h2, h3 { color: #ffffff !important; }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: white;
        border: 1px solid #e0e7ff;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 8px rgba(0,51,102,0.08);
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        color: #003366 !important;
    }
    [data-testid="metric-container"] [data-testid="stMetricLabel"] {
        font-size: 0.75rem !important;
        font-weight: 600 !important;
        color: #6b7280 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Headers */
    h1 { color: #003366 !important; font-weight: 700 !important; }
    h2 { color: #1a3a5c !important; font-weight: 600 !important; }
    h3 { color: #2c5282 !important; }

    /* Decision badges */
    .badge-approve {
        background: #d1fae5; color: #065f46; padding: 6px 16px;
        border-radius: 20px; font-weight: 700; font-size: 0.9rem;
        border: 2px solid #34d399; display: inline-block;
    }
    .badge-decline {
        background: #fee2e2; color: #991b1b; padding: 6px 16px;
        border-radius: 20px; font-weight: 700; font-size: 0.9rem;
        border: 2px solid #f87171; display: inline-block;
    }
    .badge-refer {
        background: #fef3c7; color: #92400e; padding: 6px 16px;
        border-radius: 20px; font-weight: 700; font-size: 0.9rem;
        border: 2px solid #fbbf24; display: inline-block;
    }

    /* RAG indicators */
    .rag-green { color: #16a34a; font-weight: 700; }
    .rag-amber { color: #d97706; font-weight: 700; }
    .rag-red   { color: #dc2626; font-weight: 700; }

    /* Divider */
    hr { border-color: #e0e7ff; margin: 1rem 0; }

    /* Buttons */
    .stButton > button {
        background: #003366 !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        padding: 0.5rem 2rem !important;
    }
    .stButton > button:hover {
        background: #1a3a5c !important;
        box-shadow: 0 4px 12px rgba(0,51,102,0.3) !important;
    }

    /* Info boxes */
    .info-box {
        background: #eff6ff; border-left: 4px solid #3b82f6;
        padding: 12px 16px; border-radius: 4px; margin: 8px 0;
        font-size: 0.85rem; color: #1e40af;
    }
    .warning-box {
        background: #fffbeb; border-left: 4px solid #f59e0b;
        padding: 12px 16px; border-radius: 4px; margin: 8px 0;
        font-size: 0.85rem; color: #92400e;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Landing Page
# ─────────────────────────────────────────────
def main():
    # Header
    col_logo, col_title = st.columns([1, 8])
    with col_logo:
        st.markdown("## 🏦")
    with col_title:
        st.markdown("# CreditIQ — Canadian Risk Platform")
        st.caption("OSFI E-23 · IFRS 9 · Basel III AIRB | Credit Risk Analytics")

    st.divider()

    # Navigation cards
    st.markdown("### 🗂️ Platform Modules")
    st.info("Use the **left sidebar** to navigate between modules, or click the module cards below.")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div style="background:white; border:1px solid #e0e7ff; border-radius:12px; padding:20px;
                    border-top:4px solid #003366; box-shadow:0 2px 8px rgba(0,51,102,0.08);">
            <h3 style="color:#003366; margin-top:0;">📋 Loan Origination</h3>
            <p style="color:#6b7280; font-size:0.85rem;">
                Real-time credit scoring for loan officers.
                Enter applicant details, get instant score,
                PD estimate, and adverse action reasons.
            </p>
            <p style="color:#003366; font-weight:600; font-size:0.8rem;">
                ➤ Score 300–850 · SHAP Explanations · Decision Rules
            </p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div style="background:white; border:1px solid #e0e7ff; border-radius:12px; padding:20px;
                    border-top:4px solid #1a3a5c; box-shadow:0 2px 8px rgba(0,51,102,0.08);">
            <h3 style="color:#1a3a5c; margin-top:0;">📊 Portfolio Monitoring</h3>
            <p style="color:#6b7280; font-size:0.85rem;">
                Live model performance dashboard.
                PSI, CSI, AUC decay, score distribution
                shift over time.
            </p>
            <p style="color:#1a3a5c; font-weight:600; font-size:0.8rem;">
                ➤ PSI/CSI · ROC · Vintage Analysis · RAG Status
            </p>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div style="background:white; border:1px solid #e0e7ff; border-radius:12px; padding:20px;
                    border-top:4px solid #2c5282; box-shadow:0 2px 8px rgba(0,51,102,0.08);">
            <h3 style="color:#2c5282; margin-top:0;">🏛️ IFRS 9 Provisioning</h3>
            <p style="color:#6b7280; font-size:0.85rem;">
                Expected Credit Loss engine.
                Stage 1/2/3 migration, ECL calculation,
                FLI macro overlay scenarios.
            </p>
            <p style="color:#2c5282; font-weight:600; font-size:0.8rem;">
                ➤ PD/LGD/EAD · ECL Staging · BoC Scenarios
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Regulatory framework banner
    st.markdown("""
    <div style="background:linear-gradient(135deg,#003366,#1a3a5c); color:white;
                border-radius:10px; padding:16px 24px; margin:8px 0;">
        <div style="display:flex; align-items:center; gap:24px; flex-wrap:wrap;">
            <div><strong>🇨🇦 Regulatory Framework</strong></div>
            <div>📋 OSFI E-23 Model Risk Management</div>
            <div>📊 IFRS 9 ECL Provisioning</div>
            <div>🏛️ Basel III AIRB</div>
            <div>🔒 OSFI B-20 Mortgage Standards</div>
            <div>⚖️ PIPEDA Compliant</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar nav
    with st.sidebar:
        st.markdown("## 🏦 CreditIQ")
        st.caption("Canadian Risk Platform")
        st.divider()
        st.markdown("**Navigation**")
        st.markdown("↗️ Use **Pages** above to navigate")
        st.divider()
        st.markdown("**Model Status**")
        st.markdown("🟢 **Champion (LR):** v1.0 Active")
        st.markdown("🟡 **Challenger (XGB):** v1.0 In Validation")
        st.divider()
        st.markdown("**Last Updated**")
        st.caption("March 2025 | OSFI E-23 Compliant")


if __name__ == "__main__":
    main()
