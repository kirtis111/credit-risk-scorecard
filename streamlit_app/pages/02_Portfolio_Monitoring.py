"""
02_Portfolio_Monitoring.py
──────────────────────────
Model performance monitoring dashboard.
PSI, CSI, AUC decay, score distributions — OSFI E-23 aligned.
Mirrors the MRM monitoring dashboards at Canadian Schedule I banks.
"""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import gaussian_kde
from sklearn.metrics import roc_auc_score, roc_curve

st.set_page_config(page_title="Portfolio Monitoring | CreditIQ", page_icon="📊", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
* { font-family:'Inter',sans-serif; }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#003366,#1a3a5c); }
[data-testid="stSidebar"] * { color:#fff !important; }
h1,h2 { color:#003366 !important; }
[data-testid="metric-container"] { background:white; border:1px solid #e0e7ff;
    border-radius:10px; padding:12px; box-shadow:0 2px 8px rgba(0,51,102,0.08); }
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size:1.5rem !important; font-weight:700 !important; color:#003366 !important; }
</style>""", unsafe_allow_html=True)


# ── Synthetic monitoring data (replace with model.predict in production) ──
@st.cache_data
def generate_monitoring_data():
    np.random.seed(42)
    n = 5960

    # Development population
    dev_scores  = np.random.normal(635, 68, n).clip(300, 850)
    dev_pd      = 1 / (1 + np.exp((dev_scores - 600) / 28.85))
    dev_bad     = (np.random.rand(n) < dev_pd).astype(int)

    # Monthly monitoring slices (12 months — gradual drift)
    months, mon_data = pd.date_range("2024-01-01", periods=12, freq="MS"), []
    for i, mo in enumerate(months):
        drift = i * 1.8
        mn   = np.random.normal(635 - drift, 68 + drift * 0.3, 450).clip(300, 850)
        mpd  = 1 / (1 + np.exp((mn - 600) / 28.85))
        mbad = (np.random.rand(450) < mpd).astype(int)
        try:
            auc = roc_auc_score(mbad, mpd)
        except Exception:
            auc = 0.78
        mon_data.append(dict(
            month=mo, n=450,
            mean_score=mn.mean(), std_score=mn.std(),
            default_rate=mbad.mean(), expected_pd=mpd.mean(),
            auc_roc=auc, gini=2*auc-1,
            psi=i * 0.012 + np.random.uniform(0, 0.008),
            scores=mn, pds=mpd, bads=mbad
        ))

    return dict(dev_scores=dev_scores, dev_pd=dev_pd, dev_bad=dev_bad,
                mon_data=mon_data, months=months)


data = generate_monitoring_data()
mon  = data["mon_data"]
months_labels = [m["month"].strftime("%b %Y") for m in mon]

# ── Sidebar controls ──────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Monitoring Controls")
    st.divider()
    sel_month = st.selectbox("Monitoring Period", months_labels, index=len(months_labels)-1)
    model_ver = st.selectbox("Model Version", ["v1.0 — Champion (LR)", "v1.0 — Challenger (XGB)"])
    st.divider()
    st.markdown("**Thresholds**")
    psi_amber = st.slider("PSI Amber", 0.05, 0.20, 0.10, 0.01)
    psi_red   = st.slider("PSI Red",   0.10, 0.40, 0.25, 0.01)
    st.divider()
    st.markdown("**Export**")
    if st.button("📥 Download Monitoring Report", use_container_width=True):
        st.info("In production: generates PDF/Excel report")

sel_idx = months_labels.index(sel_month)
cur     = mon[sel_idx]

# ── Header ────────────────────────────────────────────────────
st.markdown("# 📊 Portfolio Monitoring Dashboard")
st.caption(f"OSFI E-23 Monthly Model Monitoring | Period: {sel_month} | Model: {model_ver}")

# ── KPI Cards ─────────────────────────────────────────────────
psi_now = cur["psi"]
rag = "🟢 Green" if psi_now < psi_amber else ("🟡 Amber" if psi_now < psi_red else "🔴 Red")
c1,c2,c3,c4,c5,c6 = st.columns(6)
c1.metric("AUC-ROC",      f"{cur['auc_roc']:.4f}", f"{cur['auc_roc']-mon[0]['auc_roc']:+.4f}")
c2.metric("Gini",         f"{cur['gini']:.4f}",    f"{cur['gini']-mon[0]['gini']:+.4f}")
c3.metric("Mean Score",   f"{cur['mean_score']:.0f}", f"{cur['mean_score']-mon[0]['mean_score']:+.0f}")
c4.metric("Default Rate", f"{cur['default_rate']*100:.2f}%")
c5.metric("PSI",          f"{psi_now:.4f}",         rag)
c6.metric("Population N", f"{cur['n']:,}")

st.divider()

# ── Row 1: AUC Decay + PSI Chart ──────────────────────────────
col_auc, col_psi = st.columns(2)

with col_auc:
    st.markdown("### 📈 AUC-ROC Over Time")
    fig, ax = plt.subplots(figsize=(7, 3.8))
    aucs   = [m["auc_roc"] for m in mon]
    colors = ["#dc2626" if a < 0.70 else "#d97706" if a < 0.75 else "#16a34a" for a in aucs]
    ax.bar(range(len(mon)), aucs, color=colors, edgecolor="white", width=0.7)
    ax.axhline(0.75, color="#d97706", ls="--", lw=1.5, label="Amber threshold (0.75)")
    ax.axhline(0.70, color="#dc2626", ls="--", lw=1.5, label="Red threshold (0.70)")
    ax.axvline(sel_idx, color="#003366", ls="-", lw=2, alpha=0.6, label="Selected period")
    ax.set_xticks(range(len(mon))); ax.set_xticklabels(months_labels, rotation=45, fontsize=7)
    ax.set_ylabel("AUC-ROC"); ax.set_ylim(0.60, 0.95)
    ax.set_title("AUC-ROC Monthly Trend", fontweight="bold")
    ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.3)
    st.pyplot(fig, use_container_width=True); plt.close("all")

with col_psi:
    st.markdown("### 📉 PSI — Population Stability Index")
    fig, ax = plt.subplots(figsize=(7, 3.8))
    psis   = [m["psi"] for m in mon]
    pcolors= ["#dc2626" if p >= psi_red else "#d97706" if p >= psi_amber else "#16a34a" for p in psis]
    bars   = ax.bar(range(len(mon)), psis, color=pcolors, edgecolor="white", width=0.7)
    ax.axhline(psi_amber, color="#d97706", ls="--", lw=1.5, label=f"Amber ({psi_amber:.2f})")
    ax.axhline(psi_red,   color="#dc2626", ls="--", lw=1.5, label=f"Red ({psi_red:.2f})")
    ax.axvline(sel_idx, color="#003366", ls="-", lw=2, alpha=0.6)
    for bar, val in zip(bars, psis):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
                f"{val:.3f}", ha="center", fontsize=6.5)
    ax.set_xticks(range(len(mon))); ax.set_xticklabels(months_labels, rotation=45, fontsize=7)
    ax.set_ylabel("PSI"); ax.set_title("PSI Monthly Trend — Score Distribution Stability", fontweight="bold")
    ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.3)
    st.pyplot(fig, use_container_width=True); plt.close("all")

# ── Row 2: Score Dist + ROC Curve ─────────────────────────────
col_sd, col_roc = st.columns(2)

with col_sd:
    st.markdown("### 🎯 Score Distribution — Dev vs Monitor")
    fig, ax = plt.subplots(figsize=(7, 3.8))
    dev_s = data["dev_scores"]
    cur_s = cur["scores"]
    ax.hist(dev_s, bins=40, alpha=0.55, color="#003366", density=True, label="Development", edgecolor="white")
    ax.hist(cur_s, bins=40, alpha=0.55, color="#f97316", density=True, label=f"Monitor ({sel_month})", edgecolor="white")
    for arr, col in [(dev_s,"#003366"),(cur_s,"#f97316")]:
        if len(arr) > 1:
            kde = gaussian_kde(arr)
            xs  = np.linspace(300, 850, 400)
            ax.plot(xs, kde(xs), color=col, lw=2)
    ax.axvline(580, color="red",  ls="--", lw=1, alpha=0.7, label="Decline threshold (580)")
    ax.axvline(660, color="green",ls="--", lw=1, alpha=0.7, label="Approve threshold (660)")
    ax.set_xlabel("Credit Score"); ax.set_ylabel("Density")
    ax.set_title("Score Distribution Shift", fontweight="bold")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    st.pyplot(fig, use_container_width=True); plt.close("all")

with col_roc:
    st.markdown("### 📐 ROC Curve — Current Period")
    fig, ax = plt.subplots(figsize=(7, 3.8))
    dev_b  = data["dev_bad"]; dev_p = data["dev_pd"]
    cur_b  = cur["bads"];     cur_p = cur["pds"]
    for yb, yp, label, col in [
        (dev_b, dev_p, "Development",     "#003366"),
        (cur_b, cur_p, f"Monitor {sel_month}", "#f97316"),
    ]:
        try:
            fpr, tpr, _ = roc_curve(yb, yp)
            auc = roc_auc_score(yb, yp)
            gini = 2*auc - 1
            ax.plot(fpr, tpr, color=col, lw=2.2,
                    label=f"{label} (AUC={auc:.4f}, Gini={gini:.4f})")
            ax.fill_between(fpr, tpr, alpha=0.07, color=col)
        except Exception:
            pass
    ax.plot([0,1],[0,1],"k--",lw=1,label="Random (AUC=0.50)")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Champion Model", fontweight="bold")
    ax.legend(fontsize=7.5); ax.grid(alpha=0.3)
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    st.pyplot(fig, use_container_width=True); plt.close("all")

st.divider()

# ── Monitoring Summary Table ──────────────────────────────────
st.markdown("### 📋 Monthly Monitoring Summary")
summary_rows = []
for m in mon:
    p = m["psi"]
    rag_v = "🟢 Green" if p < psi_amber else ("🟡 Amber" if p < psi_red else "🔴 Red")
    auc_v = "🟢" if m["auc_roc"] >= 0.75 else ("🟡" if m["auc_roc"] >= 0.70 else "🔴")
    summary_rows.append({
        "Period":         m["month"].strftime("%b %Y"),
        "N":              m["n"],
        "Mean Score":     f"{m['mean_score']:.0f}",
        "Default Rate %": f"{m['default_rate']*100:.2f}%",
        "Expected PD %":  f"{m['expected_pd']*100:.2f}%",
        "AUC-ROC":        f"{auc_v} {m['auc_roc']:.4f}",
        "Gini":           f"{m['gini']:.4f}",
        "PSI":            f"{m['psi']:.4f}",
        "RAG":            rag_v,
    })
st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True,
             column_config={"Period": st.column_config.TextColumn(width="small"),
                            "RAG":    st.column_config.TextColumn(width="small")})

st.markdown("""
<div style="background:#f8f9fa;border-radius:8px;padding:10px 14px;
            font-size:0.74rem;color:#6b7280;margin-top:8px;">
📋 <strong>OSFI E-23:</strong> PSI ≥ 0.10 triggers amber review. PSI ≥ 0.25 requires model rebuild escalation to CRO.
AUC below 0.70 requires immediate model redevelopment. All monitoring must be logged in the Model Risk Register.
</div>""", unsafe_allow_html=True)
