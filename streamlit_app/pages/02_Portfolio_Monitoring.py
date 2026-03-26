"""
Portfolio Monitoring — Model Performance Tracking
Monthly PSI, AUC decay, score distribution. OSFI E-23 aligned.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

st.set_page_config(page_title="Portfolio Monitoring", page_icon="📈", layout="wide")

st.title("Portfolio Monitoring")
st.caption("Monthly model performance | OSFI E-23 | PSI thresholds: Amber > 0.10 | Red > 0.25")


# ── Generate 12 months of synthetic monitoring data ────────────────────────
@st.cache_data
def build_monitoring_data():
    np.random.seed(42)
    months = pd.date_range("2024-01-01", periods=12, freq="MS")
    rows   = []
    for i, mo in enumerate(months):
        drift      = i * 1.6
        n          = 420 + i * 8
        scores_dev = np.random.normal(638, 67, 4000).clip(300, 850)
        scores_mon = np.random.normal(638 - drift, 67 + drift * 0.25, n).clip(300, 850)
        pds        = 1 / (1 + np.exp((scores_mon - 600) / 28.85))
        bads       = (np.random.rand(n) < pds).astype(int)

        try:
            auc = roc_auc_score(bads, pds) if bads.sum() > 0 else 0.80
        except Exception:
            auc = 0.80

        psi = float(i * 0.011 + np.random.uniform(0, 0.007))

        rows.append({
            "Month":        mo,
            "Period":       mo.strftime("%b %Y"),
            "N":            n,
            "Mean_Score":   round(float(scores_mon.mean()), 1),
            "Default_Rate": round(float(bads.mean() * 100), 2),
            "Exp_PD":       round(float(pds.mean() * 100), 2),
            "AUC":          round(auc, 4),
            "Gini":         round(2 * auc - 1, 4),
            "PSI":          round(psi, 4),
            "scores_dev":   scores_dev,
            "scores_mon":   scores_mon,
            "bads":         bads,
            "pds":          pds,
        })
    return rows


data  = build_monitoring_data()
months_labels = [r["Period"] for r in data]


# ── Sidebar controls ───────────────────────────────────────────────────────
st.sidebar.header("Controls")
sel_period = st.sidebar.selectbox("Monitoring Period", months_labels, index=len(months_labels) - 1)
psi_amber  = st.sidebar.number_input("PSI Amber threshold", 0.05, 0.20, 0.10, 0.01)
psi_red    = st.sidebar.number_input("PSI Red threshold",   0.10, 0.40, 0.25, 0.01)

sel_idx = months_labels.index(sel_period)
cur     = data[sel_idx]
dev     = data[0]  # development period


# ── RAG helper ─────────────────────────────────────────────────────────────
def rag(psi_val):
    if psi_val < psi_amber:  return "🟢 Green"
    if psi_val < psi_red:    return "🟡 Amber"
    return "🔴 Red"

def auc_rag(auc_val):
    if auc_val >= 0.75:  return "🟢"
    if auc_val >= 0.70:  return "🟡"
    return "🔴"


# ── KPIs ───────────────────────────────────────────────────────────────────
st.subheader(f"Period: {sel_period}")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("AUC-ROC",      f"{cur['AUC']:.4f}",
          delta=f"{cur['AUC'] - dev['AUC']:+.4f} vs dev")
c2.metric("Gini",         f"{cur['Gini']:.4f}",
          delta=f"{cur['Gini'] - dev['Gini']:+.4f} vs dev")
c3.metric("Mean Score",   f"{cur['Mean_Score']:.0f}",
          delta=f"{cur['Mean_Score'] - dev['Mean_Score']:+.0f} vs dev")
c4.metric("Default Rate", f"{cur['Default_Rate']:.2f}%")
c5.metric("PSI",          f"{cur['PSI']:.4f}",  help=rag(cur["PSI"]))

st.divider()

# ── Charts ─────────────────────────────────────────────────────────────────
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("AUC-ROC — Monthly Trend")
    aucs    = [r["AUC"] for r in data]
    periods = [r["Period"] for r in data]

    fig, ax = plt.subplots(figsize=(7, 3.8))
    bar_colours = ["#ef4444" if a < 0.70 else "#f97316" if a < 0.75 else "#22c55e"
                   for a in aucs]
    ax.bar(range(len(data)), aucs, color=bar_colours, edgecolor="white", width=0.7)
    ax.axhline(0.75, color="orange", ls="--", lw=1.2, label="Amber (0.75)")
    ax.axhline(0.70, color="red",    ls="--", lw=1.2, label="Red (0.70)")
    ax.axvline(sel_idx, color="navy", lw=1.8, alpha=0.6, label=f"Selected: {sel_period}")
    ax.set_xticks(range(len(data)))
    ax.set_xticklabels(periods, rotation=45, fontsize=7)
    ax.set_ylabel("AUC-ROC")
    ax.set_ylim(0.55, 0.95)
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("AUC-ROC Monthly Trend", fontsize=10)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

with col_right:
    st.subheader("PSI — Score Distribution Stability")
    psis = [r["PSI"] for r in data]

    fig, ax = plt.subplots(figsize=(7, 3.8))
    bar_colours = [
        "#ef4444" if p >= psi_red else "#f97316" if p >= psi_amber else "#22c55e"
        for p in psis
    ]
    bars = ax.bar(range(len(data)), psis, color=bar_colours, edgecolor="white", width=0.7)
    ax.axhline(psi_amber, color="orange", ls="--", lw=1.2, label=f"Amber ({psi_amber:.2f})")
    ax.axhline(psi_red,   color="red",    ls="--", lw=1.2, label=f"Red ({psi_red:.2f})")
    ax.axvline(sel_idx, color="navy", lw=1.8, alpha=0.6)
    for bar, v in zip(bars, psis):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{v:.3f}", ha="center", fontsize=6.5)
    ax.set_xticks(range(len(data)))
    ax.set_xticklabels(periods, rotation=45, fontsize=7)
    ax.set_ylabel("PSI")
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("PSI Monthly Trend", fontsize=10)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

st.divider()

col_left2, col_right2 = st.columns(2)

with col_left2:
    st.subheader("Score Distribution — Dev vs Monitor")
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.hist(cur["scores_dev"], bins=40, alpha=0.55, color="steelblue",
            density=True, label="Development", edgecolor="white")
    ax.hist(cur["scores_mon"], bins=40, alpha=0.55, color="darkorange",
            density=True, label=f"{sel_period}", edgecolor="white")
    ax.axvline(580, color="red",   ls="--", lw=1, label="Decline (580)")
    ax.axvline(660, color="green", ls="--", lw=1, label="Approve (660)")
    ax.set_xlabel("Credit Score")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_title("Score Distribution Shift", fontsize=10)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

with col_right2:
    st.subheader("ROC Curve — Current Period vs Development")
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for d, label, colour in [
        (dev, "Development",    "steelblue"),
        (cur, sel_period,       "darkorange"),
    ]:
        if d["bads"].sum() > 0:
            fpr, tpr, _ = roc_curve(d["bads"], d["pds"])
            ax.plot(fpr, tpr, color=colour, lw=1.8,
                    label=f"{label} (AUC={d['AUC']:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("ROC Curve Comparison", fontsize=10)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

st.divider()

# ── Monthly summary table ──────────────────────────────────────────────────
st.subheader("Monthly Summary Table")

table_rows = []
for r in data:
    table_rows.append({
        "Period":        r["Period"],
        "N":             r["N"],
        "Mean Score":    r["Mean_Score"],
        "Default Rate %":r["Default_Rate"],
        "Exp PD %":      r["Exp_PD"],
        "AUC-ROC":       f"{auc_rag(r['AUC'])} {r['AUC']:.4f}",
        "Gini":          f"{r['Gini']:.4f}",
        "PSI":           r["PSI"],
        "RAG":           rag(r["PSI"]),
    })

st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

st.caption(
    "OSFI E-23: PSI ≥ 0.10 → increase monitoring frequency. "
    "PSI ≥ 0.25 → model rebuild required, escalate to Model Risk / CRO."
)
