"""
03_IFRS9_Provisioning.py
─────────────────────────
IFRS 9 Expected Credit Loss engine — interactive provisioning tool.
Stage 1/2/3 ECL, FLI macro overlay, scenario analysis.
Mirrors provisioning tools at Canadian Schedule I banks.
"""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

st.set_page_config(page_title="IFRS 9 Provisioning | CreditIQ", page_icon="🏛️", layout="wide")
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


# ── ECL Calculation Engine ─────────────────────────────────────
def compute_ecl(pd12, lgd, ead, discount_rate=0.065, horizon=5, stage=1):
    """ECL = PD × LGD × EAD × DiscountFactor"""
    if stage == 1:
        return pd12 * lgd * ead * (1/(1+discount_rate))
    else:
        pdlt = min(pd12 * 3.5, 0.99)
        ecl  = 0.0
        surv = 1.0
        for t in range(1, horizon+1):
            ecl  += surv * pd12 * lgd * ead / ((1+discount_rate)**t)
            surv *= (1 - pd12)
        return ecl

def assign_stage(pd12, dpd=0, watchlisted=False):
    pdlt = min(pd12 * 3.5, 0.99)
    if dpd >= 90 or pdlt > 0.20:   return 3
    if dpd >= 30 or pd12 > 0.03 or watchlisted: return 2
    return 1

def fli_pd(pd12, optimistic_w, base_w, adverse_w):
    pd_opt = pd12 * 0.75
    pd_adv = pd12 * 1.65
    return pd_opt*optimistic_w + pd12*base_w + pd_adv*adverse_w


# ── Portfolio generator ────────────────────────────────────────
@st.cache_data
def generate_portfolio(n=500, seed=42):
    np.random.seed(seed)
    loans    = np.random.lognormal(np.log(30000), 0.6, n).clip(5000, 300000)
    values   = loans * np.random.uniform(1.5, 4.5, n)
    pds      = np.random.beta(2, 18, n)
    delinqs  = np.random.choice([0,0,0,1,2,3], n, p=[0.70,0.10,0.08,0.06,0.04,0.02])
    return pd.DataFrame({"EAD":loans,"PROPERTY_VALUE":values,
                          "PD12":pds,"DELINQ":delinqs})


# ── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏛️ IFRS 9 Parameters")
    st.divider()
    st.markdown("**Scenario Weights**")
    st.caption("Must sum to 100%")
    opt_w   = st.slider("Optimistic Weight %",  0,  60, 25) / 100
    base_w  = st.slider("Base Case Weight %",   20, 80, 50) / 100
    adv_w   = round(1 - opt_w - base_w, 2)
    adv_w   = max(adv_w, 0.0)
    st.markdown(f"**Adverse Weight: {adv_w*100:.0f}%**")
    if abs(opt_w + base_w + adv_w - 1.0) > 0.01:
        st.warning("⚠️ Weights don't sum to 100%")
    st.divider()
    st.markdown("**Model Parameters**")
    discount_rate  = st.slider("Discount Rate (BoC + spread) %", 3.0, 12.0, 6.5, 0.1) / 100
    downturn_lgd_s = st.slider("Downturn LGD Scalar (OSFI min 1.25)", 1.0, 2.0, 1.25, 0.05)
    horizon_yrs    = st.slider("Lifetime Horizon (years)", 3, 10, 5)
    st.divider()
    reporting_dt   = st.date_input("Reporting Date")
    if st.button("📥 Export Provision Register", use_container_width=True):
        st.info("In production: exports to Excel/OFSAA")

# ── Compute portfolio ECL ──────────────────────────────────────
df = generate_portfolio()
df["STAGE"]      = df.apply(lambda r: assign_stage(r["PD12"], r["DELINQ"]*30), axis=1)
df["PD12_FLI"]   = df["PD12"].apply(lambda p: fli_pd(p, opt_w, base_w, adv_w))
df["LGD"]        = (1 - np.minimum(df["PROPERTY_VALUE"]*0.75, df["EAD"])*0.92
                    / df["EAD"]).clip(0, 1)
df["LGD_DT"]     = (df["LGD"] * downturn_lgd_s).clip(0, 1)
df["ECL"]        = df.apply(lambda r: compute_ecl(r["PD12_FLI"], r["LGD_DT"],
                             r["EAD"], discount_rate, horizon_yrs, r["STAGE"]), axis=1)
df["ECL_RATE"]   = df["ECL"] / df["EAD"]

# Scenario ECLs
df["ECL_OPT"]  = df.apply(lambda r: compute_ecl(r["PD12"]*0.75,  r["LGD_DT"], r["EAD"],
                           discount_rate, horizon_yrs, r["STAGE"]), axis=1)
df["ECL_BASE"] = df.apply(lambda r: compute_ecl(r["PD12"],        r["LGD_DT"], r["EAD"],
                           discount_rate, horizon_yrs, r["STAGE"]), axis=1)
df["ECL_ADV"]  = df.apply(lambda r: compute_ecl(r["PD12"]*1.65,  r["LGD_DT"], r["EAD"],
                           discount_rate, horizon_yrs, r["STAGE"]), axis=1)

stage_summary = df.groupby("STAGE").agg(
    N=("EAD","count"), EAD=("EAD","sum"), ECL=("ECL","sum"),
    Avg_PD=("PD12_FLI","mean"), Avg_LGD=("LGD_DT","mean"),
).reset_index()
stage_summary["Coverage_%"] = (stage_summary["ECL"]/stage_summary["EAD"]*100).round(2)
stage_summary["Stage_Label"] = stage_summary["STAGE"].map({
    1:"Stage 1 — Performing (12M ECL)",
    2:"Stage 2 — Underperforming (Lifetime ECL)",
    3:"Stage 3 — Non-Performing (Lifetime ECL)"})

total_ead = df["EAD"].sum(); total_ecl = df["ECL"].sum()
total_opt = (df["ECL_OPT"]*opt_w + df["ECL_BASE"]*base_w + df["ECL_ADV"]*adv_w).sum()

# ── Header ────────────────────────────────────────────────────
st.markdown("# 🏛️ IFRS 9 ECL Provisioning Engine")
st.caption("Expected Credit Loss | 3-Stage Model | Forward-Looking Information | OSFI E-6 Compliant")
st.divider()

# ── KPI Cards ─────────────────────────────────────────────────
k1,k2,k3,k4,k5,k6 = st.columns(6)
k1.metric("Total EAD",      f"${total_ead/1e6:.1f}M")
k2.metric("Total ECL",      f"${total_ecl/1e6:.2f}M")
k3.metric("Coverage Ratio", f"{total_ecl/total_ead*100:.2f}%")
k4.metric("Portfolio Avg PD",f"{df['PD12'].mean()*100:.2f}%")
k5.metric("Avg Downturn LGD",f"{df['LGD_DT'].mean()*100:.1f}%")
k6.metric("FLI-Adj ECL",    f"${total_opt/1e6:.2f}M",
          delta=f"{(total_opt-total_ecl)/total_ecl*100:+.1f}% vs base")

st.divider()

# ── Stage Breakdown + Scenario Chart ──────────────────────────
col_s, col_c = st.columns(2)

with col_s:
    st.markdown("### 📊 IFRS 9 Stage Breakdown")
    stage_colors = {1:"#16a34a", 2:"#d97706", 3:"#dc2626"}
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.8))

    # EAD by stage (donut)
    ax = axes[0]
    ead_vals = [df[df["STAGE"]==s]["EAD"].sum() for s in [1,2,3]]
    wedge_colors = ["#16a34a","#d97706","#dc2626"]
    wedges, texts, autotexts = ax.pie(
        ead_vals, labels=["Stage 1","Stage 2","Stage 3"],
        autopct="%1.1f%%", colors=wedge_colors,
        wedgeprops=dict(width=0.55), startangle=90,
        textprops={"fontsize":8})
    for at in autotexts: at.set_fontsize(8); at.set_color("white"); at.set_fontweight("bold")
    ax.set_title("EAD by Stage", fontweight="bold", fontsize=9)

    # ECL coverage by stage (bar)
    ax2 = axes[1]
    cov  = stage_summary["Coverage_%"].values
    bars = ax2.bar(["S1","S2","S3"], cov, color=wedge_colors, edgecolor="white", width=0.5)
    for bar, v in zip(bars, cov):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
                 f"{v:.1f}%", ha="center", fontsize=8, fontweight="bold")
    ax2.set_ylabel("Coverage Ratio %")
    ax2.set_title("ECL Coverage by Stage", fontweight="bold", fontsize=9)
    ax2.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close("all")

with col_c:
    st.markdown("### 🌐 Scenario ECL Analysis")
    fig, ax = plt.subplots(figsize=(7, 3.8))
    scenario_ecls = [df["ECL_OPT"].sum(), df["ECL_BASE"].sum(),
                     total_opt, df["ECL_ADV"].sum()]
    scenario_lbls = ["Optimistic\n(PD ×0.75)", "Base Case\n(PD ×1.00)",
                     "Probability\nWeighted", "Adverse\n(PD ×1.65)"]
    bar_cols      = ["#16a34a","#2563eb","#7c3aed","#dc2626"]
    bars = ax.bar(scenario_lbls, [v/1e6 for v in scenario_ecls],
                  color=bar_cols, edgecolor="white", width=0.55)
    for bar, val in zip(bars, scenario_ecls):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                f"${val/1e6:.2f}M", ha="center", fontsize=8.5, fontweight="bold")
    ax.set_ylabel("ECL ($M)")
    ax.set_title("Probability-Weighted ECL — BoC Macro Scenarios\n"
                 f"Weights: Opt {opt_w*100:.0f}% | Base {base_w*100:.0f}% | Adv {adv_w*100:.0f}%",
                 fontweight="bold", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close("all")

# ── Stage Migration Matrix ─────────────────────────────────────
st.divider()
col_mig, col_tbl = st.columns(2)

with col_mig:
    st.markdown("### 🔄 Stage Migration Heatmap")
    st.caption("Simulated quarter-over-quarter migration")
    np.random.seed(99)
    migration = np.array([
        [0.88, 0.10, 0.02],
        [0.12, 0.74, 0.14],
        [0.02, 0.08, 0.90],
    ])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(migration, cmap="RdYlGn", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{migration[i,j]:.0%}", ha="center", va="center",
                    fontsize=13, fontweight="bold",
                    color="white" if migration[i,j] < 0.4 or migration[i,j] > 0.8 else "black")
    ax.set_xticks([0,1,2]); ax.set_yticks([0,1,2])
    ax.set_xticklabels(["→ Stage 1","→ Stage 2","→ Stage 3"], fontsize=9)
    ax.set_yticklabels(["From S1","From S2","From S3"], fontsize=9)
    ax.set_title("Stage Migration Matrix (Q-o-Q)", fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046, label="Migration %")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close("all")

with col_tbl:
    st.markdown("### 📋 Provision Register")
    disp = stage_summary.copy()
    disp["EAD"]      = disp["EAD"].apply(lambda x: f"${x:,.0f}")
    disp["ECL"]      = disp["ECL"].apply(lambda x: f"${x:,.0f}")
    disp["Avg_PD"]   = disp["Avg_PD"].apply(lambda x: f"{x*100:.2f}%")
    disp["Avg_LGD"]  = disp["Avg_LGD"].apply(lambda x: f"{x*100:.1f}%")
    disp["Coverage_%"]=disp["Coverage_%"].apply(lambda x: f"{x:.2f}%")
    st.dataframe(
        disp[["Stage_Label","N","EAD","ECL","Avg_PD","Avg_LGD","Coverage_%"]],
        use_container_width=True, hide_index=True,
        column_config={"Stage_Label": st.column_config.TextColumn("IFRS 9 Stage", width="large")})

    # Totals
    st.markdown(f"""
    <div style="background:#eff6ff;border:1px solid #3b82f6;border-radius:8px;padding:12px;margin-top:8px;">
      <strong>Portfolio Total</strong><br>
      EAD: <strong>${total_ead:,.0f}</strong> &nbsp;|&nbsp;
      ECL: <strong>${total_ecl:,.0f}</strong> &nbsp;|&nbsp;
      Coverage: <strong>{total_ecl/total_ead*100:.3f}%</strong> &nbsp;|&nbsp;
      FLI-Adj ECL: <strong>${total_opt:,.0f}</strong>
    </div>""", unsafe_allow_html=True)

st.divider()
st.markdown("""
<div style="background:#f8f9fa;border-radius:8px;padding:12px 16px;font-size:0.74rem;color:#6b7280;">
🏛️ <strong>IFRS 9 / OSFI E-6:</strong> Stage 1 → 12-month ECL.
Stage 2/3 → Lifetime ECL. Downturn LGD scalar = {:.2f}× (OSFI minimum 1.25×).
FLI macro overlay uses Bank of Canada economic scenarios.
Discount rate = BoC overnight + credit spread = {:.2f}%.
</div>""".format(downturn_lgd_s, discount_rate*100), unsafe_allow_html=True)
