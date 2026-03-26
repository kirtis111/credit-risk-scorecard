"""
IFRS 9 ECL Provisioning
Expected Credit Loss engine. Stage 1/2/3, FLI macro overlay, scenario analysis.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="IFRS 9 Provisioning", page_icon="🏛️", layout="wide")

st.title("IFRS 9 Expected Credit Loss — Provisioning Engine")
st.caption("ECL = PD × LGD × EAD × DF  |  Stage 1: 12-month ECL  |  Stage 2/3: Lifetime ECL  |  OSFI E-6")


# ── Sidebar parameters ─────────────────────────────────────────────────────
st.sidebar.header("IFRS 9 Parameters")

st.sidebar.subheader("FLI Scenario Weights")
st.sidebar.caption("Must sum to 100%")
opt_w  = st.sidebar.slider("Optimistic weight %",  0,  60, 25) / 100
base_w = st.sidebar.slider("Base case weight %",  20,  80, 50) / 100
adv_w  = round(1.0 - opt_w - base_w, 2)
adv_w  = max(adv_w, 0.0)
st.sidebar.markdown(f"**Adverse weight: {adv_w*100:.0f}%**")
if abs(opt_w + base_w + adv_w - 1.0) > 0.02:
    st.sidebar.warning("Weights don't sum to 100%")

st.sidebar.subheader("Model Parameters")
discount_rate  = st.sidebar.slider("Discount rate (BoC + spread) %", 3.0, 12.0, 6.5, 0.25) / 100
downturn_scalar = st.sidebar.slider("Downturn LGD scalar (OSFI min 1.25)", 1.0, 2.0, 1.25, 0.05)
horizon_yrs    = st.sidebar.slider("Lifetime ECL horizon (years)", 3, 10, 5)

st.sidebar.subheader("Staging Thresholds")
stage2_pd_thresh = st.sidebar.slider("Stage 1→2 PD threshold %", 1, 20, 10) / 100
stage3_pd_thresh = st.sidebar.slider("Stage 2→3 PD threshold %", 20, 70, 40) / 100


# ── ECL engine ─────────────────────────────────────────────────────────────
def compute_ecl(pd_annual, lgd, ead, stage, disc_rate, horizon):
    if stage == 1:
        return pd_annual * lgd * ead / (1 + disc_rate)
    # Lifetime ECL — simplified annual roll-forward
    ecl, surv = 0.0, 1.0
    for t in range(1, horizon + 1):
        ecl  += surv * pd_annual * lgd * ead / ((1 + disc_rate) ** t)
        surv *= (1 - pd_annual)
    return ecl


def fli_weighted_pd(pd_base, opt_w, base_w, adv_w):
    return pd_base * 0.75 * opt_w + pd_base * base_w + pd_base * 1.65 * adv_w


@st.cache_data
def build_portfolio(seed=42):
    np.random.seed(seed)
    n = 500
    return pd.DataFrame({
        "EAD":   np.random.lognormal(np.log(30_000), 0.6, n).clip(5_000, 300_000),
        "VALUE": np.random.lognormal(np.log(30_000), 0.6, n).clip(5_000, 300_000)
                 * np.random.uniform(1.5, 4.5, n),
        "PD":    np.random.beta(2, 18, n),
    })


df = build_portfolio()

# Compute LGD + EAD
df["LGD"]        = (1 - np.minimum(df["VALUE"] * 0.75, df["EAD"]) * 0.92
                    / df["EAD"]).clip(0, 1)
df["LGD_DT"]     = (df["LGD"] * downturn_scalar).clip(0, 1)
df["PD_FLI"]     = df["PD"].apply(lambda p: fli_weighted_pd(p, opt_w, base_w, adv_w))

# Stage assignment
df["Stage"] = df["PD"].apply(
    lambda p: (3 if min(p * 2.0, 0.99) > stage3_pd_thresh
               else 2 if p > stage2_pd_thresh else 1)
)

# ECL
df["ECL"] = df.apply(
    lambda r: compute_ecl(r["PD_FLI"], r["LGD_DT"], r["EAD"],
                          int(r["Stage"]), discount_rate, horizon_yrs),
    axis=1
)
df["ECL_Rate"] = df["ECL"] / df["EAD"]

# Scenario ECLs
for label, scalar in [("ECL_Opt", 0.75), ("ECL_Base", 1.00), ("ECL_Adv", 1.65)]:
    df[label] = df.apply(
        lambda r, s=scalar: compute_ecl(
            r["PD"] * s, r["LGD_DT"], r["EAD"],
            int(r["Stage"]), discount_rate, horizon_yrs
        ), axis=1
    )

total_ead  = df["EAD"].sum()
total_ecl  = df["ECL"].sum()
ecl_pw     = (df["ECL_Opt"] * opt_w + df["ECL_Base"] * base_w + df["ECL_Adv"] * adv_w).sum()


# ── KPI row ────────────────────────────────────────────────────────────────
st.subheader("Portfolio Summary")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total EAD",       f"${total_ead/1e6:.1f}M")
c2.metric("Total ECL",       f"${total_ecl/1e6:.2f}M")
c3.metric("Coverage Ratio",  f"{total_ecl/total_ead*100:.2f}%")
c4.metric("Avg PD",          f"{df['PD'].mean()*100:.2f}%")
c5.metric("FLI-Adj ECL",     f"${ecl_pw/1e6:.2f}M",
          delta=f"{(ecl_pw - total_ecl)/total_ecl*100:+.1f}% vs base",
          help=f"Opt {opt_w*100:.0f}% | Base {base_w*100:.0f}% | Adv {adv_w*100:.0f}%")

st.divider()

# ── Stage breakdown table ──────────────────────────────────────────────────
st.subheader("IFRS 9 Stage Breakdown")

stage_summary = (
    df.groupby("Stage")
    .agg(N=("EAD", "count"), EAD=("EAD", "sum"), ECL=("ECL", "sum"),
         Avg_PD=("PD_FLI", "mean"), Avg_LGD=("LGD_DT", "mean"))
    .reset_index()
)
stage_summary["Coverage_%"] = (stage_summary["ECL"] / stage_summary["EAD"] * 100).round(2)
stage_summary["Stage_Label"] = stage_summary["Stage"].map({
    1: "Stage 1 — Performing (12-month ECL)",
    2: "Stage 2 — Underperforming (Lifetime ECL)",
    3: "Stage 3 — Non-Performing (Lifetime ECL)",
})
stage_summary["EAD"]    = stage_summary["EAD"].apply(lambda x: f"${x:,.0f}")
stage_summary["ECL"]    = stage_summary["ECL"].apply(lambda x: f"${x:,.0f}")
stage_summary["Avg_PD"] = stage_summary["Avg_PD"].apply(lambda x: f"{x*100:.2f}%")
stage_summary["Avg_LGD"]= stage_summary["Avg_LGD"].apply(lambda x: f"{x*100:.1f}%")

st.dataframe(
    stage_summary[["Stage_Label", "N", "EAD", "ECL", "Avg_PD", "Avg_LGD", "Coverage_%"]],
    use_container_width=True, hide_index=True,
    column_config={"Stage_Label": st.column_config.TextColumn("IFRS 9 Stage", width="large")}
)

st.divider()

# ── Charts ─────────────────────────────────────────────────────────────────
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("EAD by Stage")
    raw = df.groupby("Stage")["EAD"].sum()
    stage_colors = {1: "#22c55e", 2: "#f97316", 3: "#ef4444"}
    colours = [stage_colors[s] for s in raw.index]
    labels  = [f"Stage {s}" for s in raw.index]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, raw.values / 1e6, color=colours, edgecolor="white", width=0.5)
    for bar, v in zip(bars, raw.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"${v/1e6:.1f}M", ha="center", fontsize=9, fontweight="bold")
    ax.set_ylabel("EAD ($M)")
    ax.set_title("EAD by IFRS 9 Stage", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

with col_right:
    st.subheader("FLI Scenario ECL Comparison")
    scenario_ecls = [
        df["ECL_Opt"].sum(),
        df["ECL_Base"].sum(),
        ecl_pw,
        df["ECL_Adv"].sum(),
    ]
    scenario_labels = [
        f"Optimistic\n(PD ×0.75)",
        f"Base Case\n(PD ×1.00)",
        f"Prob-Weighted\n({opt_w*100:.0f}/{base_w*100:.0f}/{adv_w*100:.0f})",
        f"Adverse\n(PD ×1.65)",
    ]
    scenario_colours = ["#22c55e", "#3b82f6", "#7c3aed", "#ef4444"]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(scenario_labels, [v / 1e6 for v in scenario_ecls],
                  color=scenario_colours, edgecolor="white", width=0.55)
    for bar, v in zip(bars, scenario_ecls):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"${v/1e6:.2f}M", ha="center", fontsize=9, fontweight="bold")
    ax.set_ylabel("ECL ($M)")
    ax.set_title("FLI Scenario ECL — Bank of Canada Macro Scenarios", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

st.divider()

# ── Stage migration heatmap ────────────────────────────────────────────────
st.subheader("Stage Migration Matrix (Quarter-on-Quarter — Illustrative)")

migration = pd.DataFrame(
    [[0.88, 0.10, 0.02],
     [0.12, 0.74, 0.14],
     [0.02, 0.08, 0.90]],
    index=["From Stage 1", "From Stage 2", "From Stage 3"],
    columns=["→ Stage 1", "→ Stage 2", "→ Stage 3"],
)
st.dataframe(
    migration.applymap(lambda x: f"{x:.0%}"),
    use_container_width=False,
)

st.caption(
    "IFRS 9 / OSFI E-6: Stage 1 → 12-month ECL. Stage 2/3 → Lifetime ECL.  "
    f"Downturn LGD scalar = {downturn_scalar:.2f}× (OSFI minimum 1.25×).  "
    f"Discount rate = {discount_rate*100:.2f}%."
)
