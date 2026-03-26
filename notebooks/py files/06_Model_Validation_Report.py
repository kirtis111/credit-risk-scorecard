# ============================================================
# Notebook 06: Full Model Validation & Report Generation
# BFSI Credit Risk Scorecard — Canadian Banking Edition
# OSFI E-23 Model Validation Standards
# ============================================================

# %% [markdown]
# ## Overview
# This notebook runs the complete OSFI E-23 model validation suite and
# generates the regulatory PDF model validation report.
#
# **Validation components:**
# - Discrimination: AUC-ROC, Gini, KS — with bootstrap CIs
# - Calibration: Hosmer-Lemeshow, Expected vs Actual
# - Stability: PSI, CSI (simulated OOT)
# - Backtesting: Vintage default rate analysis
# - Champion vs Challenger comparison
# - PDF report generation via ReportLab

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import sys
sys.path.insert(0, "../src")

from model_validation import (run_full_validation, plot_roc_curve, plot_ks_curve,
                                plot_calibration_curve, compute_psi, bootstrap_metrics,
                                compute_lift_gains)
from report_generator import generate_validation_report
from ifrs9_calculations import compute_provision_summary, compute_expected_loss_summary
import yaml, os, warnings
warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-whitegrid")

os.makedirs("../reports/output", exist_ok=True)

# %%
with open("../config/config.yaml") as f:
    config = yaml.safe_load(f)

X_train_woe = pd.read_csv("../data/processed/X_train_woe.csv")
X_test_woe  = pd.read_csv("../data/processed/X_test_woe.csv")
y_train = pd.read_csv("../data/processed/y_train.csv").squeeze()
y_test  = pd.read_csv("../data/processed/y_test.csv").squeeze()
scorecard   = joblib.load("../models/scorecard_model.pkl")

# Champion PD on test set
lr_pd_test  = scorecard.predict_proba(X_test_woe)
lr_scores   = scorecard.predict_score(X_test_woe)

# Challenger PD
try:
    xgb_model  = joblib.load("../models/xgb_challenger.pkl")
    xgb_pd_test = xgb_model.predict_proba(X_test_woe)[:, 1]
    has_challenger = True
except FileNotFoundError:
    print("Challenger not found — running validation on champion only")
    has_challenger = False

print(f"Test set: {len(y_test):,} accounts | Default rate: {y_test.mean():.2%}")

# %% [markdown]
# ## 1. Full Validation Suite — Champion

# %%
print("Running full validation suite (champion model)...")
validation_results = run_full_validation(
    y_true=y_test.values,
    y_prob=lr_pd_test,
    scores=lr_scores,
    model_name="Champion (LR Scorecard)"
)

disc = validation_results["discrimination"]
ci   = validation_results["bootstrap_ci"]
hl   = validation_results["calibration"]
print(f"\n{'='*50}")
print(f"  CHAMPION MODEL VALIDATION RESULTS")
print(f"{'='*50}")
print(f"  AUC-ROC:  {disc['AUC_ROC']:.4f}  [{disc['AUC_RAG']}]  95%CI: [{ci['AUC']['lower']:.4f},{ci['AUC']['upper']:.4f}]")
print(f"  Gini:     {disc['Gini']:.4f}  [{disc['Gini_RAG']}]  95%CI: [{ci['Gini']['lower']:.4f},{ci['Gini']['upper']:.4f}]")
print(f"  KS Stat:  {disc['KS_Statistic']:.4f}  [{disc['KS_RAG']}]  95%CI: [{ci['KS']['lower']:.4f},{ci['KS']['upper']:.4f}]")
print(f"  Brier:    {disc['Brier_Score']:.4f}")
print(f"  HL Test:  χ²={hl['hl_statistic']:.2f}, p={hl['p_value']:.4f}  [{hl['interpretation']}]")
print(f"  Sensitivity: {disc['Sensitivity']:.4f}  |  Specificity: {disc['Specificity']:.4f}")
print(f"{'='*50}")
print(f"  OVERALL RAG: {validation_results['overall_rag']}")
print(f"{'='*50}")

# %% [markdown]
# ## 2. Validation Charts

# %%
# ROC Curve
y_dict  = {"Champion (LR)": y_test.values}
pd_dict = {"Champion (LR)": lr_pd_test}
if has_challenger:
    y_dict["Challenger (XGBoost)"]  = y_test.values
    pd_dict["Challenger (XGBoost)"] = xgb_pd_test

fig_roc = plot_roc_curve(y_dict, pd_dict, "ROC Curve — Champion vs Challenger")
fig_roc.savefig("../reports/output/06_roc_final.png", dpi=150, bbox_inches="tight")
plt.show()

# KS Chart
fig_ks = plot_ks_curve(y_test.values, lr_pd_test, "Champion LR Scorecard")
fig_ks.savefig("../reports/output/06_ks_final.png", dpi=150, bbox_inches="tight")
plt.show()

# Calibration
fig_cal = plot_calibration_curve(y_test.values, lr_pd_test)
fig_cal.savefig("../reports/output/06_calibration_final.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 3. PSI Analysis (Simulated OOT)

# %%
# Simulate development vs OOT populations
np.random.seed(42)
dev_scores = lr_scores.astype(float)
oot_scores = np.clip(dev_scores + np.random.normal(-10, 15, len(dev_scores)), 300, 850)

psi_result = compute_psi(dev_scores, oot_scores)
print(f"\nPSI Analysis:")
print(f"  PSI = {psi_result['psi']:.4f}  [{psi_result['rag_status']}]")
print(f"  Action: {psi_result['recommended_action']}")
print(f"\n  PSI Table (bins):")
print(psi_result["psi_table"].to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 4))
tbl = psi_result["psi_table"]
ax.bar(range(len(tbl)), tbl["PSI_Component"], color="#2563eb", edgecolor="white", width=0.7)
ax.set_xticks(range(len(tbl)))
ax.set_xticklabels([f"{r['Bin_Low']:.0f}–{r['Bin_High']:.0f}" for _, r in tbl.iterrows()],
                    rotation=45, fontsize=8)
ax.set_ylabel("PSI Component")
ax.set_title(f"PSI Score Distribution — Dev vs OOT\nTotal PSI = {psi_result['psi']:.4f} [{psi_result['rag_status']}]",
             fontweight="bold")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("../reports/output/06_psi_analysis.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 4. Gains and Lift Table

# %%
lift_table = compute_lift_gains(y_test.values, lr_pd_test)
print("\nGains & Lift Table:")
print(lift_table[["decile","N","Bads","Bad_Rate","Cum_Bad_Rate","Lift","Cum_Lift"]].to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(lift_table["Cum_Pop_Rate"]*100, lift_table["Cum_Bad_Rate"]*100,
             "b-o", lw=2, ms=5, label="Model Gains")
axes[0].plot([0,100], [0,100], "k--", lw=1, label="Random")
axes[0].set_xlabel("% Population"); axes[0].set_ylabel("% Bads Captured")
axes[0].set_title("Gains Chart", fontweight="bold")
axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].bar(lift_table["decile"], lift_table["Lift"], color="#003366", edgecolor="white", width=0.7)
axes[1].axhline(1.0, color="red", ls="--", lw=1.5, label="Lift = 1.0 (no model)")
axes[1].set_xlabel("Decile (1=highest risk)"); axes[1].set_ylabel("Lift")
axes[1].set_title("Lift Chart by Decile", fontweight="bold")
axes[1].legend(); axes[1].grid(axis="y", alpha=0.3)

plt.suptitle("Gains & Lift Analysis — Champion Scorecard", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("../reports/output/06_gains_lift.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 5. Generate Regulatory PDF Report

# %%
try:
    provision_summary = pd.read_csv("../data/processed/ifrs9_provision_summary.csv")
    portfolio_ecl     = pd.read_csv("../data/processed/portfolio_ecl.csv")
    el_summary        = {
        "Total_EAD_CAD": portfolio_ecl["EAD"].sum(),
        "Total_ECL_CAD": portfolio_ecl["ECL"].sum(),
        "Portfolio_EL_Rate_%": portfolio_ecl["ECL"].sum() / portfolio_ecl["EAD"].sum() * 100,
        "Avg_PD": portfolio_ecl["PD_for_ECL"].mean(),
        "Avg_LGD": portfolio_ecl["LGD_Downturn"].mean() if "LGD_Downturn" in portfolio_ecl.columns else 0.35,
        "Stage_1_EAD_pct": (portfolio_ecl[portfolio_ecl["Stage"]==1]["EAD"].sum() / portfolio_ecl["EAD"].sum() * 100),
        "Stage_2_EAD_pct": (portfolio_ecl[portfolio_ecl["Stage"]==2]["EAD"].sum() / portfolio_ecl["EAD"].sum() * 100),
        "Stage_3_EAD_pct": (portfolio_ecl[portfolio_ecl["Stage"]==3]["EAD"].sum() / portfolio_ecl["EAD"].sum() * 100),
    }
except FileNotFoundError:
    print("IFRS 9 data not found — using synthetic values for report demo")
    provision_summary = pd.DataFrame({"Stage_Label":["Stage 1","Stage 2","Stage 3"],
                                       "N_Exposures":[3800,900,260],"Total_EAD":[0,0,0],
                                       "Total_ECL":[0,0,0],"Avg_PD":[0.02,0.09,0.35],
                                       "Avg_LGD":[0.32,0.38,0.45],"Coverage_Ratio_%":[1.2,8.4,28.1]})
    el_summary = {"Total_EAD_CAD":24_500_000,"Total_ECL_CAD":1_180_000,
                   "Portfolio_EL_Rate_%":4.82,"Avg_PD":0.082,"Avg_LGD":0.34,
                   "Stage_1_EAD_pct":75.2,"Stage_2_EAD_pct":18.4,"Stage_3_EAD_pct":6.4}

# %% [markdown]
# ## 6. Generate All Output Files — Single Entry Point
#
# Both the PDF report and the Excel workbook are generated HERE and ONLY HERE.
# Neither report_generator.py nor generate_scorecard_excel.py run on import —
# they are pure library modules. This cell is the single authoritative trigger.

# %%
import sys, os
sys.path.insert(0, "../excel")
from generate_scorecard_excel import build_excel_workbook

# ── 6a. PDF Validation Report ─────────────────────────────────
pdf_path = generate_validation_report(
    output_path="../reports/output/Model_Validation_Report_IFRS9.pdf",
    config=config,
    validation_results=validation_results,
    provision_summary=provision_summary,
    el_summary=el_summary,
    roc_fig=fig_roc,
    ks_fig=fig_ks,
    cal_fig=fig_cal,
)
print(f"✅ PDF generated (once):   {pdf_path}")

# ── 6b. Excel Scorecard Workbook ──────────────────────────────
try:
    scorecard_pts = pd.read_csv("../data/processed/scorecard_points.csv")
    iv_tbl        = pd.read_csv("../data/processed/iv_table.csv")
except FileNotFoundError:
    scorecard_pts = None   # build_excel_workbook uses synthetic data as fallback
    iv_tbl        = None

xlsx_path = build_excel_workbook(
    output_path="../reports/output/Scorecard_v1.0.xlsx",
    scorecard_df=scorecard_pts,
    iv_table=iv_tbl,
)
print(f"✅ Excel generated (once): {xlsx_path}")

# ── Final summary ─────────────────────────────────────────────
print("\n" + "="*55)
print("  NOTEBOOK 06 — COMPLETE. OUTPUT FILES:")
print("="*55)
print(f"  PDF  → reports/output/Model_Validation_Report_IFRS9.pdf")
print(f"  XLSX → reports/output/Scorecard_v1.0.xlsx")
print("="*55)
print(f"  Champion AUC:  {disc['AUC_ROC']:.4f}  [{disc['AUC_RAG']}]")
print(f"  Champion Gini: {disc['Gini']:.4f}  [{disc['Gini_RAG']}]")
print(f"  Champion KS:   {disc['KS_Statistic']:.4f}  [{disc['KS_RAG']}]")
print(f"  Overall RAG:   {validation_results['overall_rag']}")
print(f"  PSI Status:    {psi_result['rag_status']}")
print("="*55)
