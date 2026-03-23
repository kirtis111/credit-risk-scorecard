"""
model_validation.py
───────────────────
Model validation metrics aligned with OSFI E-23 and Canadian bank MRM standards.

Implements:
  - Discrimination: AUC-ROC, Gini, KS Statistic
  - Calibration: Hosmer-Lemeshow, Brier Score, Expected vs Actual
  - Stability: PSI (Population Stability Index), CSI (Characteristic Stability Index)
  - Backtesting: Default rate monitoring across vintage cohorts
  
This module mirrors the validation frameworks used by model risk teams at
RBC, TD, BMO, and OSFI-regulated credit unions.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    brier_score_loss, confusion_matrix, classification_report
)
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import logging
import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 1. Discrimination Metrics
# ─────────────────────────────────────────────
def compute_gini(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Gini = 2 × AUC - 1. Primary metric at Canadian banks."""
    auc = roc_auc_score(y_true, y_prob)
    return 2 * auc - 1


def compute_ks_statistic(y_true: np.ndarray, y_prob: np.ndarray,
                          n_bins: int = 10) -> dict:
    """
    Kolmogorov-Smirnov statistic — measures max separation between
    cumulative good and bad distributions.
    
    Canadian bank threshold: KS > 0.40 for production models.
    """
    df = pd.DataFrame({"y": y_true, "prob": y_prob})
    df = df.sort_values("prob", ascending=False)

    total_bad = df["y"].sum()
    total_good = (df["y"] == 0).sum()

    df["cum_bad"] = df["y"].cumsum() / total_bad
    df["cum_good"] = (df["y"] == 0).cumsum() / total_good
    df["ks_diff"] = (df["cum_bad"] - df["cum_good"]).abs()

    ks_stat = df["ks_diff"].max()
    ks_row = df.loc[df["ks_diff"].idxmax()]

    return {
        "ks_statistic": round(ks_stat, 4),
        "ks_threshold": round(ks_row["prob"], 4),
        "cumulative_df": df,
    }


def compute_discrimination_metrics(y_true: np.ndarray, y_prob: np.ndarray,
                                    y_pred: np.ndarray = None,
                                    threshold: float = 0.5) -> dict:
    """Full discrimination metrics suite — OSFI E-23 required."""
    if y_pred is None:
        y_pred = (y_prob >= threshold).astype(int)

    auc = roc_auc_score(y_true, y_prob)
    gini = compute_gini(y_true, y_prob)
    ks_result = compute_ks_statistic(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred)

    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn + 1e-10)   # True Positive Rate
    specificity = tn / (tn + fp + 1e-10)   # True Negative Rate
    precision = tp / (tp + fp + 1e-10)
    f1 = 2 * precision * sensitivity / (precision + sensitivity + 1e-10)

    metrics = {
        "AUC_ROC": round(auc, 4),
        "Gini": round(gini, 4),
        "KS_Statistic": ks_result["ks_statistic"],
        "KS_Threshold_PD": ks_result["ks_threshold"],
        "Brier_Score": round(brier, 4),
        "Sensitivity": round(sensitivity, 4),
        "Specificity": round(specificity, 4),
        "Precision": round(precision, 4),
        "F1_Score": round(f1, 4),
        "Confusion_Matrix": cm,
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
    }

    # Traffic light assessment (Canadian bank standard)
    metrics["AUC_RAG"] = "Green" if auc >= 0.75 else ("Amber" if auc >= 0.65 else "Red")
    metrics["Gini_RAG"] = "Green" if gini >= 0.50 else ("Amber" if gini >= 0.30 else "Red")
    metrics["KS_RAG"] = "Green" if ks_result["ks_statistic"] >= 0.40 else \
                        ("Amber" if ks_result["ks_statistic"] >= 0.25 else "Red")

    logger.info(f"Discrimination — AUC: {auc:.4f} [{metrics['AUC_RAG']}] | "
                f"Gini: {gini:.4f} [{metrics['Gini_RAG']}] | "
                f"KS: {ks_result['ks_statistic']:.4f} [{metrics['KS_RAG']}]")

    return metrics


# ─────────────────────────────────────────────
# 2. Calibration Tests
# ─────────────────────────────────────────────
def hosmer_lemeshow_test(y_true: np.ndarray, y_prob: np.ndarray,
                          n_groups: int = 10) -> dict:
    """
    Hosmer-Lemeshow test for calibration.
    H0: Model is well-calibrated.
    
    Required for PD calibration validation per OSFI/Basel IRB guidance.
    """
    df = pd.DataFrame({"y": y_true, "prob": y_prob})
    df["decile"] = pd.qcut(df["prob"], q=n_groups, duplicates="drop", labels=False)

    grouped = df.groupby("decile").agg(
        N=("y", "count"),
        Observed_Bad=("y", "sum"),
        Expected_Bad=("prob", "sum"),
    ).reset_index()

    grouped["Observed_Good"] = grouped["N"] - grouped["Observed_Bad"]
    grouped["Expected_Good"] = grouped["N"] - grouped["Expected_Bad"]

    # HL chi-square statistic
    hl_stat = (
        ((grouped["Observed_Bad"] - grouped["Expected_Bad"]) ** 2 /
         (grouped["Expected_Bad"] + 1e-10)) +
        ((grouped["Observed_Good"] - grouped["Expected_Good"]) ** 2 /
         (grouped["Expected_Good"] + 1e-10))
    ).sum()

    dof = n_groups - 2
    p_value = 1 - stats.chi2.cdf(hl_stat, dof)

    return {
        "hl_statistic": round(hl_stat, 4),
        "p_value": round(p_value, 4),
        "degrees_of_freedom": dof,
        "calibrated": p_value >= 0.05,  # Cannot reject H0 at 5% → well-calibrated
        "decile_table": grouped,
        "interpretation": "Well-calibrated" if p_value >= 0.05 else "Miscalibrated — review PD scaling",
    }


def expected_vs_actual_rates(y_true: np.ndarray, y_prob: np.ndarray,
                              n_bins: int = 10) -> pd.DataFrame:
    """Expected vs Actual default rates by PD decile."""
    df = pd.DataFrame({"y": y_true, "prob": y_prob})
    df["decile"] = pd.qcut(df["prob"], q=n_bins, duplicates="drop", labels=False)

    result = df.groupby("decile").agg(
        N=("y", "count"),
        Actual_DR=("y", "mean"),
        Expected_PD=("prob", "mean"),
    ).reset_index()

    result["Actual_DR_pct"] = (result["Actual_DR"] * 100).round(2)
    result["Expected_PD_pct"] = (result["Expected_PD"] * 100).round(2)
    result["Ratio"] = (result["Actual_DR"] / (result["Expected_PD"] + 1e-10)).round(3)

    return result


# ─────────────────────────────────────────────
# 3. Population Stability Index (PSI)
# ─────────────────────────────────────────────
def compute_psi(expected: np.ndarray, actual: np.ndarray,
                n_bins: int = 10) -> dict:
    """
    Population Stability Index — monitors score/feature distribution shift.
    
    PSI Interpretation (Canadian Bank Standard, OSFI E-23):
      < 0.10: No significant change (Green)
      0.10–0.25: Minor shift — monitor (Amber)
      > 0.25: Major shift — model rebuild required (Red)
    """
    # Bin using expected distribution
    _, bin_edges = np.histogram(expected, bins=n_bins)
    bin_edges[0] -= 1e-10
    bin_edges[-1] += 1e-10

    expected_counts, _ = np.histogram(expected, bins=bin_edges)
    actual_counts, _ = np.histogram(actual, bins=bin_edges)

    expected_pct = expected_counts / (len(expected) + 1e-10)
    actual_pct = actual_counts / (len(actual) + 1e-10)

    # Avoid zeros
    expected_pct = np.where(expected_pct == 0, 1e-6, expected_pct)
    actual_pct = np.where(actual_pct == 0, 1e-6, actual_pct)

    psi_values = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
    psi_total = psi_values.sum()

    psi_table = pd.DataFrame({
        "Bin_Low": bin_edges[:-1].round(4),
        "Bin_High": bin_edges[1:].round(4),
        "Expected_N": expected_counts,
        "Actual_N": actual_counts,
        "Expected_Pct": (expected_pct * 100).round(2),
        "Actual_Pct": (actual_pct * 100).round(2),
        "PSI_Component": psi_values.round(6),
    })

    rag = "Green" if psi_total < 0.10 else ("Amber" if psi_total < 0.25 else "Red")
    action = {
        "Green": "No action required",
        "Amber": "Increase monitoring frequency",
        "Red": "Escalate to Model Risk — rebuild required",
    }[rag]

    logger.info(f"PSI = {psi_total:.4f} [{rag}] — {action}")

    return {
        "psi": round(psi_total, 4),
        "rag_status": rag,
        "recommended_action": action,
        "psi_table": psi_table,
    }


# ─────────────────────────────────────────────
# 4. Characteristic Stability Index (CSI)
# ─────────────────────────────────────────────
def compute_csi_all_features(X_dev: pd.DataFrame, X_monitor: pd.DataFrame,
                              n_bins: int = 10) -> pd.DataFrame:
    """CSI for all features between development and monitoring periods."""
    results = []
    for col in X_dev.columns:
        psi_result = compute_psi(X_dev[col].values, X_monitor[col].values, n_bins)
        results.append({
            "Feature": col,
            "CSI": psi_result["psi"],
            "RAG": psi_result["rag_status"],
            "Action": psi_result["recommended_action"],
        })
    return pd.DataFrame(results).sort_values("CSI", ascending=False)


# ─────────────────────────────────────────────
# 5. Lift and Gains Analysis
# ─────────────────────────────────────────────
def compute_lift_gains(y_true: np.ndarray, y_prob: np.ndarray,
                        n_bins: int = 10) -> pd.DataFrame:
    """
    Gains and Lift table — standard in Canadian BFSI model validation.
    Assesses model's ability to rank-order risk.
    """
    df = pd.DataFrame({"y": y_true, "prob": y_prob})
    df = df.sort_values("prob", ascending=False).reset_index(drop=True)
    df["decile"] = pd.qcut(df.index, q=n_bins, labels=False) + 1

    total_bad = df["y"].sum()
    total_n = len(df)
    base_rate = total_bad / total_n

    result = df.groupby("decile").agg(
        N=("y", "count"),
        Bads=("y", "sum"),
        Avg_Score=("prob", "mean"),
    ).reset_index()

    result["Bad_Rate"] = result["Bads"] / result["N"]
    result["Cum_Bads"] = result["Bads"].cumsum()
    result["Cum_N"] = result["N"].cumsum()
    result["Cum_Bad_Rate"] = result["Cum_Bads"] / total_bad
    result["Cum_Pop_Rate"] = result["Cum_N"] / total_n
    result["Lift"] = result["Bad_Rate"] / base_rate
    result["Cum_Lift"] = (result["Cum_Bads"] / result["Cum_N"]) / base_rate

    return result


# ─────────────────────────────────────────────
# 6. Bootstrap Confidence Intervals
# ─────────────────────────────────────────────
def bootstrap_metrics(y_true: np.ndarray, y_prob: np.ndarray,
                       n_bootstrap: int = 1000, ci: float = 0.95) -> dict:
    """
    Bootstrap confidence intervals for AUC, Gini, KS.
    Required for model uncertainty quantification (OSFI E-23).
    """
    np.random.seed(42)
    metrics = {"AUC": [], "Gini": [], "KS": []}
    n = len(y_true)

    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        y_t = y_true[idx]
        y_p = y_prob[idx]

        if y_t.sum() == 0 or y_t.sum() == n:
            continue  # Skip degenerate samples

        auc = roc_auc_score(y_t, y_p)
        gini = 2 * auc - 1
        ks = compute_ks_statistic(y_t, y_p)["ks_statistic"]

        metrics["AUC"].append(auc)
        metrics["Gini"].append(gini)
        metrics["KS"].append(ks)

    alpha = (1 - ci) / 2
    ci_result = {}
    for metric, values in metrics.items():
        ci_result[metric] = {
            "mean": round(np.mean(values), 4),
            "lower": round(np.percentile(values, alpha * 100), 4),
            "upper": round(np.percentile(values, (1 - alpha) * 100), 4),
        }

    logger.info(f"Bootstrap CI ({ci:.0%}) — AUC: [{ci_result['AUC']['lower']:.4f}, "
                f"{ci_result['AUC']['upper']:.4f}]")
    return ci_result


# ─────────────────────────────────────────────
# 7. Validation Plots
# ─────────────────────────────────────────────
def plot_roc_curve(y_true_dict: dict, y_prob_dict: dict,
                   title: str = "ROC Curve — Model Comparison") -> plt.Figure:
    """ROC curve for champion vs challenger comparison."""
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = {"Champion (LR)": "#2980b9", "Challenger (XGBoost)": "#e67e22",
              "Train": "#27ae60", "Test": "#e74c3c", "OOT": "#9b59b6"}

    for model_name, y_prob in y_prob_dict.items():
        y_true = y_true_dict[model_name]
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        gini = 2 * auc - 1
        color = colors.get(model_name, "gray")
        ax.plot(fpr, tpr, lw=2, color=color,
                label=f"{model_name} (AUC={auc:.4f}, Gini={gini:.4f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC=0.50)")
    ax.fill_between([0, 1], [0, 1], alpha=0.05, color="gray")
    ax.set_xlabel("False Positive Rate (1 — Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    plt.tight_layout()
    return fig


def plot_ks_curve(y_true: np.ndarray, y_prob: np.ndarray,
                  model_name: str = "Model") -> plt.Figure:
    """KS separation chart — standard Canadian bank validation output."""
    ks_result = compute_ks_statistic(y_true, y_prob)
    df = ks_result["cumulative_df"].reset_index(drop=True)
    df["pct_ranked"] = (df.index + 1) / len(df) * 100

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df["pct_ranked"], df["cum_bad"] * 100, color="#e74c3c", lw=2, label="Cumulative % Bads")
    ax.plot(df["pct_ranked"], df["cum_good"] * 100, color="#27ae60", lw=2, label="Cumulative % Goods")
    ax.plot(df["pct_ranked"], df["pct_ranked"], "k--", lw=1, alpha=0.5, label="Random")

    # KS line
    ks_idx = df["ks_diff"].idxmax()
    x_ks = df.loc[ks_idx, "pct_ranked"]
    y1_ks = df.loc[ks_idx, "cum_bad"] * 100
    y2_ks = df.loc[ks_idx, "cum_good"] * 100
    ax.annotate("", xy=(x_ks, y1_ks), xytext=(x_ks, y2_ks),
                arrowprops=dict(arrowstyle="<->", color="navy", lw=2))
    ax.text(x_ks + 2, (y1_ks + y2_ks) / 2,
            f"KS = {ks_result['ks_statistic']:.4f}", fontsize=10, color="navy", fontweight="bold")

    ax.set_xlabel("% Population (Ranked by Score, Highest Risk First)")
    ax.set_ylabel("Cumulative %")
    ax.set_title(f"KS Separation Chart — {model_name}\n"
                 f"KS Statistic = {ks_result['ks_statistic']:.4f}")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def plot_calibration_curve(y_true: np.ndarray, y_prob: np.ndarray,
                            n_bins: int = 10) -> plt.Figure:
    """Calibration curve — required for IFRS 9 PD validation."""
    result = expected_vs_actual_rates(y_true, y_prob, n_bins)
    hl = hosmer_lemeshow_test(y_true, y_prob, n_bins)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(result["Expected_PD_pct"], result["Actual_DR_pct"],
               s=result["N"] / 5, color="#2980b9", alpha=0.7, edgecolors="navy")

    max_val = max(result["Expected_PD_pct"].max(), result["Actual_DR_pct"].max()) * 1.1
    ax.plot([0, max_val], [0, max_val], "r--", lw=1.5, label="Perfect Calibration")

    ax.set_xlabel("Expected PD (Model) %")
    ax.set_ylabel("Actual Default Rate %")
    ax.set_title(f"Calibration Plot — Expected vs Actual Default Rate\n"
                 f"Hosmer-Lemeshow: χ²={hl['hl_statistic']:.2f}, p={hl['p_value']:.4f} "
                 f"({'✓ Calibrated' if hl['calibrated'] else '✗ Miscalibrated'})")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def plot_psi_over_time(psi_series: pd.Series,
                       title: str = "PSI Monitoring Over Time") -> plt.Figure:
    """PSI time series — monthly model monitoring dashboard."""
    fig, ax = plt.subplots(figsize=(12, 5))

    colors = ["#27ae60" if v < 0.10 else "#f39c12" if v < 0.25 else "#e74c3c"
              for v in psi_series.values]
    bars = ax.bar(psi_series.index.astype(str), psi_series.values, color=colors, edgecolor="white")

    ax.axhline(0.10, color="#f39c12", linestyle="--", lw=1.5, label="Amber threshold (0.10)")
    ax.axhline(0.25, color="#e74c3c", linestyle="--", lw=1.5, label="Red threshold (0.25)")

    for bar, val in zip(bars, psi_series.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", fontsize=8)

    ax.set_xlabel("Monitoring Period")
    ax.set_ylabel("PSI")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# 8. Full Validation Suite
# ─────────────────────────────────────────────
def run_full_validation(y_true: np.ndarray, y_prob: np.ndarray,
                         scores: np.ndarray = None,
                         X_dev: pd.DataFrame = None,
                         X_monitor: pd.DataFrame = None,
                         model_name: str = "Champion") -> dict:
    """Run complete OSFI E-23 aligned validation suite."""
    logger.info(f"Running full validation for {model_name}...")

    results = {
        "model_name": model_name,
        "discrimination": compute_discrimination_metrics(y_true, y_prob),
        "calibration": hosmer_lemeshow_test(y_true, y_prob),
        "bootstrap_ci": bootstrap_metrics(y_true, y_prob),
        "lift_gains": compute_lift_gains(y_true, y_prob),
    }

    if scores is not None:
        results["psi_scores"] = compute_psi(scores, scores)  # Placeholder for OOT

    if X_dev is not None and X_monitor is not None:
        results["csi"] = compute_csi_all_features(X_dev, X_monitor)

    # Overall model status
    disc = results["discrimination"]
    all_green = all([
        disc["AUC_RAG"] == "Green",
        disc["Gini_RAG"] == "Green",
        disc["KS_RAG"] == "Green",
    ])
    any_red = any([
        disc["AUC_RAG"] == "Red",
        disc["Gini_RAG"] == "Red",
        disc["KS_RAG"] == "Red",
    ])

    results["overall_rag"] = "Green" if all_green else ("Red" if any_red else "Amber")
    logger.info(f"Validation complete. Overall RAG: {results['overall_rag']}")

    return results


if __name__ == "__main__":
    # Smoke test
    np.random.seed(42)
    y_true = np.random.binomial(1, 0.20, 1000)
    y_prob = np.clip(y_true * 0.6 + np.random.beta(2, 8, 1000), 0, 1)

    results = run_full_validation(y_true, y_prob, model_name="Test Model")
    print(f"AUC: {results['discrimination']['AUC_ROC']}")
    print(f"Gini: {results['discrimination']['Gini']}")
    print(f"KS: {results['discrimination']['KS_Statistic']}")
    print(f"Overall RAG: {results['overall_rag']}")
