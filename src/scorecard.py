"""
scorecard.py
────────────
Logistic Regression Scorecard — Points Allocation and Score Scaling.

Implements the standard Canadian bank scorecard methodology:
  Score = Offset + Factor × log(odds)
  
Scaling parameters (RBC/TD/BMO standard):
  Base Score = 600 at odds 50:1 (good:bad)
  PDO = 20 (points to double the odds)
  Score range: 300–850 (mirrors FICO-like Canadian bureau scale)

References:
  - Siddiqi (2006): Credit Risk Scorecards
  - OSFI E-23: Internal Model Risk Management
  - Basel III: IRB Approach
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import logging
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Scorecard Scaling Constants
# ─────────────────────────────────────────────
def compute_scaling_factors(base_score: int = 600, base_odds: float = 50,
                             pdo: int = 20) -> tuple:
    """
    Derive Factor and Offset for score scaling.
    
    Score = Offset + Factor × ln(Odds)
    where Odds = P(Good) / P(Bad)
    
    Standard Canadian bank parameters (RBC/BMO/TD):
      Base Score = 600 (industry-standard mid-point)
      PDO = 20 (points to double odds — tighter than US 40)
      Base Odds = 50:1
    """
    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(base_odds)
    logger.info(f"Scaling: Factor={factor:.4f}, Offset={offset:.4f}")
    return factor, offset


# ─────────────────────────────────────────────
# Individual Feature Score Contribution
# ─────────────────────────────────────────────
def compute_scorecard_points(lr_model: LogisticRegression,
                              feature_names: list,
                              woe_transformer,
                              factor: float,
                              offset: float,
                              n_features: int) -> pd.DataFrame:
    """
    Compute point contribution for each WoE bin.
    
    Points(bin) = -(β_i × WoE_bin + β_0/n_vars) × Factor
    
    This decomposes the overall log-odds into additive scorecard points
    — required for regulatory transparency and OSFI E-23 documentation.
    """
    coefs = lr_model.coef_[0]
    intercept = lr_model.intercept_[0]

    records = []
    for i, (fname_woe, coef) in enumerate(zip(feature_names, coefs)):
        fname = fname_woe.replace("_WoE", "")
        if fname not in woe_transformer.binners_:
            continue

        binner = woe_transformer.binners_[fname]
        intercept_contribution = intercept / n_features  # Distribute intercept

        for _, row in binner.bin_stats_.iterrows():
            woe_val = row["WoE"]
            # Score = -(β × WoE + intercept/n) × Factor + Offset/n
            points = -(coef * woe_val + intercept_contribution) * factor + (offset / n_features)
            records.append({
                "Feature": fname,
                "Bin": row["Bin"],
                "N": row["N"],
                "EventRate": round(row["EventRate"] * 100, 2),
                "WoE": round(woe_val, 4),
                "Coefficient": round(coef, 6),
                "Points": round(points, 1),
            })

    scorecard_df = pd.DataFrame(records)
    logger.info(f"Scorecard points computed for {scorecard_df['Feature'].nunique()} features")
    return scorecard_df


# ─────────────────────────────────────────────
# Score Generator
# ─────────────────────────────────────────────
class ScorecardModel:
    """
    End-to-end scorecard model:
    1. Logistic Regression on WoE features
    2. Score scaling (300–850)
    3. PD calibration via Platt scaling
    4. Score band assignment
    """

    def __init__(self, config: dict):
        sc_cfg = config["scorecard"]
        lr_cfg = config["logistic_regression"]

        self.base_score = sc_cfg["base_score"]
        self.base_odds = sc_cfg["base_odds"]
        self.pdo = sc_cfg["pdo"]
        self.min_score = sc_cfg["min_score"]
        self.max_score = sc_cfg["max_score"]
        self.decision_thresholds = sc_cfg["decision_thresholds"]

        self.lr = LogisticRegression(
            solver=lr_cfg["solver"],
            max_iter=lr_cfg["max_iter"],
            C=lr_cfg["C"],
            class_weight=lr_cfg["class_weight"],
            random_state=42,
        )
        self.calibrated_lr = None
        self.factor, self.offset = compute_scaling_factors(
            self.base_score, self.base_odds, self.pdo
        )
        self.scorecard_points_ = None
        self.feature_names_ = None

    def fit(self, X_woe: pd.DataFrame, y: pd.Series,
            X_val_woe: pd.DataFrame = None, y_val: pd.Series = None):
        """Train logistic regression and calibrate probabilities."""
        self.feature_names_ = list(X_woe.columns)
        logger.info(f"Training Logistic Regression on {len(self.feature_names_)} WoE features...")

        self.lr.fit(X_woe, y)

        # Calibrate for IFRS 9 PD estimation
        # sklearn 1.8: cv="prefit" removed — fit calibrator on validation fold directly
        if X_val_woe is not None and y_val is not None:
            from sklearn.linear_model import LogisticRegression as _BaseLR
            _base = _BaseLR(solver=self.lr.solver, max_iter=self.lr.max_iter,
                            C=self.lr.C, class_weight=self.lr.class_weight, random_state=42)
            self.calibrated_lr = CalibratedClassifierCV(_base, method="isotonic", cv=5)
            self.calibrated_lr.fit(X_val_woe, y_val)
            logger.info("PD calibration complete (isotonic, cv=5)")
        else:
            self.calibrated_lr = self.lr

        return self

    def predict_proba(self, X_woe: pd.DataFrame) -> np.ndarray:
        """Return calibrated PD estimates."""
        return self.calibrated_lr.predict_proba(X_woe)[:, 1]

    def predict_score(self, X_woe: pd.DataFrame) -> np.ndarray:
        """
        Convert log-odds to scorecard points (300–850 scale).
        Score = Offset + Factor × log(P_good/P_bad)
        """
        log_odds_good = self.lr.predict_log_proba(X_woe)[:, 0]  # log P(good)
        # ln(odds_good) = log P(good) - log P(bad)
        log_odds = self.lr.predict_log_proba(X_woe)[:, 0] - \
                   self.lr.predict_log_proba(X_woe)[:, 1]
        scores = self.offset + self.factor * log_odds
        scores = np.clip(scores, self.min_score, self.max_score)
        return scores.round().astype(int)

    def predict_decision(self, scores: np.ndarray) -> pd.Series:
        """
        Apply Canadian bank decision rules:
        Decline / Refer to Credit Officer / Approve
        """
        thresholds = self.decision_thresholds
        decisions = pd.cut(
            scores,
            bins=[-np.inf, thresholds["decline"], thresholds["refer"],
                  thresholds["approve"], np.inf],
            labels=["Decline", "Refer", "Conditional Approve", "Approve"],
        )
        return decisions

    def build_scorecard_table(self, woe_transformer) -> pd.DataFrame:
        """Build point allocation table — core regulatory deliverable."""
        return compute_scorecard_points(
            lr_model=self.lr,
            feature_names=self.feature_names_,
            woe_transformer=woe_transformer,
            factor=self.factor,
            offset=self.offset,
            n_features=len(self.feature_names_),
        )

    def save(self, path: str = "models/scorecard_model.pkl"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"Model saved to {path}")

    @staticmethod
    def load(path: str = "models/scorecard_model.pkl") -> "ScorecardModel":
        return joblib.load(path)


# ─────────────────────────────────────────────
# Score Distribution Analysis
# ─────────────────────────────────────────────
def score_distribution_analysis(scores: np.ndarray, y: pd.Series,
                                 decision_thresholds: dict) -> dict:
    """
    Analyse score distributions by outcome.
    Standard output in Canadian bank model documentation.
    """
    df = pd.DataFrame({"Score": scores, "BAD": y})

    bands = [
        (300, 549, "Very High Risk"),
        (550, 579, "High Risk"),
        (580, 619, "Medium Risk"),
        (620, 659, "Acceptable Risk"),
        (660, 699, "Low Risk"),
        (700, 749, "Very Low Risk"),
        (750, 850, "Minimal Risk"),
    ]

    records = []
    for low, high, band in bands:
        mask = (df["Score"] >= low) & (df["Score"] <= high)
        n = mask.sum()
        if n == 0:
            continue
        n_bad = df.loc[mask, "BAD"].sum()
        records.append({
            "Score Band": f"{low}–{high}",
            "Risk Category": band,
            "N": n,
            "N_Good": n - n_bad,
            "N_Bad": n_bad,
            "Default_Rate_%": round(n_bad / n * 100, 2),
            "Cumulative_%_Pop": round(mask.cumsum().iloc[-1] / len(df) * 100, 2),
        })

    return {
        "score_bands": pd.DataFrame(records),
        "overall_default_rate": y.mean(),
        "mean_score": scores.mean(),
        "std_score": scores.std(),
        "p5_score": np.percentile(scores, 5),
        "p95_score": np.percentile(scores, 95),
    }


# ─────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────
def plot_score_distribution(scores: np.ndarray, y: pd.Series,
                             decision_thresholds: dict,
                             title: str = "Score Distribution") -> plt.Figure:
    """Score distribution plot — standard in Canadian BFSI model packages."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram by outcome
    ax = axes[0]
    scores_good = scores[y == 0]
    scores_bad = scores[y == 1]
    ax.hist(scores_good, bins=40, alpha=0.6, color="#27ae60", label="Good (No Default)",
            density=True, edgecolor="white")
    ax.hist(scores_bad, bins=40, alpha=0.6, color="#e74c3c", label="Bad (Default)",
            density=True, edgecolor="white")

    for key, val in decision_thresholds.items():
        ax.axvline(val, color="navy", linestyle="--", linewidth=1.5, alpha=0.8)
        ax.text(val, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0 else 0.01,
                f"{key}\n{val}", ha="center", fontsize=7, color="navy")

    ax.set_xlabel("Credit Score")
    ax.set_ylabel("Density")
    ax.set_title(f"{title}\nScore Distribution by Outcome")
    ax.legend()
    ax.grid(alpha=0.3)

    # KDE by outcome
    ax2 = axes[1]
    from scipy.stats import gaussian_kde
    score_range = np.linspace(300, 850, 500)
    if len(scores_good) > 1:
        kde_good = gaussian_kde(scores_good)
        ax2.plot(score_range, kde_good(score_range), color="#27ae60", lw=2, label="Good")
        ax2.fill_between(score_range, kde_good(score_range), alpha=0.2, color="#27ae60")
    if len(scores_bad) > 1:
        kde_bad = gaussian_kde(scores_bad)
        ax2.plot(score_range, kde_bad(score_range), color="#e74c3c", lw=2, label="Bad")
        ax2.fill_between(score_range, kde_bad(score_range), alpha=0.2, color="#e74c3c")

    ax2.set_xlabel("Credit Score")
    ax2.set_ylabel("Density")
    ax2.set_title("Score KDE — Separation Quality")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.suptitle("Scorecard Score Analysis — Canadian BFSI Standard", fontsize=12, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_scorecard_table(scorecard_df: pd.DataFrame, feature: str) -> plt.Figure:
    """Visualise scorecard points for a specific feature."""
    data = scorecard_df[scorecard_df["Feature"] == feature].copy()

    fig, ax = plt.subplots(figsize=(10, max(4, len(data) * 0.5)))
    ax.axis("off")

    table_data = [["Bin", "N", "Default Rate %", "WoE", "Points"]]
    for _, row in data.iterrows():
        table_data.append([
            row["Bin"], f"{int(row['N']):,}",
            f"{row['EventRate']:.1f}%",
            f"{row['WoE']:.4f}",
            f"{row['Points']:.1f}",
        ])

    colors = [["#2c3e50"] * 5] + [
        ["#ecf0f1" if i % 2 == 0 else "#dfe6e9"] * 5
        for i in range(len(table_data) - 1)
    ]

    tbl = ax.table(
        cellText=table_data,
        cellLoc="center",
        loc="center",
        cellColours=colors,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 1.8)

    # Style header
    for j in range(5):
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    ax.set_title(f"Scorecard Points Table — {feature}", fontweight="bold", pad=20)
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    print("Scorecard module loaded successfully.")
    print(f"Scaling example: Factor={compute_scaling_factors()[0]:.4f}, Offset={compute_scaling_factors()[1]:.4f}")
