"""
woe_binning.py
──────────────
Weight of Evidence (WoE) binning and Information Value (IV) calculation engine.

This is the core feature transformation module for logistic regression scorecards.
Implements best practices from Canadian bank scorecard development (RBC/TD/BMO).

Theory:
  WoE(bin_i) = ln(Distribution_of_Events / Distribution_of_Non_Events)
  IV = Σ (Distribution_Events - Distribution_Non_Events) × WoE

IV Interpretation (Canadian Bank Standards):
  < 0.02  : Useless predictor
  0.02–0.1: Weak
  0.1–0.3 : Medium
  0.3–0.5 : Strong
  > 0.5   : Suspiciously strong (check for target leakage)
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import logging
import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

# Small epsilon to avoid log(0)
EPSILON = 1e-10


# ─────────────────────────────────────────────
# Core WoE Calculator
# ─────────────────────────────────────────────
class WoEBinner:
    """
    Computes WoE bins and IV for a single feature.
    Supports both numeric (quantile binning) and categorical features.
    Enforces monotonicity for scorecard compliance.
    """

    def __init__(self, feature_name: str, n_bins: int = 10,
                 enforce_monotonicity: bool = True):
        self.feature_name = feature_name
        self.n_bins = n_bins
        self.enforce_monotonicity = enforce_monotonicity
        self.bins_ = None
        self.woe_map_ = {}
        self.iv_ = None
        self.bin_stats_ = None

    def fit(self, X: pd.Series, y: pd.Series) -> "WoEBinner":
        """Fit WoE bins on training data only (never fit on test!)."""
        df = pd.DataFrame({"X": X, "y": y}).dropna()

        if df["X"].dtype == "object" or df["X"].nunique() <= self.n_bins:
            # Categorical or low-cardinality: group by value
            bins = self._compute_woe_categorical(df)
        else:
            # Numeric: quantile binning with merge for monotonicity
            bins = self._compute_woe_numeric(df)

        self.bin_stats_ = bins
        self.iv_ = bins["IV_Contribution"].sum()
        self.woe_map_ = dict(zip(bins["Bin"], bins["WoE"]))
        return self

    def transform(self, X: pd.Series) -> pd.Series:
        """Map raw values to WoE values using fitted bins."""
        if self.bins_ is not None:
            X_binned = pd.cut(X, bins=self.bins_, labels=False, include_lowest=True)
            bin_labels = self.bin_stats_["Bin"].values
            label_map = {i: woe for i, (_, woe) in
                         enumerate(zip(bin_labels, self.bin_stats_["WoE"].values))}
            return X_binned.map(label_map).fillna(0)
        else:
            return X.map(self.woe_map_).fillna(0)

    def fit_transform(self, X: pd.Series, y: pd.Series) -> pd.Series:
        return self.fit(X, y).transform(X)

    def _compute_woe_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """Quantile-based WoE for numeric features."""
        # Initial quantile binning
        try:
            df["bin"], bin_edges = pd.qcut(df["X"], q=self.n_bins,
                                            retbins=True, duplicates="drop")
            self.bins_ = bin_edges
        except ValueError:
            # Fall back to equal-width if quantile fails
            df["bin"], bin_edges = pd.cut(df["X"], bins=self.n_bins,
                                           retbins=True, include_lowest=True)
            self.bins_ = bin_edges

        stats = self._aggregate_bins(df, "bin")

        if self.enforce_monotonicity:
            stats = self._enforce_monotonicity(stats)

        return stats

    def _compute_woe_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """WoE for categorical features."""
        stats = self._aggregate_bins(df, "X")
        return stats

    def _aggregate_bins(self, df: pd.DataFrame, bin_col: str) -> pd.DataFrame:
        """Core WoE / IV calculation."""
        total_events = df["y"].sum()
        total_non_events = (df["y"] == 0).sum()

        stats = (
            df.groupby(bin_col)["y"]
            .agg(
                N="count",
                Events=lambda x: (x == 1).sum(),
                NonEvents=lambda x: (x == 0).sum(),
            )
            .reset_index()
            .rename(columns={bin_col: "Bin"})
        )

        stats["Bin"] = stats["Bin"].astype(str)
        stats["EventRate"] = stats["Events"] / stats["N"]
        stats["Dist_Events"] = stats["Events"] / (total_events + EPSILON)
        stats["Dist_NonEvents"] = stats["NonEvents"] / (total_non_events + EPSILON)

        # Avoid log(0)
        stats["Dist_Events"] = stats["Dist_Events"].clip(lower=EPSILON)
        stats["Dist_NonEvents"] = stats["Dist_NonEvents"].clip(lower=EPSILON)

        stats["WoE"] = np.log(stats["Dist_Events"] / stats["Dist_NonEvents"])
        stats["IV_Contribution"] = (stats["Dist_Events"] - stats["Dist_NonEvents"]) * stats["WoE"]

        return stats

    def _enforce_monotonicity(self, stats: pd.DataFrame) -> pd.DataFrame:
        """
        Merge adjacent bins to achieve monotonic WoE pattern.
        Required for scorecard interpretability and OSFI E-23 compliance.
        """
        # Detect direction (increasing or decreasing WoE)
        woe_vals = stats["WoE"].values
        direction = np.sign(np.corrcoef(np.arange(len(woe_vals)), woe_vals)[0, 1])
        if direction == 0:
            return stats

        # Iteratively merge non-monotonic adjacent bins
        max_iterations = 20
        for _ in range(max_iterations):
            is_monotonic = True
            for i in range(1, len(stats)):
                prev_woe = stats.iloc[i - 1]["WoE"]
                curr_woe = stats.iloc[i]["WoE"]
                if direction > 0 and curr_woe < prev_woe:
                    stats = self._merge_bins(stats, i - 1, i)
                    is_monotonic = False
                    break
                elif direction < 0 and curr_woe > prev_woe:
                    stats = self._merge_bins(stats, i - 1, i)
                    is_monotonic = False
                    break
            if is_monotonic:
                break

        return stats

    def _merge_bins(self, stats: pd.DataFrame, idx1: int, idx2: int) -> pd.DataFrame:
        """Merge two adjacent bins into one."""
        row1 = stats.iloc[idx1]
        row2 = stats.iloc[idx2]

        merged_bin = f"{row1['Bin']} + {row2['Bin']}"
        merged_N = row1["N"] + row2["N"]
        merged_events = row1["Events"] + row2["Events"]
        merged_non_events = row1["NonEvents"] + row2["NonEvents"]
        merged_event_rate = merged_events / (merged_N + EPSILON)

        total_events = stats["Events"].sum()
        total_non_events = stats["NonEvents"].sum()
        dist_e = (merged_events + EPSILON) / (total_events + EPSILON)
        dist_ne = (merged_non_events + EPSILON) / (total_non_events + EPSILON)
        woe = np.log(dist_e / dist_ne)
        iv_contrib = (dist_e - dist_ne) * woe

        new_row = pd.DataFrame([{
            "Bin": merged_bin, "N": merged_N, "Events": merged_events,
            "NonEvents": merged_non_events, "EventRate": merged_event_rate,
            "Dist_Events": dist_e, "Dist_NonEvents": dist_ne,
            "WoE": woe, "IV_Contribution": iv_contrib,
        }])

        stats = pd.concat([
            stats.iloc[:idx1],
            new_row,
            stats.iloc[idx2 + 1:]
        ], ignore_index=True)

        return stats


# ─────────────────────────────────────────────
# WoE Transformation Pipeline
# ─────────────────────────────────────────────
class WoETransformer:
    """
    Fits WoE binners for all features and transforms dataset.
    Also computes IV for feature selection.
    """

    def __init__(self, n_bins: int = 10, enforce_monotonicity: bool = True,
                 iv_minimum: float = 0.02):
        self.n_bins = n_bins
        self.enforce_monotonicity = enforce_monotonicity
        self.iv_minimum = iv_minimum
        self.binners_ = {}
        self.iv_table_ = None
        self.selected_features_ = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "WoETransformer":
        """Fit WoE binners on training data."""
        logger.info(f"Fitting WoE binners for {X.shape[1]} features...")
        iv_records = []

        for col in X.columns:
            binner = WoEBinner(
                feature_name=col,
                n_bins=self.n_bins,
                enforce_monotonicity=self.enforce_monotonicity,
            )
            binner.fit(X[col], y)
            self.binners_[col] = binner

            iv_records.append({
                "Feature": col,
                "IV": round(binner.iv_, 4),
                "Strength": self._iv_label(binner.iv_),
                "N_Bins": len(binner.bin_stats_),
            })

        self.iv_table_ = (
            pd.DataFrame(iv_records)
            .sort_values("IV", ascending=False)
            .reset_index(drop=True)
        )

        self.selected_features_ = (
            self.iv_table_[self.iv_table_["IV"] >= self.iv_minimum]["Feature"].tolist()
        )

        logger.info(f"WoE fitting complete. Selected {len(self.selected_features_)} features (IV ≥ {self.iv_minimum})")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform all features to WoE values."""
        X_woe = X.copy()
        for col in self.selected_features_:
            if col in X.columns:
                X_woe[f"{col}_WoE"] = self.binners_[col].transform(X[col])
        # Return only WoE columns
        woe_cols = [f"{col}_WoE" for col in self.selected_features_ if col in X.columns]
        return X_woe[woe_cols]

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    @staticmethod
    def _iv_label(iv: float) -> str:
        if iv < 0.02:
            return "Useless"
        elif iv < 0.10:
            return "Weak"
        elif iv < 0.30:
            return "Medium"
        elif iv < 0.50:
            return "Strong"
        else:
            return "Suspicious (check leakage)"


# ─────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────
def plot_woe_bins(binner: WoEBinner, figsize: tuple = (10, 5)) -> plt.Figure:
    """
    Plot WoE pattern for a single feature.
    Standard chart in Canadian bank model documentation packages.
    """
    stats = binner.bin_stats_
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # WoE bar chart
    colors = ["#c0392b" if w < 0 else "#27ae60" for w in stats["WoE"]]
    ax1.barh(range(len(stats)), stats["WoE"], color=colors, edgecolor="white", linewidth=0.5)
    ax1.set_yticks(range(len(stats)))
    ax1.set_yticklabels(stats["Bin"].values, fontsize=8)
    ax1.axvline(0, color="black", linewidth=1)
    ax1.set_xlabel("Weight of Evidence (WoE)")
    ax1.set_title(f"{binner.feature_name}\nWoE by Bin (IV = {binner.iv_:.4f})")
    ax1.grid(axis="x", alpha=0.3)

    # Event rate bar chart
    ax2.barh(range(len(stats)), stats["EventRate"] * 100,
             color="#2980b9", edgecolor="white", linewidth=0.5)
    ax2.set_yticks(range(len(stats)))
    ax2.set_yticklabels(stats["Bin"].values, fontsize=8)
    ax2.set_xlabel("Default Rate (%)")
    ax2.set_title(f"{binner.feature_name}\nDefault Rate by Bin")
    ax2.grid(axis="x", alpha=0.3)

    # Add N labels
    for i, row in stats.iterrows():
        ax2.text(stats["EventRate"].max() * 100 * 0.02, i,
                 f"N={row['N']:,}", va="center", fontsize=7, color="white")

    plt.tight_layout()
    return fig


def plot_iv_summary(iv_table: pd.DataFrame, top_n: int = 15) -> plt.Figure:
    """IV summary chart — standard in Canadian bank model development packages."""
    fig, ax = plt.subplots(figsize=(10, 6))
    data = iv_table.head(top_n).sort_values("IV")

    colors_map = {
        "Useless": "#bdc3c7",
        "Weak": "#f39c12",
        "Medium": "#3498db",
        "Strong": "#27ae60",
        "Suspicious (check leakage)": "#e74c3c",
    }
    bar_colors = [colors_map.get(s, "#3498db") for s in data["Strength"]]

    bars = ax.barh(data["Feature"], data["IV"], color=bar_colors, edgecolor="white")
    ax.axvline(0.02, color="red", linestyle="--", linewidth=1, label="Min threshold (0.02)")
    ax.axvline(0.10, color="orange", linestyle="--", linewidth=1, label="Medium threshold (0.10)")
    ax.axvline(0.30, color="green", linestyle="--", linewidth=1, label="Strong threshold (0.30)")

    for bar, iv in zip(bars, data["IV"]):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{iv:.4f}", va="center", fontsize=9)

    ax.set_xlabel("Information Value (IV)")
    ax.set_title("Feature Predictive Power — Information Value Summary\n(Canadian Bank Scorecard Standard)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # Quick smoke test
    np.random.seed(42)
    n = 1000
    X_test = pd.DataFrame({
        "DEBTINC": np.random.exponential(30, n),
        "DELINQ": np.random.poisson(0.5, n),
        "CLAGE": np.random.normal(200, 80, n),
    })
    y_test = pd.Series((np.random.rand(n) < 0.20).astype(int))

    transformer = WoETransformer(n_bins=5, iv_minimum=0.02)
    X_woe = transformer.fit_transform(X_test, y_test)
    print(transformer.iv_table_)
    print(f"\nTransformed shape: {X_woe.shape}")
    print("WoE binning test passed.")
