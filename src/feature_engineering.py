"""
feature_engineering.py
───────────────────────
Derived feature construction for the HMEQ credit risk scorecard.

This module sits between data_preprocessing.py (cleaning) and
woe_binning.py (transformation). It builds domain-driven features
that Canadian bank credit risk analysts construct manually before
running WoE binning.

Pipeline position:
  data_preprocessing  →  feature_engineering  →  woe_binning  →  model_training

Feature categories built here:
  1. Collateral & LTV ratios           (OSFI B-20 aligned)
  2. Debt serviceability ratios        (GDS / TDS proxies)
  3. Employment stability tiers
  4. Credit bureau behaviour features
  5. Interaction features              (DEBTINC × DELINQ, LTV × DEROG)
  6. Missing indicator flags           (MCAR assumption testing)
  7. Near-zero variance / high-corr removal
  8. Final feature selection helper
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging
import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 1. Collateral & LTV Features (OSFI B-20)
# ─────────────────────────────────────────────
def build_ltv_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Loan-to-Value and Combined-LTV ratios.

    OSFI B-20 thresholds (Canadian mortgage underwriting):
      LTV  > 80%  → mandatory CMHC insurance (high-ratio mortgage)
      CLTV > 85%  → elevated risk flag in credit policy
      Equity < 0  → negative equity (underwater) → Stage 2/3 flag

    These ratios are primary drivers in home equity scorecard models
    at RBC, TD, and BMO retail credit.
    """
    df = df.copy()

    # LTV = Loan / Property Value
    df["LTV"] = np.where(
        df["VALUE"] > 0,
        (df["LOAN"] / df["VALUE"]).clip(0, 3),
        np.nan,
    )

    # Combined LTV = (New Loan + Existing Mortgage) / Property Value
    df["CLTV"] = np.where(
        df["VALUE"] > 0,
        ((df["LOAN"] + df["MORTDUE"].fillna(0)) / df["VALUE"]).clip(0, 3),
        np.nan,
    )

    # Available equity (CAD)
    df["EQUITY"] = np.where(
        df["VALUE"].notna() & df["MORTDUE"].notna(),
        df["VALUE"] - df["MORTDUE"],
        np.nan,
    )

    # Equity-to-Value (positive = equity cushion)
    df["EQUITY_RATIO"] = np.where(
        df["VALUE"] > 0,
        df["EQUITY"] / df["VALUE"],
        np.nan,
    )

    # OSFI B-20 flags
    df["HIGH_RATIO_FLAG"]    = (df["LTV"]  > 0.80).astype(int)   # CMHC insurance required
    df["HIGH_CLTV_FLAG"]     = (df["CLTV"] > 0.85).astype(int)   # Policy alert
    df["NEGATIVE_EQUITY_FLAG"] = (df["EQUITY"].fillna(0) < 0).astype(int)

    logger.info("LTV features built: LTV, CLTV, EQUITY, EQUITY_RATIO, HIGH_RATIO_FLAG")
    return df


# ─────────────────────────────────────────────
# 2. Debt Serviceability (GDS / TDS Proxies)
# ─────────────────────────────────────────────
def build_serviceability_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Debt serviceability features aligned with OSFI B-20 stress-test rules.

    Canadian bank underwriting uses:
      GDS (Gross Debt Service) ratio ≤ 35%
      TDS (Total Debt Service)  ratio ≤ 44%

    DEBTINC in HMEQ is a TDS proxy.
    We construct additional bins and flags.
    """
    df = df.copy()

    # DTI band (ordinal encoding — preserves monotonicity for WoE)
    df["DEBTINC_BAND"] = pd.cut(
        df["DEBTINC"],
        bins=[-np.inf, 20, 30, 36, 44, np.inf],
        labels=[1, 2, 3, 4, 5],   # 1=Low … 5=Breach
    ).astype("float")

    # OSFI B-20 TDS breach (>44%)
    df["OSFI_B20_BREACH"] = (df["DEBTINC"] > 44).astype(int)

    # GDS proxy breach (>35%)
    df["GDS_ELEVATED"] = ((df["DEBTINC"] > 35) & (df["DEBTINC"] <= 44)).astype(int)

    # Loan-to-income proxy (LOAN / implied income)
    # Implied income = LOAN / (DEBTINC / 100) — rough proxy when income not available
    df["LOAN_TO_INCOME_PROXY"] = np.where(
        df["DEBTINC"] > 0,
        df["LOAN"] / (df["DEBTINC"] / 100 + 1e-10),
        np.nan,
    )

    logger.info("Serviceability features built: DEBTINC_BAND, OSFI_B20_BREACH, GDS_ELEVATED")
    return df


# ─────────────────────────────────────────────
# 3. Employment Stability
# ─────────────────────────────────────────────
def build_employment_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Employment stability tiers — standard in Canadian bank scorecards.

    Longer tenure = lower default probability.
    Self-employed applicants (JOB=Self) get a separate adjustment in many
    Canadian bank credit policies (income volatility risk).
    """
    df = df.copy()

    # Employment stability ordinal (1=very new … 5=very stable)
    df["EMPLOYMENT_STABILITY"] = pd.cut(
        df["YOJ"],
        bins=[-np.inf, 1, 3, 7, 15, np.inf],
        labels=[1, 2, 3, 4, 5],
    ).astype("float")

    # Short-tenure flag (<2 years — higher delinquency in Canadian bureau data)
    df["SHORT_TENURE_FLAG"] = (df["YOJ"] < 2).astype(int)

    # Self-employed flag (if JOB is already encoded, use raw column before encoding)
    if "JOB" in df.columns:
        # Works whether JOB is the original string or label-encoded int
        df["SELF_EMPLOYED_FLAG"] = (df["JOB"].astype(str).str.lower() == "self").astype(int)

    logger.info("Employment features built: EMPLOYMENT_STABILITY, SHORT_TENURE_FLAG")
    return df


# ─────────────────────────────────────────────
# 4. Credit Bureau Behaviour Features
# ─────────────────────────────────────────────
def build_bureau_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Credit bureau-derived behavioural features.

    Canadian bureaus (Equifax, TransUnion) report:
      - Derogatory marks (collections, write-offs, bankruptcies)
      - Delinquency severity (30/60/90 DPD)
      - Credit inquiries (hard pulls in last 12 months)
      - Account age (oldest open account)
      - Trade line count

    HMEQ proxies for these are: DEROG, DELINQ, NINQ, CLAGE, CLNO.
    """
    df = df.copy()

    # Any adverse mark (derog OR delinquent)
    df["ANY_ADVERSE_FLAG"] = (
        (df["DEROG"].fillna(0) > 0) | (df["DELINQ"].fillna(0) > 0)
    ).astype(int)

    # Severity score (weighted adverse count)
    df["ADVERSE_SEVERITY"] = (
        df["DEROG"].fillna(0) * 2 +    # derogatory = 2× weight
        df["DELINQ"].fillna(0) * 1     # delinquent  = 1× weight
    ).clip(0, 10)

    # Credit age in years (CLAGE is months)
    df["CREDIT_AGE_YRS"] = df["CLAGE"] / 12.0

    # Thin file flag (< 2 years credit history = limited bureau data)
    df["THIN_FILE_FLAG"] = (df["CLAGE"].fillna(0) < 24).astype(int)

    # Inquiry intensity (≥4 inquiries in lookup period = active credit shopping)
    df["HIGH_INQUIRY_FLAG"] = (df["NINQ"].fillna(0) >= 4).astype(int)

    # Delinquency rate (delinquent lines / total lines)
    df["DELINQ_RATE"] = np.where(
        df["CLNO"].fillna(0) > 0,
        df["DELINQ"].fillna(0) / df["CLNO"],
        0.0,
    ).clip(0, 1)

    logger.info("Bureau features built: ANY_ADVERSE_FLAG, ADVERSE_SEVERITY, "
                "CREDIT_AGE_YRS, THIN_FILE_FLAG, HIGH_INQUIRY_FLAG, DELINQ_RATE")
    return df


# ─────────────────────────────────────────────
# 5. Interaction Features
# ─────────────────────────────────────────────
def build_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-product interaction terms that capture compounding risk signals.

    Rationale (Canadian bank credit policy):
      - High DTI *and* delinquent history = far worse than either alone
      - High LTV *and* derogatory history = collateral risk + willingness-to-pay risk
      - New employer *and* thin bureau file = uncertainty stacking

    These interactions are typically tested for IV lift before inclusion.
    """
    df = df.copy()

    # DTI × Delinquency (financial stress + past behaviour)
    df["DEBTINC_X_DELINQ"] = (
        df["DEBTINC"].fillna(df["DEBTINC"].median()) *
        np.log1p(df["DELINQ"].fillna(0))
    )

    # LTV × Adverse (collateral + credit quality compounding risk)
    df["LTV_X_ADVERSE"] = (
        df["LTV"].fillna(df["LTV"].median() if "LTV" in df.columns else 0.7) *
        df["ADVERSE_SEVERITY"] if "ADVERSE_SEVERITY" in df.columns else 0
    )

    # Employment instability × Bureau thinness
    df["INSTABILITY_SCORE"] = (
        df.get("SHORT_TENURE_FLAG", 0) +
        df.get("THIN_FILE_FLAG", 0) +
        df.get("HIGH_INQUIRY_FLAG", 0)
    ).clip(0, 3)

    logger.info("Interaction features built: DEBTINC_X_DELINQ, LTV_X_ADVERSE, INSTABILITY_SCORE")
    return df


# ─────────────────────────────────────────────
# 6. Missing Indicator Flags
# ─────────────────────────────────────────────
def build_missing_indicators(df: pd.DataFrame,
                              target_col: str = "BAD",
                              threshold_pct: float = 0.05) -> pd.DataFrame:
    """
    Add binary flags for missingness on features with >threshold% missing.

    Missing data in credit applications is itself a risk signal:
    - Missing DEBTINC → applicant may have withheld income info
    - Missing YOJ     → applicant may have unstable employment
    - Missing VALUE   → property valuation unavailable (risk)

    These flags allow the model to capture MNAR (missing not at random)
    patterns — standard practice in Canadian bank scorecards.
    """
    df = df.copy()
    cols = [c for c in df.columns if c != target_col]

    flags_added = []
    for col in cols:
        miss_pct = df[col].isna().mean()
        if miss_pct > threshold_pct:
            flag_col = f"{col}_MISSING"
            if flag_col not in df.columns:
                df[flag_col] = df[col].isna().astype(int)
                flags_added.append(flag_col)

    logger.info(f"Missing indicator flags added: {flags_added}")
    return df


# ─────────────────────────────────────────────
# 7. Feature Selection Helpers
# ─────────────────────────────────────────────
def remove_near_zero_variance(df: pd.DataFrame,
                               target_col: str = "BAD",
                               threshold: float = 0.01) -> pd.DataFrame:
    """
    Drop features where variance < threshold (near-constant columns).
    These provide no discriminatory power and can destabilise WoE binning.
    """
    df = df.copy()
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric = [c for c in numeric if c != target_col]

    to_drop = []
    for col in numeric:
        col_range = df[col].max() - df[col].min()
        if col_range == 0 or df[col].var() < threshold:
            to_drop.append(col)

    if to_drop:
        logger.info(f"Dropping near-zero variance features: {to_drop}")
        df = df.drop(columns=to_drop)

    return df


def remove_high_correlation(df: pd.DataFrame,
                              target_col: str = "BAD",
                              threshold: float = 0.85,
                              protected_cols: list = None) -> pd.DataFrame:
    """
    Remove one of each highly correlated pair (|r| > threshold).
    Original HMEQ columns are protected — only derived features are candidates for removal.
    """
    df = df.copy()
    # Original HMEQ columns are never dropped — only derived features are candidates
    HMEQ_ORIGINALS = {"LOAN","MORTDUE","VALUE","REASON","JOB","YOJ",
                       "DEROG","DELINQ","CLAGE","NINQ","CLNO","DEBTINC"}
    if protected_cols:
        HMEQ_ORIGINALS.update(protected_cols)

    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric = [c for c in numeric if c != target_col]

    corr_matrix = df[numeric].corr().abs()
    upper_tri   = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    target_corr = df[numeric + [target_col]].corr()[target_col].abs()

    to_drop = set()
    for col in upper_tri.columns:
        high_corr_features = upper_tri.index[upper_tri[col] > threshold].tolist()
        for feat in high_corr_features:
            # Never drop an original HMEQ feature
            col_is_original  = col  in HMEQ_ORIGINALS
            feat_is_original = feat in HMEQ_ORIGINALS
            if col_is_original and feat_is_original:
                continue  # both original — keep both
            elif col_is_original:
                to_drop.add(feat)   # drop the derived one
            elif feat_is_original:
                to_drop.add(col)    # drop the derived one
            else:
                # both derived — drop lower target-corr
                drop = feat if target_corr.get(feat,0) < target_corr.get(col,0) else col
                to_drop.add(drop)

    if to_drop:
        logger.info(f"Dropping high-correlation features (|r|>{threshold}): {sorted(to_drop)}")
        df = df.drop(columns=list(to_drop))
    return df


def get_feature_summary(df: pd.DataFrame,
                          target_col: str = "BAD") -> pd.DataFrame:
    """
    Summary table of all engineered features with basic stats.
    Useful for documentation in OSFI E-23 model submissions.
    """
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric = [c for c in numeric if c != target_col]

    rows = []
    for col in numeric:
        corr_target = df[[col, target_col]].dropna().corr().iloc[0, 1]
        rows.append({
            "Feature":        col,
            "Missing_%":      round(df[col].isna().mean() * 100, 2),
            "Mean":           round(df[col].mean(), 4),
            "Std":            round(df[col].std(), 4),
            "Min":            round(df[col].min(), 4),
            "Max":            round(df[col].max(), 4),
            "Corr_Target":    round(corr_target, 4),
            "Abs_Corr":       round(abs(corr_target), 4),
        })

    return (pd.DataFrame(rows)
              .sort_values("Abs_Corr", ascending=False)
              .reset_index(drop=True))


# ─────────────────────────────────────────────
# 8. Full Feature Engineering Pipeline
# ─────────────────────────────────────────────
def run_feature_engineering(df: pd.DataFrame,
                              target_col: str = "BAD",
                              config: dict = None) -> pd.DataFrame:
    """
    Execute the complete feature engineering pipeline in order.

    Call this AFTER data_preprocessing.run_preprocessing_pipeline()
    and BEFORE woe_binning.WoETransformer.fit_transform().

    Pipeline:
      raw cleaned data
        → LTV / collateral features
        → Debt serviceability features
        → Employment stability features
        → Credit bureau behaviour features
        → Interaction features
        → Missing indicator flags
        → Near-zero variance removal
        → High-correlation removal
        → Feature summary

    Returns:
      Enriched DataFrame ready for WoE binning.
    """
    if config is None:
        config = {}

    fe_cfg         = config.get("feature_engineering", {})
    corr_threshold = fe_cfg.get("correlation_threshold", 0.85)
    nzv_threshold  = fe_cfg.get("near_zero_variance_threshold", 0.01)

    n_start = df.shape[1]
    logger.info(f"Feature engineering start — {df.shape[0]:,} rows × {n_start} features")

    df = build_ltv_features(df)
    df = build_serviceability_features(df)
    df = build_employment_features(df)
    df = build_bureau_features(df)
    df = build_interaction_features(df)
    df = build_missing_indicators(df, target_col)
    df = remove_near_zero_variance(df, target_col, nzv_threshold)
    df = remove_high_correlation(df, target_col, corr_threshold)

    n_end = df.shape[1]
    logger.info(f"Feature engineering complete — {n_end} features "
                f"({n_end - n_start:+d} net from {n_start})")

    return df


# ─────────────────────────────────────────────
# Quick smoke test
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    # Build a small synthetic frame mimicking HMEQ
    np.random.seed(42)
    n = 200
    test_df = pd.DataFrame({
        "BAD":     np.random.binomial(1, 0.20, n),
        "LOAN":    np.random.uniform(10_000, 80_000, n),
        "MORTDUE": np.random.uniform(50_000, 200_000, n),
        "VALUE":   np.random.uniform(120_000, 500_000, n),
        "YOJ":     np.random.exponential(6, n),
        "DEROG":   np.random.poisson(0.3, n),
        "DELINQ":  np.random.poisson(0.4, n),
        "CLAGE":   np.random.normal(180, 80, n).clip(0),
        "NINQ":    np.random.poisson(1.2, n),
        "CLNO":    np.random.poisson(8, n),
        "DEBTINC": np.random.normal(33, 12, n).clip(5, 80),
        "JOB":     np.random.choice(["Mgr","ProfExe","Office","Self","Other"], n),
    })

    result = run_feature_engineering(test_df)

    print(f"\n✅ feature_engineering.py smoke test passed")
    print(f"   Input features:  {test_df.shape[1]}")
    print(f"   Output features: {result.shape[1]}")
    print(f"\n   New derived features:")
    new_cols = [c for c in result.columns if c not in test_df.columns]
    for c in new_cols:
        print(f"     {c}")

    summary = get_feature_summary(result)
    print(f"\n   Top 5 by |correlation with BAD|:")
    print(summary.head(5)[["Feature", "Corr_Target", "Missing_%"]].to_string(index=False))
