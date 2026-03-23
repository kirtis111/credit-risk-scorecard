"""
data_preprocessing.py
─────────────────────
Data cleaning and preparation pipeline for the HMEQ credit risk dataset.
Mirrors data quality standards used at RBC / TD / BMO model development teams.

OSFI E-23 compliance: All data transformations documented and reproducible.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging
import yaml
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Load configuration
# ─────────────────────────────────────────────
def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────
# 1. Load Raw Data
# ─────────────────────────────────────────────
def load_hmeq(path: str = "data/raw/hmeq.csv") -> pd.DataFrame:
    """Load HMEQ dataset from Kaggle. Source: https://www.kaggle.com/datasets/ajay1735/hmeq-data"""
    logger.info(f"Loading HMEQ dataset from {path}")
    df = pd.read_csv(path)
    logger.info(f"Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


# ─────────────────────────────────────────────
# 2. Data Quality Report
# ─────────────────────────────────────────────
def data_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a data quality report — required by OSFI E-23 model documentation.
    Reports missing rate, cardinality, and basic statistics.
    """
    report = []
    for col in df.columns:
        missing_n = df[col].isna().sum()
        missing_pct = missing_n / len(df) * 100
        dtype = str(df[col].dtype)
        cardinality = df[col].nunique()
        # treat object AND pandas StringDtype as non-numeric
        is_numeric = pd.api.types.is_numeric_dtype(df[col])
        report.append({
            "Feature": col,
            "DType": dtype,
            "Missing_N": missing_n,
            "Missing_Pct": round(missing_pct, 2),
            "Cardinality": cardinality,
            "Min": round(float(df[col].min()), 4) if is_numeric else None,
            "Max": round(float(df[col].max()), 4) if is_numeric else None,
            "Mean": round(float(df[col].mean()), 4) if is_numeric else None,
            "Std": round(float(df[col].std()), 4) if is_numeric else None,
        })
    dq = pd.DataFrame(report)
    logger.info(f"Data quality report generated for {len(dq)} features")
    return dq


# ─────────────────────────────────────────────
# 3. PIPEDA Compliance — Remove Prohibited Variables
# ─────────────────────────────────────────────
PROHIBITED_VARIABLES = [
    "race", "ethnicity", "gender", "sex", "religion",
    "national_origin", "marital_status", "postal_code_proxy",
    "age_discriminatory",  # Age can be used in limited contexts only
]

def remove_prohibited_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove any variables that violate PIPEDA / Canadian human rights legislation.
    Canadian banks cannot use protected characteristics in credit decisions.
    """
    to_drop = [col for col in df.columns if col.lower() in PROHIBITED_VARIABLES]
    if to_drop:
        logger.warning(f"Removing PIPEDA-prohibited variables: {to_drop}")
        df = df.drop(columns=to_drop)
    return df


# ─────────────────────────────────────────────
# 4. Handle Missing Values
# ─────────────────────────────────────────────
def handle_missing_values(df: pd.DataFrame, target_col: str = "BAD",
                           missing_threshold: float = 0.60) -> pd.DataFrame:
    """
    Imputation strategy aligned with Basel III IRB guidance:
    - Features with > threshold% missing: dropped
    - Numeric: median imputation (conservative, not mean)
    - Categorical: mode + 'Unknown' category for missings
    - Missing indicator flags added for MCAR assumption testing
    """
    df = df.copy()

    # Drop rows where target is missing
    before = len(df)
    df = df.dropna(subset=[target_col])
    logger.info(f"Dropped {before - len(df)} rows with missing target")

    # Drop high-missing features
    missing_rates = df.isnull().mean()
    high_missing = missing_rates[missing_rates > missing_threshold].index.tolist()
    if high_missing:
        logger.info(f"Dropping {len(high_missing)} high-missing features: {high_missing}")
        df = df.drop(columns=high_missing)

    # Add missing indicators for features with > 5% missing
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != target_col]
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

    for col in numeric_cols:
        if df[col].isnull().mean() > 0.05:
            df[f"{col}_MISSING"] = df[col].isnull().astype(int)

    # Median imputation for numerics
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logger.debug(f"Imputed {col} with median={median_val:.4f}")

    # Mode + 'Unknown' for categoricals
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna("Unknown")

    logger.info(f"Missing value handling complete. Shape: {df.shape}")
    return df


# ─────────────────────────────────────────────
# 5. Outlier Treatment (Winsorization)
# ─────────────────────────────────────────────
def winsorize_features(df: pd.DataFrame, target_col: str = "BAD",
                       lower_pct: float = 0.01, upper_pct: float = 0.99) -> pd.DataFrame:
    """
    Winsorize numeric features at 1st and 99th percentile.
    Standard practice in Canadian bank scorecard development to handle data entry errors.
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != target_col and not c.endswith("_MISSING")]

    for col in numeric_cols:
        lower = df[col].quantile(lower_pct)
        upper = df[col].quantile(upper_pct)
        original_range = (df[col].min(), df[col].max())
        df[col] = df[col].clip(lower=lower, upper=upper)
        logger.debug(f"Winsorized {col}: {original_range} → ({lower:.2f}, {upper:.2f})")

    logger.info(f"Winsorization complete for {len(numeric_cols)} numeric features")
    return df


# ─────────────────────────────────────────────
# 6. Encode Categorical Variables
# ─────────────────────────────────────────────
def encode_categoricals(df: pd.DataFrame, target_col: str = "BAD") -> pd.DataFrame:
    """
    Encode categorical variables using label encoding.
    Actual WoE encoding is done in the woe_binning module.
    """
    df = df.copy()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

    encoding_map = {}
    for col in cat_cols:
        codes, uniques = pd.factorize(df[col])
        df[col] = codes
        encoding_map[col] = {v: k for k, v in enumerate(uniques)}
        logger.debug(f"Encoded {col}: {dict(list(encoding_map[col].items())[:5])}")

    logger.info(f"Encoded {len(cat_cols)} categorical features")
    return df, encoding_map


# ─────────────────────────────────────────────
# 7. Train / Validation / Test Split
# ─────────────────────────────────────────────
def temporal_split(df: pd.DataFrame, target_col: str = "BAD",
                   test_size: float = 0.20, val_size: float = 0.10,
                   random_state: int = 42):
    """
    Stratified split maintaining class proportions.
    In production (RBC/TD) this would be temporal (OOT) split.
    """
    from sklearn.model_selection import train_test_split

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # First split: train+val vs test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Second split: train vs val
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_ratio, stratify=y_trainval,
        random_state=random_state
    )

    logger.info(f"Split sizes — Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
    logger.info(f"Default rates — Train: {y_train.mean():.2%} | Val: {y_val.mean():.2%} | Test: {y_test.mean():.2%}")

    return X_train, X_val, X_test, y_train, y_val, y_test


# ─────────────────────────────────────────────
# 8. Full Pipeline
# ─────────────────────────────────────────────
def run_preprocessing_pipeline(config_path: str = "config/config.yaml") -> dict:
    """Execute complete preprocessing pipeline."""
    cfg = load_config(config_path)

    # Load
    df = load_hmeq(cfg["data"]["raw_path"])

    # Quality report
    dq_report = data_quality_report(df)
    dq_report.to_csv("data/processed/data_quality_report.csv", index=False)
    logger.info("Data quality report saved to data/processed/data_quality_report.csv")

    # Clean
    df = remove_prohibited_variables(df)
    df = handle_missing_values(df, cfg["data"]["target_column"],
                                cfg["feature_engineering"]["missing_threshold"])
    df = winsorize_features(df, cfg["data"]["target_column"])
    df, encoding_map = encode_categoricals(df, cfg["data"]["target_column"])

    # Save processed
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    df.to_csv("data/processed/hmeq_clean.csv", index=False)
    logger.info(f"Processed data saved. Final shape: {df.shape}")

    # Split
    splits = temporal_split(
        df,
        cfg["data"]["target_column"],
        cfg["data"]["test_size"],
        cfg["data"]["validation_size"],
        cfg["data"]["random_state"],
    )

    return {
        "df": df,
        "splits": splits,
        "encoding_map": encoding_map,
        "dq_report": dq_report,
        "config": cfg,
    }


if __name__ == "__main__":
    results = run_preprocessing_pipeline()
    print("Preprocessing complete.")
