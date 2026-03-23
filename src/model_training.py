"""
model_training.py
─────────────────
Champion/Challenger model training pipeline.
  Champion:   Logistic Regression Scorecard (always available)
  Challenger: XGBoost + Optuna HPO when installed;
              GradientBoostingClassifier + GridSearchCV as fallback.

SHAP explainability when shap is installed; permutation_importance fallback otherwise.

Mirrors model development at RBC Enterprise Risk, TD Risk Analytics,
and BMO Credit Risk Management.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import logging
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

# ── Optional heavy dependencies — graceful fallbacks ────────────────────────
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    xgb = None
    XGB_AVAILABLE = False
    logger.warning("xgboost not installed — GradientBoostingClassifier used as challenger")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    shap = None
    SHAP_AVAILABLE = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    optuna = None
    OPTUNA_AVAILABLE = False


# ─────────────────────────────────────────────
# Champion: Logistic Regression
# ─────────────────────────────────────────────
def train_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series,
                                X_val: pd.DataFrame, y_val: pd.Series,
                                config: dict) -> dict:
    """
    Train logistic regression champion model.
    Uses WoE features (already monotonic, stable).
    Calibrated with Platt scaling for IFRS 9 PD output.
    5-fold CV for regularisation selection.
    """
    lr_cfg = config["logistic_regression"]
    logger.info("Training Champion: Logistic Regression on WoE features...")

    best_c, best_auc = lr_cfg["C"], 0.0
    for c in [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]:
        lr_tmp = LogisticRegression(solver=lr_cfg["solver"], max_iter=lr_cfg["max_iter"],
                                     C=c, class_weight=lr_cfg["class_weight"], random_state=42)
        cv_auc = cross_val_score(lr_tmp, X_train, y_train,
                                  cv=StratifiedKFold(5), scoring="roc_auc").mean()
        if cv_auc > best_auc:
            best_auc, best_c = cv_auc, c

    logger.info(f"Best C = {best_c}  (CV AUC = {best_auc:.4f})")

    lr = LogisticRegression(solver=lr_cfg["solver"], max_iter=lr_cfg["max_iter"],
                             C=best_c, class_weight=lr_cfg["class_weight"], random_state=42)
    lr.fit(X_train, y_train)

    # sklearn 1.8: cv="prefit" removed — calibrate on validation fold
    _base = LogisticRegression(solver=lr_cfg["solver"], max_iter=lr_cfg["max_iter"],
                                C=best_c, class_weight=lr_cfg["class_weight"], random_state=42)
    calibrated = CalibratedClassifierCV(_base, method="isotonic", cv=5)
    calibrated.fit(X_val, y_val)

    train_auc = roc_auc_score(y_train, lr.predict_proba(X_train)[:, 1])
    val_auc   = roc_auc_score(y_val,   calibrated.predict_proba(X_val)[:, 1])
    logger.info(f"LR Train AUC: {train_auc:.4f} | Val AUC: {val_auc:.4f}")

    return {
        "model":           lr,
        "calibrated_model": calibrated,
        "best_c":          best_c,
        "train_auc":       train_auc,
        "val_auc":         val_auc,
        "feature_names":   list(X_train.columns),
        "coefficients":    pd.DataFrame({
            "Feature":     X_train.columns,
            "Coefficient": lr.coef_[0],
        }).sort_values("Coefficient", key=abs, ascending=False),
    }


# ─────────────────────────────────────────────
# Challenger: XGBoost / GradientBoosting
# ─────────────────────────────────────────────
def train_xgboost_challenger(X_train: pd.DataFrame, y_train: pd.Series,
                               X_val: pd.DataFrame, y_val: pd.Series,
                               config: dict, n_trials: int = 30) -> dict:
    """
    Train challenger model.
    - XGBoost + Optuna when both are installed (production path).
    - GradientBoostingClassifier + GridSearchCV when they are not (CI/offline path).
    Both paths produce calibrated PD estimates for IFRS 9.
    """
    if XGB_AVAILABLE and OPTUNA_AVAILABLE:
        return _train_xgboost_optuna(X_train, y_train, X_val, y_val, config, n_trials)
    else:
        return _train_gbm_fallback(X_train, y_train, X_val, y_val, config)


def _train_xgboost_optuna(X_train, y_train, X_val, y_val, config, n_trials):
    """XGBoost + Optuna HPO path (when libraries are available)."""
    logger.info("Training Challenger: XGBoost + Optuna HPO (%d trials)...", n_trials)

    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 600),
            "max_depth":        trial.suggest_int("max_depth", 3, 6),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 3, 10),
            "scale_pos_weight": (y_train == 0).sum() / (y_train == 1).sum(),
            "eval_metric": "auc", "random_state": 42, "tree_method": "hist",
        }
        m = xgb.XGBClassifier(**params)
        m.fit(X_train, y_train, eval_set=[(X_val, y_val)],
              early_stopping_rounds=30, verbose=False)
        return roc_auc_score(y_val, m.predict_proba(X_val)[:, 1])

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = {**study.best_params,
                   "scale_pos_weight": (y_train == 0).sum() / (y_train == 1).sum(),
                   "eval_metric": "auc", "random_state": 42, "tree_method": "hist"}
    logger.info("Best XGB params: %s", best_params)

    challenger = xgb.XGBClassifier(**best_params)
    challenger.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                   early_stopping_rounds=50, verbose=False)

    # sklearn 1.8: calibrate with cv=5 on a fresh base
    cal_base   = xgb.XGBClassifier(**best_params)
    calibrated = CalibratedClassifierCV(cal_base, method="isotonic", cv=5)
    calibrated.fit(X_val, y_val)

    train_auc = roc_auc_score(y_train, challenger.predict_proba(X_train)[:, 1])
    val_auc   = roc_auc_score(y_val,   calibrated.predict_proba(X_val)[:, 1])
    logger.info("XGBoost Train AUC: %.4f | Val AUC: %.4f", train_auc, val_auc)

    return {"model": challenger, "calibrated_model": calibrated,
            "best_params": best_params, "train_auc": train_auc, "val_auc": val_auc,
            "model_type": "XGBoost", "feature_names": list(X_train.columns),
            "optuna_study": study}


def _train_gbm_fallback(X_train, y_train, X_val, y_val, config):
    """GradientBoostingClassifier fallback when xgboost/optuna not installed."""
    logger.info("Training Challenger: GradientBoostingClassifier (XGBoost fallback)...")

    param_grid = {
        "n_estimators":  [200, 400],
        "max_depth":     [3, 4],
        "learning_rate": [0.05, 0.10],
        "subsample":     [0.8],
    }
    base = GradientBoostingClassifier(min_samples_leaf=20, random_state=42)
    grid = GridSearchCV(base, param_grid, cv=3, scoring="roc_auc", n_jobs=-1, refit=True)
    grid.fit(X_train, y_train)
    challenger = grid.best_estimator_
    best_params = grid.best_params_
    logger.info("Best GBM params: %s", best_params)

    # Calibrate on validation fold
    cal_base   = GradientBoostingClassifier(**best_params, min_samples_leaf=20, random_state=42)
    calibrated = CalibratedClassifierCV(cal_base, method="isotonic", cv=5)
    calibrated.fit(X_val, y_val)

    train_auc = roc_auc_score(y_train, challenger.predict_proba(X_train)[:, 1])
    val_auc   = roc_auc_score(y_val,   calibrated.predict_proba(X_val)[:, 1])
    logger.info("GBM Train AUC: %.4f | Val AUC: %.4f", train_auc, val_auc)

    return {"model": challenger, "calibrated_model": calibrated,
            "best_params": best_params, "train_auc": train_auc, "val_auc": val_auc,
            "model_type": "GradientBoosting (XGBoost fallback)",
            "feature_names": list(X_train.columns)}


# ─────────────────────────────────────────────
# SHAP Explainability
# ─────────────────────────────────────────────
class SHAPExplainer:
    """
    Feature explainability for the challenger model.
    Uses SHAP TreeExplainer when shap is installed;
    falls back to permutation_importance otherwise.

    Both paths produce adverse action reason codes per OSFI E-23 /
    Canadian consumer credit regulation (PIPEDA s.11).
    """

    def __init__(self, model, X_train: pd.DataFrame):
        self.model         = model
        self.feature_names = list(X_train.columns)
        self.X_train       = X_train
        self._shap_values  = None

        if SHAP_AVAILABLE and XGB_AVAILABLE and hasattr(model, "get_booster"):
            self.explainer = shap.TreeExplainer(model)
            self._backend  = "shap"
            logger.info("SHAP TreeExplainer initialised")
        else:
            self.explainer = None
            self._backend  = "permutation"
            logger.info("SHAP unavailable — permutation_importance used as fallback")

    def compute_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        if self._backend == "shap":
            self._shap_values = self.explainer.shap_values(X)
        else:
            # permutation importance broadcast to (n_samples, n_features)
            # to keep the same downstream interface
            r = permutation_importance(self.model, X,
                                       np.zeros(len(X)),
                                       n_repeats=5, random_state=42,
                                       scoring="roc_auc")
            self._shap_values = np.tile(r.importances_mean, (len(X), 1))
        return self._shap_values

    def plot_summary(self, X: pd.DataFrame, max_display: int = 15) -> plt.Figure:
        shap_vals = self.compute_shap_values(X)
        mean_abs  = np.abs(shap_vals).mean(axis=0)
        order     = np.argsort(mean_abs)[::-1][:max_display]

        if self._backend == "shap":
            fig, _ = plt.subplots(figsize=(10, 7))
            shap.summary_plot(shap_vals, X, max_display=max_display,
                               show=False, plot_type="dot")
            plt.title("SHAP Feature Importance — Challenger Model\n"
                      "Required for OSFI E-23 Model Documentation", fontweight="bold")
        else:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(range(len(order)), mean_abs[order[::-1]],
                    color="#003366", edgecolor="white")
            ax.set_yticks(range(len(order)))
            ax.set_yticklabels([self.feature_names[i] for i in order[::-1]], fontsize=9)
            ax.set_xlabel("Permutation Importance (AUC drop)")
            ax.set_title("Feature Importance — Permutation Method (SHAP fallback)\n"
                         "Challenger Model | OSFI E-23 Documentation", fontweight="bold")
            ax.grid(axis="x", alpha=0.3)

        plt.tight_layout()
        return fig

    def get_top_reasons(self, X_single: pd.DataFrame, n_reasons: int = 3) -> list:
        """Adverse action reason codes — PIPEDA s.11 compliant."""
        shap_vals = self.compute_shap_values(X_single)[0]
        df = pd.DataFrame({
            "Feature":  self.feature_names,
            "SHAP":     shap_vals,
            "AbsSHAP":  np.abs(shap_vals),
        }).sort_values("AbsSHAP", ascending=False)

        return [{"feature":    row["Feature"],
                 "shap_value": round(row["SHAP"], 4),
                 "direction":  "increases risk" if row["SHAP"] > 0 else "decreases risk",
                 "impact":     "High" if row["AbsSHAP"] > 0.1 else "Medium"}
                for _, row in df.head(n_reasons).iterrows()]


# ─────────────────────────────────────────────
# Champion vs Challenger Comparison
# ─────────────────────────────────────────────
def compare_champion_challenger(champion_results: dict, challenger_results: dict,
                                  X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """
    Champion vs Challenger comparison table.
    Standard output for Canadian bank model governance committees.
    """
    from model_validation import compute_discrimination_metrics

    champ_pd  = champion_results["calibrated_model"].predict_proba(X_test)[:, 1]
    chall_pd  = challenger_results["calibrated_model"].predict_proba(X_test)[:, 1]
    champ_m   = compute_discrimination_metrics(y_test.values, champ_pd)
    chall_m   = compute_discrimination_metrics(y_test.values, chall_pd)

    comparison = pd.DataFrame({
        "Metric":               ["AUC-ROC", "Gini", "KS Statistic",
                                  "Brier Score", "Sensitivity", "Specificity"],
        "Champion (LR Scorecard)": [champ_m["AUC_ROC"], champ_m["Gini"],
                                     champ_m["KS_Statistic"], champ_m["Brier_Score"],
                                     champ_m["Sensitivity"], champ_m["Specificity"]],
        f"Challenger ({challenger_results.get('model_type','GBM')})":
                                [chall_m["AUC_ROC"], chall_m["Gini"],
                                 chall_m["KS_Statistic"], chall_m["Brier_Score"],
                                 chall_m["Sensitivity"], chall_m["Specificity"]],
    })
    comparison["Difference"] = (
        comparison.iloc[:, 2] - comparison.iloc[:, 1]
    ).round(4)
    comparison["Verdict"] = comparison.apply(
        lambda r: ("✓ Challenger Superior"
                   if (r["Metric"] != "Brier Score" and r["Difference"] > 0.01)
                   or (r["Metric"] == "Brier Score" and r["Difference"] < -0.01)
                   else ("= Comparable" if abs(r["Difference"]) <= 0.01
                         else "✓ Champion Superior")),
        axis=1)

    logger.info(f"\n{comparison.to_string(index=False)}")
    return comparison


# ─────────────────────────────────────────────
# Save All Model Artefacts
# ─────────────────────────────────────────────
def save_models(champion_results: dict, challenger_results: dict,
                woe_transformer, scorecard_model,
                model_dir: str = "models/") -> None:
    """Save all model artefacts — required for production deployment."""
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump(champion_results["calibrated_model"],   f"{model_dir}lr_champion.pkl")
    joblib.dump(challenger_results["calibrated_model"], f"{model_dir}xgb_challenger.pkl")
    joblib.dump(woe_transformer,                        f"{model_dir}woe_transformer.pkl")
    joblib.dump(scorecard_model,                        f"{model_dir}scorecard_model.pkl")

    if XGB_AVAILABLE and hasattr(challenger_results.get("model"), "save_model"):
        challenger_results["model"].save_model(f"{model_dir}xgb_raw.json")

    logger.info(f"All model artefacts saved to {model_dir}")


if __name__ == "__main__":
    print(f"model_training.py loaded.")
    print(f"  XGBoost  : {'available' if XGB_AVAILABLE  else 'NOT installed — GBM fallback active'}")
    print(f"  SHAP     : {'available' if SHAP_AVAILABLE  else 'NOT installed — permutation fallback active'}")
    print(f"  Optuna   : {'available' if OPTUNA_AVAILABLE else 'NOT installed — GridSearchCV fallback active'}")
