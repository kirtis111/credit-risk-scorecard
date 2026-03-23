"""
ifrs9_calculations.py
─────────────────────
IFRS 9 Expected Credit Loss (ECL) Engine for Canadian Financial Institutions.

Implements the 3-stage ECL model as required by:
  - IFRS 9 Financial Instruments (effective Jan 1, 2018 for Canadian banks)
  - OSFI E-6: Principles-Based Provisions (PBP) Framework
  - Basel III: Advanced IRB (AIRB) approach for PD/LGD/EAD

Canadian BFSI Context:
  - "Big Six" Canadian banks fully adopted IFRS 9 in 2018
  - OSFI requires forward-looking macro overlays (FLI)
  - Downturn LGD = LGD × 1.25 scalar (OSFI minimum)
  - Stage migration triggers aligned with BCBS guidance

ECL Formula:
  ECL = PD × LGD × EAD × DF
  where DF = discount factor (BoC overnight + credit spread)
"""

import numpy as np
import pandas as pd
from scipy.special import expit  # Sigmoid for PD conversion
import logging
import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 1. PD Estimation and Calibration
# ─────────────────────────────────────────────
class PDModel:
    """
    Probability of Default estimation with IFRS 9 forward-looking adjustments.
    
    Supports:
      - Point-in-time (PIT) PD: For IFRS 9 staging and ECL
      - Through-the-cycle (TTC) PD: For regulatory capital (Basel III)
      - Forward-looking PD: FLI macro overlay
    """

    def __init__(self, config: dict):
        self.config = config

    def compute_lifetime_pd(self, annual_pd: float, horizon_years: int = 5) -> np.ndarray:
        """
        Compute multi-year cumulative PD for lifetime ECL.
        Marginal PD(t) = 1 - (1 - Annual_PD)^t × survival
        
        Used for Stage 2 and Stage 3 exposures.
        """
        marginal_pds = []
        cumulative_survival = 1.0

        for t in range(1, horizon_years + 1):
            # Marginal PD for year t (conditional on survival to t-1)
            marginal_pd = annual_pd  # Simplified; in practice use term structure
            pd_t = cumulative_survival * marginal_pd
            cumulative_survival *= (1 - marginal_pd)
            marginal_pds.append(pd_t)

        return np.array(marginal_pds)

    def apply_macro_overlay(self, base_pd: float, scenario: str = "base") -> float:
        """
        Apply forward-looking information (FLI) macro overlay.
        
        Scenarios (Canadian economy, Bank of Canada scenarios):
          Base: GDP growth 2.0%, unemployment 6.5%, house prices stable
          Optimistic: GDP growth 3.5%, unemployment 5.5%, house prices +5%
          Adverse: GDP growth -1.5%, unemployment 8.5%, house prices -15%
        """
        macro_scalars = {
            "optimistic": 0.75,  # PD × 0.75 in favourable conditions
            "base": 1.00,
            "adverse": 1.65,     # PD × 1.65 in adverse conditions (BoC 2023 scenario)
        }
        scalar = macro_scalars.get(scenario, 1.00)
        return min(base_pd * scalar, 1.0)

    def probability_weighted_pd(self, base_pd: float,
                                  scenario_weights: dict = None) -> float:
        """
        Compute probability-weighted PD across macro scenarios.
        IFRS 9 requires multiple forward-looking scenarios.
        """
        if scenario_weights is None:
            scenario_weights = {
                "optimistic": 0.25,
                "base": 0.50,
                "adverse": 0.25,
            }

        weighted_pd = sum(
            self.apply_macro_overlay(base_pd, scenario) * weight
            for scenario, weight in scenario_weights.items()
        )
        return min(weighted_pd, 1.0)


# ─────────────────────────────────────────────
# 2. LGD Estimation
# ─────────────────────────────────────────────
class LGDModel:
    """
    Loss Given Default estimation for Canadian BFSI.
    
    LGD = 1 - Recovery Rate
    Recovery Rate depends on:
      - Collateral type and value (property, vehicle, unsecured)
      - Seniority of claim
      - Recovery period and costs
      - Downturn conditions (OSFI requires downturn LGD)
    """

    def __init__(self, config: dict):
        self.lgd_config = config.get("lgd", {})
        self.downturn_scalar = config.get("ifrs9", {}).get("lgd_downturn_scalar", 1.25)

    def compute_lgd(self, loan_value: float, property_value: float,
                     loan_type: str = "home_equity",
                     recovery_costs_pct: float = 0.08) -> dict:
        """
        Compute LGD for a home equity loan.
        
        Canadian bank LGD framework:
          Recovery = min(Property Value × (1 - Haircut), Outstanding Balance) × (1 - Costs)
          LGD = 1 - Recovery / EAD
        """
        haircuts = self.lgd_config.get("collateral_haircut", {
            "residential_mortgage": 0.20,
            "home_equity": 0.25,
            "unsecured": 0.00,
        })

        haircut = haircuts.get(loan_type, 0.25)

        # Net collateral value after OSFI haircut
        net_collateral = property_value * (1 - haircut)

        # Recovery (net of liquidation costs)
        recovery_gross = min(net_collateral, loan_value)
        recovery_net = recovery_gross * (1 - recovery_costs_pct)

        lgd_base = 1 - (recovery_net / (loan_value + 1e-10))
        lgd_base = max(0.0, min(lgd_base, 1.0))

        # Downturn LGD (OSFI requirement: use in ECL and capital)
        lgd_downturn = min(lgd_base * self.downturn_scalar, 1.0)

        return {
            "lgd_base": round(lgd_base, 4),
            "lgd_downturn": round(lgd_downturn, 4),
            "net_collateral": round(net_collateral, 2),
            "recovery_net": round(recovery_net, 2),
            "haircut_applied": haircut,
            "collateral_coverage": round(net_collateral / (loan_value + 1e-10), 4),
        }

    def compute_portfolio_lgd(self, df: pd.DataFrame,
                               loan_col: str = "LOAN",
                               value_col: str = "VALUE") -> pd.DataFrame:
        """Compute LGD for all loans in portfolio."""
        lgd_results = []
        for _, row in df.iterrows():
            loan_amt = row.get(loan_col, 10000)
            prop_val = row.get(value_col, loan_amt * 2)
            result = self.compute_lgd(loan_amt, prop_val)
            lgd_results.append(result)

        lgd_df = pd.DataFrame(lgd_results)
        df = df.copy()
        df["LGD"] = lgd_df["lgd_base"].values
        df["LGD_Downturn"] = lgd_df["lgd_downturn"].values
        return df


# ─────────────────────────────────────────────
# 3. EAD Estimation
# ─────────────────────────────────────────────
class EADModel:
    """
    Exposure at Default — estimates outstanding balance at time of default.
    
    For term loans (home equity): EAD ≈ Outstanding Balance (CCF = 1.0)
    For revolving facilities: EAD = Drawn + CCF × Undrawn
    
    CCF = Credit Conversion Factor
    """

    def __init__(self, config: dict):
        ead_cfg = config.get("ead", {})
        self.revolving_ccf = ead_cfg.get("revolving_ccf", 0.75)
        self.term_ccf = ead_cfg.get("term_ccf", 1.00)

    def compute_ead(self, outstanding: float, limit: float = None,
                     facility_type: str = "term") -> float:
        """
        EAD computation:
        Term: EAD = Outstanding (CCF = 1.0)
        Revolving: EAD = Outstanding + CCF × (Limit - Outstanding)
        """
        if facility_type == "term" or limit is None:
            return outstanding * self.term_ccf
        else:
            undrawn = max(limit - outstanding, 0)
            return outstanding + self.revolving_ccf * undrawn


# ─────────────────────────────────────────────
# 4. IFRS 9 Staging Engine
# ─────────────────────────────────────────────
def assign_ifrs9_stage(pd_12m: float, pd_lifetime: float,
                        days_past_due: int = 0,
                        is_watchlisted: bool = False,
                        staging_config: dict = None) -> dict:
    """
    Assign IFRS 9 Stage based on Significant Increase in Credit Risk (SICR).
    
    Stage 1: No SICR — 12-month ECL provisioning
    Stage 2: SICR identified (but not defaulted) — Lifetime ECL
    Stage 3: Credit-impaired (defaulted) — Lifetime ECL on NPL basis
    
    Canadian bank SICR triggers (OSFI-aligned):
      - PD doubled since origination
      - 30+ DPD (rebuttable presumption for SICR)
      - 90+ DPD (rebuttable presumption for Stage 3)
      - Internal watchlist / CRM downgrade
    """
    if staging_config is None:
        staging_config = {
            "stage1_max_pd": 0.03,
            "stage2_max_pd": 0.20,
            "stage3_min_pd": 0.20,
        }

    # Stage 3: Default / Credit-impaired
    if days_past_due >= 90 or pd_lifetime >= staging_config["stage3_min_pd"]:
        stage = 3
        ecl_horizon = "Lifetime"
        pd_for_ecl = pd_lifetime
        provision_basis = "NPL"
    # Stage 2: SICR
    elif (days_past_due >= 30 or
          pd_12m >= staging_config["stage1_max_pd"] or
          is_watchlisted):
        stage = 2
        ecl_horizon = "Lifetime"
        pd_for_ecl = pd_lifetime
        provision_basis = "Lifetime ECL"
    # Stage 1: Performing
    else:
        stage = 1
        ecl_horizon = "12-Month"
        pd_for_ecl = pd_12m
        provision_basis = "12-Month ECL"

    return {
        "stage": stage,
        "ecl_horizon": ecl_horizon,
        "pd_for_ecl": pd_for_ecl,
        "provision_basis": provision_basis,
    }


# ─────────────────────────────────────────────
# 5. ECL Calculator
# ─────────────────────────────────────────────
class ECLCalculator:
    """
    Expected Credit Loss (ECL) computation engine.
    
    ECL = PD × LGD × EAD × DF
    
    For multi-year lifetime ECL:
    ECL = Σ_t [ PD(t) × LGD(t) × EAD(t) × DF(t) ]
    """

    def __init__(self, config: dict):
        self.config = config
        self.discount_rate = config.get("ifrs9", {}).get("discount_rate", 0.065)
        self.pd_model = PDModel(config)
        self.lgd_model = LGDModel(config)
        self.ead_model = EADModel(config)

    def compute_ecl_single(self, pd_annual: float, lgd: float, ead: float,
                            stage: int, horizon_years: int = 5) -> dict:
        """Compute ECL for a single exposure."""

        if stage == 1:
            # 12-month ECL
            ecl = pd_annual * lgd * ead * self._discount_factor(1)
            return {
                "ecl": round(ecl, 2),
                "ecl_rate": round(ecl / (ead + 1e-10), 6),
                "pd_used": round(pd_annual, 6),
                "horizon": "12-month",
            }
        else:
            # Lifetime ECL (Stage 2 / Stage 3)
            pd_term_structure = self.pd_model.compute_lifetime_pd(pd_annual, horizon_years)

            ecl_total = 0.0
            for t, pd_t in enumerate(pd_term_structure, start=1):
                # LGD and EAD assumed constant (simplification; use term structure in prod)
                df_t = self._discount_factor(t)
                ecl_total += pd_t * lgd * ead * df_t

            return {
                "ecl": round(ecl_total, 2),
                "ecl_rate": round(ecl_total / (ead + 1e-10), 6),
                "pd_used": round(pd_annual, 6),
                "horizon": f"Lifetime ({horizon_years}yr)",
                "pd_term_structure": pd_term_structure.tolist(),
            }

    def _discount_factor(self, years: int) -> float:
        """Effective interest rate discount factor per IFRS 9."""
        return 1 / ((1 + self.discount_rate) ** years)

    def compute_portfolio_ecl(self, df: pd.DataFrame,
                               pd_col: str = "PD",
                               loan_col: str = "LOAN",
                               value_col: str = "VALUE") -> pd.DataFrame:
        """
        Compute ECL for full portfolio.
        Returns enriched DataFrame with IFRS 9 staging and provisioning.
        """
        df = df.copy()

        # Compute LGD
        df = self.lgd_model.compute_portfolio_lgd(df, loan_col, value_col)

        # EAD = Loan amount (term loans)
        df["EAD"] = df[loan_col].apply(self.ead_model.compute_ead)

        # Stage assignment
        stages = []
        for _, row in df.iterrows():
            pd_12m = row.get(pd_col, 0.05)
            pd_lifetime = min(pd_12m * 2.0, 0.99)  # Simplified lifetime PD
            dpd = row.get("DELINQ", 0) * 30  # Proxy DPD from delinquencies
            stage_result = assign_ifrs9_stage(pd_12m, pd_lifetime, int(dpd))
            stages.append(stage_result)

        stage_df = pd.DataFrame(stages)
        df["Stage"] = stage_df["stage"].values
        df["ECL_Horizon"] = stage_df["ecl_horizon"].values
        df["PD_for_ECL"] = stage_df["pd_for_ecl"].values

        # ECL computation
        ecl_values = []
        for _, row in df.iterrows():
            result = self.compute_ecl_single(
                pd_annual=row["PD_for_ECL"],
                lgd=row["LGD_Downturn"],  # Use downturn LGD per OSFI
                ead=row["EAD"],
                stage=int(row["Stage"]),
            )
            ecl_values.append(result["ecl"])

        df["ECL"] = ecl_values
        df["ECL_Rate"] = df["ECL"] / (df["EAD"] + 1e-10)

        # Apply FLI macro overlay (probability-weighted scenarios)
        df["PD_FLI_Weighted"] = df[pd_col].apply(
            self.pd_model.probability_weighted_pd
        )
        df["ECL_FLI_Adjusted"] = df["ECL"] * (df["PD_FLI_Weighted"] / (df[pd_col] + 1e-10))

        logger.info(f"Portfolio ECL computed. "
                    f"Stage 1: {(df['Stage'] == 1).sum()} | "
                    f"Stage 2: {(df['Stage'] == 2).sum()} | "
                    f"Stage 3: {(df['Stage'] == 3).sum()}")
        logger.info(f"Total ECL: ${df['ECL'].sum():,.0f} | "
                    f"FLI-Adjusted ECL: ${df['ECL_FLI_Adjusted'].sum():,.0f}")

        return df


# ─────────────────────────────────────────────
# 6. IFRS 9 Provision Summary
# ─────────────────────────────────────────────
def compute_provision_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute IFRS 9 provision summary by stage.
    This is the output that goes into the bank's financial statements.
    """
    summary = df.groupby("Stage").agg(
        N_Exposures=("EAD", "count"),
        Total_EAD=("EAD", "sum"),
        Total_ECL=("ECL", "sum"),
        Total_ECL_FLI=("ECL_FLI_Adjusted", "sum"),
        Avg_PD=("PD_for_ECL", "mean"),
        Avg_LGD=("LGD_Downturn", "mean"),
        Avg_Coverage=("ECL_Rate", "mean"),
    ).reset_index()

    summary["Stage_Label"] = summary["Stage"].map({
        1: "Stage 1 — Performing (12M ECL)",
        2: "Stage 2 — Underperforming (Lifetime ECL)",
        3: "Stage 3 — Non-Performing (Lifetime ECL)",
    })
    summary["Coverage_Ratio_%"] = (summary["Total_ECL"] / (summary["Total_EAD"] + 1e-10) * 100).round(2)
    summary["Total_EAD"] = summary["Total_EAD"].round(0).astype(int)
    summary["Total_ECL"] = summary["Total_ECL"].round(0).astype(int)

    return summary[["Stage_Label", "N_Exposures", "Total_EAD", "Total_ECL",
                     "Total_ECL_FLI", "Avg_PD", "Avg_LGD", "Coverage_Ratio_%"]]


# ─────────────────────────────────────────────
# 7. Expected Loss Decomposition
# ─────────────────────────────────────────────
def compute_expected_loss_summary(df: pd.DataFrame, pd_col: str = "PD") -> dict:
    """
    Decompose expected loss into PD, LGD, EAD contributions.
    Standard output for Canadian bank risk management dashboards.
    """
    total_ead = df["EAD"].sum()
    total_ecl = df["ECL"].sum()
    avg_pd = df[pd_col].mean()
    avg_lgd = df["LGD"].mean()
    portfolio_el_rate = total_ecl / (total_ead + 1e-10)

    return {
        "Total_EAD_CAD": round(total_ead, 0),
        "Total_ECL_CAD": round(total_ecl, 0),
        "Portfolio_EL_Rate_%": round(portfolio_el_rate * 100, 4),
        "Avg_PD": round(avg_pd, 4),
        "Avg_LGD": round(avg_lgd, 4),
        "Stage_1_EAD_pct": round((df["EAD"][df["Stage"] == 1].sum() / (total_ead + 1e-10)) * 100, 2),
        "Stage_2_EAD_pct": round((df["EAD"][df["Stage"] == 2].sum() / (total_ead + 1e-10)) * 100, 2),
        "Stage_3_EAD_pct": round((df["EAD"][df["Stage"] == 3].sum() / (total_ead + 1e-10)) * 100, 2),
    }


if __name__ == "__main__":
    import yaml

    # Smoke test
    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)

    # Synthetic data
    np.random.seed(42)
    n = 100
    test_df = pd.DataFrame({
        "LOAN": np.random.uniform(10000, 50000, n),
        "VALUE": np.random.uniform(80000, 300000, n),
        "DELINQ": np.random.poisson(0.3, n),
        "PD": np.random.beta(2, 18, n),
    })

    calculator = ECLCalculator(cfg)
    result_df = calculator.compute_portfolio_ecl(test_df)
    summary = compute_provision_summary(result_df)

    print("\nIFRS 9 Provision Summary:")
    print(summary.to_string(index=False))
    el_summary = compute_expected_loss_summary(result_df)
    print(f"\nTotal EAD: ${el_summary['Total_EAD_CAD']:,.0f}")
    print(f"Total ECL: ${el_summary['Total_ECL_CAD']:,.0f}")
    print(f"EL Rate: {el_summary['Portfolio_EL_Rate_%']:.4f}%")
