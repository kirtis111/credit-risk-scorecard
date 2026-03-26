-- ============================================================
-- 02_feature_engineering.sql
-- SQL-Based Feature Engineering for Credit Risk Model
-- ============================================================

-- ─────────────────────────────────────────────
-- 1. Base Feature Set with Derived Variables
-- ─────────────────────────────────────────────
CREATE OR REPLACE VIEW credit_risk.v_features AS
SELECT
    application_id,
    application_date,
    product_type,
    province,

    -- ── Loan Features ──────────────────────────
    loan_amount,
    loan_purpose,

    -- Loan-to-Value ratio (critical for home equity)
    CASE
        WHEN property_value > 0 THEN loan_amount / property_value
        ELSE NULL
    END AS ltv_ratio,

    -- Combined LTV (existing mortgage + new loan)
    CASE
        WHEN property_value > 0 THEN (COALESCE(mortgage_due, 0) + loan_amount) / property_value
        ELSE NULL
    END AS cltv_ratio,

    -- Equity available
    CASE
        WHEN property_value IS NOT NULL AND mortgage_due IS NOT NULL
        THEN property_value - mortgage_due
        ELSE NULL
    END AS equity_amount,

    -- ── Applicant Financials ────────────────────
    debt_income_ratio,

    -- Debt-to-income buckets (Canadian bank standard)
    CASE
        WHEN debt_income_ratio IS NULL THEN 'Unknown'
        WHEN debt_income_ratio < 20    THEN 'Low (<20%)'
        WHEN debt_income_ratio < 35    THEN 'Moderate (20-35%)'
        WHEN debt_income_ratio < 43    THEN 'Elevated (35-43%)'
        WHEN debt_income_ratio < 50    THEN 'High (43-50%)'
        ELSE 'Very High (>50%)'
    END AS dti_band,

    -- OSFI B-20: GDS/TDS caps — flag if breaching
    CASE WHEN debt_income_ratio > 44 THEN 1 ELSE 0 END AS osfi_b20_breach_flag,

    -- ── Employment ─────────────────────────────
    job_category,
    years_on_job,

    -- Employment stability tiers
    CASE
        WHEN years_on_job IS NULL THEN 'Unknown'
        WHEN years_on_job < 1    THEN 'Very New (<1yr)'
        WHEN years_on_job < 3    THEN 'New (1-3yr)'
        WHEN years_on_job < 7    THEN 'Established (3-7yr)'
        WHEN years_on_job < 15   THEN 'Stable (7-15yr)'
        ELSE 'Very Stable (15+yr)'
    END AS employment_stability,

    -- ── Credit Bureau Features ──────────────────
    derogatory_reports,
    delinquent_lines,
    credit_line_age_mths,
    num_inquiries,
    num_credit_lines,

    -- Credit behaviour flags
    CASE WHEN derogatory_reports > 0 THEN 1 ELSE 0 END AS has_derogatory,
    CASE WHEN delinquent_lines > 0   THEN 1 ELSE 0 END AS has_delinquency,
    CASE WHEN num_inquiries >= 4     THEN 1 ELSE 0 END AS high_inquiry_flag,

    -- Credit vintage (months)
    credit_line_age_mths / 12.0 AS credit_age_years,

    -- Utilization proxy (if available)
    CASE
        WHEN num_credit_lines > 0 THEN delinquent_lines::FLOAT / num_credit_lines
        ELSE 0
    END AS delinquency_rate,

    -- ── Risk Score Buckets ──────────────────────
    -- Rule-based risk tier (pre-model heuristic)
    CASE
        WHEN derogatory_reports >= 2 OR delinquent_lines >= 2 THEN 'High Risk'
        WHEN derogatory_reports = 1 OR delinquent_lines = 1   THEN 'Elevated Risk'
        WHEN debt_income_ratio > 43                            THEN 'Watch'
        ELSE 'Standard'
    END AS rule_based_risk_tier,

    -- ── Missing Indicators ──────────────────────
    CASE WHEN debt_income_ratio IS NULL  THEN 1 ELSE 0 END AS dti_missing,
    CASE WHEN years_on_job IS NULL       THEN 1 ELSE 0 END AS yoj_missing,
    CASE WHEN mortgage_due IS NULL       THEN 1 ELSE 0 END AS mort_missing,
    CASE WHEN property_value IS NULL     THEN 1 ELSE 0 END AS value_missing,

    -- Target
    bad_flag,
    created_at

FROM credit_risk.loan_applications;


-- ─────────────────────────────────────────────
-- 2. Aggregated Vintage Analysis
-- ─────────────────────────────────────────────
CREATE OR REPLACE VIEW credit_risk.v_vintage_analysis AS
SELECT
    DATE_TRUNC('quarter', application_date)::DATE AS vintage_quarter,
    product_type,
    province,
    COUNT(*)                                        AS n_applications,
    SUM(bad_flag)                                   AS n_defaults,
    AVG(bad_flag::FLOAT)                            AS default_rate,
    AVG(loan_amount)                                AS avg_loan_cad,
    AVG(debt_income_ratio)                          AS avg_dti,
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY loan_amount) AS median_loan_cad,
    SUM(loan_amount)                                AS total_ead_cad
FROM credit_risk.loan_applications
WHERE application_date IS NOT NULL
GROUP BY 1, 2, 3;


-- ─────────────────────────────────────────────
-- 3. Portfolio Risk Summary View
-- ─────────────────────────────────────────────
CREATE OR REPLACE VIEW credit_risk.v_portfolio_risk AS
SELECT
    s.model_version,
    s.score_date,
    la.province,
    la.product_type,
    la.job_category,
    s.ifrs9_stage,
    s.decision_band,
    s.risk_grade,
    COUNT(*)                    AS n_accounts,
    SUM(s.ead_estimate)         AS total_ead,
    SUM(s.ecl_amount)           AS total_ecl,
    AVG(s.pd_calibrated)        AS avg_pd,
    AVG(s.lgd_downturn)         AS avg_lgd,
    AVG(s.ead_estimate)         AS avg_ead,
    AVG(s.credit_score)         AS avg_score,
    STDDEV(s.credit_score)      AS score_stddev,
    MIN(s.credit_score)         AS min_score,
    MAX(s.credit_score)         AS max_score,
    SUM(s.ecl_amount) / NULLIF(SUM(s.ead_estimate), 0) AS ecl_coverage_ratio
FROM credit_risk.scored_applications s
JOIN credit_risk.loan_applications la
    ON s.application_id = la.application_id
GROUP BY 1, 2, 3, 4, 5, 6, 7, 8;


-- ─────────────────────────────────────────────
-- 4. Score Band Distribution (Monthly)
-- ─────────────────────────────────────────────
CREATE OR REPLACE VIEW credit_risk.v_score_distribution AS
SELECT
    DATE_TRUNC('month', score_date)::DATE   AS score_month,
    model_version,
    CASE
        WHEN credit_score < 500  THEN '300-499 Very High Risk'
        WHEN credit_score < 550  THEN '500-549 High Risk'
        WHEN credit_score < 580  THEN '550-579 Elevated Risk'
        WHEN credit_score < 620  THEN '580-619 Medium Risk'
        WHEN credit_score < 660  THEN '620-659 Acceptable'
        WHEN credit_score < 700  THEN '660-699 Low Risk'
        WHEN credit_score < 750  THEN '700-749 Very Low Risk'
        ELSE                          '750-850 Minimal Risk'
    END AS score_band,
    COUNT(*)                                AS n_accounts,
    AVG(pd_calibrated)                      AS avg_pd,
    SUM(ead_estimate)                       AS total_ead,
    SUM(ecl_amount)                         AS total_ecl,
    COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (
        PARTITION BY DATE_TRUNC('month', score_date), model_version
    )                                       AS pct_of_population
FROM credit_risk.scored_applications
GROUP BY 1, 2, 3;


-- ─────────────────────────────────────────────
-- 5. PSI Computation Query
-- ─────────────────────────────────────────────
-- Computes PSI between development population and monitoring month
-- Usage: Replace :dev_month and :monitor_month with actual dates

WITH dev_distribution AS (
    SELECT
        model_version,
        NTILE(10) OVER (PARTITION BY model_version ORDER BY credit_score) AS decile,
        COUNT(*) AS n_dev
    FROM credit_risk.scored_applications
    WHERE score_date >= '2023-01-01'   -- Development period
      AND score_date <  '2023-12-31'
    GROUP BY 1, NTILE(10) OVER (PARTITION BY model_version ORDER BY credit_score)
),
monitor_distribution AS (
    SELECT
        model_version,
        NTILE(10) OVER (PARTITION BY model_version ORDER BY credit_score) AS decile,
        COUNT(*) AS n_monitor
    FROM credit_risk.scored_applications
    WHERE score_date >= '2024-01-01'   -- Monitoring period
      AND score_date <  '2024-01-31'
    GROUP BY 1, NTILE(10) OVER (PARTITION BY model_version ORDER BY credit_score)
),
psi_calc AS (
    SELECT
        d.model_version,
        d.decile,
        d.n_dev,
        COALESCE(m.n_monitor, 0) AS n_monitor,
        d.n_dev::FLOAT / SUM(d.n_dev) OVER (PARTITION BY d.model_version)   AS pct_dev,
        COALESCE(m.n_monitor::FLOAT / NULLIF(SUM(m.n_monitor) OVER
            (PARTITION BY m.model_version), 0), 0.0001)                      AS pct_monitor
    FROM dev_distribution d
    LEFT JOIN monitor_distribution m
        ON d.model_version = m.model_version AND d.decile = m.decile
)
SELECT
    model_version,
    SUM(
        (pct_monitor - pct_dev) * LN(NULLIF(pct_monitor, 0) / NULLIF(pct_dev, 0))
    ) AS psi_total,
    CASE
        WHEN SUM((pct_monitor - pct_dev) * LN(NULLIF(pct_monitor, 0) / NULLIF(pct_dev, 0))) < 0.10
        THEN 'Green — No Action'
        WHEN SUM((pct_monitor - pct_dev) * LN(NULLIF(pct_monitor, 0) / NULLIF(pct_dev, 0))) < 0.25
        THEN 'Amber — Increase Monitoring'
        ELSE 'Red — Model Rebuild Required'
    END AS rag_status
FROM psi_calc
GROUP BY model_version;
