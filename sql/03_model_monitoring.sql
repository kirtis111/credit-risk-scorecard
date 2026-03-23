-- ============================================================
-- 03_model_monitoring.sql
-- Monthly Model Monitoring Queries
-- OSFI E-23 — Required ongoing performance monitoring
-- ============================================================

-- ─────────────────────────────────────────────
-- 1. Insert Monthly PSI Monitoring Record
-- ─────────────────────────────────────────────
INSERT INTO credit_risk.model_monitoring (
    model_version, monitoring_period, monitoring_type,
    psi_score, rag_status, recommended_action,
    n_applications, analyst_id
)
SELECT
    'v1.0'                                  AS model_version,
    DATE_TRUNC('month', CURRENT_DATE)       AS monitoring_period,
    'PSI'                                   AS monitoring_type,
    /* PSI computation inline */
    (
        SELECT SUM(
            (pct_curr - pct_dev) * LN(NULLIF(pct_curr, 0) / NULLIF(pct_dev, 0))
        )
        FROM (
            SELECT
                NTILE(10) OVER (ORDER BY credit_score) AS decile,
                COUNT(*) * 1.0 / (SELECT COUNT(*) FROM credit_risk.scored_applications
                                   WHERE model_version = 'v1.0'
                                     AND score_date < '2024-01-01') AS pct_dev,
                COUNT(*) * 1.0 / (SELECT COUNT(*) FROM credit_risk.scored_applications
                                   WHERE model_version = 'v1.0'
                                     AND DATE_TRUNC('month', score_date) = DATE_TRUNC('month', CURRENT_DATE)) AS pct_curr
            FROM credit_risk.scored_applications
            WHERE model_version = 'v1.0'
            GROUP BY NTILE(10) OVER (ORDER BY credit_score)
        ) psi_sub
    )                                       AS psi_score,
    'Green'                                 AS rag_status,
    'No action required'                    AS recommended_action,
    COUNT(*)                                AS n_applications,
    'SYSTEM'                                AS analyst_id
FROM credit_risk.scored_applications
WHERE model_version = 'v1.0'
  AND DATE_TRUNC('month', score_date) = DATE_TRUNC('month', CURRENT_DATE);


-- ─────────────────────────────────────────────
-- 2. Default Rate Monitoring (Vintage)
-- ─────────────────────────────────────────────
CREATE OR REPLACE VIEW credit_risk.v_default_rate_monitoring AS
WITH outcomes AS (
    SELECT
        s.application_id,
        s.model_version,
        s.credit_score,
        s.pd_calibrated,
        s.ifrs9_stage,
        la.bad_flag,
        la.application_date,
        -- Months on book
        EXTRACT(YEAR FROM AGE(CURRENT_DATE, la.application_date)) * 12 +
        EXTRACT(MONTH FROM AGE(CURRENT_DATE, la.application_date)) AS months_on_book
    FROM credit_risk.scored_applications s
    JOIN credit_risk.loan_applications la
        ON s.application_id = la.application_id
    WHERE la.bad_flag IS NOT NULL
)
SELECT
    DATE_TRUNC('quarter', application_date)::DATE   AS origination_quarter,
    model_version,
    NTILE(10) OVER (
        PARTITION BY DATE_TRUNC('quarter', application_date), model_version
        ORDER BY credit_score
    )                                               AS score_decile,
    months_on_book,
    COUNT(*)                                        AS n_accounts,
    SUM(bad_flag)                                   AS n_defaults,
    AVG(bad_flag::FLOAT)                            AS actual_default_rate,
    AVG(pd_calibrated)                              AS expected_pd,
    -- Accuracy ratio: actual vs expected
    AVG(bad_flag::FLOAT) / NULLIF(AVG(pd_calibrated), 0) AS pd_accuracy_ratio
FROM outcomes
GROUP BY 1, 2, NTILE(10) OVER (
    PARTITION BY DATE_TRUNC('quarter', application_date), model_version
    ORDER BY credit_score
), 4;


-- ─────────────────────────────────────────────
-- 3. Stage Migration Analysis (IFRS 9)
-- ─────────────────────────────────────────────
CREATE OR REPLACE VIEW credit_risk.v_stage_migration AS
WITH current_stage AS (
    SELECT application_id, ifrs9_stage AS stage_current,
           ROW_NUMBER() OVER (PARTITION BY application_id ORDER BY score_date DESC) AS rn
    FROM credit_risk.scored_applications
),
prior_stage AS (
    SELECT application_id, ifrs9_stage AS stage_prior,
           ROW_NUMBER() OVER (PARTITION BY application_id ORDER BY score_date DESC) AS rn
    FROM credit_risk.scored_applications
    WHERE score_date < DATE_TRUNC('month', CURRENT_DATE)
)
SELECT
    p.stage_prior,
    c.stage_current,
    COUNT(*) AS n_accounts,
    COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY p.stage_prior) AS migration_pct
FROM current_stage c
JOIN prior_stage p ON c.application_id = p.application_id
WHERE c.rn = 1 AND p.rn = 1
GROUP BY 1, 2;


-- ─────────────────────────────────────────────
-- 4. Monthly IFRS 9 Provision Change
-- ─────────────────────────────────────────────
SELECT
    a.reporting_period,
    a.total_ead,
    a.total_ecl,
    a.total_ecl - LAG(a.total_ecl) OVER (ORDER BY a.reporting_period) AS ecl_movement,
    (a.total_ecl - LAG(a.total_ecl) OVER (ORDER BY a.reporting_period))
        / NULLIF(LAG(a.total_ecl) OVER (ORDER BY a.reporting_period), 0) AS ecl_change_pct,
    a.stage1_ecl,
    a.stage2_ecl,
    a.stage3_ecl,
    a.probability_weighted_ecl,
    a.el_rate
FROM credit_risk.ifrs9_provisions a
WHERE a.model_version = 'v1.0'
  AND a.is_final = TRUE
ORDER BY reporting_period;


-- ─────────────────────────────────────────────
-- 5. Model Performance Decay (Backtesting)
-- ─────────────────────────────────────────────
SELECT
    m.monitoring_period,
    m.model_version,
    m.monitoring_type,
    m.auc_roc,
    m.gini,
    m.ks_statistic,
    m.default_rate,
    m.expected_default,
    m.psi_score,
    m.rag_status,
    -- Rolling 3-month average AUC
    AVG(m.auc_roc) OVER (
        PARTITION BY m.model_version
        ORDER BY m.monitoring_period
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ) AS auc_3m_rolling,
    -- Month-over-month AUC change
    m.auc_roc - LAG(m.auc_roc) OVER (
        PARTITION BY m.model_version ORDER BY m.monitoring_period
    ) AS auc_mom_change
FROM credit_risk.model_monitoring m
WHERE m.monitoring_type = 'PERFORMANCE'
ORDER BY m.model_version, m.monitoring_period;
