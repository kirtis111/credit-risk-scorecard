-- ============================================================
-- 01_create_tables.sql
-- Credit Risk Feature Store Schema
-- Canadian BFSI — OSFI E-23 / IFRS 9 Aligned
-- Mirrors feature store designs at RBC, TD, BMO
-- ============================================================

-- ─────────────────────────────────────────────
-- Schema Setup
-- ─────────────────────────────────────────────
CREATE SCHEMA IF NOT EXISTS credit_risk;

-- ─────────────────────────────────────────────
-- 1. Application Table (Raw Data)
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS credit_risk.loan_applications (
    application_id      BIGSERIAL PRIMARY KEY,
    application_date    DATE NOT NULL,
    product_type        VARCHAR(50) NOT NULL DEFAULT 'HOME_EQUITY',  -- HOME_EQUITY | HELOC | MORTGAGE
    
    -- Loan Characteristics
    loan_amount         NUMERIC(15, 2) NOT NULL,
    loan_purpose        VARCHAR(50),        -- DEBTCON (debt consolidation) | HOMEIMP (home improvement)
    
    -- Property
    property_value      NUMERIC(15, 2),
    mortgage_due        NUMERIC(15, 2),
    ltv_ratio           NUMERIC(8, 4),      -- Loan-to-Value (calculated)
    
    -- Applicant
    job_category        VARCHAR(50),        -- Mgr | Office | ProfExe | Sales | Self | Other
    years_on_job        NUMERIC(6, 2),
    debt_income_ratio   NUMERIC(8, 4),
    
    -- Credit Bureau
    derogatory_reports  INTEGER DEFAULT 0,
    delinquent_lines    INTEGER DEFAULT 0,
    credit_line_age_mths NUMERIC(8, 2),
    num_inquiries       INTEGER DEFAULT 0,
    num_credit_lines    INTEGER DEFAULT 0,
    
    -- Target
    bad_flag            SMALLINT,          -- 1 = Default, 0 = No Default
    
    -- Metadata
    branch_id           VARCHAR(20),
    province            CHAR(2),           -- ON, BC, QC, AB, etc.
    risk_analyst_id     VARCHAR(20),
    created_at          TIMESTAMPTZ DEFAULT NOW(),
    updated_at          TIMESTAMPTZ DEFAULT NOW()
);

COMMENT ON TABLE credit_risk.loan_applications IS
    'Raw loan application data — source of truth for model development. PIPEDA compliant.';

-- ─────────────────────────────────────────────
-- 2. WoE Bin Definitions (Version Controlled)
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS credit_risk.woe_bins (
    bin_id              BIGSERIAL PRIMARY KEY,
    model_version       VARCHAR(20) NOT NULL,
    feature_name        VARCHAR(100) NOT NULL,
    bin_number          INTEGER NOT NULL,
    bin_lower           NUMERIC(20, 6),    -- NULL for categorical
    bin_upper           NUMERIC(20, 6),    -- NULL for categorical
    bin_label           VARCHAR(200),      -- Categorical value or range label
    n_observations      INTEGER NOT NULL,
    n_events            INTEGER NOT NULL,  -- Defaults
    n_non_events        INTEGER NOT NULL,
    event_rate          NUMERIC(8, 6),
    woe_value           NUMERIC(12, 6),
    iv_contribution     NUMERIC(12, 6),
    is_missing_bin      BOOLEAN DEFAULT FALSE,
    created_at          TIMESTAMPTZ DEFAULT NOW(),
    
    CONSTRAINT uq_woe_bin UNIQUE (model_version, feature_name, bin_number)
);

COMMENT ON TABLE credit_risk.woe_bins IS
    'WoE bin definitions by model version — required for OSFI E-23 reproducibility.';

-- ─────────────────────────────────────────────
-- 3. Scorecard Points Table
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS credit_risk.scorecard_points (
    points_id           BIGSERIAL PRIMARY KEY,
    model_version       VARCHAR(20) NOT NULL,
    feature_name        VARCHAR(100) NOT NULL,
    bin_label           VARCHAR(200) NOT NULL,
    woe_value           NUMERIC(12, 6),
    coefficient         NUMERIC(12, 8),
    score_points        NUMERIC(8, 2) NOT NULL,
    cumulative_min      NUMERIC(8, 2),
    cumulative_max      NUMERIC(8, 2),
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

-- ─────────────────────────────────────────────
-- 4. Model Registry (Model Governance)
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS credit_risk.model_registry (
    model_id            BIGSERIAL PRIMARY KEY,
    model_version       VARCHAR(20) UNIQUE NOT NULL,
    model_name          VARCHAR(200) NOT NULL,
    model_type          VARCHAR(50) NOT NULL,  -- SCORECARD | XGBOOST | NEURAL_NET
    champion_flag       BOOLEAN DEFAULT FALSE,
    
    -- Performance metrics
    train_auc           NUMERIC(8, 4),
    val_auc             NUMERIC(8, 4),
    test_auc            NUMERIC(8, 4),
    gini                NUMERIC(8, 4),
    ks_statistic        NUMERIC(8, 4),
    brier_score         NUMERIC(8, 4),
    
    -- Governance
    development_date    DATE,
    validation_date     DATE,
    approval_date       DATE,
    deployment_date     DATE,
    retirement_date     DATE,
    
    status              VARCHAR(30) DEFAULT 'DEVELOPMENT',
    -- DEVELOPMENT | IN_VALIDATION | APPROVED | PRODUCTION | RETIRED | REJECTED
    
    author              VARCHAR(100),
    validator           VARCHAR(100),
    approver            VARCHAR(100),
    
    -- Regulatory
    osfi_e23_compliant  BOOLEAN DEFAULT FALSE,
    ifrs9_certified     BOOLEAN DEFAULT FALSE,
    pipeda_reviewed     BOOLEAN DEFAULT FALSE,
    
    notes               TEXT,
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

-- ─────────────────────────────────────────────
-- 5. Scored Applications (Predictions Store)
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS credit_risk.scored_applications (
    score_id            BIGSERIAL PRIMARY KEY,
    application_id      BIGINT REFERENCES credit_risk.loan_applications(application_id),
    model_version       VARCHAR(20) NOT NULL,
    score_date          DATE NOT NULL,
    
    -- Model outputs
    credit_score        INTEGER NOT NULL,   -- 300–850 scorecard scale
    pd_estimate         NUMERIC(8, 6),      -- Probability of Default
    pd_calibrated       NUMERIC(8, 6),      -- Platt-scaled PD (for IFRS 9)
    lgd_estimate        NUMERIC(8, 6),      -- Loss Given Default
    lgd_downturn        NUMERIC(8, 6),      -- Downturn LGD (OSFI scalar)
    ead_estimate        NUMERIC(15, 2),     -- Exposure at Default
    ecl_12month         NUMERIC(15, 2),     -- 12-month ECL (Stage 1)
    ecl_lifetime        NUMERIC(15, 2),     -- Lifetime ECL (Stage 2/3)
    ecl_fli_adjusted    NUMERIC(15, 2),     -- FLI macro-adjusted ECL
    
    -- IFRS 9
    ifrs9_stage         SMALLINT,           -- 1, 2, or 3
    ecl_amount          NUMERIC(15, 2),     -- ECL used for provisioning
    
    -- Decision
    decision_band       VARCHAR(30),        -- APPROVE | CONDITIONAL | REFER | DECLINE
    risk_grade          VARCHAR(5),         -- A+ | A | B+ | B | C+ | C | D
    
    -- Explainability (top 3 SHAP reasons)
    reason_1_feature    VARCHAR(100),
    reason_1_direction  VARCHAR(20),
    reason_2_feature    VARCHAR(100),
    reason_2_direction  VARCHAR(20),
    reason_3_feature    VARCHAR(100),
    reason_3_direction  VARCHAR(20),
    
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

-- ─────────────────────────────────────────────
-- 6. Model Monitoring (PSI / CSI Tracking)
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS credit_risk.model_monitoring (
    monitoring_id       BIGSERIAL PRIMARY KEY,
    model_version       VARCHAR(20) NOT NULL,
    monitoring_period   DATE NOT NULL,       -- First day of monitoring month
    monitoring_type     VARCHAR(10) NOT NULL, -- PSI | CSI | PERFORMANCE
    
    -- Stability metrics
    psi_score           NUMERIC(8, 6),
    csi_score           NUMERIC(8, 6),
    feature_name        VARCHAR(100),        -- NULL for overall PSI
    
    -- Performance (if available with outcomes)
    auc_roc             NUMERIC(8, 4),
    gini                NUMERIC(8, 4),
    ks_statistic        NUMERIC(8, 4),
    default_rate        NUMERIC(8, 6),
    expected_default    NUMERIC(8, 6),
    
    -- Traffic light
    rag_status          CHAR(5) DEFAULT 'Green',  -- Green | Amber | Red
    recommended_action  TEXT,
    
    -- Metadata
    n_applications      INTEGER,
    analyst_id          VARCHAR(50),
    reviewed_by         VARCHAR(50),
    notes               TEXT,
    created_at          TIMESTAMPTZ DEFAULT NOW(),
    
    CONSTRAINT uq_monitoring UNIQUE (model_version, monitoring_period, monitoring_type, feature_name)
);

-- ─────────────────────────────────────────────
-- 7. IFRS 9 Provision Register
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS credit_risk.ifrs9_provisions (
    provision_id        BIGSERIAL PRIMARY KEY,
    reporting_period    DATE NOT NULL,
    model_version       VARCHAR(20) NOT NULL,
    
    -- Stage breakdown
    stage1_ead          NUMERIC(20, 2) NOT NULL,
    stage1_ecl          NUMERIC(20, 2) NOT NULL,
    stage1_coverage     NUMERIC(8, 6),
    
    stage2_ead          NUMERIC(20, 2) NOT NULL,
    stage2_ecl          NUMERIC(20, 2) NOT NULL,
    stage2_coverage     NUMERIC(8, 6),
    
    stage3_ead          NUMERIC(20, 2) NOT NULL,
    stage3_ecl          NUMERIC(20, 2) NOT NULL,
    stage3_coverage     NUMERIC(8, 6),
    
    total_ead           NUMERIC(20, 2) GENERATED ALWAYS AS
                        (stage1_ead + stage2_ead + stage3_ead) STORED,
    total_ecl           NUMERIC(20, 2) GENERATED ALWAYS AS
                        (stage1_ecl + stage2_ecl + stage3_ecl) STORED,
    
    -- FLI scenario breakdown
    base_ecl            NUMERIC(20, 2),
    optimistic_ecl      NUMERIC(20, 2),
    adverse_ecl         NUMERIC(20, 2),
    probability_weighted_ecl NUMERIC(20, 2),
    
    -- Portfolio metrics
    avg_pd              NUMERIC(8, 6),
    avg_lgd             NUMERIC(8, 6),
    el_rate             NUMERIC(8, 6),
    
    -- Governance
    prepared_by         VARCHAR(100),
    reviewed_by         VARCHAR(100),
    approved_by         VARCHAR(100),
    is_final            BOOLEAN DEFAULT FALSE,
    
    notes               TEXT,
    created_at          TIMESTAMPTZ DEFAULT NOW(),
    
    CONSTRAINT uq_provision UNIQUE (reporting_period, model_version)
);

-- ─────────────────────────────────────────────
-- Indexes for Performance
-- ─────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_apps_date      ON credit_risk.loan_applications(application_date);
CREATE INDEX IF NOT EXISTS idx_apps_product   ON credit_risk.loan_applications(product_type);
CREATE INDEX IF NOT EXISTS idx_apps_province  ON credit_risk.loan_applications(province);
CREATE INDEX IF NOT EXISTS idx_scores_date    ON credit_risk.scored_applications(score_date);
CREATE INDEX IF NOT EXISTS idx_scores_stage   ON credit_risk.scored_applications(ifrs9_stage);
CREATE INDEX IF NOT EXISTS idx_monitor_period ON credit_risk.model_monitoring(monitoring_period);
CREATE INDEX IF NOT EXISTS idx_provision_period ON credit_risk.ifrs9_provisions(reporting_period);

COMMENT ON SCHEMA credit_risk IS
    'Credit Risk Analytics schema — OSFI E-23 / IFRS 9 compliant. Managed by Credit Risk Analytics.';
