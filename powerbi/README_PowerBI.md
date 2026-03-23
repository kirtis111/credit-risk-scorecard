# 📊 Power BI Dashboard — Canadian BFSI Credit Risk

## Overview

This guide walks through building the Power BI credit risk dashboard that mirrors
analytics dashboards used at RBC, TD, BMO, and major Canadian credit unions.

**Dashboard Pages:**
1. 📋 Executive Risk Summary (RAG status, KPIs)
2. 📈 Model Performance (AUC/Gini/KS trend)
3. 🏛️ IFRS 9 Provisioning (ECL staging, scenario comparison)
4. 🗺️ Provincial Portfolio Map (heatmap by province)
5. 📉 Score Distribution & Vintage Analysis

---

## 🔌 Data Connection Setup

### Option A — Excel Source (quickest)
1. Open Power BI Desktop → **Get Data → Excel**
2. Import `excel/Scorecard_v1.0.xlsx`
3. Select sheets: `Scorecard Points`, `IFRS9 Provisions`, `PSI Monitoring`, `Score Distribution`

### Option B — SQLite (recommended for dev)
```
1. pip install sqlalchemy
2. Run: python sql/load_sqlite.py  (creates credit_risk.db)
3. Power BI → Get Data → ODBC → sqlite3
4. Connection string: Driver={SQLite3};Database=credit_risk.db
```

### Option C — PostgreSQL (production)
```
Server:   localhost
Port:     5432
Database: credit_risk
Schema:   credit_risk
Tables:   loan_applications, scored_applications, ifrs9_provisions, model_monitoring
```

---

## 📐 Data Model (Star Schema)

```
dim_date
  ├── date_id (PK)
  ├── year, quarter, month
  └── is_month_end

dim_model
  ├── model_id (PK)
  ├── model_version
  └── model_type (Champion/Challenger)

fact_scored_applications
  ├── score_id (PK)
  ├── application_id (FK)
  ├── date_id (FK)
  ├── model_id (FK)
  ├── credit_score
  ├── pd_calibrated
  ├── lgd_downturn
  ├── ead_estimate
  ├── ecl_amount
  ├── ifrs9_stage
  └── decision_band

fact_monitoring
  ├── monitoring_id (PK)
  ├── date_id (FK)
  ├── model_id (FK)
  ├── psi_score
  ├── auc_roc
  └── rag_status

fact_provisions
  ├── provision_id (PK)
  ├── date_id (FK)
  ├── stage1_ead, stage1_ecl
  ├── stage2_ead, stage2_ecl
  ├── stage3_ead, stage3_ecl
  └── probability_weighted_ecl
```

---

## 📏 DAX Measures

Paste these into Power BI's **New Measure** dialog:

### Core Portfolio Measures

```dax
-- Total EAD
Total EAD = 
    SUMX(fact_scored_applications, fact_scored_applications[ead_estimate])

-- Total ECL
Total ECL = 
    SUMX(fact_scored_applications, fact_scored_applications[ecl_amount])

-- ECL Coverage Ratio
ECL Coverage % = 
    DIVIDE([Total ECL], [Total EAD], 0) * 100

-- Portfolio Average PD
Avg PD = 
    AVERAGEX(fact_scored_applications, fact_scored_applications[pd_calibrated])

-- Portfolio Average Score
Avg Credit Score = 
    AVERAGEX(fact_scored_applications, fact_scored_applications[credit_score])

-- Default Rate
Actual Default Rate = 
    DIVIDE(
        COUNTROWS(FILTER(fact_scored_applications, fact_scored_applications[bad_flag] = 1)),
        COUNTROWS(fact_scored_applications),
        0
    ) * 100
```

### IFRS 9 Staging Measures

```dax
-- Stage 1 EAD
Stage 1 EAD = 
    CALCULATE([Total EAD], fact_scored_applications[ifrs9_stage] = 1)

-- Stage 2 EAD
Stage 2 EAD = 
    CALCULATE([Total EAD], fact_scored_applications[ifrs9_stage] = 2)

-- Stage 3 EAD
Stage 3 EAD = 
    CALCULATE([Total EAD], fact_scored_applications[ifrs9_stage] = 3)

-- Stage 1 EAD %
Stage 1 EAD % = 
    DIVIDE([Stage 1 EAD], [Total EAD], 0) * 100

-- ECL by Stage
ECL Stage 1 = CALCULATE([Total ECL], fact_scored_applications[ifrs9_stage] = 1)
ECL Stage 2 = CALCULATE([Total ECL], fact_scored_applications[ifrs9_stage] = 2)
ECL Stage 3 = CALCULATE([Total ECL], fact_scored_applications[ifrs9_stage] = 3)

-- Stage 2 Migration Rate
Stage 2 Migration % = 
    VAR CurrentS2 = CALCULATE([Stage 2 EAD], dim_date[is_month_end] = TRUE)
    VAR PriorS1   = CALCULATE(
                       [Stage 1 EAD],
                       DATEADD(dim_date[date_id], -1, MONTH)
                   )
    RETURN DIVIDE(CurrentS2, PriorS1, 0) * 100
```

### Model Performance Measures

```dax
-- Gini from monitoring table
Current Gini = 
    CALCULATE(
        LASTNONBLANK(fact_monitoring[gini], 1),
        FILTER(fact_monitoring, fact_monitoring[model_version] = "v1.0")
    )

-- PSI — Latest
Latest PSI = 
    CALCULATE(
        LASTNONBLANK(fact_monitoring[psi_score], 1),
        fact_monitoring[monitoring_type] = "PSI"
    )

-- RAG Status Text
PSI RAG Status = 
    VAR psi = [Latest PSI]
    RETURN
        IF(psi < 0.10, "🟢 Green — No Action",
        IF(psi < 0.25, "🟡 Amber — Monitor Closely",
                       "🔴 Red — Rebuild Required"))

-- AUC MoM Change
AUC MoM Change = 
    VAR CurrentAUC = CALCULATE(LASTNONBLANK(fact_monitoring[auc_roc], 1))
    VAR PriorAUC   = CALCULATE(
                         LASTNONBLANK(fact_monitoring[auc_roc], 1),
                         DATEADD(dim_date[date_id], -1, MONTH)
                     )
    RETURN CurrentAUC - PriorAUC
```

### Score Band Distribution

```dax
-- Count by decision band
Approve Count = 
    CALCULATE(
        COUNTROWS(fact_scored_applications),
        fact_scored_applications[decision_band] = "APPROVE"
    )

Decline Rate % = 
    DIVIDE(
        CALCULATE(COUNTROWS(fact_scored_applications),
                  fact_scored_applications[decision_band] = "DECLINE"),
        COUNTROWS(fact_scored_applications),
        0
    ) * 100

-- Refer rate
Refer Rate % = 
    DIVIDE(
        CALCULATE(COUNTROWS(fact_scored_applications),
                  fact_scored_applications[decision_band] IN {"REFER", "CONDITIONAL APPROVE"}),
        COUNTROWS(fact_scored_applications),
        0
    ) * 100
```

### FLI Scenario Analysis

```dax
-- FLI Weighted ECL
FLI Weighted ECL = 
    VAR OptimisticECL = SUMX(fact_scored_applications,
                              fact_scored_applications[pd_calibrated] * 0.75 *
                              fact_scored_applications[lgd_downturn] *
                              fact_scored_applications[ead_estimate])
    VAR BaseECL      = SUMX(fact_scored_applications,
                              fact_scored_applications[pd_calibrated] *
                              fact_scored_applications[lgd_downturn] *
                              fact_scored_applications[ead_estimate])
    VAR AdverseECL   = SUMX(fact_scored_applications,
                              fact_scored_applications[pd_calibrated] * 1.65 *
                              fact_scored_applications[lgd_downturn] *
                              fact_scored_applications[ead_estimate])
    RETURN OptimisticECL * 0.25 + BaseECL * 0.50 + AdverseECL * 0.25

-- FLI overlay impact
FLI vs Base ECL Uplift = 
    DIVIDE([FLI Weighted ECL] - [Total ECL], [Total ECL], 0) * 100
```

---

## 🎨 Dashboard Layout Recommendations

### Page 1 — Executive Risk Summary
- **Card visuals**: Total EAD, Total ECL, ECL Coverage %, Avg Credit Score
- **KPI card**: PSI RAG Status (conditional formatting Green/Amber/Red)
- **Donut chart**: EAD by IFRS 9 Stage (Stage 1/2/3)
- **Multi-row card**: AUC, Gini, KS with trend arrows

### Page 2 — Model Performance
- **Line chart**: AUC-ROC trend (12-month) with threshold reference lines (0.70, 0.75)
- **Bar chart**: PSI monthly trend with colour rules (< 0.10 green, 0.10-0.25 amber, > 0.25 red)
- **Table**: Monthly monitoring summary with conditional formatting
- **Scatter**: PD Expected vs Actual (calibration drift)

### Page 3 — IFRS 9 Provisioning
- **Stacked bar**: ECL by Stage per month (stage migration)
- **Waterfall**: ECL movement (new/resolved/migrated)
- **Clustered bar**: FLI scenario comparison (Opt/Base/Adverse/PW)
- **Matrix heatmap**: Stage migration matrix

### Page 4 — Provincial Heatmap
- **Filled map (Canada)**: EAD or default rate by province (ON/BC/QC/AB/SK/MB/NS)
- **Bar chart**: Default rate by province vs national average
- **Decomposition tree**: Default → Province → Score Band → Job Category

### Page 5 — Vintage Analysis
- **Area chart**: Default rate by origination quarter
- **Line chart**: Score distribution shift (mean score over time)
- **Table**: Score band distribution with EAD and ECL by band

---

## 🎨 Colour Theme (Corporate Canadian Banking)

| Element | Hex |
|---------|-----|
| Primary Navy | `#003366` |
| Secondary Blue | `#2563EB` |
| Positive/Good | `#16A34A` |
| Warning/Amber | `#D97706` |
| Alert/Red | `#DC2626` |
| Background | `#F8FAFC` |
| Card Background | `#FFFFFF` |
| Border | `#E2E8F0` |

Set in Power BI: **View → Themes → Customize → Paste JSON**

```json
{
  "name": "Canadian BFSI Risk",
  "dataColors": ["#003366","#2563EB","#16A34A","#D97706","#DC2626","#7C3AED","#0EA5E9"],
  "background": "#F8FAFC",
  "foreground": "#003366",
  "tableAccent": "#003366"
}
```

---

## 📤 Refresh Schedule

| Data Source | Refresh Frequency | Method |
|-------------|------------------|--------|
| Scored Applications | Daily (overnight) | Scheduled Refresh |
| IFRS 9 Provisions | Monthly (business day 3) | Manual + Approval |
| Model Monitoring | Monthly | Automated script |
| PSI/CSI | Monthly | Python → Excel → Power BI |

---

*Built for OSFI E-23 Model Risk Management governance. All dashboards subject to MRM review.*
