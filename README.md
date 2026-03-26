# BFSI Credit Risk Scorecard Model — Canadian Banking Edition

## Project Overview

This repository implements a production-grade credit risk scorecard for the Canadian BFSI sector, covering the full model development lifecycle as practiced across major financial institutions and credit unions.

| Phase | Description |
|-------|-------------|
| **Data Ingestion** | Kaggle HMEQ dataset + SQL feature store |
| **Feature Engineering** | WoE binning, IV selection, monotonicity |
| **Champion Model** | Logistic Regression Scorecard (300–850 scale) |
| **Challenger Model** | XGBoost + SHAP explainability |
| **Risk Parameters** | PD, LGD, EAD per IFRS 9 / Basel III |
| **Provisioning** | Stage 1/2/3 ECL calculations |
| **Validation** | Gini, KS, AUC, PSI, CSI — OSFI E-23 aligned |
| **Reporting** | Regulatory PDF + Power BI dashboard |
| **Deployment** | Streamlit loan officer simulation app |

---

## Dataset

**Primary:** [Home Equity Loan Default (HMEQ)](https://www.kaggle.com/datasets/ajay1735/hmeq-data)

```bash
pip install kaggle
kaggle datasets download -d ajay1735/hmeq-data -p data/raw/ --unzip
```

**Alternative (richer features):** [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk)

| Variable | Description |
|----------|-------------|
| BAD | Target: 1=Default, 0=Repaid |
| LOAN | Loan request amount (CAD) |
| MORTDUE | Amount due on existing mortgage |
| VALUE | Current property value |
| REASON | Loan reason (DebtCon / HomeImp) |
| JOB | Occupational category |
| YOJ | Years at present job |
| DEROG | Major derogatory reports |
| DELINQ | Delinquent credit lines |
| CLAGE | Age of oldest credit line (months) |
| NINQ | Recent credit inquiries |
| CLNO | Number of credit lines |
| DEBTINC | Debt-to-income ratio |

---

## Repository Structure

```
credit-risk-scorecard/
├── notebooks/
│   ├── 01_EDA_and_Data_Quality.ipynb
│   ├── 02_Feature_Engineering_WoE.ipynb
│   ├── 03_Logistic_Regression_Scorecard.ipynb
│   ├── 04_XGBoost_Challenger_SHAP.ipynb
│   ├── 05_PD_LGD_EAD_IFRS9.ipynb
│   └── 06_Model_Validation_Report.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── woe_binning.py
│   ├── feature_engineering.py
│   ├── scorecard.py
│   ├── model_training.py
│   ├── model_validation.py
│   ├── ifrs9_calculations.py
│   └── report_generator.py
├── sql/
│   ├── 01_create_tables.sql
│   ├── 02_feature_engineering.sql
│   └── 03_model_monitoring.sql
├── streamlit_app/
│   ├── app.py
│   └── pages/
│       ├── 01_loan_origination.py
│       ├── 02_portfolio_monitoring.py
│       └── 03_ifrs9_provisioning.py
├── reports/
│   └── model_validation_report.py
├── excel/
│   └── generate_scorecard_excel.py
├── powerbi/
│   └── README_PowerBI.md
├── config/config.yaml
├── requirements.txt
└── setup.py
```

---

## Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/credit-risk-scorecard-canada.git
cd credit-risk-scorecard-canada
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
kaggle datasets download -d ajay1735/hmeq-data -p data/raw/ --unzip
jupyter lab                        # Run notebooks 01-06 in order
cd streamlit_app && streamlit run app.py
```

---

## 🇨🇦 Canadian Regulatory Alignment

| Regulation | Implementation |
|-----------|----------------|
| OSFI E-23 | Model risk governance, validation independence |
| OSFI B-20 | Residential mortgage underwriting standards |
| IFRS 9 | 3-stage ECL provisioning (12-month vs lifetime) |
| Basel III / AIRB | PD/LGD/EAD parameter estimation |
| PIPEDA | Prohibited variable exclusion |

---

## Expected Performance Benchmarks

| Metric | Champion (LR) | Challenger (XGB) | Threshold |
|--------|---------------|------------------|-----------|
| AUC-ROC | ~0.83 | ~0.87 | > 0.75 |
| Gini | ~0.66 | ~0.74 | > 0.50 |
| KS Statistic | ~0.52 | ~0.58 | > 0.40 |

---

## Author 

Kirti Sinha | Data Analyst | Credit Risk Analyst
