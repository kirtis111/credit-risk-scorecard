"""
setup.py — BFSI Credit Risk Scorecard Package
Canadian Banking Edition
"""
from setuptools import setup, find_packages

setup(
    name="credit-risk-scorecard-canada",
    version="1.0.0",
    description="End-to-end credit risk scorecard — Canadian BFSI (OSFI E-23 / IFRS 9)",
    author="Credit Risk Analytics Team",
    python_requires=">=3.10",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.26",
        "pandas>=2.2",
        "scikit-learn>=1.4",
        "xgboost>=2.0",
        "shap>=0.45",
        "optuna>=3.6",
        "imbalanced-learn>=0.12",
        "matplotlib>=3.8",
        "seaborn>=0.13",
        "plotly>=5.21",
        "streamlit>=1.34",
        "openpyxl>=3.1",
        "reportlab>=4.2",
        "pyyaml>=6.0",
        "joblib>=1.4",
        "scipy>=1.13",
    ],
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Financial and Insurance Industry",
    ],
)
