"""
Feature engineering for credit default prediction.

Creates ratio features that capture repayment capacity better than the raw
columns, notably:
  * income-to-loan ratio
  * EMI-to-income ratio (debt-service burden)
plus a few supporting interaction / behavioural features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RAW_FEATURES = [
    "age",
    "annual_income",
    "monthly_income",
    "employment_years",
    "loan_amount",
    "loan_term_months",
    "interest_rate",
    "emi",
    "num_existing_loans",
    "credit_utilization",
    "num_late_payments_12m",
    "credit_history_years",
]

ENGINEERED_FEATURES = [
    "income_to_loan",
    "emi_to_income",
    "loan_to_income",
    "disposable_income",
    "delinquency_rate",
    "debt_load",
]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    eps = 1e-6

    # Core repayment-capacity ratios highlighted in the resume
    df["income_to_loan"] = df["annual_income"] / (df["loan_amount"] + eps)
    df["emi_to_income"] = df["emi"] / (df["monthly_income"] + eps)

    # Supporting features
    df["loan_to_income"] = df["loan_amount"] / (df["annual_income"] + eps)
    df["disposable_income"] = df["monthly_income"] - df["emi"]
    df["delinquency_rate"] = df["num_late_payments_12m"] / (
        df["credit_history_years"] + 1
    )
    df["debt_load"] = df["num_existing_loans"] * df["credit_utilization"]

    # Clip pathological ratios so the network sees a sane range
    df["income_to_loan"] = df["income_to_loan"].clip(upper=50)
    df["emi_to_income"] = df["emi_to_income"].clip(upper=5)
    return df


def feature_matrix(df: pd.DataFrame):
    """Return (X, y) using raw + engineered features."""
    df = add_features(df)
    cols = RAW_FEATURES + ENGINEERED_FEATURES
    X = df[cols].astype(np.float32).values
    y = df["default"].astype(np.float32).values
    return X, y, cols
