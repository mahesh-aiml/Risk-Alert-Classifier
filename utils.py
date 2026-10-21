"""Shared feature-engineering helpers used by train.py and app.py."""
import pandas as pd
import numpy as np

REFERENCE_DATE = pd.Timestamp("2025-12-31")  # fixed reference so train/app behave the same

NUMERIC_COLS = [
    "age", "annual_income_inr", "credit_score", "credit_utilization_ratio",
    "missed_payments_12m", "avg_late_payment_days", "monthly_transaction_count",
    "monthly_spend_inr", "cash_advance_count_6m", "complaints_last_6m",
    "failed_login_attempts_3m", "account_tenure_months", "debt_balance_inr",
]
CATEGORIES = {
    "gender": ["Male", "Female", "Other"],
    "region": ["North", "South", "East", "West", "Central"],
    "employment_type": ["Salaried", "Self-Employed", "Student", "Unemployed", "Retired"],
}


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Raw customer dataframe -> model feature matrix (NaNs kept for KNN imputer)."""
    out = pd.DataFrame(index=df.index)
    for c in NUMERIC_COLS:
        out[c] = pd.to_numeric(df[c], errors="coerce")
    d = pd.to_datetime(df["last_transaction_date"], errors="coerce")
    out["days_since_last_txn"] = (REFERENCE_DATE - d).dt.days
    # Domain ratios (NaN propagates if inputs missing)
    out["debt_to_income"] = out["debt_balance_inr"] / out["annual_income_inr"].replace(0, np.nan)
    # One-hot with fixed categories; a missing value -> all zeros for that field
    for col, cats in CATEGORIES.items():
        for cat in cats:
            out[f"{col}_{cat}"] = (df[col] == cat).astype(int)
    return out


FEATURE_COLUMNS = (
    NUMERIC_COLS + ["days_since_last_txn", "debt_to_income"]
    + [f"{c}_{v}" for c, vs in CATEGORIES.items() for v in vs]
)
