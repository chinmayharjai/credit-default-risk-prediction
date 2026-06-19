"""
Synthetic credit dataset generator.

Produces a realistic, *imbalanced* customer financial & behavioural dataset that
mirrors the kind of data used for credit default risk modelling. Real lending
data cannot be shared, so this generator creates statistically plausible records
where default probability depends on income, existing debt burden and behaviour.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)


def generate(n: int = 30_000) -> pd.DataFrame:
    age = RNG.integers(21, 70, size=n)

    # Monthly income (right-skewed)
    annual_income = np.round(RNG.lognormal(mean=12.6, sigma=0.5, size=n), 2)
    annual_income = np.clip(annual_income, 80_000, 5_000_000)

    employment_years = np.clip(RNG.normal(8, 5, size=n), 0, 45).round(1)

    # Loan characteristics
    loan_amount = np.round(annual_income * RNG.uniform(0.2, 2.5, size=n), 2)
    loan_term_months = RNG.choice([12, 24, 36, 48, 60, 84], size=n)
    interest_rate = np.round(RNG.uniform(8.0, 24.0, size=n), 2)

    # Approx EMI using standard amortisation formula
    monthly_rate = interest_rate / 1200.0
    emi = (
        loan_amount
        * monthly_rate
        * (1 + monthly_rate) ** loan_term_months
        / ((1 + monthly_rate) ** loan_term_months - 1)
    )
    emi = np.round(emi, 2)

    monthly_income = annual_income / 12.0

    # Behavioural features
    num_existing_loans = RNG.poisson(1.2, size=n)
    credit_utilization = np.clip(RNG.beta(2, 5, size=n), 0, 1).round(3)
    num_late_payments_12m = RNG.poisson(0.8, size=n)
    credit_history_years = np.clip(age - 20 - RNG.integers(0, 5, size=n), 0, None)

    df = pd.DataFrame(
        {
            "customer_id": np.arange(1, n + 1),
            "age": age,
            "annual_income": annual_income,
            "monthly_income": monthly_income.round(2),
            "employment_years": employment_years,
            "loan_amount": loan_amount,
            "loan_term_months": loan_term_months,
            "interest_rate": interest_rate,
            "emi": emi,
            "num_existing_loans": num_existing_loans,
            "credit_utilization": credit_utilization,
            "num_late_payments_12m": num_late_payments_12m,
            "credit_history_years": credit_history_years,
        }
    )

    # Latent default risk score -> probability -> imbalanced label
    z = (
        -6.5
        + 2.6 * (df.emi / df.monthly_income)
        + 1.8 * (df.loan_amount / df.annual_income)
        + 1.5 * df.credit_utilization
        + 0.35 * df.num_late_payments_12m
        + 0.20 * df.num_existing_loans
        - 0.04 * df.employment_years
        - 0.015 * df.credit_history_years
        + RNG.normal(0, 0.6, size=n)  # irreducible noise
    )
    prob_default = 1 / (1 + np.exp(-z))
    df["default"] = (RNG.uniform(size=n) < prob_default).astype(int)

    return df


if __name__ == "__main__":
    import os

    out = os.path.join(os.path.dirname(__file__), "..", "data", "credit_data.csv")
    data = generate()
    data.to_csv(out, index=False)
    rate = data["default"].mean()
    print(f"Wrote {len(data):,} rows to {os.path.abspath(out)}")
    print(f"Default rate: {rate:.2%}  (class imbalance is intentional)")
