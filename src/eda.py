"""
Exploratory Data Analysis for the credit dataset.

Generates summary statistics and a set of plots (saved to reports/) covering
class balance, feature distributions by default status, and a correlation
heatmap of the engineered features.
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from feature_engineering import ENGINEERED_FEATURES, add_features

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data", "credit_data.csv")
REPORTS = os.path.join(HERE, "..", "reports")


def main() -> None:
    os.makedirs(REPORTS, exist_ok=True)
    df = add_features(pd.read_csv(DATA))

    print("=" * 60)
    print("SHAPE:", df.shape)
    print("\nCLASS BALANCE:")
    print(df["default"].value_counts(normalize=True).rename("proportion"))
    print("\nSUMMARY (key columns):")
    print(
        df[["annual_income", "loan_amount", "emi", "emi_to_income", "income_to_loan"]]
        .describe()
        .round(2)
    )

    # 1) Class balance
    plt.figure(figsize=(5, 4))
    sns.countplot(x="default", data=df)
    plt.title("Class balance (0 = repaid, 1 = default)")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS, "class_balance.png"), dpi=120)
    plt.close()

    # 2) EMI-to-income by default status
    plt.figure(figsize=(6, 4))
    sns.kdeplot(data=df, x="emi_to_income", hue="default", common_norm=False, fill=True)
    plt.title("EMI-to-income by default status")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS, "emi_to_income.png"), dpi=120)
    plt.close()

    # 3) Correlation heatmap of engineered features + target
    plt.figure(figsize=(7, 6))
    corr = df[ENGINEERED_FEATURES + ["default"]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Engineered feature correlations")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS, "feature_correlation.png"), dpi=120)
    plt.close()

    print(f"\nPlots written to {os.path.abspath(REPORTS)}")


if __name__ == "__main__":
    main()
