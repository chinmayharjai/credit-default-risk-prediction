# Credit Default Risk Prediction

> A supervised machine-learning project that predicts the probability a loan customer will **default**,
> using their financial and behavioural data. Built with an **Artificial Neural Network (ANN)** in PyTorch
> and evaluated with **ROC-AUC** — the right metric for an imbalanced risk problem.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-ANN-red)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

📓 **Start here:** [`notebooks/01_credit_default_risk.ipynb`](notebooks/01_credit_default_risk.ipynb) — a
fully-explained, **already-executed** walkthrough that renders with all charts directly on GitHub
(problem → EDA → feature engineering → ANN → ROC-AUC → Q&A).

---

## Table of contents
1. [The problem](#1-the-problem)
2. [Why this matters (business context)](#2-why-this-matters-business-context)
3. [The dataset](#3-the-dataset)
4. [Approach & methodology](#4-approach--methodology)
5. [Model architecture](#5-model-architecture)
6. [Results](#6-results)
7. [Project structure](#7-project-structure)
8. [Setup & installation](#8-setup--installation)
9. [How to run](#9-how-to-run)
10. [FAQ — questions an interviewer might ask](#10-faq--questions-an-interviewer-might-ask)
11. [Limitations](#11-limitations)
12. [Future work](#12-future-work)
13. [Tech stack](#13-tech-stack)

---

## 1. The problem

Lenders make money when borrowers repay and lose money when they default. The goal of this project is to
answer one question at the moment of loan application:

> *"Given everything we know about a customer, how likely are they to default on this loan?"*

This is framed as a **binary classification** problem (`default = 1` vs `repaid = 0`). But the true
objective is not a hard yes/no label — it is a **risk score** that lets the credit team **rank** customers
from safest to riskiest. That ranking drives real decisions: approve/decline, credit limit, and interest
rate (risk-based pricing).

## 2. Why this matters (business context)

- **Risk-based pricing:** higher predicted risk → higher interest rate to compensate for expected losses.
- **Approval decisions:** customers above a risk threshold are declined or asked for collateral.
- **Capital & provisioning:** expected default rates feed loss provisioning and regulatory capital.
- **Portfolio monitoring:** ranking customers helps target collections and early-warning interventions.

Because the cost of a missed defaulter (money lost) usually far exceeds the cost of declining a good
customer (opportunity lost), the model is optimised for **ranking quality**, not raw accuracy.

## 3. The dataset

Real lending data is confidential and cannot be shared, so this project ships a **synthetic data
generator** ([`src/generate_data.py`](src/generate_data.py)) that produces a statistically realistic and
**intentionally imbalanced** dataset (~24% defaulters, like a real loan book). Crucially, the default
label is **not random** — it is driven by a latent risk function of debt-service burden, credit
utilisation and past behaviour, plus irreducible noise. This means the patterns the model learns are
genuine, not artefacts.

The generated file is committed at [`data/credit_data.csv`](data/credit_data.csv) (30,000 rows).

### Data dictionary

| Column | Type | Meaning |
|---|---|---|
| `customer_id` | int | Unique identifier |
| `age` | int | Customer age (years) |
| `annual_income` / `monthly_income` | float | Income (annual / monthly) |
| `employment_years` | float | Length of current employment |
| `loan_amount` | float | Requested loan principal |
| `loan_term_months` | int | Loan tenure in months |
| `interest_rate` | float | Annual interest rate (%) |
| `emi` | float | Equated Monthly Instalment (computed via the standard amortisation formula) |
| `num_existing_loans` | int | Other live loans |
| `credit_utilization` | float | Fraction of available credit in use (0–1) |
| `num_late_payments_12m` | int | Late payments in the last 12 months |
| `credit_history_years` | int | Length of credit history |
| **`default`** | int | **Target** — 1 if the customer defaulted, else 0 |

## 4. Approach & methodology

The pipeline mirrors how a credit-risk data scientist would actually work:

1. **Exploratory Data Analysis** ([`src/eda.py`](src/eda.py)) — confirm the class imbalance, compare
   feature distributions for defaulters vs non-defaulters, and inspect correlations. EDA shows defaulters
   skew toward a **high EMI-to-income ratio** and **high credit utilisation**.
2. **Feature engineering** ([`src/feature_engineering.py`](src/feature_engineering.py)) — raw columns
   don't capture *repayment capacity* well, so we add ratio features that express burden relative to
   ability to pay:
   - **`emi_to_income`** = EMI ÷ monthly income (how stretched the monthly budget is)
   - **`income_to_loan`** = annual income ÷ loan amount (ability to cover the loan)
   - plus `loan_to_income`, `disposable_income`, `delinquency_rate`, `debt_load`
3. **Modelling** — a class-weighted **ANN** learns the non-linear interactions between these features
   (see below).
4. **Evaluation** — **ROC-AUC** (primary), PR-AUC, a ROC curve, and a confusion matrix.

### Why a neural network instead of logistic regression?

A linear model assumes risk is a straight-line combination of features. Real default risk is
**non-linear and interactive** — a large EMI is only dangerous *when income is low*; utilisation interacts
with late payments. An ANN learns these interactions automatically, improving the separation between
defaulters and non-defaulters. (Logistic regression remains an excellent, interpretable baseline.)

## 5. Model architecture

A compact multilayer perceptron defined in [`src/model.py`](src/model.py):

```
Input (18 features)
  → [Linear(128) → BatchNorm → ReLU → Dropout(0.3)]
  → [Linear(64)  → BatchNorm → ReLU → Dropout(0.3)]
  → [Linear(32)  → BatchNorm → ReLU → Dropout(0.3)]
  → Linear(1)  (logit)
```

Training details ([`src/train.py`](src/train.py)):
- **Loss:** `BCEWithLogitsLoss` with a **`pos_weight`** to counter class imbalance.
- **Optimiser:** Adam (lr 1e-3) with weight decay; `ReduceLROnPlateau` on validation ROC-AUC.
- **Regularisation:** BatchNorm + Dropout + weight decay; **early stopping** keeps the best
  validation-AUC checkpoint.
- **Splits:** stratified 70/15/15 train/val/test; features standardised with `StandardScaler` fit on
  train only.

## 6. Results

| Metric | Value |
|---|---|
| **Test ROC-AUC** | **≈ 0.90** |
| Test PR-AUC | ≈ 0.77 |
| Resume target (real data) | ~0.87 |

An ROC-AUC of ~0.90 means that, given a random defaulter and a random non-defaulter, the model assigns the
defaulter a higher risk score ~90% of the time. The notebook visualises the ROC curve and confusion
matrix and discusses threshold selection.

> **Note on the number:** ~0.90 is achieved on the *synthetic* test set. The resume figure of 0.87 comes
> from the original work on real data. The pipeline is identical — drop a real dataset into
> `data/credit_data.csv` and re-run to reproduce on real data.

## 7. Project structure

```
credit-default-risk-prediction/
├── data/
│   └── credit_data.csv             # generated dataset (30k rows)
├── models/                         # saved weights (git-ignored, recreated by training)
├── reports/                        # EDA plots (generated)
├── notebooks/
│   └── 01_credit_default_risk.ipynb  # explained, executed walkthrough
├── src/
│   ├── generate_data.py            # synthetic imbalanced credit dataset
│   ├── feature_engineering.py      # repayment-capacity ratio features
│   ├── eda.py                      # exploratory analysis & plots
│   ├── model.py                    # ANN (MLP) architecture
│   └── train.py                    # train + evaluate (ROC-AUC)
├── requirements.txt
├── LICENSE
└── README.md
```

## 8. Setup & installation

```bash
git clone https://github.com/chinmayharjai/credit-default-risk-prediction.git
cd credit-default-risk-prediction
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 9. How to run

```bash
cd src
python generate_data.py     # -> data/credit_data.csv  (skip if already present)
python eda.py               # -> reports/*.png
python train.py             # trains the ANN, prints ROC-AUC, saves models/credit_mlp.pt
```

Or open the notebook for the full narrative:
```bash
jupyter notebook notebooks/01_credit_default_risk.ipynb
```

## 10. FAQ — questions an interviewer might ask

**Q: Why ROC-AUC instead of accuracy?**
The data is imbalanced (~24% defaulters). A model that always predicts "no default" scores ~76% accuracy
while catching zero defaulters. ROC-AUC is threshold-independent and measures how well the model *ranks*
risky customers above safe ones — exactly what the business needs.

**Q: What's the difference between ROC-AUC and PR-AUC, and why report both?**
ROC-AUC summarises the trade-off between true-positive and false-positive rates across all thresholds.
PR-AUC focuses on the positive (minority) class and is more sensitive when positives are rare. Reporting
both gives a fuller picture on imbalanced data.

**Q: How did you handle class imbalance?**
A class-weighted loss (`pos_weight` = #negatives / #positives) so the rarer defaulter class contributes
proportionally more to the loss. Stratified splits keep the default rate consistent across train/val/test.

**Q: Why engineer ratio features instead of feeding raw columns?**
Ratios encode domain knowledge about *repayment capacity*. `emi_to_income` directly expresses
debt-service burden; `income_to_loan` expresses coverage. They give the model a strong, interpretable
signal and showed the clearest separation in EDA.

**Q: How do you prevent overfitting?**
Dropout, BatchNorm and weight decay during training; early stopping on validation ROC-AUC; and saving the
best checkpoint rather than the last epoch.

**Q: How would you turn a probability into an approve/decline decision?**
Choose a threshold from business economics, not 0.5. Because the loss from a missed defaulter (loss given
default) usually dwarfs the cost of declining a good customer, the optimal threshold is tuned on the
expected cost of false negatives vs false positives.

**Q: How would you make this production-ready?**
Probability calibration (e.g. isotonic/Platt), monitoring for data/concept drift, periodic retraining,
fairness auditing across protected groups, and explainability (SHAP) so each decision has a reason code
for regulators and customers.

**Q: Is the synthetic data a limitation?**
Yes — real labels would shift the exact numbers. But the methodology (EDA → feature engineering → ANN →
ROC-AUC) is identical, and the data generator encodes realistic, non-trivial risk relationships so the
learned patterns are meaningful.

**Q: Could a simpler model work?**
Often yes — logistic regression or gradient-boosted trees (XGBoost/LightGBM) are strong, interpretable
baselines and frequently competitive. The ANN is used here to model non-linear interactions; in practice
you'd benchmark all three and pick the best risk/interpretability trade-off.

## 11. Limitations

- Synthetic data: relationships are designed, so absolute metrics won't match a specific real portfolio.
- No probability calibration step yet (scores rank well but aren't calibrated probabilities).
- No explainability/fairness module yet (see future work).
- Single model; no formal benchmarking against tree-based baselines in the committed code.

## 12. Future work

- Add **SHAP** explainability for per-decision reason codes.
- **Calibrate** probabilities (isotonic regression / Platt scaling).
- **Threshold optimisation** driven by an explicit cost matrix.
- Benchmark against **XGBoost/LightGBM** and ensemble.
- **Fairness auditing** across demographic groups.

## 13. Tech stack

**Python**, **PyTorch** (ANN), **scikit-learn** (splits, scaling, metrics), **pandas/NumPy** (data),
**matplotlib/seaborn** (visualisation), **Jupyter** (walkthrough).

---

*Real banking data cannot be shared; this project uses realistic synthetic data so the full pipeline is
reproducible end-to-end. Licensed under [MIT](LICENSE).*
