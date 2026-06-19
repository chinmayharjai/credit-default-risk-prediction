# Credit Default Risk Prediction

A supervised credit-default prediction model built on customer financial and
behavioural data. The project performs EDA and feature engineering, then models
non-linear risk patterns with an **artificial neural network** (instead of a
linear model) to improve separation between defaulters and non-defaulters.
Performance is measured with **ROC-AUC** to handle class imbalance and focus on
risk ranking rather than raw accuracy.

> **Target outcome:** ROC-AUC ≈ **0.87** on held-out test data.

📓 **Start here:** [`notebooks/01_credit_default_risk.ipynb`](notebooks/01_credit_default_risk.ipynb) — a
fully-explained, executed walkthrough (problem → EDA → features → ANN → ROC-AUC → Q&A) that renders with
charts directly on GitHub.

## Highlights
- **EDA** — class-balance, distribution and correlation analysis (`src/eda.py`).
- **Feature engineering** — repayment-capacity ratios including
  **income-to-loan** and **EMI-to-income**, plus disposable income, delinquency
  rate and debt load (`src/feature_engineering.py`).
- **Model** — class-weighted MLP (PyTorch) with BatchNorm + Dropout
  (`src/model.py`, `src/train.py`).
- **Evaluation** — ROC-AUC (primary), PR-AUC and classification report.

## Project structure
```
credit-default-risk-prediction/
├── data/                 # generated dataset (csv)
├── models/               # saved model weights
├── reports/              # EDA plots
├── src/
│   ├── generate_data.py      # synthetic imbalanced credit dataset
│   ├── feature_engineering.py# ratio + behavioural features
│   ├── eda.py                # exploratory analysis & plots
│   ├── model.py              # ANN (MLP) architecture
│   └── train.py              # train + evaluate (ROC-AUC)
└── requirements.txt
```

> Real lending data cannot be shared, so `generate_data.py` produces a
> statistically plausible, intentionally **imbalanced** dataset where default
> probability depends on debt-service burden, utilisation and behaviour.

## Setup
```bash
pip install -r requirements.txt
```

## Run
```bash
cd src
python generate_data.py     # -> data/credit_data.csv
python eda.py               # -> reports/*.png
python train.py             # trains ANN, prints ROC-AUC, saves models/credit_mlp.pt
```

## Why ROC-AUC?
With an imbalanced target, accuracy is misleading (a model predicting "no
default" for everyone scores high). ROC-AUC measures how well the model *ranks*
risky customers above safe ones, which is what a credit risk team actually
needs.
