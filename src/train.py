"""
Train and evaluate the credit-default ANN.

Pipeline:
  1. Load data + engineer features (income-to-loan, EMI-to-income, ...).
  2. Stratified train/val/test split + standardisation.
  3. Train an MLP with class-weighted BCE loss to handle class imbalance.
  4. Evaluate with ROC-AUC (primary metric) plus PR-AUC and a classification
     report. ROC-AUC is the focus because it measures *risk ranking* rather
     than raw accuracy on an imbalanced target.
"""
from __future__ import annotations

import os

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from feature_engineering import feature_matrix
from generate_data import generate
from model import CreditDefaultMLP

import pandas as pd

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data", "credit_data.csv")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


def load_data() -> "pd.DataFrame":
    if os.path.exists(DATA):
        return pd.read_csv(DATA)
    df = generate()
    os.makedirs(os.path.dirname(DATA), exist_ok=True)
    df.to_csv(DATA, index=False)
    return df


def make_loaders(X, y, batch_size=512):
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=SEED
    )
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=SEED
    )

    scaler = StandardScaler().fit(X_tr)
    X_tr, X_val, X_te = (scaler.transform(a) for a in (X_tr, X_val, X_te))

    def loader(Xa, ya, shuffle):
        ds = TensorDataset(torch.tensor(Xa), torch.tensor(ya))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    return (
        loader(X_tr, y_tr, True),
        loader(X_val, y_val, False),
        (torch.tensor(X_te), torch.tensor(y_te)),
        y_tr,
    )


@torch.no_grad()
def predict_proba(model, X):
    model.eval()
    return torch.sigmoid(model(X.to(DEVICE))).cpu().numpy()


def main(epochs: int = 40):
    df = load_data()
    X, y, cols = feature_matrix(df)
    print(f"Features ({len(cols)}): {cols}")

    train_loader, val_loader, (X_te, y_te), y_tr = make_loaders(X, y)

    model = CreditDefaultMLP(in_features=X.shape[1]).to(DEVICE)

    # Class-weighted loss for imbalance
    pos_weight = torch.tensor([(y_tr == 0).sum() / max((y_tr == 1).sum(), 1)])
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    best_auc, best_state = 0.0, None
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        # Validation ROC-AUC
        val_X = torch.cat([xb for xb, _ in val_loader])
        val_y = torch.cat([yb for _, yb in val_loader]).numpy()
        val_auc = roc_auc_score(val_y, predict_proba(model, val_X))
        scheduler.step(val_auc)

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1:
            print(f"epoch {epoch:3d} | loss {loss.item():.4f} | val ROC-AUC {val_auc:.4f}")

    # Restore best and evaluate on held-out test set
    model.load_state_dict(best_state)
    proba = predict_proba(model, X_te)
    y_true = y_te.numpy()

    roc = roc_auc_score(y_true, proba)
    pr = average_precision_score(y_true, proba)
    print("\n" + "=" * 60)
    print(f"TEST ROC-AUC : {roc:.4f}   (target ~0.87)")
    print(f"TEST PR-AUC  : {pr:.4f}")
    print("\nClassification report @ 0.5 threshold:")
    print(classification_report(y_true, (proba >= 0.5).astype(int), digits=3))

    os.makedirs(os.path.join(HERE, "..", "models"), exist_ok=True)
    torch.save(model.state_dict(), os.path.join(HERE, "..", "models", "credit_mlp.pt"))
    print("Saved model -> models/credit_mlp.pt")


if __name__ == "__main__":
    main()
