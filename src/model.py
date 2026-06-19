"""Artificial Neural Network (MLP) for credit default classification."""
from __future__ import annotations

import torch
import torch.nn as nn


class CreditDefaultMLP(nn.Module):
    """A small feed-forward network that models non-linear risk patterns.

    Chosen over a linear model so that interactions between debt-service burden,
    utilisation and behaviour can be captured without manual interaction terms.
    """

    def __init__(self, in_features: int, hidden=(128, 64, 32), dropout: float = 0.3):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_features
        for h in hidden:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, 1))  # logit
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
