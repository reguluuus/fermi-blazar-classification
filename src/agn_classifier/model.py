from __future__ import annotations

import torch
from torch import nn


class BlazarClassifier(nn.Module):
    """Small fully connected binary classifier for BLL vs FSRQ."""

    def __init__(
        self,
        input_features: int,
        hidden_size: int = 128,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
