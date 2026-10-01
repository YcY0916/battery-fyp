"""LSTM SOC estimator: LSTM -> output of the last time step -> Linear(H, 1)."""

from __future__ import annotations

import torch
from torch import nn


class LSTMSOC(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int, input_size: int = 3):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=0.0)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, W, 3) -> (batch,)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
