"""Q-network: flattened board (16) → 256 → 256 → 4 Q-values."""

from __future__ import annotations

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    def __init__(
        self,
        state_dim: int = 16,
        action_dim: int = 4,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
