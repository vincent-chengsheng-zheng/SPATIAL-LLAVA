"""
core/model/regression_head.py

MLP regression head: maps the [LOC] token hidden state → [xc, yc, w, h].

Architecture:
    hidden_state (1, hidden_dim)
        → Linear(hidden_dim, 512) → GELU → Dropout
        → Linear(512, 256)        → GELU → Dropout
        → Linear(256, 4)
        → Sigmoid                 → [xc, yc, w, h] in (0, 1)

Why Sigmoid at the end:
    - bbox coordinates must be in [0, 1] (normalized)
    - Sigmoid naturally constrains output without clipping
    - SmoothL1 loss works well with this range

Usage:
    from core.model.regression_head import RegressionHead

    head = RegressionHead(hidden_dim=4096)
    bbox = head(hidden_state)   # (B, 4)
"""

import torch
import torch.nn as nn
from torch import Tensor


class RegressionHead(nn.Module):
    """
    MLP that maps the [LOC] token hidden state to a normalized bounding box.

    Args:
        hidden_dim : LLaVA hidden state dimension.
                     LLaVA-1.5-7B  → 4096
                     LLaVA-1.5-13B → 5120
        dropout    : Dropout probability (default 0.1)

    Input:
        x : Tensor(B, hidden_dim)  — [LOC] token hidden states

    Output:
        Tensor(B, 4)  — [xc, yc, w, h] in (0, 1)
    """

    def __init__(self, hidden_dim: int = 4096, dropout: float = 0.1):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 4),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier init for linear layers, zero bias."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x : Tensor(B, hidden_dim)

        Returns:
            Tensor(B, 4) — [xc, yc, w, h] in (0, 1)
        """
        return self.mlp(x)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
    