"""
core/model/regression_head.py

MLP regression head that maps LLM hidden states to bounding box coordinates.

Takes the hidden state at the [LOC] token position and regresses it into
4 continuous values [x_center, y_center, width, height], all normalized to [0, 1].

Architecture:
    hidden_state (4096,) → Linear(4096, 512) → ReLU → Linear(512, 4) → Sigmoid

Usage:
    from core.model.regression_head import RegressionHead

    head = RegressionHead(hidden_size=4096)
    hidden = torch.randn(8, 4096)   # batch of 8 [LOC] hidden states
    bbox = head(hidden)             # (8, 4) — normalized [x, y, w, h]
"""

import torch.nn as nn
from torch import Tensor


class RegressionHead(nn.Module):
    """
    Lightweight MLP that converts a single hidden state vector into
    a normalized bounding box [x_center, y_center, width, height].

    All output values are in [0, 1] via Sigmoid activation.
    Parameter count: ~0.5M — negligible overhead on top of LLaVA-7B.

    Args:
        hidden_size  : Dimensionality of the LLM hidden state (default: 4096
                       for LLaVA-v1.5-7B based on Vicuna-7B backbone)
        intermediate : Size of the hidden layer (default: 512)
        dropout      : Dropout rate applied before the output layer (default: 0.1)
    """

    def __init__(
        self,
        hidden_size: int = 4096,
        intermediate: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate = intermediate

        self.net = nn.Sequential(
            nn.Linear(hidden_size, intermediate),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate, 4),
            nn.Sigmoid(),           # output in [0, 1]
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """
        Initialize weights with small values so initial predictions
        are near the center of the image, not random extreme values.
        This gives a more stable starting point for training.
        """
        for module in self.net.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, hidden_state: Tensor) -> Tensor:
        """
        Args:
            hidden_state : Tensor of shape (batch_size, hidden_size)
                           Hidden state extracted at the [LOC] token position.

        Returns:
            Tensor of shape (batch_size, 4)
            Each row is [x_center, y_center, width, height] in [0, 1].
        """
        return self.net(hidden_state)

    def count_parameters(self) -> int:
        """Return number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
