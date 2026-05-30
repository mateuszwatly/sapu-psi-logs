"""Linear TPSAPU decoder."""

from __future__ import annotations

import torch
import torch.nn as nn


class LinearDecoder(nn.Module):
    """Single-layer classifier over pooled membrane features."""

    input_state = "membrane"
    input_multiplier = 1
    needs_sequence = False

    def __init__(self, input_dim: int, num_classes: int = 10) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)
