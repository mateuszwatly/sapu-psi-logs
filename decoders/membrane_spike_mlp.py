"""Membrane+spike MLP TPSAPU decoder."""

from __future__ import annotations

import torch
import torch.nn as nn

from .common import mlp_layers


class MembraneSpikeMLPDecoder(nn.Module):
    """Three-layer MLP over concatenated membrane and spike features."""

    input_state = "both"
    input_multiplier = 2
    needs_sequence = False

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 10,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = mlp_layers(
            input_dim,
            hidden_dim,
            num_classes,
            depth=3,
            dropout=dropout,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)
