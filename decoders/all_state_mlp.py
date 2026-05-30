"""All-state MLP TPSAPU decoder."""

from __future__ import annotations

import torch
import torch.nn as nn

from .common import mlp_layers


class AllStateMLPDecoder(nn.Module):
    """
    Four-layer MLP over membrane, spikes, membrane dynamics, and spike history.

    The input is the concatenation of:
        membrane readout, current spikes, membrane delta, cumulative spikes.
    """

    input_state = "all"
    input_multiplier = 4
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
            depth=4,
            dropout=dropout,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)
