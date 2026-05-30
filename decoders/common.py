"""Shared helpers for TPSAPU decoders."""

from __future__ import annotations

import torch.nn as nn


def mlp_layers(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    *,
    depth: int,
    dropout: float,
) -> nn.Sequential:
    if depth < 2:
        raise ValueError("MLP depth must be at least 2.")

    layers: list[nn.Module] = [nn.LayerNorm(input_dim)]
    in_dim = input_dim
    for _ in range(depth - 1):
        layers.extend(
            [
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
        )
        in_dim = hidden_dim
    layers.append(nn.Linear(in_dim, output_dim))
    return nn.Sequential(*layers)
