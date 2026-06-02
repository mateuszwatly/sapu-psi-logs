"""Transformer sequence readouts for SAPU state trajectories."""

from __future__ import annotations

import torch
import torch.nn as nn


class StateTransformerDecoder(nn.Module):
    """Transformer encoder readout over a full state sequence."""

    input_state = "membrane"
    input_multiplier = 1
    needs_sequence = True

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 10,
        model_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_multiplier: float = 4.0,
        dropout: float = 0.1,
        max_steps: int = 256,
    ) -> None:
        super().__init__()
        if model_dim <= 0:
            raise ValueError("model_dim must be positive.")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive.")
        if model_dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads.")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        if max_steps <= 0:
            raise ValueError("max_steps must be positive.")

        self.max_steps = max_steps
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = (
            nn.Identity() if input_dim == model_dim else nn.Linear(input_dim, model_dim)
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, model_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, max_steps + 1, model_dim))
        ff_dim = max(model_dim, int(round(model_dim * ff_multiplier)))
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(model_dim)
        self.classifier = nn.Linear(model_dim, num_classes)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, sequence_features: torch.Tensor) -> torch.Tensor:
        if sequence_features.dim() != 3:
            raise ValueError(
                "StateTransformerDecoder expects (batch, steps, features)."
            )
        batch_size, steps, _ = sequence_features.shape
        if steps > self.max_steps:
            raise ValueError(
                f"Received {steps} steps, but max_steps={self.max_steps}."
            )

        x = self.input_proj(self.input_norm(sequence_features))
        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed[:, : steps + 1, :]
        x = self.transformer(x)
        return self.classifier(self.output_norm(x[:, 0, :]))


class MembraneTransformerDecoder(StateTransformerDecoder):
    """Transformer readout over the membrane sequence."""

    input_state = "membrane"
    input_multiplier = 1


class SpikeTransformerDecoder(StateTransformerDecoder):
    """Transformer readout over the spike sequence."""

    input_state = "spike"
    input_multiplier = 1


class MembraneSpikeTransformerDecoder(StateTransformerDecoder):
    """Transformer readout over concatenated membrane and spike sequences."""

    input_state = "both"
    input_multiplier = 2


class AllStateTransformerDecoder(StateTransformerDecoder):
    """Transformer readout over membrane, spike, dynamics, and spike history."""

    input_state = "all"
    input_multiplier = 4
