"""Class LIF spike-count TPSAPU decoder."""

from __future__ import annotations

import torch
import torch.nn as nn

from .spiking import functional, neuron, require_spikingjelly, surrogate


class ClassLIFSpikeCountDecoder(nn.Module):
    """
    Ten class LIF neurons with spike-count output.

    Each timestep projects concatenated membrane+spike features into 10 class
    currents. The class LIF neurons run over the whole sequence and the output
    logits are the per-class spike counts.
    """

    input_state = "both"
    input_multiplier = 2
    needs_sequence = True

    def __init__(self, input_dim: int, num_classes: int = 10) -> None:
        super().__init__()
        require_spikingjelly(self.__class__.__name__)
        self.num_classes = num_classes
        self.norm = nn.LayerNorm(input_dim)
        self.current_proj = nn.Linear(input_dim, num_classes)
        self.class_lif = neuron.LIFNode(
            tau=2.0,
            v_threshold=1.0,
            v_reset=0.0,
            detach_reset=True,
            surrogate_function=surrogate.ATan(),
        )

    def forward(self, sequence_features: torch.Tensor) -> torch.Tensor:
        if sequence_features.dim() != 3:
            raise ValueError(
                "ClassLIFSpikeCountDecoder expects (batch, steps, features)."
            )

        functional.reset_net(self.class_lif)
        class_spikes = []
        for step in range(sequence_features.size(1)):
            currents = self.current_proj(self.norm(sequence_features[:, step, :]))
            class_spikes.append(self.class_lif(currents))
        return torch.stack(class_spikes, dim=1).sum(dim=1)
