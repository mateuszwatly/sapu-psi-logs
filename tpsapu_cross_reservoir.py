"""
Compressed cross-reservoir communication for TPSAPU.

This module keeps the base TPSAPU recurrent update, then adds a cheap
low-rank current that lets spikes from one tau reservoir influence the others.
The dense all-neuron alternative would scale with ``(taus * dim)^2``; this
factorization scales with ``taus * dim * rank`` plus a small tau mixing matrix.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tpsapu import TPSAPUBackbone


class LowRankCrossReservoir(nn.Module):
    """Low-rank cross-tau projection for spike tensors shaped (batch, tau, dim)."""

    def __init__(self, tau_count: int, reservoir_dim: int, rank: int = 16) -> None:
        super().__init__()
        if tau_count <= 1:
            raise ValueError("cross-reservoir communication requires at least two taus.")
        if reservoir_dim <= 0:
            raise ValueError("reservoir_dim must be positive.")
        if rank <= 0:
            raise ValueError("rank must be positive.")

        self.tau_count = int(tau_count)
        self.reservoir_dim = int(reservoir_dim)
        self.rank = min(int(rank), self.reservoir_dim)

        self.down = nn.Parameter(torch.empty(self.rank, self.reservoir_dim))
        self.tau_mix = nn.Parameter(torch.empty(self.tau_count, self.tau_count))
        self.up = nn.Parameter(torch.empty(self.reservoir_dim, self.rank))
        self.register_buffer(
            "_offdiag_tau_mask",
            torch.ones(self.tau_count, self.tau_count)
            - torch.eye(self.tau_count),
            persistent=False,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.down)
        nn.init.xavier_uniform_(self.up)
        nn.init.uniform_(self.tau_mix, -0.01, 0.01)

    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        compressed = F.linear(spikes, self.down)
        tau_weight = self.tau_mix * self._offdiag_tau_mask.to(
            device=spikes.device,
            dtype=spikes.dtype,
        )
        mixed = torch.einsum("bkr,tk->btr", compressed, tau_weight)
        return F.linear(mixed, self.up)


class CompressedCrossReservoirTPSAPUBackbone(TPSAPUBackbone):
    """
    TPSAPU variant with compressed communication between tau reservoirs.

    The normal recurrent projection still runs inside each tau reservoir. This
    variant adds a low-rank off-diagonal tau projection from previous spikes:

        spikes -> rank channels -> tau mixing -> reservoir channels

    ``cross_gain`` scales that extra current and defaults to 0.1.
    """

    def __init__(
        self,
        *args,
        cross_rank: int = 16,
        cross_gain: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if cross_gain < 0.0:
            raise ValueError("cross_gain must be non-negative.")
        self.cross_gain = float(cross_gain)
        self.cross_reservoir = LowRankCrossReservoir(
            tau_count=len(self.taus),
            reservoir_dim=self.reservoir_dim,
            rank=cross_rank,
        )

    def _recurrent_projection(
        self,
        spikes: torch.Tensor,
        recurrent_weight: torch.Tensor,
    ) -> torch.Tensor:
        recurrent = super()._recurrent_projection(spikes, recurrent_weight)
        return recurrent + self.cross_gain * self.cross_reservoir(spikes)


__all__ = [
    "CompressedCrossReservoirTPSAPUBackbone",
    "LowRankCrossReservoir",
]
