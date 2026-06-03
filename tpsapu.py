"""
Reusable TPSAPU backbone.

TPSAPU is a shared-topology parallel spiking unit: several LIF reservoirs run
in parallel with different time constants while sharing one input projection
and one recurrent matrix. This file is intentionally import-safe; it defines
model classes only and does not download data or start training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from spikingjelly.activation_based import functional, neuron, surrogate
except ImportError as exc:  # pragma: no cover - exercised only without dependency
    functional = None
    neuron = None
    surrogate = None
    _SPIKINGJELLY_IMPORT_ERROR = exc
else:
    _SPIKINGJELLY_IMPORT_ERROR = None


def _require_spikingjelly() -> None:
    if _SPIKINGJELLY_IMPORT_ERROR is not None:
        raise ImportError(
            "TPSAPUBackbone requires spikingjelly. Install it with "
            "`pip install spikingjelly` before constructing the backbone."
        ) from _SPIKINGJELLY_IMPORT_ERROR


@dataclass(frozen=True)
class TPSAPUBackboneConfig:
    """Configuration for :class:`TPSAPUBackbone`."""

    input_dim: int
    reservoir_dim: int = 64
    taus: tuple[float, ...] = (1.1, 8.0, 64.0)
    recurrent_drop_p: float = 0.0
    input_hidden_dim: int | None = None
    detach_recurrent_state: bool = True
    output_norm: bool = True


class SharedReservoir(nn.Module):
    """
    One LIF reservoir using externally supplied shared weights.

    Each reservoir owns its neuron state and tau, but not its input or
    recurrent weights. This is what lets all reservoirs share topology while
    still operating at different time scales.
    """

    def __init__(
        self,
        reservoir_dim: int,
        tau: float,
        *,
        detach_recurrent_state: bool = True,
    ) -> None:
        super().__init__()
        _require_spikingjelly()

        self.reservoir_dim = reservoir_dim
        self.detach_recurrent_state = detach_recurrent_state
        self.lif = neuron.LIFNode(
            tau=tau,
            v_threshold=1.0,
            v_reset=0.0,
            detach_reset=True,
            surrogate_function=surrogate.ATan(),
        )
        self._last_spikes: torch.Tensor | None = None

    def forward(
        self,
        x: torch.Tensor,
        input_proj: nn.Linear,
        recurrent_weight: torch.Tensor,
        neuron_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = x.size(0)
        self._reset_if_batch_changed(batch_size)

        if self._last_spikes is None:
            recurrent = x.new_zeros(batch_size, self.reservoir_dim)
        else:
            recurrent = F.linear(self._last_spikes, recurrent_weight)

        current = input_proj(x) + recurrent
        if neuron_mask is not None:
            current = current * neuron_mask.to(dtype=current.dtype)

        spikes = self.lif(current)
        if neuron_mask is not None:
            spikes = spikes * neuron_mask.to(dtype=spikes.dtype)
        self._last_spikes = spikes.detach() if self.detach_recurrent_state else spikes
        return spikes

    def membrane(self) -> torch.Tensor:
        v = self.lif.v
        if isinstance(v, torch.Tensor):
            return v
        if self._last_spikes is None:
            raise RuntimeError("Reservoir membrane is unavailable before forward().")
        return self._last_spikes.new_zeros(self._last_spikes.shape)

    def reset_state(self) -> None:
        self._last_spikes = None
        functional.reset_net(self.lif)

    def _reset_if_batch_changed(self, batch_size: int) -> None:
        last_batch = self._last_spikes.size(0) if self._last_spikes is not None else None
        membrane = self.lif.v
        membrane_batch = membrane.size(0) if isinstance(membrane, torch.Tensor) else None

        if last_batch not in (None, batch_size) or membrane_batch not in (
            None,
            batch_size,
        ):
            self.reset_state()


class TPSAPUBackbone(nn.Module):
    """
    Generic encoder-ready TPSAPU feature extractor.

    Expected input shape is ``(batch, steps, input_dim)``. A task-specific
    encoder should convert raw data into that sequence format; a task-specific
    decoder can then consume the pooled backbone features.
    """

    def __init__(
        self,
        input_dim: int,
        reservoir_dim: int = 64,
        taus: Sequence[float] = (1.1, 8.0, 64.0),
        recurrent_drop_p: float = 0.0,
        input_hidden_dim: int | None = None,
        detach_recurrent_state: bool = True,
        output_norm: bool = True,
    ) -> None:
        super().__init__()
        _require_spikingjelly()

        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if reservoir_dim <= 0:
            raise ValueError("reservoir_dim must be positive.")
        if not taus:
            raise ValueError("taus must contain at least one reservoir tau.")
        if not 0.0 <= recurrent_drop_p < 1.0:
            raise ValueError("recurrent_drop_p must be in [0.0, 1.0).")

        self.config = TPSAPUBackboneConfig(
            input_dim=input_dim,
            reservoir_dim=reservoir_dim,
            taus=tuple(float(tau) for tau in taus),
            recurrent_drop_p=recurrent_drop_p,
            input_hidden_dim=input_hidden_dim,
            detach_recurrent_state=detach_recurrent_state,
            output_norm=output_norm,
        )
        self.input_dim = input_dim
        self.reservoir_dim = reservoir_dim
        self.taus = self.config.taus
        self.recurrent_drop_p = recurrent_drop_p
        self.out_features = reservoir_dim * len(self.taus)

        hidden_dim = input_hidden_dim or reservoir_dim * 2
        self.nl_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, reservoir_dim, bias=False),
        )

        self.shared_input_proj = nn.Linear(reservoir_dim, reservoir_dim, bias=False)
        self.shared_recurrent = nn.Linear(reservoir_dim, reservoir_dim, bias=False)
        self.register_buffer("neuron_mask", torch.ones(reservoir_dim))
        self.last_states: dict[str, torch.Tensor] | None = None
        self.reservoirs = nn.ModuleList(
            [
                SharedReservoir(
                    reservoir_dim,
                    tau,
                    detach_recurrent_state=detach_recurrent_state,
                )
                for tau in self.taus
            ]
        )
        self.output_norm = nn.LayerNorm(self.out_features) if output_norm else nn.Identity()

        self.reset_parameters()

    @classmethod
    def from_config(cls, config: TPSAPUBackboneConfig) -> "TPSAPUBackbone":
        return cls(
            input_dim=config.input_dim,
            reservoir_dim=config.reservoir_dim,
            taus=config.taus,
            recurrent_drop_p=config.recurrent_drop_p,
            input_hidden_dim=config.input_hidden_dim,
            detach_recurrent_state=config.detach_recurrent_state,
            output_norm=config.output_norm,
        )

    def reset_parameters(self) -> None:
        for module in self.nl_proj:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        nn.init.xavier_uniform_(self.shared_input_proj.weight)
        nn.init.uniform_(self.shared_recurrent.weight, -0.001, 0.001)

    def reset_state(self) -> None:
        self.last_states = None
        for reservoir in self.reservoirs:
            reservoir.reset_state()

    def freeze_shared_topology(self) -> None:
        """Freeze the shared input and recurrent topology weights."""

        self.shared_input_proj.weight.requires_grad_(False)
        self.shared_recurrent.weight.requires_grad_(False)

    @torch.no_grad()
    def set_neuron_mask(self, mask: torch.Tensor) -> None:
        if mask.numel() != self.reservoir_dim:
            raise ValueError(
                f"Expected neuron mask with {self.reservoir_dim} entries, "
                f"received {mask.numel()}."
            )
        self.neuron_mask.copy_(mask.reshape(-1).to(self.neuron_mask))

    def neuron_sparsity(self) -> float:
        return 1.0 - self.neuron_mask.float().mean().item()

    def set_recurrent_dropout(self, drop_p: float) -> None:
        if not 0.0 <= drop_p < 1.0:
            raise ValueError("drop_p must be in [0.0, 1.0).")
        self.recurrent_drop_p = drop_p

    def recurrent_l1(self) -> torch.Tensor:
        return self.shared_recurrent.weight.abs().sum()

    def forward(
        self,
        x: torch.Tensor,
        *,
        pooling: str = "last",
        state: str = "membrane",
        reset_state: bool = True,
    ) -> torch.Tensor:
        """
        Run encoded tokens through TPSAPU.

        Args:
            x: Tensor shaped ``(batch, steps, input_dim)``. A 2-D tensor is
                treated as a single-step sequence.
            pooling: ``"last"`` returns the last step, ``"mean"`` averages all
                steps, and ``"none"`` returns the full sequence.
            state: ``"membrane"``, ``"spike"``, ``"both"``, or ``"all"``.
            reset_state: Reset reservoir state before processing this sequence.
        """

        states = self.forward_states(x, reset_state=reset_state)
        if state == "membrane":
            sequence = states["membrane"]
        elif state == "spike":
            sequence = states["spike"]
        elif state == "both":
            sequence = torch.cat([states["membrane"], states["spike"]], dim=-1)
        elif state == "all":
            sequence = torch.cat(
                [
                    states["membrane"],
                    states["spike"],
                    states["dynamics"],
                    states["spike_history"],
                ],
                dim=-1,
            )
        else:
            raise ValueError(
                'state must be one of "membrane", "spike", "both", or "all".'
            )
        return self._pool(sequence, pooling)

    def forward_states(
        self,
        x: torch.Tensor,
        *,
        reset_state: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Return membrane, spike, membrane-dynamics, and spike-history sequences."""

        x = self._normalize_input(x)
        if reset_state:
            self.reset_state()

        projected = self.nl_proj(x)
        recurrent_weight = self._masked_recurrent_weight()
        neuron_mask = self.neuron_mask.to(device=projected.device)
        membrane_readouts = []
        spike_readouts = []

        for step in range(projected.size(1)):
            step_input = projected[:, step, :]
            membranes = []
            spikes = []
            for reservoir in self.reservoirs:
                spike = reservoir(
                    step_input,
                    self.shared_input_proj,
                    recurrent_weight,
                    neuron_mask,
                )
                spikes.append(spike)
                membranes.append(
                    reservoir.membrane() * neuron_mask.to(dtype=step_input.dtype)
                )
            membrane_readouts.append(torch.cat(membranes, dim=-1))
            spike_readouts.append(torch.cat(spikes, dim=-1))

        feature_mask = neuron_mask.repeat(len(self.taus)).view(1, 1, -1)
        membranes = self.output_norm(torch.stack(membrane_readouts, dim=1))
        membranes = membranes * feature_mask.to(dtype=membranes.dtype)
        spikes = torch.stack(spike_readouts, dim=1)
        spikes = spikes * feature_mask.to(dtype=spikes.dtype)
        dynamics = torch.cat(
            [membranes[:, :1, :], membranes[:, 1:, :] - membranes[:, :-1, :]],
            dim=1,
        )
        spike_history = spikes.cumsum(dim=1)
        states = {
            "membrane": membranes,
            "spike": spikes,
            "dynamics": dynamics,
            "spike_history": spike_history,
        }
        self.last_states = {key: value.detach() for key, value in states.items()}
        return states

    def _pool(self, sequence: torch.Tensor, pooling: str) -> torch.Tensor:
        if pooling == "last":
            return sequence[:, -1, :]
        if pooling == "mean":
            return sequence.mean(dim=1)
        if pooling in {"none", "sequence"}:
            return sequence
        raise ValueError('pooling must be one of "last", "mean", or "none".')

    def _normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if x.dim() != 3:
            raise ValueError(
                "TPSAPUBackbone input must have shape (batch, steps, input_dim)."
            )
        if x.size(-1) != self.input_dim:
            raise ValueError(
                f"Expected input_dim={self.input_dim}, received {x.size(-1)}."
            )
        return x

    def _masked_recurrent_weight(self) -> torch.Tensor:
        weight = self.shared_recurrent.weight
        neuron_mask = self.neuron_mask.to(device=weight.device, dtype=weight.dtype)
        weight = weight * neuron_mask.view(-1, 1) * neuron_mask.view(1, -1)
        if self.training and self.recurrent_drop_p > 0.0:
            mask = torch.rand_like(weight) > self.recurrent_drop_p
            return weight * mask.to(dtype=weight.dtype)
        return weight


def build_tpsapu_backbone(input_dim: int, **kwargs) -> TPSAPUBackbone:
    """Convenience factory for code that expects a backbone builder."""

    return TPSAPUBackbone(input_dim=input_dim, **kwargs)


TPSAPU = TPSAPUBackbone

__all__ = [
    "SharedReservoir",
    "TPSAPU",
    "TPSAPUBackbone",
    "TPSAPUBackboneConfig",
    "build_tpsapu_backbone",
]
