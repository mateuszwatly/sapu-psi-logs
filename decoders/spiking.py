"""SpikingJelly import helpers for decoders."""

from __future__ import annotations

try:
    from spikingjelly.activation_based import functional, neuron, surrogate
except ImportError as exc:  # pragma: no cover - exercised only without dependency
    functional = None
    neuron = None
    surrogate = None
    _SPIKINGJELLY_IMPORT_ERROR = exc
else:
    _SPIKINGJELLY_IMPORT_ERROR = None


def require_spikingjelly(name: str) -> None:
    if _SPIKINGJELLY_IMPORT_ERROR is not None:
        raise ImportError(
            f"{name} requires spikingjelly. Install it with `pip install spikingjelly`."
        ) from _SPIKINGJELLY_IMPORT_ERROR
