"""Registry mapping model names to their concrete ``DenoiserModel`` subclass."""

from __future__ import annotations

from sine_denoiser.models.base import DenoiserModel
from sine_denoiser.models.lstm import LSTM
from sine_denoiser.models.mlp import MLP
from sine_denoiser.models.rnn import RNN

_REGISTRY: dict[str, type[DenoiserModel]] = {
    MLP.name: MLP,
    RNN.name: RNN,
    LSTM.name: LSTM,
}


def available() -> list[str]:
    """Return the sorted list of registered model names."""
    return sorted(_REGISTRY)


def build(name: str, cfg: dict | None = None) -> DenoiserModel:
    """Construct a registered model by name.

    ``cfg`` is forwarded to the subclass via ``from_config``; ``None`` is
    treated as an empty dict so callers can request defaults.
    """
    try:
        cls = _REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"unknown model name {name!r}; available: {available()}"
        ) from None
    return cls.from_config(dict(cfg) if cfg else {})
