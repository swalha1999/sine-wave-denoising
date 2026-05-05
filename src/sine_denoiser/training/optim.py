"""Optimizer factory: map cfg name -> ``torch.optim.Optimizer`` instance."""

from __future__ import annotations

import torch
from torch import nn

_OPTIMIZERS: dict[str, type[torch.optim.Optimizer]] = {
    "adam": torch.optim.Adam,
    "sgd": torch.optim.SGD,
}


def available() -> list[str]:
    """Return the sorted list of registered optimizer names."""
    return sorted(_OPTIMIZERS)


def build(model: nn.Module, cfg: dict | None = None) -> torch.optim.Optimizer:
    """Construct an optimizer for ``model`` from ``cfg``.

    ``cfg`` keys: ``optimizer`` (``"adam"``/``"sgd"``, default ``"adam"``)
    and ``lr`` (default ``1e-3``). Unknown names raise ``ValueError``.
    """
    cfg = dict(cfg) if cfg else {}
    name = str(cfg.get("optimizer", "adam")).lower()
    try:
        cls = _OPTIMIZERS[name]
    except KeyError:
        raise ValueError(
            f"unknown optimizer {name!r}; available: {available()}"
        ) from None
    return cls(model.parameters(), lr=float(cfg.get("lr", 1e-3)))
