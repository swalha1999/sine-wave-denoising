"""Abstract base class shared by all denoising models."""

from __future__ import annotations

import abc

from torch import Tensor, nn


class DenoiserModel(nn.Module, abc.ABC):
    """Common interface for sine wave denoising models.

    Subclasses must implement ``forward(x_ctx, c) -> y_hat`` where
    ``x_ctx`` is a ``(B, W)`` float tensor of noisy context windows,
    ``c`` is a ``(B,)`` long tensor of component selectors, and
    ``y_hat`` is the predicted clean window for the chosen component.
    """

    name: str = ""

    @abc.abstractmethod
    def forward(self, x_ctx: Tensor, c: Tensor) -> Tensor: ...

    @classmethod
    def from_config(cls, cfg: dict) -> "DenoiserModel":
        """Construct an instance from a configuration dict.

        Default implementation forwards ``cfg`` as keyword arguments to
        ``cls``. Subclasses may override to apply validation or defaults.
        """
        return cls(**cfg)
