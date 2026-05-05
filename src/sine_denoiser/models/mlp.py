"""MLP denoiser per ``PRD_models.md`` §2."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from sine_denoiser.models.base import DenoiserModel

_NUM_COMPONENTS = 4
_WINDOW = 10

_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "gelu": nn.GELU,
}


class MLP(DenoiserModel):
    """Fully-connected denoiser.

    Input is ``concat(one_hot(c, 4), x_ctx)`` of length 14; output is the
    predicted clean window of shape ``(B, 10)``.
    """

    name = "mlp"

    def __init__(
        self,
        hidden_size: int = 64,
        num_layers: int = 2,
        activation: str = "relu",
        dropout: float = 0.0,
        num_components: int = _NUM_COMPONENTS,
        context_window: int = _WINDOW,
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be > 0")
        if num_layers <= 0:
            raise ValueError("num_layers must be > 0")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if activation not in _ACTIVATIONS:
            raise ValueError(
                f"activation must be one of {sorted(_ACTIVATIONS)}; got {activation!r}"
            )
        act_cls = _ACTIVATIONS[activation]
        self._num_components = num_components
        self._window = context_window

        layers: list[nn.Module] = []
        in_dim = num_components + context_window
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(act_cls())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_size
        layers.append(nn.Linear(in_dim, context_window))
        self.net = nn.Sequential(*layers)

    def forward(self, x_ctx: Tensor, c: Tensor) -> Tensor:
        if x_ctx.ndim != 2 or x_ctx.shape[1] != self._window:
            raise ValueError(
                f"x_ctx must have shape (B, {self._window}); got {tuple(x_ctx.shape)}"
            )
        if c.ndim != 1 or c.shape[0] != x_ctx.shape[0]:
            raise ValueError(
                f"c must have shape ({x_ctx.shape[0]},); got {tuple(c.shape)}"
            )
        one_hot = torch.nn.functional.one_hot(c, self._num_components).to(x_ctx.dtype)
        x = torch.cat([one_hot, x_ctx], dim=1)
        return self.net(x)
