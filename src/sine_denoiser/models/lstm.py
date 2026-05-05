"""LSTM denoiser per ``PRD_models.md`` §4."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from sine_denoiser.models.base import DenoiserModel

_NUM_COMPONENTS = 4
_WINDOW = 10


class LSTM(DenoiserModel):
    """LSTM denoiser.

    The window is treated as a length-``W`` sequence; at each timestep the
    model consumes ``[x_ctx[t]] ++ one_hot(c, num_components)``. A linear
    projection maps the per-timestep hidden state to a scalar, giving an
    output of shape ``(B, W)``.
    """

    name = "lstm"

    def __init__(
        self,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
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
        self._num_components = num_components
        self._window = context_window

        self.lstm = nn.LSTM(
            input_size=1 + num_components,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        head_in = hidden_size * (2 if bidirectional else 1)
        self.head = nn.Linear(head_in, 1)

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
        one_hot_seq = one_hot.unsqueeze(1).expand(-1, self._window, -1)
        seq = torch.cat([x_ctx.unsqueeze(-1), one_hot_seq], dim=-1)
        out, _ = self.lstm(seq)
        return self.head(out).squeeze(-1)
