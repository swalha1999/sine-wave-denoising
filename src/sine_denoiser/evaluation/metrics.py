"""MSE metrics: aggregate total and per-component breakdown."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MseReport:
    """Aggregate MSE plus per-component breakdown.

    ``per_component[k]`` is ``NaN`` when no samples have ``c == k``.
    """

    total: float
    per_component: np.ndarray


def mse(y_hat: np.ndarray, y: np.ndarray) -> float:
    """Mean squared error over every element of ``y_hat`` and ``y``."""
    y_hat = np.asarray(y_hat)
    y = np.asarray(y)
    if y_hat.shape != y.shape:
        raise ValueError(
            f"shape mismatch: y_hat {y_hat.shape} vs y {y.shape}"
        )
    if y_hat.size == 0:
        raise ValueError("cannot compute MSE over empty arrays")
    return float(np.mean((y_hat - y) ** 2))


def mse_per_component(
    y_hat: np.ndarray,
    y: np.ndarray,
    c: np.ndarray,
    num_components: int,
) -> np.ndarray:
    """Per-component MSE: ``out[k]`` averages squared error over rows with
    ``c == k``. Components with no samples get ``NaN``.
    """
    y_hat = np.asarray(y_hat)
    y = np.asarray(y)
    c = np.asarray(c)
    if y_hat.shape != y.shape:
        raise ValueError(
            f"shape mismatch: y_hat {y_hat.shape} vs y {y.shape}"
        )
    if y_hat.shape[0] != c.shape[0]:
        raise ValueError(
            f"batch mismatch: y_hat has {y_hat.shape[0]} rows, "
            f"c has {c.shape[0]}"
        )
    if c.ndim != 1:
        raise ValueError(f"c must be 1-D, got shape {c.shape}")
    if num_components <= 0:
        raise ValueError("num_components must be > 0")

    sq = (y_hat - y) ** 2
    out = np.full(num_components, np.nan, dtype=np.float64)
    for k in range(num_components):
        mask = c == k
        if mask.any():
            out[k] = float(np.mean(sq[mask]))
    return out


def compute(
    y_hat: np.ndarray,
    y: np.ndarray,
    c: np.ndarray,
    num_components: int,
) -> MseReport:
    """Compute aggregate MSE and per-component MSE in one call."""
    return MseReport(
        total=mse(y_hat, y),
        per_component=mse_per_component(y_hat, y, c, num_components),
    )
