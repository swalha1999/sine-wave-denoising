"""Add per-signal Gaussian noise and build the mixed observation."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def add_noise(
    signals: np.ndarray,
    sigma: float | Sequence[float],
    *,
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """Return signals plus N(0, σ²) noise drawn independently per sample, per component.

    `signals` is a 2-D array shaped `(num_components, num_samples)`. `sigma` is either a
    scalar (applied to every component) or a length-`num_components` sequence (one σ per
    component). `rng` may be a numpy `Generator`, an integer seed, or `None`.
    """
    if signals.ndim != 2:
        raise ValueError(
            f"signals must be 2-D (num_components, num_samples); got shape {signals.shape}"
        )
    n_components = signals.shape[0]
    sigma_arr = np.asarray(sigma, dtype=np.float64)
    if sigma_arr.ndim == 0:
        sigma_arr = np.full((n_components,), float(sigma_arr))
    elif sigma_arr.shape != (n_components,):
        raise ValueError(
            f"sigma must be a scalar or length-{n_components} sequence; got shape {sigma_arr.shape}"
        )
    if np.any(sigma_arr < 0):
        raise ValueError("sigma must be non-negative")
    generator = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    noise = generator.standard_normal(signals.shape) * sigma_arr[:, None]
    return signals + noise


def mix(signals: np.ndarray) -> np.ndarray:
    """Sum components along axis 0, returning a 1-D array of length `num_samples`."""
    if signals.ndim != 2:
        raise ValueError(f"signals must be 2-D (num_components, num_samples); got shape {signals.shape}")
    return signals.sum(axis=0)
