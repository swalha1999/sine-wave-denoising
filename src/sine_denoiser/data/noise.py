"""Per-signal Gaussian noise and mixed-signal construction."""

from __future__ import annotations

import numpy as np


def add_gaussian_noise(
    signals: np.ndarray,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add independent N(0, sigma**2) noise to every element of ``signals``.

    ``signals`` is typically ``(K, T)``: ``K`` pure component waveforms,
    ``T`` samples each. Each element gets its own noise draw.
    """
    if sigma < 0:
        raise ValueError(f"sigma must be non-negative, got {sigma}")
    if sigma == 0:
        return signals.copy()
    noise = rng.normal(loc=0.0, scale=sigma, size=signals.shape)
    return signals + noise


def build_mixed(signals: np.ndarray) -> np.ndarray:
    """Sum component signals along axis 0 to produce one mixed waveform.

    ``(K, T)`` -> ``(T,)``.
    """
    return signals.sum(axis=0)
