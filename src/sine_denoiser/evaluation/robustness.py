"""Robustness sweep: test MSE as a function of noise sigma."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from sine_denoiser.data.loader import build_loaders
from sine_denoiser.data.noise import add_gaussian_noise, build_mixed


@dataclass(frozen=True)
class SweepPoint:
    """Single (sigma, test MSE) entry of a robustness curve."""

    sigma: float
    mse: float


@dataclass(frozen=True)
class SweepResult:
    """Robustness curve: aligned ``sigmas`` and per-sigma test MSEs."""

    points: tuple[SweepPoint, ...]

    @property
    def sigmas(self) -> np.ndarray:
        return np.array([p.sigma for p in self.points], dtype=np.float64)

    @property
    def mses(self) -> np.ndarray:
        return np.array([p.mse for p in self.points], dtype=np.float64)


def evaluate_test_mse(
    model: nn.Module,
    pure: np.ndarray,
    sigma: float,
    *,
    context_window: int,
    split: dict,
    batch_size: int,
    seed: int,
) -> float:
    """Build a noisy mixed signal at level ``sigma`` and return MSE on the
    test loader.

    The split (and therefore the test windows) is fixed by ``seed`` so that
    points across the sweep are directly comparable.
    """
    if sigma < 0:
        raise ValueError(f"sigma must be non-negative, got {sigma}")
    rng = np.random.default_rng(seed)
    noisy_pure = add_gaussian_noise(pure, sigma, rng)
    mixed = build_mixed(noisy_pure)
    loaders = build_loaders(
        mixed,
        pure,
        context_window=context_window,
        split=split,
        batch_size=batch_size,
        seed=seed,
        shuffle_train=False,
    )
    model.eval()
    total_sse = 0.0
    total_n = 0
    with torch.no_grad():
        for x_np, c_np, y_np in loaders.test:
            x = torch.as_tensor(x_np, dtype=torch.float32)
            c = torch.as_tensor(c_np, dtype=torch.long)
            y = torch.as_tensor(y_np, dtype=torch.float32)
            y_hat = model(x, c)
            total_sse += float(((y_hat - y) ** 2).sum().item())
            total_n += y.numel()
    if total_n == 0:
        raise ValueError("test loader produced no batches")
    return total_sse / total_n


def sweep_noise(
    model: nn.Module,
    pure: np.ndarray,
    sigmas: Iterable[float],
    *,
    context_window: int,
    split: dict,
    batch_size: int,
    seed: int = 0,
) -> SweepResult:
    """Evaluate test MSE at each sigma and return the assembled curve."""
    points: list[SweepPoint] = []
    for sigma in sigmas:
        mse = evaluate_test_mse(
            model,
            pure,
            float(sigma),
            context_window=context_window,
            split=split,
            batch_size=batch_size,
            seed=seed,
        )
        points.append(SweepPoint(sigma=float(sigma), mse=mse))
    return SweepResult(points=tuple(points))
