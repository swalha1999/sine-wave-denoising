"""Unit tests for ``evaluation.robustness.sweep_noise``."""

from __future__ import annotations

import numpy as np
import pytest
from torch import Tensor, nn

from sine_denoiser.evaluation.robustness import (
    SweepPoint,
    SweepResult,
    evaluate_test_mse,
    sweep_noise,
)


class _IdentityDenoiser(nn.Module):
    """Toy "denoiser" that returns the noisy input unchanged.

    With this model, MSE = E[(noisy_mixed - clean_component)^2], whose
    noise term scales with sigma^2 — so the curve must rise with sigma.
    """

    def forward(self, x_ctx: Tensor, c: Tensor) -> Tensor:  # noqa: ARG002
        return x_ctx


def _toy_pure(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((4, 400))


def test_sweep_returns_aligned_sigmas_and_mses():
    pure = _toy_pure()
    sigmas = [0.0, 0.5, 1.0]
    result = sweep_noise(
        _IdentityDenoiser(),
        pure,
        sigmas,
        context_window=10,
        split={"train": 0.8, "val": 0.1, "test": 0.1},
        batch_size=32,
        seed=0,
    )
    assert isinstance(result, SweepResult)
    assert len(result.points) == 3
    assert all(isinstance(p, SweepPoint) for p in result.points)
    assert result.sigmas.tolist() == sigmas
    assert result.mses.shape == (3,)
    assert np.all(np.isfinite(result.mses))


def test_sweep_curve_is_monotone_ish_on_toy_data():
    pure = _toy_pure()
    sigmas = [0.0, 0.25, 0.5, 1.0, 2.0]
    result = sweep_noise(
        _IdentityDenoiser(),
        pure,
        sigmas,
        context_window=10,
        split={"train": 0.8, "val": 0.1, "test": 0.1},
        batch_size=64,
        seed=0,
    )
    mses = result.mses
    diffs = np.diff(mses)
    assert np.all(diffs >= -1e-3)
    assert mses[-1] > mses[0] + 1.0


def test_sweep_at_zero_sigma_recovers_noiseless_mse():
    pure = _toy_pure()
    result = sweep_noise(
        _IdentityDenoiser(),
        pure,
        [0.0],
        context_window=10,
        split={"train": 0.8, "val": 0.1, "test": 0.1},
        batch_size=32,
        seed=0,
    )
    direct = evaluate_test_mse(
        _IdentityDenoiser(),
        pure,
        0.0,
        context_window=10,
        split={"train": 0.8, "val": 0.1, "test": 0.1},
        batch_size=32,
        seed=0,
    )
    assert result.points[0].mse == pytest.approx(direct)


def test_evaluate_rejects_negative_sigma():
    pure = _toy_pure()
    with pytest.raises(ValueError, match="sigma"):
        evaluate_test_mse(
            _IdentityDenoiser(),
            pure,
            -0.1,
            context_window=10,
            split={"train": 0.8, "val": 0.1, "test": 0.1},
            batch_size=32,
            seed=0,
        )


def test_sweep_with_empty_sigmas_returns_empty_result():
    pure = _toy_pure()
    result = sweep_noise(
        _IdentityDenoiser(),
        pure,
        [],
        context_window=10,
        split={"train": 0.8, "val": 0.1, "test": 0.1},
        batch_size=32,
        seed=0,
    )
    assert result.points == ()
    assert result.sigmas.shape == (0,)
    assert result.mses.shape == (0,)
