import numpy as np
import pytest

from sine_denoiser.data.noise import add_gaussian_noise, build_mixed


def _pure_signals(rng: np.random.Generator, k: int = 4, t: int = 200) -> np.ndarray:
    return rng.standard_normal((k, t))


def test_sigma_zero_mixed_equals_sum_of_pure():
    rng = np.random.default_rng(0)
    pure = _pure_signals(rng)
    noisy = add_gaussian_noise(pure, sigma=0.0, rng=rng)
    np.testing.assert_array_equal(build_mixed(noisy), pure.sum(axis=0))


def test_noise_mean_approx_zero_over_many_seeds():
    pure = np.zeros((4, 1000))
    sigma = 0.5
    means = np.array(
        [
            add_gaussian_noise(pure, sigma=sigma, rng=np.random.default_rng(s)).mean()
            for s in range(200)
        ]
    )
    assert abs(means.mean()) < 0.01


def test_add_noise_preserves_shape_and_does_not_mutate_input():
    rng = np.random.default_rng(1)
    pure = _pure_signals(rng)
    snapshot = pure.copy()
    noisy = add_gaussian_noise(pure, sigma=0.3, rng=rng)
    assert noisy.shape == pure.shape
    np.testing.assert_array_equal(pure, snapshot)


def test_noise_std_matches_sigma():
    pure = np.zeros((4, 5000))
    sigma = 0.7
    noisy = add_gaussian_noise(pure, sigma=sigma, rng=np.random.default_rng(42))
    assert noisy.std() == pytest.approx(sigma, rel=0.05)


def test_per_signal_noise_is_independent():
    pure = np.zeros((2, 10000))
    noisy = add_gaussian_noise(pure, sigma=1.0, rng=np.random.default_rng(7))
    corr = np.corrcoef(noisy[0], noisy[1])[0, 1]
    assert abs(corr) < 0.05


def test_build_mixed_reduces_component_axis():
    signals = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
    np.testing.assert_array_equal(build_mixed(signals), np.array([11.0, 22.0, 33.0]))


def test_negative_sigma_raises():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        add_gaussian_noise(np.zeros((2, 4)), sigma=-0.1, rng=rng)
