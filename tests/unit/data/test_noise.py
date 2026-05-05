import numpy as np
import pytest

from sine_denoiser.data.noise import add_noise, mix
from sine_denoiser.data.signals import generate_signals


DEFAULT_CFG = {
    "num_components": 4,
    "duration_s": 10.0,
    "sample_rate_hz": 1000,
    "frequencies_hz": [2.0, 5.0, 11.0, 17.0],
    "phases_rad": [0.0, 0.7, 1.5, 2.3],
    "amplitudes": [1.0, 1.0, 1.0, 1.0],
}


def test_zero_sigma_returns_input_unchanged():
    sigs = generate_signals(DEFAULT_CFG)
    noisy = add_noise(sigs, 0.0, rng=0)
    assert np.array_equal(sigs, noisy)


def test_zero_sigma_mixed_equals_sum_of_pure():
    sigs = generate_signals(DEFAULT_CFG)
    noisy = add_noise(sigs, 0.0, rng=42)
    assert np.allclose(mix(noisy), sigs.sum(axis=0))


def test_noise_mean_is_zero_over_many_seeds():
    sigs = generate_signals(DEFAULT_CFG)
    sigma = 0.5
    n_trials = 200
    diffs = np.empty((n_trials, *sigs.shape), dtype=np.float64)
    for s in range(n_trials):
        diffs[s] = add_noise(sigs, sigma, rng=s) - sigs
    assert abs(float(diffs.mean())) < 0.01


def test_noise_std_matches_sigma():
    sigs = np.zeros((4, 10000))
    noisy = add_noise(sigs, 0.5, rng=0)
    assert noisy.std() == pytest.approx(0.5, rel=0.05)


def test_per_component_sigma_applied_independently():
    sigs = np.zeros((4, 100000))
    sigmas = [0.1, 0.5, 1.0, 2.0]
    noisy = add_noise(sigs, sigmas, rng=0)
    for k, s in enumerate(sigmas):
        assert noisy[k].std() == pytest.approx(s, rel=0.05)


def test_same_seed_is_reproducible():
    sigs = generate_signals(DEFAULT_CFG)
    a = add_noise(sigs, 0.3, rng=7)
    b = add_noise(sigs, 0.3, rng=7)
    assert np.array_equal(a, b)


def test_different_seeds_produce_different_noise():
    sigs = generate_signals(DEFAULT_CFG)
    a = add_noise(sigs, 0.3, rng=0)
    b = add_noise(sigs, 0.3, rng=1)
    assert not np.array_equal(a, b)


def test_explicit_generator_consumes_state():
    sigs = np.zeros((2, 1000))
    rng = np.random.default_rng(123)
    noisy = add_noise(sigs, 1.0, rng=rng)
    expected = np.random.default_rng(123).standard_normal(sigs.shape)
    assert np.allclose(noisy, expected)


def test_sigma_wrong_length_raises():
    sigs = np.zeros((4, 100))
    with pytest.raises(ValueError, match="length-4"):
        add_noise(sigs, [0.1, 0.2], rng=0)


def test_negative_sigma_raises():
    sigs = np.zeros((4, 100))
    with pytest.raises(ValueError, match="non-negative"):
        add_noise(sigs, -0.1, rng=0)


def test_add_noise_requires_2d_input():
    with pytest.raises(ValueError, match="2-D"):
        add_noise(np.zeros((10,)), 0.1, rng=0)


def test_mix_shape_is_num_samples():
    sigs = generate_signals(DEFAULT_CFG)
    assert mix(sigs).shape == (10000,)


def test_mix_requires_2d_input():
    with pytest.raises(ValueError, match="2-D"):
        mix(np.zeros((10,)))


def test_mix_equals_sum_along_axis_zero():
    rng = np.random.default_rng(0)
    arr = rng.standard_normal((4, 50))
    assert np.array_equal(mix(arr), arr.sum(axis=0))
