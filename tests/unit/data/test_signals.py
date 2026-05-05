import numpy as np
import pytest

from sine_denoiser.data.signals import SignalConfig, generate_signals


DEFAULT_CFG = {
    "num_components": 4,
    "duration_s": 10.0,
    "sample_rate_hz": 1000,
    "frequencies_hz": [2.0, 5.0, 11.0, 17.0],
    "phases_rad": [0.0, 0.7, 1.5, 2.3],
    "amplitudes": [1.0, 1.0, 1.0, 1.0],
}


def test_shape_is_4_by_10000():
    sigs = generate_signals(DEFAULT_CFG)
    assert sigs.shape == (4, 10000)


def test_known_values_match_closed_form():
    sigs = generate_signals(DEFAULT_CFG)
    sample_rate = DEFAULT_CFG["sample_rate_hz"]
    indices = [0, 1, 250, 9999]
    for k, (f, p, a) in enumerate(
        zip(
            DEFAULT_CFG["frequencies_hz"],
            DEFAULT_CFG["phases_rad"],
            DEFAULT_CFG["amplitudes"],
            strict=True,
        )
    ):
        for i in indices:
            t = i / sample_rate
            expected = a * np.sin(2.0 * np.pi * f * t + p)
            assert sigs[k, i] == pytest.approx(expected, abs=1e-12)


def test_first_sample_is_amplitude_times_sin_phase():
    sigs = generate_signals(DEFAULT_CFG)
    for k, (a, p) in enumerate(
        zip(DEFAULT_CFG["amplitudes"], DEFAULT_CFG["phases_rad"], strict=True)
    ):
        assert sigs[k, 0] == pytest.approx(a * np.sin(p), abs=1e-12)


def test_dataclass_path_matches_dict_path():
    sc = SignalConfig.from_dict(DEFAULT_CFG)
    assert np.array_equal(generate_signals(sc), generate_signals(DEFAULT_CFG))


def test_mismatched_array_lengths_raise():
    bad = dict(DEFAULT_CFG)
    bad["amplitudes"] = [1.0, 1.0, 1.0]
    with pytest.raises(ValueError, match="length 4"):
        generate_signals(bad)


def test_deterministic_no_seed_needed():
    a = generate_signals(DEFAULT_CFG)
    b = generate_signals(DEFAULT_CFG)
    assert np.array_equal(a, b)
