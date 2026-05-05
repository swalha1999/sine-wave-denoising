"""Generate the 4 pure sine wave components from a data config."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SignalConfig:
    num_components: int
    duration_s: float
    sample_rate_hz: int
    frequencies_hz: tuple[float, ...]
    phases_rad: tuple[float, ...]
    amplitudes: tuple[float, ...]

    @classmethod
    def from_dict(cls, cfg: dict) -> "SignalConfig":
        return cls(
            num_components=int(cfg["num_components"]),
            duration_s=float(cfg["duration_s"]),
            sample_rate_hz=int(cfg["sample_rate_hz"]),
            frequencies_hz=tuple(float(f) for f in cfg["frequencies_hz"]),
            phases_rad=tuple(float(p) for p in cfg["phases_rad"]),
            amplitudes=tuple(float(a) for a in cfg["amplitudes"]),
        )


def generate_signals(cfg: dict | SignalConfig) -> np.ndarray:
    """Return clean sine components, shape (num_components, num_samples).

    Sample i of component k is `amplitudes[k] * sin(2π · frequencies_hz[k] · t_i + phases_rad[k])`
    where `t_i = i / sample_rate_hz` for `i in [0, duration_s · sample_rate_hz)`.
    """
    sc = cfg if isinstance(cfg, SignalConfig) else SignalConfig.from_dict(cfg)
    n_components = sc.num_components
    if not (len(sc.frequencies_hz) == len(sc.phases_rad) == len(sc.amplitudes) == n_components):
        raise ValueError(
            f"frequencies_hz/phases_rad/amplitudes must each have length {n_components}"
        )
    num_samples = int(round(sc.duration_s * sc.sample_rate_hz))
    t = np.arange(num_samples, dtype=np.float64) / sc.sample_rate_hz
    freqs = np.asarray(sc.frequencies_hz, dtype=np.float64)[:, None]
    phases = np.asarray(sc.phases_rad, dtype=np.float64)[:, None]
    amps = np.asarray(sc.amplitudes, dtype=np.float64)[:, None]
    return amps * np.sin(2.0 * np.pi * freqs * t + phases)
