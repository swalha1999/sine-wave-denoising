"""Unit tests for ``plotting.curves.plot_training_curve``."""

from __future__ import annotations

import struct

import pytest

from sine_denoiser.plotting.curves import plot_training_curve
from sine_denoiser.training.loop import EpochMetrics, FitResult


def _history(*pairs: tuple[float, float]) -> list[EpochMetrics]:
    return [
        EpochMetrics(epoch=i + 1, train_mse=t, val_mse=v)
        for i, (t, v) in enumerate(pairs)
    ]


def _is_png(path) -> bool:
    with path.open("rb") as fh:
        head = fh.read(8)
    return head == b"\x89PNG\r\n\x1a\n"


def _png_size(path) -> tuple[int, int]:
    with path.open("rb") as fh:
        fh.seek(16)
        width, height = struct.unpack(">II", fh.read(8))
    return width, height


def test_plot_training_curve_writes_png_to_target(tmp_path):
    history = _history((1.0, 1.2), (0.5, 0.6), (0.3, 0.4))
    out = tmp_path / "runs" / "abc" / "training_curve.png"

    returned = plot_training_curve(history, out)

    assert returned == out
    assert out.exists()
    assert _is_png(out)
    width, height = _png_size(out)
    assert width > 0 and height > 0


def test_plot_training_curve_accepts_fit_result(tmp_path):
    history = _history((1.0, 0.9), (0.4, 0.5))
    result = FitResult(
        best_epoch=1,
        best_val_mse=0.5,
        history=history,
        best_state_dict={},
    )
    out = tmp_path / "training_curve.png"

    plot_training_curve(result, out)

    assert _is_png(out)


def test_plot_training_curve_creates_parent_directories(tmp_path):
    history = _history((0.7, 0.8))
    out = tmp_path / "deep" / "nested" / "dir" / "curve.png"

    plot_training_curve(history, out)

    assert out.exists()


def test_plot_training_curve_rejects_empty_history(tmp_path):
    out = tmp_path / "curve.png"
    with pytest.raises(ValueError, match="empty"):
        plot_training_curve([], out)
    assert not out.exists()


def test_plot_training_curve_accepts_str_path(tmp_path):
    history = _history((0.6, 0.7), (0.3, 0.35))
    out = tmp_path / "curve.png"

    returned = plot_training_curve(history, str(out))

    assert returned == out
    assert _is_png(out)


def test_sdk_train_writes_training_curve_png(tmp_path):
    pytest.importorskip("torch")
    from sine_denoiser.sdk import SDK

    cfg = {
        "version": "1.00",
        "data": {
            "num_components": 4,
            "duration_s": 0.2,
            "sample_rate_hz": 1000,
            "noise_sigma": 0.1,
            "frequencies_hz": [2.0, 5.0, 11.0, 17.0],
            "phases_rad": [0.0, 0.7, 1.5, 2.3],
            "amplitudes": [1.0, 1.0, 1.0, 1.0],
            "context_window": 10,
            "split": {"train": 0.8, "val": 0.1, "test": 0.1},
            "data_seed": 0,
        },
        "model": {"mlp": {"hidden_size": 8, "num_layers": 1}},
        "training": {
            "optimizer": "adam",
            "lr": 1e-3,
            "batch_size": 32,
            "epochs": 2,
            "early_stopping_patience": 0,
        },
    }
    run_dir = tmp_path / "run"
    sdk = SDK(config=cfg)
    sdk.train("mlp", seed=0, run_dir=run_dir)

    png = run_dir / "training_curve.png"
    assert png.exists()
    assert _is_png(png)
