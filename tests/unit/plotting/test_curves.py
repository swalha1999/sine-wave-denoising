"""Unit tests for ``plotting.curves.plot_training_curves``."""

from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path

import pytest

from sine_denoiser.plotting.curves import plot_training_curves
from sine_denoiser.training.loop import EpochMetrics, FitResult


def _png_dimensions(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    length = struct.unpack(">I", data[8:12])[0]
    chunk_type = data[12:16]
    assert chunk_type == b"IHDR"
    chunk_data = data[16 : 16 + length]
    crc = struct.unpack(">I", data[16 + length : 20 + length])[0]
    assert zlib.crc32(chunk_type + chunk_data) == crc
    width, height = struct.unpack(">II", chunk_data[:8])
    return width, height


def _history() -> list[EpochMetrics]:
    return [
        EpochMetrics(epoch=1, train_mse=1.0, val_mse=1.2),
        EpochMetrics(epoch=2, train_mse=0.5, val_mse=0.7),
        EpochMetrics(epoch=3, train_mse=0.3, val_mse=0.6),
    ]


def test_writes_valid_png_from_fit_result(tmp_path):
    result = FitResult(best_epoch=3, best_val_mse=0.6, history=_history())
    out = tmp_path / "training_curve.png"

    returned = plot_training_curves(result, out)

    assert returned == out
    assert out.is_file()
    width, height = _png_dimensions(out)
    assert width > 100 and height > 100


def test_creates_parent_directory(tmp_path):
    nested = tmp_path / "runs" / "lstm__seed0" / "training_curve.png"
    plot_training_curves(_history(), nested)
    assert nested.is_file()


def test_accepts_run_directory_with_metrics_json(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    payload = {
        "best_epoch": 3,
        "best_val_mse": 0.6,
        "history": [
            {"epoch": 1, "train_mse": 1.0, "val_mse": 1.2},
            {"epoch": 2, "train_mse": 0.5, "val_mse": 0.7},
            {"epoch": 3, "train_mse": 0.3, "val_mse": 0.6},
        ],
    }
    (run_dir / "metrics.json").write_text(json.dumps(payload))

    out = run_dir / "training_curve.png"
    plot_training_curves(run_dir, out)
    assert out.is_file()


def test_accepts_metrics_json_path(tmp_path):
    metrics = tmp_path / "metrics.json"
    payload = {
        "best_epoch": 1,
        "best_val_mse": 1.2,
        "history": [{"epoch": 1, "train_mse": 1.0, "val_mse": 1.2}],
    }
    metrics.write_text(json.dumps(payload))

    out = tmp_path / "curve.png"
    plot_training_curves(metrics, out)
    assert out.is_file()


def test_accepts_iterable_of_dicts(tmp_path):
    history = [
        {"epoch": 1, "train_mse": 1.0, "val_mse": 1.1},
        {"epoch": 2, "train_mse": 0.4, "val_mse": 0.5},
    ]
    out = tmp_path / "curve.png"
    plot_training_curves(history, out)
    assert out.is_file()


def test_empty_history_raises(tmp_path):
    out = tmp_path / "curve.png"
    with pytest.raises(ValueError, match="empty"):
        plot_training_curves([], out)
    assert not out.exists()


def test_title_override_does_not_break_output(tmp_path):
    out = tmp_path / "curve.png"
    plot_training_curves(_history(), out, title="LSTM seed=0")
    assert out.is_file()
