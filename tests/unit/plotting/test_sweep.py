"""Unit tests for ``plotting.sweep``."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np
import pytest

from sine_denoiser.evaluation.robustness import SweepPoint, SweepResult
from sine_denoiser.plotting.sweep import plot_sweep_heatmap, plot_sweep_line


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


def test_line_plot_from_sweep_result(tmp_path):
    result = SweepResult(
        points=(
            SweepPoint(sigma=0.0, mse=0.01),
            SweepPoint(sigma=0.5, mse=0.04),
            SweepPoint(sigma=1.0, mse=0.16),
        )
    )
    out = tmp_path / "noise_sigma.png"

    returned = plot_sweep_line(result, out_path=out, axis_name="noise_sigma")

    assert returned == out
    assert out.is_file()
    width, height = _png_dimensions(out)
    assert width > 100 and height > 100


def test_line_plot_from_values_and_mses(tmp_path):
    out = tmp_path / "hidden.png"
    plot_sweep_line(
        [32, 64, 128],
        [0.05, 0.03, 0.02],
        out_path=out,
        axis_name="hidden_size",
    )
    assert out.is_file()


def test_line_plot_multi_series_with_labels(tmp_path):
    out = tmp_path / "lr.png"
    plot_sweep_line(
        [1e-4, 1e-3, 1e-2],
        [[0.10, 0.04, 0.20], [0.12, 0.05, 0.18]],
        out_path=out,
        axis_name="lr",
        series_labels=["mlp", "lstm"],
    )
    assert out.is_file()


def test_line_plot_creates_parent_directory(tmp_path):
    nested = tmp_path / "runs" / "sweeps" / "noise_sigma.png"
    plot_sweep_line(
        [0.0, 0.5],
        [0.01, 0.04],
        out_path=nested,
        axis_name="noise_sigma",
    )
    assert nested.is_file()


def test_line_plot_handles_zero_mse_without_log_scale(tmp_path):
    out = tmp_path / "zero.png"
    plot_sweep_line(
        [0.0, 1.0],
        [0.0, 0.5],
        out_path=out,
        axis_name="noise_sigma",
    )
    assert out.is_file()


def test_line_plot_rejects_mses_with_sweep_result(tmp_path):
    result = SweepResult(points=(SweepPoint(sigma=0.0, mse=0.01),))
    with pytest.raises(ValueError, match="mses must be None"):
        plot_sweep_line(result, [0.01], out_path=tmp_path / "bad.png")


def test_line_plot_requires_mses_when_values_is_sequence(tmp_path):
    with pytest.raises(ValueError, match="mses is required"):
        plot_sweep_line([1, 2, 3], None, out_path=tmp_path / "bad.png")


def test_line_plot_rejects_length_mismatch(tmp_path):
    with pytest.raises(ValueError, match="does not match"):
        plot_sweep_line(
            [1, 2, 3],
            [0.1, 0.2],
            out_path=tmp_path / "bad.png",
        )


def test_line_plot_rejects_label_count_mismatch(tmp_path):
    with pytest.raises(ValueError, match="series_labels"):
        plot_sweep_line(
            [1, 2],
            [[0.1, 0.2], [0.2, 0.3]],
            out_path=tmp_path / "bad.png",
            series_labels=["only-one"],
        )


def test_line_plot_rejects_empty_values(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        plot_sweep_line([], [], out_path=tmp_path / "bad.png")


def test_heatmap_writes_valid_png(tmp_path):
    grid = np.array([[0.10, 0.08, 0.06], [0.07, 0.05, 0.04]], dtype=np.float64)
    out = tmp_path / "hidden_x_layers.png"

    returned = plot_sweep_heatmap(
        [32, 64, 128],
        [1, 2],
        grid,
        out_path=out,
        axis_x_name="hidden_size",
        axis_y_name="num_layers",
    )

    assert returned == out
    assert out.is_file()
    width, height = _png_dimensions(out)
    assert width > 100 and height > 100


def test_heatmap_creates_parent_directory(tmp_path):
    nested = tmp_path / "runs" / "sweeps" / "grid.png"
    plot_sweep_heatmap(
        [1, 2],
        [10, 20],
        [[0.1, 0.2], [0.05, 0.06]],
        out_path=nested,
    )
    assert nested.is_file()


def test_heatmap_rejects_shape_mismatch(tmp_path):
    with pytest.raises(ValueError, match="mse_grid shape"):
        plot_sweep_heatmap(
            [1, 2, 3],
            [10, 20],
            [[0.1, 0.2], [0.05, 0.06]],
            out_path=tmp_path / "bad.png",
        )


def test_heatmap_rejects_empty_axes(tmp_path):
    with pytest.raises(ValueError, match="non-empty"):
        plot_sweep_heatmap(
            [],
            [10, 20],
            np.zeros((2, 0)),
            out_path=tmp_path / "bad.png",
        )
