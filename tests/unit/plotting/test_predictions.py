"""Unit tests for ``plotting.predictions.plot_predictions``."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np
import pytest
from torch import Tensor, nn

from sine_denoiser.plotting.predictions import plot_predictions


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


class _IdentityModel(nn.Module):
    """Returns ``x_ctx`` unchanged, ignoring ``c``."""

    def forward(self, x_ctx: Tensor, c: Tensor) -> Tensor:  # noqa: ARG002
        return x_ctx.clone()


def _make_loader(
    *,
    num_components: int,
    num_batches: int,
    batch_size: int,
    window: int = 10,
    seed: int = 0,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    batches: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for _ in range(num_batches):
        x = rng.standard_normal((batch_size, window)).astype(np.float32)
        c = rng.integers(0, num_components, size=batch_size).astype(np.int64)
        y = rng.standard_normal((batch_size, window)).astype(np.float32)
        batches.append((x, c, y))
    return batches


def test_writes_one_png_per_component(tmp_path):
    loader = _make_loader(num_components=4, num_batches=4, batch_size=8)
    paths = plot_predictions(
        _IdentityModel(),
        loader,
        out_dir=tmp_path,
        num_components=4,
        model_name="mlp",
        examples_per_component=3,
    )

    assert len(paths) == 4
    for k, p in enumerate(paths):
        assert p == tmp_path / f"mlp_component_{k}.png"
        assert p.is_file()
        width, height = _png_dimensions(p)
        assert width > 100 and height > 100


def test_skips_components_without_examples(tmp_path):
    rng = np.random.default_rng(1)
    only_c0 = [
        (
            rng.standard_normal((4, 10)).astype(np.float32),
            np.zeros(4, dtype=np.int64),
            rng.standard_normal((4, 10)).astype(np.float32),
        )
    ]

    paths = plot_predictions(
        _IdentityModel(),
        only_c0,
        out_dir=tmp_path,
        num_components=4,
        model_name="lstm",
        examples_per_component=2,
    )

    assert len(paths) == 1
    assert paths[0].name == "lstm_component_0.png"
    assert paths[0].is_file()
    assert not (tmp_path / "lstm_component_1.png").exists()


def test_creates_output_directory(tmp_path):
    nested = tmp_path / "runs" / "lstm" / "predictions"
    loader = _make_loader(num_components=2, num_batches=2, batch_size=4)

    paths = plot_predictions(
        _IdentityModel(),
        loader,
        out_dir=nested,
        num_components=2,
        examples_per_component=2,
    )

    assert nested.is_dir()
    assert len(paths) == 2
    for p in paths:
        assert p.is_file()


def test_caps_examples_per_component(tmp_path):
    loader = _make_loader(num_components=2, num_batches=10, batch_size=16, seed=2)

    paths = plot_predictions(
        _IdentityModel(),
        loader,
        out_dir=tmp_path,
        num_components=2,
        examples_per_component=2,
        model_name="rnn",
    )

    assert {p.name for p in paths} == {
        "rnn_component_0.png",
        "rnn_component_1.png",
    }


def test_validates_arguments(tmp_path):
    loader = _make_loader(num_components=2, num_batches=1, batch_size=2)
    with pytest.raises(ValueError, match="num_components"):
        plot_predictions(
            _IdentityModel(), loader, out_dir=tmp_path, num_components=0
        )
    with pytest.raises(ValueError, match="examples_per_component"):
        plot_predictions(
            _IdentityModel(),
            loader,
            out_dir=tmp_path,
            num_components=2,
            examples_per_component=0,
        )


def test_empty_loader_raises(tmp_path):
    with pytest.raises(ValueError, match="no windows"):
        plot_predictions(
            _IdentityModel(),
            [],
            out_dir=tmp_path,
            num_components=2,
        )
