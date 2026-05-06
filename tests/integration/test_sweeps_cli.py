"""Integration tests for the ``python -m sine_denoiser.sweeps`` CLI."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from sine_denoiser.sweeps import main


def _tiny_sweep_config() -> dict:
    return {
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
        "model": {
            "mlp": {"hidden_size": 8, "num_layers": 1},
            "rnn": {"hidden_size": 8, "num_layers": 1},
            "lstm": {"hidden_size": 8, "num_layers": 1},
        },
        "training": {
            "optimizer": "adam",
            "lr": 1e-3,
            "batch_size": 32,
            "epochs": 1,
            "early_stopping_patience": 0,
        },
        "sweep": {
            "models": ["mlp", "rnn", "lstm"],
            "axes": [
                {"name": "lr", "target": "training.lr", "values": [1e-3, 5e-3]},
                {"name": "hidden_size", "target": "model.{model}.hidden_size", "values": [4, 8]},
                {"name": "num_layers", "target": "model.{model}.num_layers", "values": [1, 2]},
            ],
        },
    }


def _write_config(tmp_path) -> str:
    cfg_path = tmp_path / "sweep.json"
    cfg_path.write_text(json.dumps(_tiny_sweep_config()))
    return str(cfg_path)


def test_cli_runs_all_axes_and_writes_artifacts(tmp_path, capsys):
    cfg_path = _write_config(tmp_path)
    out_dir = tmp_path / "sweeps"
    rc = main(["--config", cfg_path, "--out-dir", str(out_dir)])
    assert rc == 0
    out_lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    reported = [json.loads(line)["axis"] for line in out_lines]
    assert reported == ["lr", "hidden_size", "num_layers"]
    for axis in ("lr", "hidden_size", "num_layers"):
        json_path = out_dir / f"{axis}.json"
        png_path = out_dir / f"{axis}.png"
        assert json_path.is_file()
        assert png_path.is_file()
        payload = json.loads(json_path.read_text())
        assert payload["axis"] == axis
        assert set(payload["test_mse"]) == {"mlp", "rnn", "lstm"}
        for series in payload["test_mse"].values():
            assert len(series) == len(payload["values"])
            assert all(v >= 0 for v in series)


def test_cli_can_filter_to_subset_of_axes_and_models(tmp_path, capsys):
    cfg_path = _write_config(tmp_path)
    out_dir = tmp_path / "sweeps"
    rc = main([
        "--config", cfg_path,
        "--out-dir", str(out_dir),
        "--axis", "lr",
        "--model", "mlp",
    ])
    assert rc == 0
    out_lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert len(out_lines) == 1
    payload = json.loads(out_lines[0])
    assert payload["axis"] == "lr"
    assert payload["models"] == ["mlp"]
    assert (out_dir / "lr.json").is_file()
    assert not (out_dir / "hidden_size.json").exists()
    on_disk = json.loads((out_dir / "lr.json").read_text())
    assert set(on_disk["test_mse"]) == {"mlp"}


def test_cli_returns_nonzero_when_config_missing(tmp_path):
    rc = main(["--config", str(tmp_path / "nope.json"), "--out-dir", str(tmp_path / "out")])
    assert rc != 0


def test_cli_returns_nonzero_when_sweep_section_missing(tmp_path):
    cfg = _tiny_sweep_config()
    cfg.pop("sweep")
    cfg_path = tmp_path / "sweep.json"
    cfg_path.write_text(json.dumps(cfg))
    rc = main(["--config", str(cfg_path), "--out-dir", str(tmp_path / "out")])
    assert rc != 0


def test_cli_rejects_unknown_model(tmp_path):
    cfg_path = _write_config(tmp_path)
    with pytest.raises(SystemExit):
        main([
            "--config", cfg_path,
            "--out-dir", str(tmp_path / "out"),
            "--model", "transformer",
        ])


def test_module_invocation_exits_zero(tmp_path):
    cfg_path = _write_config(tmp_path)
    out_dir = tmp_path / "sweeps"
    proc = subprocess.run(
        [
            sys.executable,
            "-m", "sine_denoiser.sweeps",
            "--config", cfg_path,
            "--out-dir", str(out_dir),
            "--axis", "lr",
            "--model", "mlp",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert (out_dir / "lr.json").is_file()
