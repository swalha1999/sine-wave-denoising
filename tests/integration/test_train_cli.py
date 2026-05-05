"""Integration tests for the ``python -m sine_denoiser.train`` CLI."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from sine_denoiser.models.registry import available
from sine_denoiser.train import main


def _tiny_config() -> dict:
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
    }


def _write_config(tmp_path) -> str:
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(_tiny_config()))
    return str(cfg_path)


def test_main_trains_all_three_models_and_returns_zero(tmp_path, capsys):
    cfg_path = _write_config(tmp_path)
    run_dir = tmp_path / "runs"
    rc = main(["--config", cfg_path, "--run-dir", str(run_dir)])
    assert rc == 0
    out_lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    reported = [json.loads(line)["model"] for line in out_lines]
    assert reported == available()
    for name in available():
        assert (run_dir / name / "best.pt").exists()
        assert (run_dir / name / "metrics.json").exists()
        assert (run_dir / name / "config_snapshot.json").exists()


def test_main_supports_single_model_selection(tmp_path, capsys):
    cfg_path = _write_config(tmp_path)
    rc = main(["--config", cfg_path, "--model", "mlp"])
    assert rc == 0
    out_lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert len(out_lines) == 1
    assert json.loads(out_lines[0])["model"] == "mlp"


def test_main_returns_nonzero_when_config_missing(tmp_path, capsys):
    missing = tmp_path / "nope.json"
    rc = main(["--config", str(missing)])
    assert rc != 0


def test_main_rejects_unknown_model(tmp_path):
    cfg_path = _write_config(tmp_path)
    with pytest.raises(SystemExit):
        main(["--config", cfg_path, "--model", "transformer"])


def test_module_invocation_exits_zero(tmp_path):
    cfg_path = _write_config(tmp_path)
    proc = subprocess.run(
        [sys.executable, "-m", "sine_denoiser.train", "--config", cfg_path],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    reported = [json.loads(line)["model"] for line in proc.stdout.splitlines() if line.strip()]
    assert reported == available()
