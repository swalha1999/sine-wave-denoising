"""End-to-end pipeline tests for the public ``SDK`` class."""

from __future__ import annotations

import json

import pytest
import torch

from sine_denoiser import SDK, DataBundle, EvaluationReport, RunArtifacts


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


def test_sdk_loads_config_from_path(tmp_path):
    cfg = _tiny_config()
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(cfg))
    sdk = SDK(cfg_path)
    assert sdk.config["version"] == "1.00"


def test_sdk_requires_exactly_one_of_path_or_config():
    with pytest.raises(ValueError):
        SDK()
    with pytest.raises(ValueError):
        SDK("a.json", config=_tiny_config())


def test_generate_data_returns_cached_bundle():
    sdk = SDK(config=_tiny_config())
    bundle = sdk.generate_data()
    assert isinstance(bundle, DataBundle)
    assert bundle.pure.shape == (4, 200)
    assert bundle.mixed.shape == (200,)
    again = sdk.generate_data()
    assert again is bundle


def test_full_pipeline_train_evaluate_predict(tmp_path):
    sdk = SDK(config=_tiny_config())
    run_dir = tmp_path / "run"
    run = sdk.train("mlp", seed=0, run_dir=run_dir)

    assert isinstance(run, RunArtifacts)
    assert run.model_name == "mlp"
    assert run.seed == 0
    assert run.run_dir == run_dir
    assert (run_dir / "best.pt").exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "config_snapshot.json").exists()
    snapshot = json.loads((run_dir / "config_snapshot.json").read_text())
    assert snapshot["version"] == "1.00"

    report = sdk.evaluate(run)
    assert isinstance(report, EvaluationReport)
    assert report.model_name == "mlp"
    assert report.test_mse >= 0.0
    assert len(report.test_mse_per_component) == 4
    assert all(v >= 0.0 for v in report.test_mse_per_component)

    bundle = sdk.generate_data()
    x = bundle.mixed[:10]
    y_hat = sdk.predict(run, x, 0)
    assert isinstance(y_hat, torch.Tensor)
    assert y_hat.shape == (1, 10)


def test_train_seeding_is_deterministic():
    cfg = _tiny_config()
    sdk_a = SDK(config=cfg)
    sdk_b = SDK(config=cfg)
    run_a = sdk_a.train("mlp", seed=42)
    run_b = sdk_b.train("mlp", seed=42)
    for k in run_a.model.state_dict():
        assert torch.equal(run_a.model.state_dict()[k], run_b.model.state_dict()[k])


def test_train_supports_all_three_models():
    sdk = SDK(config=_tiny_config())
    for name in ("mlp", "rnn", "lstm"):
        run = sdk.train(name, seed=0)
        report = sdk.evaluate(run)
        assert report.model_name == name
        assert report.test_mse >= 0.0


def test_predict_accepts_batched_input():
    sdk = SDK(config=_tiny_config())
    run = sdk.train("mlp", seed=0)
    bundle = sdk.generate_data()
    x_batch = bundle.mixed[:30].reshape(3, 10)
    c_batch = [0, 1, 2]
    y_hat = sdk.predict(run, x_batch, c_batch)
    assert y_hat.shape == (3, 10)


def test_train_without_run_dir_skips_persistence(tmp_path):
    sdk = SDK(config=_tiny_config())
    run = sdk.train("mlp", seed=0)
    assert run.run_dir is None
    assert list(tmp_path.iterdir()) == []
