"""Integration tests for ``training.loop.fit``."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from sine_denoiser.data.loader import build_loaders
from sine_denoiser.models import build
from sine_denoiser.training.loop import EpochMetrics, FitResult, fit


def _tiny_loaders():
    rng = np.random.default_rng(0)
    t = 200
    pure = rng.standard_normal((4, t))
    mixed = pure.sum(axis=0) + rng.standard_normal(t) * 0.1
    return build_loaders(
        mixed,
        pure,
        context_window=10,
        split={"train": 0.8, "val": 0.1, "test": 0.1},
        batch_size=16,
        seed=0,
    )


def test_one_epoch_run_persists_checkpoint_and_metrics(tmp_path):
    torch.manual_seed(0)
    loaders = _tiny_loaders()
    model = build("mlp", {"hidden_size": 8, "num_layers": 1})
    run_dir = tmp_path / "run"

    result = fit(
        model,
        loaders.train,
        loaders.val,
        {"optimizer": "adam", "lr": 1e-3, "epochs": 1, "early_stopping_patience": 0},
        run_dir=run_dir,
    )

    assert isinstance(result, FitResult)
    assert result.best_epoch == 1
    assert len(result.history) == 1
    assert isinstance(result.history[0], EpochMetrics)
    assert result.history[0].epoch == 1
    assert result.best_val_mse == result.history[0].val_mse

    assert (run_dir / "best.pt").exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "training_curve.png").is_file()
    payload = json.loads((run_dir / "metrics.json").read_text())
    assert payload["best_epoch"] == 1
    assert len(payload["history"]) == 1
    assert payload["history"][0]["epoch"] == 1
    state = torch.load(run_dir / "best.pt", weights_only=True)
    assert set(state.keys()) == set(model.state_dict().keys())


def test_fit_without_run_dir_skips_persistence(tmp_path):
    torch.manual_seed(0)
    loaders = _tiny_loaders()
    model = build("mlp", {"hidden_size": 4, "num_layers": 1})

    result = fit(
        model,
        loaders.train,
        loaders.val,
        {"epochs": 1},
    )

    assert result.best_epoch == 1
    assert list(tmp_path.iterdir()) == []


def test_early_stopping_halts_when_val_does_not_improve(tmp_path):
    torch.manual_seed(0)
    loaders = _tiny_loaders()
    model = build("mlp", {"hidden_size": 4, "num_layers": 1})

    result = fit(
        model,
        loaders.train,
        loaders.val,
        {"optimizer": "sgd", "lr": 0.0, "epochs": 10, "early_stopping_patience": 2},
    )
    assert result.best_epoch == 1
    assert len(result.history) == 3


def test_fit_rejects_non_positive_epochs():
    loaders = _tiny_loaders()
    model = build("mlp", {"hidden_size": 4, "num_layers": 1})
    with pytest.raises(ValueError, match="epochs"):
        fit(model, loaders.train, loaders.val, {"epochs": 0})


def test_fit_rejects_unknown_optimizer():
    loaders = _tiny_loaders()
    model = build("mlp", {"hidden_size": 4, "num_layers": 1})
    with pytest.raises(ValueError, match="unknown optimizer"):
        fit(model, loaders.train, loaders.val, {"epochs": 1, "optimizer": "rmsprop"})


def test_best_state_dict_can_be_loaded_back(tmp_path):
    torch.manual_seed(0)
    loaders = _tiny_loaders()
    model = build("mlp", {"hidden_size": 8, "num_layers": 1})

    result = fit(
        model,
        loaders.train,
        loaders.val,
        {"epochs": 1},
        run_dir=tmp_path / "run",
    )
    fresh = build("mlp", {"hidden_size": 8, "num_layers": 1})
    fresh.load_state_dict(result.best_state_dict)
    for k, v in fresh.state_dict().items():
        assert torch.equal(v, result.best_state_dict[k])
