import json
from pathlib import Path

import pytest

CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "default.json"


@pytest.fixture(scope="module")
def cfg() -> dict:
    with CONFIG_PATH.open() as f:
        return json.load(f)


def test_config_file_exists():
    assert CONFIG_PATH.is_file()


def test_version_is_1_00(cfg):
    assert cfg["version"] == "1.00"


def test_data_section_matches_plan(cfg):
    data = cfg["data"]
    assert data["num_components"] == 4
    assert data["duration_s"] == 10.0
    assert data["sample_rate_hz"] == 1000
    assert data["noise_sigma"] == 0.5
    assert data["frequencies_hz"] == [2.0, 5.0, 11.0, 17.0]
    assert data["phases_rad"] == [0.0, 0.7, 1.5, 2.3]
    assert data["amplitudes"] == [1.0, 1.0, 1.0, 1.0]
    assert data["context_window"] == 10
    assert data["split"] == {"train": 0.8, "val": 0.1, "test": 0.1}
    assert data["data_seed"] == 0


def test_model_section_matches_plan(cfg):
    model = cfg["model"]
    assert model["mlp"] == {
        "hidden_size": 64,
        "num_layers": 2,
        "activation": "relu",
        "dropout": 0.0,
    }
    assert model["rnn"] == {
        "hidden_size": 64,
        "num_layers": 1,
        "nonlinearity": "tanh",
        "dropout": 0.0,
    }
    assert model["lstm"] == {
        "hidden_size": 64,
        "num_layers": 1,
        "dropout": 0.0,
        "bidirectional": False,
    }


def test_training_section_matches_plan(cfg):
    training = cfg["training"]
    assert training["optimizer"] == "adam"
    assert training["lr"] == pytest.approx(1e-3)
    assert training["batch_size"] == 256
    assert training["epochs"] == 50
    assert training["early_stopping_patience"] == 5
    assert training["seeds"] == [0, 1, 2]


def test_config_version_matches_package_version(cfg):
    import sine_denoiser

    assert cfg["version"] == sine_denoiser.__version__
