import pytest
import torch

from sine_denoiser.models import available, build
from sine_denoiser.models.lstm import LSTM
from sine_denoiser.models.mlp import MLP
from sine_denoiser.models.rnn import RNN


@pytest.mark.parametrize(
    ("name", "expected_cls"),
    [("mlp", MLP), ("rnn", RNN), ("lstm", LSTM)],
)
def test_build_returns_correct_subclass(name, expected_cls):
    model = build(name, {})
    assert isinstance(model, expected_cls)
    assert model.name == name


def test_build_accepts_none_cfg():
    model = build("mlp")
    assert isinstance(model, MLP)


def test_build_forwards_cfg_to_subclass():
    model = build("mlp", {"hidden_size": 16, "num_layers": 1, "dropout": 0.0})
    x_ctx = torch.randn(2, 10)
    c = torch.zeros(2, dtype=torch.long)
    assert model(x_ctx, c).shape == (2, 10)


def test_build_unknown_name_raises():
    with pytest.raises(ValueError, match="unknown model"):
        build("transformer")


def test_available_lists_all_models():
    assert available() == ["lstm", "mlp", "rnn"]


def test_build_does_not_mutate_caller_cfg():
    cfg = {"hidden_size": 8, "num_layers": 1}
    build("mlp", cfg)
    assert cfg == {"hidden_size": 8, "num_layers": 1}
