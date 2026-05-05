import pytest
import torch
from torch import nn

from sine_denoiser.models import DenoiserModel
from sine_denoiser.models.mlp import MLP


def _inputs(batch_size: int = 8):
    x_ctx = torch.randn(batch_size, 10)
    c = torch.randint(0, 4, (batch_size,), dtype=torch.long)
    return x_ctx, c


def test_forward_shape_default_config():
    model = MLP()
    x_ctx, c = _inputs(8)
    y = model(x_ctx, c)
    assert y.shape == (8, 10)


def test_is_denoiser_model_subclass():
    assert issubclass(MLP, DenoiserModel)
    assert MLP.name == "mlp"


def test_forward_shape_with_all_config_keys():
    model = MLP(hidden_size=32, num_layers=3, activation="gelu", dropout=0.1)
    x_ctx, c = _inputs(4)
    y = model(x_ctx, c)
    assert y.shape == (4, 10)


@pytest.mark.parametrize("activation", ["relu", "tanh", "gelu"])
def test_supported_activations_build(activation):
    model = MLP(activation=activation)
    x_ctx, c = _inputs(2)
    assert model(x_ctx, c).shape == (2, 10)


def test_unknown_activation_raises():
    with pytest.raises(ValueError, match="activation"):
        MLP(activation="silu")


@pytest.mark.parametrize("bad", [{"hidden_size": 0}, {"num_layers": 0}])
def test_invalid_size_raises(bad):
    with pytest.raises(ValueError):
        MLP(**bad)


@pytest.mark.parametrize("dropout", [-0.1, 1.0, 1.5])
def test_invalid_dropout_raises(dropout):
    with pytest.raises(ValueError, match="dropout"):
        MLP(dropout=dropout)


def test_dropout_zero_omits_dropout_layer():
    model = MLP(hidden_size=8, num_layers=2, dropout=0.0)
    assert not any(isinstance(m, nn.Dropout) for m in model.net)


def test_dropout_positive_includes_dropout_layer():
    model = MLP(hidden_size=8, num_layers=2, dropout=0.2)
    drops = [m for m in model.net if isinstance(m, nn.Dropout)]
    assert len(drops) == 2
    assert all(d.p == pytest.approx(0.2) for d in drops)


def test_num_layers_controls_linear_count():
    model = MLP(hidden_size=8, num_layers=3, dropout=0.0)
    linears = [m for m in model.net if isinstance(m, nn.Linear)]
    assert len(linears) == 4
    assert linears[0].in_features == 4 + 10
    assert linears[0].out_features == 8
    assert linears[-1].out_features == 10


def test_from_config_builds_with_all_keys():
    cfg = {"hidden_size": 16, "num_layers": 2, "activation": "tanh", "dropout": 0.0}
    model = MLP.from_config(cfg)
    assert isinstance(model, MLP)
    x_ctx, c = _inputs(2)
    assert model(x_ctx, c).shape == (2, 10)


def test_forward_rejects_wrong_x_shape():
    model = MLP()
    bad_x = torch.randn(2, 7)
    c = torch.zeros(2, dtype=torch.long)
    with pytest.raises(ValueError, match="x_ctx"):
        model(bad_x, c)


def test_forward_rejects_mismatched_batch():
    model = MLP()
    x_ctx = torch.randn(4, 10)
    c = torch.zeros(3, dtype=torch.long)
    with pytest.raises(ValueError, match="c"):
        model(x_ctx, c)


def test_backward_pass_produces_grads():
    model = MLP(hidden_size=8, num_layers=1, dropout=0.0)
    x_ctx, c = _inputs(4)
    y = model(x_ctx, c)
    y.sum().backward()
    assert all(p.grad is not None for p in model.parameters())
