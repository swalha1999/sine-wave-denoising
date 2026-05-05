import pytest
import torch
from torch import nn

from sine_denoiser.models import DenoiserModel
from sine_denoiser.models.rnn import RNN


def _inputs(batch_size: int = 8):
    x_ctx = torch.randn(batch_size, 10)
    c = torch.randint(0, 4, (batch_size,), dtype=torch.long)
    return x_ctx, c


def test_forward_shape_default_config():
    model = RNN()
    x_ctx, c = _inputs(8)
    y = model(x_ctx, c)
    assert y.shape == (8, 10)


def test_is_denoiser_model_subclass():
    assert issubclass(RNN, DenoiserModel)
    assert RNN.name == "rnn"


def test_forward_shape_with_all_config_keys():
    model = RNN(hidden_size=32, num_layers=2, nonlinearity="relu", dropout=0.1)
    x_ctx, c = _inputs(4)
    y = model(x_ctx, c)
    assert y.shape == (4, 10)


@pytest.mark.parametrize("nonlinearity", ["tanh", "relu"])
def test_supported_nonlinearities_build(nonlinearity):
    model = RNN(nonlinearity=nonlinearity)
    x_ctx, c = _inputs(2)
    assert model(x_ctx, c).shape == (2, 10)


def test_unknown_nonlinearity_raises():
    with pytest.raises(ValueError, match="nonlinearity"):
        RNN(nonlinearity="gelu")


@pytest.mark.parametrize("bad", [{"hidden_size": 0}, {"num_layers": 0}])
def test_invalid_size_raises(bad):
    with pytest.raises(ValueError):
        RNN(**bad)


@pytest.mark.parametrize("dropout", [-0.1, 1.0, 1.5])
def test_invalid_dropout_raises(dropout):
    with pytest.raises(ValueError, match="dropout"):
        RNN(dropout=dropout)


def test_rnn_input_size_matches_spec():
    model = RNN(hidden_size=8, num_layers=1)
    assert model.rnn.input_size == 1 + 4
    assert model.rnn.hidden_size == 8
    assert model.rnn.num_layers == 1
    assert model.rnn.batch_first is True


def test_head_projects_hidden_to_scalar():
    model = RNN(hidden_size=16, num_layers=1)
    assert isinstance(model.head, nn.Linear)
    assert model.head.in_features == 16
    assert model.head.out_features == 1


def test_dropout_propagated_when_multilayer():
    model = RNN(hidden_size=8, num_layers=2, dropout=0.3)
    assert model.rnn.dropout == pytest.approx(0.3)


def test_dropout_ignored_when_single_layer():
    model = RNN(hidden_size=8, num_layers=1, dropout=0.3)
    assert model.rnn.dropout == 0.0


def test_from_config_builds_with_all_keys():
    cfg = {"hidden_size": 16, "num_layers": 1, "nonlinearity": "tanh", "dropout": 0.0}
    model = RNN.from_config(cfg)
    assert isinstance(model, RNN)
    x_ctx, c = _inputs(2)
    assert model(x_ctx, c).shape == (2, 10)


def test_forward_rejects_wrong_x_shape():
    model = RNN()
    bad_x = torch.randn(2, 7)
    c = torch.zeros(2, dtype=torch.long)
    with pytest.raises(ValueError, match="x_ctx"):
        model(bad_x, c)


def test_forward_rejects_mismatched_batch():
    model = RNN()
    x_ctx = torch.randn(4, 10)
    c = torch.zeros(3, dtype=torch.long)
    with pytest.raises(ValueError, match="c"):
        model(x_ctx, c)


def test_backward_pass_produces_grads():
    model = RNN(hidden_size=8, num_layers=1, dropout=0.0)
    x_ctx, c = _inputs(4)
    y = model(x_ctx, c)
    y.sum().backward()
    assert all(p.grad is not None for p in model.parameters())


def test_selector_broadcast_to_every_timestep():
    """Same x_ctx with different selectors must produce different outputs."""
    torch.manual_seed(0)
    model = RNN(hidden_size=16, num_layers=1)
    model.eval()
    x_ctx = torch.randn(1, 10)
    with torch.no_grad():
        y0 = model(x_ctx, torch.tensor([0], dtype=torch.long))
        y1 = model(x_ctx, torch.tensor([1], dtype=torch.long))
    assert not torch.allclose(y0, y1)
