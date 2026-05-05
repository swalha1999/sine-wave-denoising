import pytest
import torch
from torch import nn

from sine_denoiser.training.optim import available, build


@pytest.fixture
def model() -> nn.Module:
    return nn.Linear(4, 2)


def test_available_lists_adam_and_sgd():
    assert available() == ["adam", "sgd"]


def test_build_adam_returns_adam_with_configured_lr(model):
    opt = build(model, {"optimizer": "adam", "lr": 5e-4})
    assert isinstance(opt, torch.optim.Adam)
    assert opt.param_groups[0]["lr"] == pytest.approx(5e-4)


def test_build_sgd_returns_sgd_with_configured_lr(model):
    opt = build(model, {"optimizer": "sgd", "lr": 1e-2})
    assert isinstance(opt, torch.optim.SGD)
    assert opt.param_groups[0]["lr"] == pytest.approx(1e-2)


def test_build_optimizer_name_is_case_insensitive(model):
    opt = build(model, {"optimizer": "Adam"})
    assert isinstance(opt, torch.optim.Adam)


def test_build_defaults_to_adam_with_default_lr(model):
    opt = build(model)
    assert isinstance(opt, torch.optim.Adam)
    assert opt.param_groups[0]["lr"] == pytest.approx(1e-3)


def test_build_accepts_none_cfg(model):
    opt = build(model, None)
    assert isinstance(opt, torch.optim.Adam)


def test_build_unknown_name_raises(model):
    with pytest.raises(ValueError, match="unknown optimizer"):
        build(model, {"optimizer": "rmsprop"})


def test_build_attaches_model_parameters(model):
    opt = build(model, {"optimizer": "sgd", "lr": 1e-2})
    optimized = {id(p) for group in opt.param_groups for p in group["params"]}
    assert optimized == {id(p) for p in model.parameters()}


def test_build_does_not_mutate_caller_cfg(model):
    cfg = {"optimizer": "sgd", "lr": 1e-2}
    build(model, cfg)
    assert cfg == {"optimizer": "sgd", "lr": 1e-2}
