import abc

import pytest
import torch
from torch import nn

from sine_denoiser.models.base import DenoiserModel


def test_subclass_without_forward_fails_to_instantiate():
    class Incomplete(DenoiserModel):
        pass

    with pytest.raises(TypeError):
        Incomplete()


def test_subclass_with_forward_instantiates():
    class Concrete(DenoiserModel):
        name = "concrete"

        def __init__(self) -> None:
            super().__init__()

        def forward(self, x_ctx, c):
            return x_ctx

    model = Concrete()
    assert isinstance(model, nn.Module)
    assert isinstance(model, DenoiserModel)
    x = torch.zeros(2, 10)
    c = torch.zeros(2, dtype=torch.long)
    y = model(x, c)
    assert y.shape == (2, 10)


def test_is_abc_and_module_subclass():
    assert issubclass(DenoiserModel, nn.Module)
    assert issubclass(DenoiserModel, abc.ABC)


def test_from_config_default_passes_kwargs():
    class Concrete(DenoiserModel):
        name = "concrete"

        def __init__(self, hidden_size: int) -> None:
            super().__init__()
            self.hidden_size = hidden_size

        def forward(self, x_ctx, c):
            return x_ctx

    model = Concrete.from_config({"hidden_size": 32})
    assert isinstance(model, Concrete)
    assert model.hidden_size == 32
