"""Unit tests for ``sine_denoiser.sweeps`` (helpers and parsing)."""

from __future__ import annotations

import pytest

from sine_denoiser.sweeps import (
    SweepAxis,
    _resolve_target,
    _set_dotted,
    parse_axes,
)


def test_set_dotted_writes_nested_value():
    cfg = {"a": {"b": {"c": 1}}}
    _set_dotted(cfg, "a.b.c", 42)
    assert cfg["a"]["b"]["c"] == 42


def test_set_dotted_overwrites_existing_leaf():
    cfg = {"training": {"lr": 1e-3}}
    _set_dotted(cfg, "training.lr", 5e-4)
    assert cfg["training"]["lr"] == 5e-4


def test_set_dotted_rejects_empty_path():
    with pytest.raises(ValueError, match="non-empty"):
        _set_dotted({"a": 1}, "", 0)


def test_set_dotted_rejects_missing_intermediate_key():
    with pytest.raises(KeyError, match="missing key"):
        _set_dotted({"a": {}}, "a.b.c", 1)


def test_set_dotted_rejects_non_dict_intermediate():
    with pytest.raises(KeyError, match="non-dict"):
        _set_dotted({"a": 5}, "a.b", 1)


def test_resolve_target_substitutes_model_placeholder():
    assert _resolve_target("model.{model}.hidden_size", "mlp") == "model.mlp.hidden_size"


def test_resolve_target_passes_through_when_no_placeholder():
    assert _resolve_target("training.lr", "rnn") == "training.lr"


def test_parse_axes_builds_sweep_axes():
    axes = parse_axes([
        {"name": "lr", "target": "training.lr", "values": [1e-4, 1e-3]},
        {"name": "h", "target": "model.{model}.hidden_size", "values": [16, 32]},
    ])
    assert len(axes) == 2
    assert isinstance(axes[0], SweepAxis)
    assert axes[0].name == "lr"
    assert axes[0].values == (1e-4, 1e-3)
    assert axes[1].target == "model.{model}.hidden_size"


def test_parse_axes_rejects_missing_keys():
    with pytest.raises(ValueError, match="missing"):
        parse_axes([{"name": "lr", "values": [1.0]}])


def test_parse_axes_rejects_empty_values():
    with pytest.raises(ValueError, match="no values"):
        parse_axes([{"name": "lr", "target": "training.lr", "values": []}])
