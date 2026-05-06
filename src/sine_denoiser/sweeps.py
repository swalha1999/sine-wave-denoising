"""Hyperparameter sweep runner: trains each model across multiple axes."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sine_denoiser.models.registry import available
from sine_denoiser.plotting.sweep import plot_sweep_line
from sine_denoiser.sdk import SDK


@dataclass(frozen=True)
class SweepAxis:
    """One axis of a sweep: a target parameter and the values to try.

    ``target`` is a dotted path into the config (e.g. ``training.lr``). It
    may contain a literal ``{model}`` placeholder, in which case the path is
    resolved per-model when the override is applied.
    """

    name: str
    target: str
    values: tuple[Any, ...]


@dataclass(frozen=True)
class AxisSweepResult:
    """Test/val MSE for each model across the values of one axis."""

    axis: SweepAxis
    test_mse: dict[str, list[float]]
    val_mse: dict[str, list[float]]


def _set_dotted(cfg: dict, dotted_path: str, value: Any) -> None:
    keys = dotted_path.split(".") if dotted_path else []
    if not keys:
        raise ValueError("dotted_path must be non-empty")
    node: Any = cfg
    for k in keys[:-1]:
        if not isinstance(node, dict) or k not in node:
            raise KeyError(f"missing key '{k}' in path '{dotted_path}'")
        node = node[k]
    if not isinstance(node, dict):
        raise KeyError(f"non-dict node at path '{dotted_path}'")
    node[keys[-1]] = value


def _resolve_target(target: str, model: str) -> str:
    return target.replace("{model}", model)


def parse_axes(specs: Iterable[dict]) -> list[SweepAxis]:
    """Build :class:`SweepAxis` objects from JSON-shaped dicts."""
    axes: list[SweepAxis] = []
    for s in specs:
        for key in ("name", "target", "values"):
            if key not in s:
                raise ValueError(f"axis spec missing '{key}': {s!r}")
        values = tuple(s["values"])
        if len(values) == 0:
            raise ValueError(f"axis '{s['name']}' has no values")
        axes.append(SweepAxis(name=str(s["name"]), target=str(s["target"]), values=values))
    return axes


def run_axis_sweep(
    base_config: dict,
    axis: SweepAxis,
    *,
    models: Sequence[str],
    seed: int = 0,
) -> AxisSweepResult:
    """Train each model at every value of ``axis`` and return the MSEs."""
    test_mse: dict[str, list[float]] = {m: [] for m in models}
    val_mse: dict[str, list[float]] = {m: [] for m in models}
    for model in models:
        for value in axis.values:
            cfg = copy.deepcopy(base_config)
            _set_dotted(cfg, _resolve_target(axis.target, model), value)
            sdk = SDK(config=cfg)
            run = sdk.train(model, seed=seed)
            report = sdk.evaluate(run)
            test_mse[model].append(float(report.test_mse))
            val_mse[model].append(float(run.fit_result.best_val_mse))
    return AxisSweepResult(axis=axis, test_mse=test_mse, val_mse=val_mse)


def write_axis_artifacts(
    result: AxisSweepResult, out_dir: Path | str
) -> tuple[Path, Path]:
    """Write the JSON results file and a multi-series PNG for one axis."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / f"{result.axis.name}.json"
    png_path = out / f"{result.axis.name}.png"
    payload = {
        "axis": result.axis.name,
        "target": result.axis.target,
        "values": list(result.axis.values),
        "test_mse": result.test_mse,
        "val_mse": result.val_mse,
    }
    json_path.write_text(json.dumps(payload, indent=2))
    models = list(result.test_mse)
    series = [result.test_mse[m] for m in models]
    plot_sweep_line(
        list(result.axis.values),
        series,
        out_path=png_path,
        axis_name=result.axis.name,
        series_labels=models,
        title=f"test MSE vs. {result.axis.name}",
    )
    return json_path, png_path


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m sine_denoiser.sweeps",
        description="Run hyperparameter sweeps for all models across ≥3 axes.",
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--axis", action="append", dest="axes", default=None)
    parser.add_argument("--model", action="append", choices=available(), default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run every axis defined in ``config["sweep"]`` and persist artifacts."""
    args = _parse_args(argv)
    if not args.config.is_file():
        print(f"config not found: {args.config}", file=sys.stderr)
        return 2
    cfg = json.loads(args.config.read_text())
    sweep_cfg = cfg.get("sweep")
    if not sweep_cfg or "axes" not in sweep_cfg:
        print(f"config missing 'sweep.axes': {args.config}", file=sys.stderr)
        return 2
    base = {k: v for k, v in cfg.items() if k != "sweep"}
    axes = parse_axes(sweep_cfg["axes"])
    if args.axes:
        axes = [a for a in axes if a.name in set(args.axes)]
    models = list(args.model) if args.model else list(sweep_cfg.get("models", available()))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for axis in axes:
        result = run_axis_sweep(base, axis, models=models, seed=args.seed)
        json_path, png_path = write_axis_artifacts(result, args.out_dir)
        print(json.dumps({
            "axis": axis.name,
            "models": models,
            "values": list(axis.values),
            "test_mse": result.test_mse,
            "json": str(json_path),
            "png": str(png_path),
        }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
