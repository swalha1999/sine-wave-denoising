"""CLI entry point: ``python -m sine_denoiser.train --config <path>``."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from sine_denoiser.models.registry import available
from sine_denoiser.sdk import SDK, EvaluationReport, RunArtifacts


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m sine_denoiser.train",
        description="Train sine-wave denoising models end-to-end from a config.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a JSON config (see config/default.json).",
    )
    parser.add_argument(
        "--model",
        action="append",
        choices=available(),
        help=(
            "Model name to train. May be passed multiple times. "
            "Defaults to all registered models."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed used for model init and training (default: 0).",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory to persist artifacts to. Each model writes "
            "to <run-dir>/<model_name>/."
        ),
    )
    return parser.parse_args(argv)


def _train_one(
    sdk: SDK,
    name: str,
    *,
    seed: int,
    run_dir: Path | None,
) -> tuple[RunArtifacts, EvaluationReport]:
    sub_dir = run_dir / name if run_dir is not None else None
    run = sdk.train(name, seed=seed, run_dir=sub_dir)
    report = sdk.evaluate(run)
    return run, report


def main(argv: Sequence[str] | None = None) -> int:
    """Train every requested model and print a one-line report each."""
    args = _parse_args(argv)
    if not args.config.is_file():
        print(f"config not found: {args.config}", file=sys.stderr)
        return 2
    sdk = SDK(args.config)
    names = args.model if args.model else available()
    for name in names:
        run, report = _train_one(sdk, name, seed=args.seed, run_dir=args.run_dir)
        line = {
            "model": name,
            "seed": args.seed,
            "best_epoch": run.fit_result.best_epoch,
            "best_val_mse": run.fit_result.best_val_mse,
            "test_mse": report.test_mse,
            "test_mse_per_component": report.test_mse_per_component,
        }
        print(json.dumps(line))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
