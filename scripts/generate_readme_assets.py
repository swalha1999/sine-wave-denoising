"""Generate the figures and summary embedded in the README.

Trains MLP / RNN / LSTM with ``config/default.json``, evaluates each on the
held-out test set, runs a noise-sigma robustness sweep, and writes:

    docs/assets/
      training_curves.png       — train/val MSE per epoch, all three models
      predictions_<model>.png   — predicted vs. clean window per component
      sweep_sigma.png           — test MSE vs. noise sigma, all three models
      test_mse_per_component.png — bar chart of per-component test MSE
      summary.json              — numbers reused in the README findings
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from sine_denoiser.evaluation.robustness import sweep_noise
from sine_denoiser.plotting.predictions import plot_predictions
from sine_denoiser.sdk import SDK

REPO = Path(__file__).resolve().parent.parent
ASSETS = REPO / "docs" / "assets"
RUNS = REPO / "runs" / "readme"
MODELS = ["mlp", "rnn", "lstm"]
SIGMAS = [0.0, 0.25, 0.5, 1.0, 2.0]


def train_all(sdk: SDK):
    runs, reports = {}, {}
    for name in MODELS:
        print(f"[train] {name}")
        run = sdk.train(name, seed=0, run_dir=RUNS / name)
        report = sdk.evaluate(run)
        runs[name] = run
        reports[name] = report
        print(f"  best_epoch={run.fit_result.best_epoch} test_mse={report.test_mse:.5f}")
    return runs, reports


def plot_combined_training_curves(runs, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for name in MODELS:
        history = [asdict(m) for m in runs[name].fit_result.history]
        epochs = [int(r["epoch"]) for r in history]
        val = [float(r["val_mse"]) for r in history]
        ax.plot(epochs, val, marker="o", label=f"{name} (val)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation MSE")
    ax.set_yscale("log")
    ax.set_title("Validation MSE per epoch")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_combined_sweep(sdk: SDK, runs, out_path: Path):
    bundle = sdk.generate_data()
    cfg = sdk.config
    series = {}
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for name in MODELS:
        print(f"[sweep] {name}")
        result = sweep_noise(
            runs[name].model,
            bundle.pure,
            sigmas=SIGMAS,
            context_window=int(cfg["data"]["context_window"]),
            split=cfg["data"]["split"],
            batch_size=int(cfg["training"]["batch_size"]),
            seed=int(cfg["data"]["data_seed"]),
        )
        ax.plot(result.sigmas, result.mses, marker="o", label=name)
        series[name] = [
            {"sigma": float(s), "mse": float(m)}
            for s, m in zip(result.sigmas, result.mses)
        ]
    ax.set_xlabel("noise sigma")
    ax.set_ylabel("test MSE")
    ax.set_yscale("log")
    ax.set_title("Robustness: test MSE vs. noise level")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return series


def plot_per_component_bars(reports, out_path: Path) -> None:
    num_components = len(next(iter(reports.values())).test_mse_per_component)
    width = 0.25
    xs = np.arange(num_components)
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for i, name in enumerate(MODELS):
        per_c = reports[name].test_mse_per_component
        ax.bar(xs + (i - 1) * width, per_c, width=width, label=name)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"comp {k}" for k in range(num_components)])
    ax.set_ylabel("test MSE")
    ax.set_title("Per-component test MSE by model")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    sdk = SDK(REPO / "config" / "default.json")
    runs, reports = train_all(sdk)

    plot_combined_training_curves(runs, ASSETS / "training_curves.png")

    bundle = sdk.generate_data()
    num_components = int(sdk.config["data"]["num_components"])
    for name in MODELS:
        out_dir = ASSETS / f"predictions_{name}"
        plot_predictions(
            runs[name].model,
            bundle.loaders.test,
            out_dir=out_dir,
            num_components=num_components,
            model_name=name,
            examples_per_component=2,
        )

    sweep_series = plot_combined_sweep(sdk, runs, ASSETS / "sweep_sigma.png")
    plot_per_component_bars(reports, ASSETS / "test_mse_per_component.png")

    summary = {
        "config": "config/default.json",
        "seed": 0,
        "noise_sigma_train": float(sdk.config["data"]["noise_sigma"]),
        "models": {
            name: {
                "best_epoch": runs[name].fit_result.best_epoch,
                "best_val_mse": float(runs[name].fit_result.best_val_mse),
                "test_mse": float(reports[name].test_mse),
                "test_mse_per_component": [
                    float(v) for v in reports[name].test_mse_per_component
                ],
            }
            for name in MODELS
        },
        "sweep_sigma": sweep_series,
    }
    (ASSETS / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[done] wrote summary + figures under {ASSETS}")


if __name__ == "__main__":
    main()
