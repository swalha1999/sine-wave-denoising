"""Training-curve plot: train and validation MSE per epoch."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sine_denoiser.training.loop import EpochMetrics, FitResult


def plot_training_curve(
    history: FitResult | Iterable[EpochMetrics],
    out_path: Path | str,
    *,
    title: str | None = None,
) -> Path:
    """Render train/val MSE per epoch and write a PNG to ``out_path``.

    Accepts either a ``FitResult`` or its raw ``history`` list. Marks the best
    epoch (lowest val MSE) with a vertical line. Returns the resolved path.
    """
    rows = list(_history_rows(history))
    if not rows:
        raise ValueError("history is empty; nothing to plot")

    epochs = [m.epoch for m in rows]
    train = [m.train_mse for m in rows]
    val = [m.val_mse for m in rows]
    best_epoch = min(rows, key=lambda m: m.val_mse).epoch

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    try:
        ax.plot(epochs, train, marker="o", label="train MSE")
        ax.plot(epochs, val, marker="o", label="val MSE")
        ax.axvline(
            best_epoch, color="grey", linestyle="--", alpha=0.6,
            label=f"best epoch ({best_epoch})",
        )
        ax.set_xlabel("epoch")
        ax.set_ylabel("MSE")
        ax.set_yscale("log")
        ax.set_title(title or "Training curve")
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out, dpi=120)
    finally:
        plt.close(fig)
    return out


def _history_rows(
    history: FitResult | Iterable[EpochMetrics],
) -> Iterable[EpochMetrics]:
    if isinstance(history, FitResult):
        return history.history
    return history
