"""Training curves: train/val MSE per epoch as a single PNG."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, is_dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sine_denoiser.training.loop import EpochMetrics, FitResult


def _coerce_history(
    source: FitResult | Iterable[EpochMetrics] | Iterable[dict] | Path | str,
) -> list[dict]:
    if isinstance(source, FitResult):
        return [asdict(m) for m in source.history]
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.is_dir():
            path = path / "metrics.json"
        payload = json.loads(path.read_text())
        return list(payload["history"])
    rows: list[dict] = []
    for row in source:
        if is_dataclass(row):
            rows.append(asdict(row))
        else:
            rows.append(dict(row))
    return rows


def plot_training_curves(
    source: FitResult | Iterable[EpochMetrics] | Iterable[dict] | Path | str,
    out_path: Path | str,
    *,
    title: str | None = None,
) -> Path:
    """Plot train/val MSE per epoch and save the PNG to ``out_path``.

    ``source`` may be a :class:`FitResult`, an iterable of
    :class:`EpochMetrics` (or equivalent dicts), or a path to a run
    directory / ``metrics.json`` written by :func:`training.loop.fit`.
    The parent directory of ``out_path`` is created if needed. Returns
    the resolved output path.
    """
    history = _coerce_history(source)
    if not history:
        raise ValueError("history is empty; nothing to plot")

    epochs = [int(row["epoch"]) for row in history]
    train = [float(row["train_mse"]) for row in history]
    val = [float(row["val_mse"]) for row in history]
    best_epoch = min(history, key=lambda row: float(row["val_mse"]))["epoch"]

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    try:
        ax.plot(epochs, train, marker="o", label="train")
        ax.plot(epochs, val, marker="o", label="val")
        ax.axvline(
            best_epoch,
            color="grey",
            linestyle="--",
            alpha=0.6,
            label=f"best epoch ({best_epoch})",
        )
        ax.set_xlabel("epoch")
        ax.set_ylabel("MSE")
        ax.set_yscale("log")
        ax.set_title(title or "Training curves")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out, dpi=120)
    finally:
        plt.close(fig)
    return out
