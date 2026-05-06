"""Sweep plots: line plot per axis, heatmap for two-axis grids."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from sine_denoiser.evaluation.robustness import SweepResult


def _coerce_xs_ys(
    values: Sequence[float] | SweepResult,
    mses: Sequence[float] | Sequence[Sequence[float]] | None,
    series_labels: Sequence[str] | None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if isinstance(values, SweepResult):
        if mses is not None:
            raise ValueError("mses must be None when values is a SweepResult")
        xs = values.sigmas
        ys = values.mses[None, :]
        labels = list(series_labels) if series_labels else ["test MSE"]
        return xs, ys, labels
    xs = np.asarray(list(values), dtype=np.float64)
    if mses is None:
        raise ValueError("mses is required when values is not a SweepResult")
    arr = np.asarray(mses, dtype=np.float64)
    if arr.ndim == 1:
        ys = arr[None, :]
        labels = list(series_labels) if series_labels else ["test MSE"]
    elif arr.ndim == 2:
        ys = arr
        labels = (
            list(series_labels)
            if series_labels
            else [f"series {i}" for i in range(ys.shape[0])]
        )
    else:
        raise ValueError(f"mses must be 1D or 2D, got ndim={arr.ndim}")
    if ys.shape[1] != xs.shape[0]:
        raise ValueError(
            f"mses last dim {ys.shape[1]} does not match values length {xs.shape[0]}"
        )
    if len(labels) != ys.shape[0]:
        raise ValueError(
            f"series_labels length {len(labels)} != number of curves {ys.shape[0]}"
        )
    return xs, ys, labels


def plot_sweep_line(
    values: Sequence[float] | SweepResult,
    mses: Sequence[float] | Sequence[Sequence[float]] | None = None,
    *,
    out_path: Path | str,
    axis_name: str = "value",
    series_labels: Sequence[str] | None = None,
    title: str | None = None,
) -> Path:
    """Plot test MSE vs. axis values and save the PNG to ``out_path``.

    ``values`` may be a :class:`SweepResult` (uses its sigmas/mses), or a
    sequence of axis values together with ``mses``. ``mses`` can be 1D for a
    single curve or 2D where each row is a separate series (e.g. one model
    per row); ``series_labels`` names the rows. The parent directory of
    ``out_path`` is created if needed. Returns the resolved output path.
    """
    xs, ys, labels = _coerce_xs_ys(values, mses, series_labels)
    if xs.shape[0] == 0:
        raise ValueError("values is empty; nothing to plot")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    try:
        for label, row in zip(labels, ys):
            ax.plot(xs, row, marker="o", label=label)
        ax.set_xlabel(axis_name)
        ax.set_ylabel("test MSE")
        if np.all(ys > 0):
            ax.set_yscale("log")
        ax.set_title(title or f"Sweep over {axis_name}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out, dpi=120)
    finally:
        plt.close(fig)
    return out


def plot_sweep_heatmap(
    values_x: Sequence[float],
    values_y: Sequence[float],
    mse_grid: Sequence[Sequence[float]] | np.ndarray,
    *,
    out_path: Path | str,
    axis_x_name: str = "x",
    axis_y_name: str = "y",
    title: str | None = None,
) -> Path:
    """Plot test MSE as a heatmap over two axes and save the PNG.

    ``mse_grid`` must have shape ``(len(values_y), len(values_x))`` — rows
    indexed by ``values_y``, columns by ``values_x``. The parent directory of
    ``out_path`` is created if needed. Returns the resolved output path.
    """
    xs = np.asarray(list(values_x), dtype=np.float64)
    ys = np.asarray(list(values_y), dtype=np.float64)
    grid = np.asarray(mse_grid, dtype=np.float64)
    if xs.shape[0] == 0 or ys.shape[0] == 0:
        raise ValueError("values_x and values_y must be non-empty")
    if grid.shape != (ys.shape[0], xs.shape[0]):
        raise ValueError(
            f"mse_grid shape {grid.shape} != ({ys.shape[0]}, {xs.shape[0]})"
        )

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    try:
        im = ax.imshow(grid, aspect="auto", origin="lower", cmap="viridis")
        ax.set_xticks(range(xs.shape[0]))
        ax.set_xticklabels([f"{v:g}" for v in xs])
        ax.set_yticks(range(ys.shape[0]))
        ax.set_yticklabels([f"{v:g}" for v in ys])
        ax.set_xlabel(axis_x_name)
        ax.set_ylabel(axis_y_name)
        ax.set_title(title or f"Sweep over {axis_x_name} × {axis_y_name}")
        fig.colorbar(im, ax=ax, label="test MSE")
        fig.tight_layout()
        fig.savefig(out, dpi=120)
    finally:
        plt.close(fig)
    return out
