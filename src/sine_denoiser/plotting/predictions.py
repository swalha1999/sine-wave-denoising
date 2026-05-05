"""Predicted vs. clean window plots: one PNG per (model, component)."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402


def _gather_examples(
    model: nn.Module,
    loader: Iterable,
    num_components: int,
    examples_per_component: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Run ``model`` over ``loader`` and bin up to ``E`` (y_hat, y) windows
    per component. Returns a list of length ``num_components``; each entry is
    a tuple of stacked ``(E, W)`` arrays, possibly empty.
    """
    yhat_bins: list[list[np.ndarray]] = [[] for _ in range(num_components)]
    y_bins: list[list[np.ndarray]] = [[] for _ in range(num_components)]
    full = lambda: all(  # noqa: E731
        len(y_bins[k]) >= examples_per_component for k in range(num_components)
    )
    model.eval()
    with torch.no_grad():
        for x_np, c_np, y_np in loader:
            if full():
                break
            x = torch.as_tensor(x_np, dtype=torch.float32)
            c = torch.as_tensor(c_np, dtype=torch.long)
            y = torch.as_tensor(y_np, dtype=torch.float32)
            y_hat = model(x, c).detach().cpu().numpy()
            y_arr = y.cpu().numpy()
            c_arr = c.cpu().numpy()
            for i, k in enumerate(c_arr):
                k_int = int(k)
                if k_int < 0 or k_int >= num_components:
                    continue
                if len(y_bins[k_int]) >= examples_per_component:
                    continue
                yhat_bins[k_int].append(y_hat[i])
                y_bins[k_int].append(y_arr[i])
    out: list[tuple[np.ndarray, np.ndarray]] = []
    empty = (np.zeros((0, 0), dtype=np.float32), np.zeros((0, 0), dtype=np.float32))
    for k in range(num_components):
        if y_bins[k]:
            out.append((np.stack(yhat_bins[k]), np.stack(y_bins[k])))
        else:
            out.append(empty)
    return out


def _plot_component_png(
    y_hat: np.ndarray,
    y: np.ndarray,
    out_path: Path,
    *,
    title: str,
) -> Path:
    """Plot ``E`` overlaid (predicted, clean) windows in a grid and save PNG."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_examples, w = y_hat.shape
    cols = min(2, n_examples)
    rows = (n_examples + cols - 1) // cols
    fig, axes = plt.subplots(
        rows, cols, figsize=(4.5 * cols, 3.0 * rows), squeeze=False
    )
    try:
        xs = np.arange(w)
        for i in range(rows * cols):
            ax = axes[i // cols][i % cols]
            if i >= n_examples:
                ax.axis("off")
                continue
            ax.plot(xs, y[i], marker="o", label="clean", color="C0")
            ax.plot(
                xs, y_hat[i], marker="x", linestyle="--", label="predicted", color="C1"
            )
            ax.set_xlabel("t")
            ax.set_ylabel("amplitude")
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.legend(loc="best")
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(out_path, dpi=120)
    finally:
        plt.close(fig)
    return out_path


def plot_predictions(
    model: nn.Module,
    loader: Iterable,
    *,
    out_dir: Path | str,
    num_components: int,
    model_name: str = "model",
    examples_per_component: int = 4,
) -> list[Path]:
    """Save one ``predicted vs. clean`` PNG per component under ``out_dir``.

    ``loader`` yields ``(x_ctx, c, y)`` batches as numpy arrays — same layout
    as :class:`data.loader.SineWindowLoader`. Up to ``examples_per_component``
    test windows are drawn per component; components missing from ``loader``
    are skipped silently. Filenames are ``<model_name>_component_<k>.png``.
    Returns the written paths in component order.
    """
    if num_components <= 0:
        raise ValueError("num_components must be > 0")
    if examples_per_component <= 0:
        raise ValueError("examples_per_component must be > 0")

    examples = _gather_examples(model, loader, num_components, examples_per_component)
    if not any(yh.shape[0] > 0 for yh, _ in examples):
        raise ValueError("loader produced no windows; nothing to plot")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for k, (y_hat, y) in enumerate(examples):
        if y_hat.shape[0] == 0:
            continue
        path = out_dir / f"{model_name}_component_{k}.png"
        _plot_component_png(
            y_hat,
            y,
            path,
            title=f"{model_name} — component {k}: predicted vs. clean",
        )
        written.append(path)
    return written
