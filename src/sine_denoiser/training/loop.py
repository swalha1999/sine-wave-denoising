"""Fit loop with MSE-based early stopping for denoising models."""

from __future__ import annotations

import copy
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch import Tensor, nn

from sine_denoiser.training.optim import build as build_optimizer


@dataclass(frozen=True)
class EpochMetrics:
    """One row of the per-epoch training history."""

    epoch: int
    train_mse: float
    val_mse: float


@dataclass
class FitResult:
    """Outcome of a ``fit`` call: best epoch + state_dict + history."""

    best_epoch: int
    best_val_mse: float
    history: list[EpochMetrics] = field(default_factory=list)
    best_state_dict: dict[str, Tensor] = field(default_factory=dict)


def _to_tensors(batch) -> tuple[Tensor, Tensor, Tensor]:
    x, c, y = batch
    return (
        torch.as_tensor(x, dtype=torch.float32),
        torch.as_tensor(c, dtype=torch.long),
        torch.as_tensor(y, dtype=torch.float32),
    )


def _run_epoch(
    model: nn.Module,
    loader: Iterable,
    *,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_sse = 0.0
    total_n = 0
    for batch in loader:
        x_ctx, c, y = _to_tensors(batch)
        with torch.set_grad_enabled(training):
            y_hat = model(x_ctx, c)
            loss = nn.functional.mse_loss(y_hat, y)
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        n = y.numel()
        total_sse += loss.item() * n
        total_n += n
    if total_n == 0:
        raise ValueError("loader produced no batches")
    return total_sse / total_n


def fit(
    model: nn.Module,
    train_loader: Iterable,
    val_loader: Iterable,
    cfg: dict,
    *,
    run_dir: Path | str | None = None,
) -> FitResult:
    """Train ``model`` on ``train_loader`` with MSE; early-stop on ``val_loader``.

    ``cfg`` keys: ``epochs`` (required >0), ``optimizer`` (``"adam"``/``"sgd"``,
    default ``"adam"``), ``lr`` (default ``1e-3``), ``early_stopping_patience``
    (default ``0`` meaning disabled). When ``run_dir`` is given, writes
    ``best.pt`` (best state_dict) and ``metrics.json`` (history + summary).
    """
    epochs = int(cfg.get("epochs", 1))
    if epochs <= 0:
        raise ValueError("epochs must be > 0")
    patience = int(cfg.get("early_stopping_patience", 0))
    optimizer = build_optimizer(model, cfg)

    best_val_mse = float("inf")
    best_epoch = 0
    best_state: dict[str, Tensor] = {}
    epochs_no_improve = 0
    history: list[EpochMetrics] = []

    for epoch in range(1, epochs + 1):
        train_mse = _run_epoch(model, train_loader, optimizer=optimizer)
        val_mse = _run_epoch(model, val_loader, optimizer=None)
        history.append(EpochMetrics(epoch=epoch, train_mse=train_mse, val_mse=val_mse))
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if patience > 0 and epochs_no_improve >= patience:
                break

    result = FitResult(
        best_epoch=best_epoch,
        best_val_mse=best_val_mse,
        history=history,
        best_state_dict=best_state,
    )
    if run_dir is not None:
        _persist(Path(run_dir), result)
    return result


def _persist(run_dir: Path, result: FitResult) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(result.best_state_dict, run_dir / "best.pt")
    payload = {
        "best_epoch": result.best_epoch,
        "best_val_mse": result.best_val_mse,
        "history": [asdict(m) for m in result.history],
    }
    (run_dir / "metrics.json").write_text(json.dumps(payload, indent=2))
    if result.history:
        from sine_denoiser.plotting.curves import plot_training_curves

        plot_training_curves(result, run_dir / "training_curve.png")
