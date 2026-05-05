"""Single public surface wiring data + model + training + eval."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from sine_denoiser.data.loader import Loaders, build_loaders
from sine_denoiser.data.noise import add_gaussian_noise, build_mixed
from sine_denoiser.data.signals import generate_signals
from sine_denoiser.models import build as build_model
from sine_denoiser.models.base import DenoiserModel
from sine_denoiser.plotting.curves import plot_training_curve
from sine_denoiser.training.loop import FitResult, fit


@dataclass(frozen=True)
class DataBundle:
    """Generated signals plus the train/val/test loaders that wrap them."""

    pure: np.ndarray
    mixed: np.ndarray
    loaders: Loaders


@dataclass
class RunArtifacts:
    """Outputs of a single ``SDK.train`` call."""

    model_name: str
    seed: int
    model: DenoiserModel
    fit_result: FitResult
    config_snapshot: dict
    run_dir: Path | None = None


@dataclass(frozen=True)
class EvaluationReport:
    """Test-set metrics for one trained model."""

    model_name: str
    test_mse: float
    test_mse_per_component: list[float]


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _evaluate_test(
    model: DenoiserModel, loader, num_components: int
) -> tuple[float, list[float]]:
    sse_total = 0.0
    n_total = 0
    sse_per_c = [0.0] * num_components
    n_per_c = [0] * num_components
    model.eval()
    with torch.no_grad():
        for x, c, y in loader:
            x_t = torch.as_tensor(x, dtype=torch.float32)
            c_t = torch.as_tensor(c, dtype=torch.long)
            y_t = torch.as_tensor(y, dtype=torch.float32)
            err = (model(x_t, c_t) - y_t).pow(2)
            sse_total += float(err.sum())
            n_total += int(y_t.numel())
            for k in range(num_components):
                mask = c_t == k
                if not mask.any():
                    continue
                sse_per_c[k] += float(err[mask].sum())
                n_per_c[k] += int(y_t[mask].numel())
    total = sse_total / n_total if n_total > 0 else float("nan")
    per_c = [
        sse_per_c[k] / n_per_c[k] if n_per_c[k] > 0 else float("nan")
        for k in range(num_components)
    ]
    return total, per_c


class SDK:
    """The single public surface for the sine wave denoising pipeline."""

    def __init__(
        self,
        config_path: Path | str | None = None,
        *,
        config: dict | None = None,
    ) -> None:
        if (config_path is None) == (config is None):
            raise ValueError("provide exactly one of config_path or config")
        if config is None:
            self.config = json.loads(Path(config_path).read_text())
        else:
            self.config = json.loads(json.dumps(config))
        self._data: DataBundle | None = None

    def generate_data(self) -> DataBundle:
        """Build pure components, the noisy mixed signal, and the loaders."""
        if self._data is not None:
            return self._data
        data_cfg = self.config["data"]
        pure = generate_signals(data_cfg)
        rng = np.random.default_rng(int(data_cfg["data_seed"]))
        noisy = add_gaussian_noise(pure, float(data_cfg["noise_sigma"]), rng)
        mixed = build_mixed(noisy)
        loaders = build_loaders(
            mixed,
            pure,
            context_window=int(data_cfg["context_window"]),
            split=data_cfg["split"],
            batch_size=int(self.config["training"]["batch_size"]),
            seed=int(data_cfg["data_seed"]),
        )
        self._data = DataBundle(pure=pure, mixed=mixed, loaders=loaders)
        return self._data

    def train(
        self,
        model_name: str,
        *,
        seed: int = 0,
        run_dir: Path | str | None = None,
    ) -> RunArtifacts:
        """Train ``model_name`` end-to-end and return run artifacts."""
        _seed_everything(seed)
        bundle = self.generate_data()
        model = build_model(model_name, self.config["model"].get(model_name, {}))
        result = fit(
            model,
            bundle.loaders.train,
            bundle.loaders.val,
            self.config["training"],
            run_dir=run_dir,
        )
        if result.best_state_dict:
            model.load_state_dict(result.best_state_dict)
        rd = Path(run_dir) if run_dir is not None else None
        snapshot = json.loads(json.dumps(self.config))
        if rd is not None:
            rd.mkdir(parents=True, exist_ok=True)
            (rd / "config_snapshot.json").write_text(json.dumps(snapshot, indent=2))
            if result.history:
                plot_training_curve(
                    result,
                    rd / "training_curve.png",
                    title=f"{model_name} (seed={seed})",
                )
        return RunArtifacts(model_name, seed, model, result, snapshot, rd)

    def evaluate(self, run: RunArtifacts) -> EvaluationReport:
        """Compute test MSE total + per-component for a trained run."""
        bundle = self.generate_data()
        num_components = int(self.config["data"]["num_components"])
        total, per_c = _evaluate_test(run.model, bundle.loaders.test, num_components)
        return EvaluationReport(run.model_name, total, per_c)

    def predict(self, run: RunArtifacts, x_ctx, c) -> Tensor:
        """Run the trained model on a window (or batch of windows)."""
        x_t = torch.as_tensor(x_ctx, dtype=torch.float32)
        c_t = torch.as_tensor(c, dtype=torch.long)
        if x_t.ndim == 1:
            x_t = x_t.unsqueeze(0)
        if c_t.ndim == 0:
            c_t = c_t.unsqueeze(0)
        run.model.eval()
        with torch.no_grad():
            return run.model(x_t, c_t)
