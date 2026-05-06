# Sine Wave Denoising

A controlled comparison of three neural architectures — fully connected (MLP),
vanilla RNN, and LSTM — on a denoising task. Given a one-hot selector and a
short noisy context window from a mixture of four sine waves, each model learns
to recover the corresponding clean window from the *selected* component.

## Task

- Generate **4** sine waves with distinct frequencies/phases/amplitudes.
- Sample each for **10 s at 1000 Hz** → 10,000 samples per signal.
- Add per-sample Gaussian noise (configurable σ) and sum to form the mixed signal.
- **Input:** one-hot selector `c ∈ {0,1,2,3}` + a 10-sample window from the
  noisy mixture.
- **Target:** the aligned 10-sample window from the chosen *clean* sine.
- Train all three models with MSE; compare per-component test MSE, robustness
  curves, and qualitative predictions.

## Install

Requires **Python ≥ 3.12** and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/swalha1999/sine-wave-denoising.git
cd sine-wave-denoising
uv sync
```

Optional: copy `.env-example` to `.env` if you want to override defaults like
output directories or pin a global seed.

```bash
cp .env-example .env
```

## Quickstart

Train all three models end-to-end with the default config:

```bash
uv run python -m sine_denoiser.train --config config/default.json
```


Each model prints a one-line JSON report with the best epoch, validation MSE,
and test MSE (total + per component):

```json
{"model": "lstm", "seed": 0, "best_epoch": 12, "best_val_mse": 0.043,
 "test_mse": 0.045, "test_mse_per_component": [0.04, 0.05, 0.05, 0.04]}
```

### CLI options

| Flag | Default | Description |
|---|---|---|
| `--config PATH` | required | JSON config (see `config/default.json`). |
| `--model NAME` | all | `mlp`, `rnn`, or `lstm`. Repeat to select several. |
| `--seed INT` | `0` | Seed for model init and training. |
| `--run-dir DIR` | none | If set, persists `best.pt`, `metrics.json`, `training_curve.png`, and `config_snapshot.json` under `<run-dir>/<model>/`. |

Examples:

```bash
# Train just the LSTM and persist artifacts.
uv run python -m sine_denoiser.train \
  --config config/default.json --model lstm --run-dir runs/quick

# Train MLP and RNN with a non-default seed.
uv run python -m sine_denoiser.train \
  --config config/default.json --model mlp --model rnn --seed 1
```

## Using the SDK

The `SDK` class is the single public surface — CLI and notebooks call only it,
not internals.

```python
from sine_denoiser import SDK

sdk = SDK("config/default.json")

run = sdk.train("lstm", seed=0, run_dir="runs/lstm-seed0")
report = sdk.evaluate(run)
print(report.test_mse, report.test_mse_per_component)

# Predict a clean window from a noisy context + component selector.
import numpy as np
x_ctx = np.zeros(10, dtype=np.float32)   # one noisy 10-sample window
y_hat = sdk.predict(run, x_ctx, c=2)
```

`SDK(config=...)` accepts an in-memory dict instead of a path, useful for
parameter sweeps from notebooks.

## Configuration

All hyperparameters live in JSON under `config/`. Each file carries a
`version` field. Override the defaults by copying and editing
`config/default.json`.

### `data` block

| Key | Type | Notes |
|---|---|---|
| `num_components` | int | Number of pure sines (default 4). |
| `duration_s` | float | Signal length in seconds. |
| `sample_rate_hz` | int | Sampling rate. |
| `noise_sigma` | float | Per-sample Gaussian noise σ. |
| `frequencies_hz` | list[float] | Length must equal `num_components`. |
| `phases_rad` | list[float] | Same length constraint. |
| `amplitudes` | list[float] | Same length constraint. |
| `context_window` | int | Window width in samples (default 10). |
| `split` | object | `{"train": 0.8, "val": 0.1, "test": 0.1}`; sums to ≤ 1. |
| `data_seed` | int | Seeds noise + split. |

### `model` block

Each model name maps to its own subdict. Keys passed through to the model:

- **`mlp`**: `hidden_size`, `num_layers`, `activation` (`"relu"`/`"tanh"`), `dropout`.
- **`rnn`**: `hidden_size`, `num_layers`, `nonlinearity` (`"tanh"`/`"relu"`), `dropout`.
- **`lstm`**: `hidden_size`, `num_layers`, `dropout`, `bidirectional`.

### `training` block

| Key | Default | Notes |
|---|---|---|
| `optimizer` | `"adam"` | Also `"sgd"`. |
| `lr` | `1e-3` | Learning rate. |
| `batch_size` | `256` | |
| `epochs` | `50` | Hard cap. |
| `early_stopping_patience` | `5` | `0` disables early stopping. |
| `seeds` | `[0,1,2]` | Seeds reported in multi-seed analyses. |

## Plots

Helpers in `sine_denoiser.plotting` write PNGs:

```python
from sine_denoiser.plotting.curves import plot_training_curves
from sine_denoiser.plotting.predictions import plot_predictions
from sine_denoiser.plotting.sweep import plot_sweep_line

plot_training_curves(run.fit_result, "runs/lstm-seed0/training_curve.png")
plot_predictions(
    run.model, sdk.generate_data().loaders.test,
    out_dir="runs/lstm-seed0/preds",
    num_components=4, model_name="lstm",
)
```

Robustness sweep over noise σ:

```python
from sine_denoiser.evaluation.robustness import sweep_noise

bundle = sdk.generate_data()
result = sweep_noise(
    run.model, bundle.pure, sigmas=[0.0, 0.25, 0.5, 1.0, 2.0],
    context_window=sdk.config["data"]["context_window"],
    split=sdk.config["data"]["split"],
    batch_size=sdk.config["training"]["batch_size"],
)
plot_sweep_line(result, out_path="runs/lstm-seed0/sweep_sigma.png",
                axis_name="noise sigma")
```

## Project layout

```
config/                 JSON configs (versioned)
docs/                   PRD, plan, model spec, TODO
src/sine_denoiser/
  data/                 signal generation, noise, dataset, loader
  models/               MLP, RNN, LSTM + registry
  training/             fit loop, optimizer factory
  evaluation/           metrics, robustness sweep
  plotting/             training curves, predictions, sweeps
  shared/               version
  sdk.py                public entry class
  train.py              `python -m sine_denoiser.train`
tests/                  unit/ mirrors src/, integration/ covers end-to-end
```

## Development

```bash
uv sync                           # install + dev deps
uv run pytest                     # full test suite
uv run pytest --cov=src --cov-fail-under=85
uv run ruff check .               # lint
uv run ruff format .              # autoformat
```

Tests follow `tests/unit/<module>/test_<file>.py` and mirror `src/`.
Integration tests under `tests/integration/` exercise the SDK and CLI on a
small-scale config.

## License

Coursework for *Orchestration of AI Agents*. No license granted for external
reuse without permission.

---

## Findings

The figures below were produced by `scripts/generate_readme_assets.py`,
which trains all three models on `config/default.json` (seed 0,
σ\_train = 0.5), evaluates them on the held-out test split, and runs a
robustness sweep over noise σ ∈ {0.0, 0.25, 0.5, 1.0, 2.0}. Numbers cited
here come from the resulting `docs/assets/summary.json`.

### 1. Validation curves — all three models hit the same floor

![Validation MSE per epoch](docs/assets/training_curves.png)

Every model converges to a validation MSE in a narrow band around
**0.36 – 0.37**. The MLP gets there fastest (best epoch **13**), the RNN
takes longer (best epoch **19**), and the LSTM keeps inching down for the
longest before early stopping fires (best epoch **33**). None of the
recurrent gains buy us a meaningfully lower floor — the bottleneck isn't
optimization, it's information.

### 2. Robustness — MLP degrades the slowest under heavy noise

![Test MSE vs. noise sigma](docs/assets/sweep_sigma.png)

When σ is small (0.0 – 0.5) the three curves are stacked within ~0.02 MSE
of each other. The interesting regime is σ ≥ 1.0:

| σ_test | MLP    | RNN    | LSTM   |
|-------:|-------:|-------:|-------:|
| 0.0    | 0.351  | 0.362  | 0.358  |
| 0.5    | 0.370  | 0.384  | 0.381  |
| 1.0    | 0.402  | 0.431  | 0.434  |
| 2.0    | **0.499** | 0.578  | 0.620  |

At σ = 2.0 the LSTM is **24 % worse** than the MLP. The RNN and LSTM, both
of which see the input as a 10-step sequence, overfit to noise structure
that doesn't generalize once the noise distribution shifts. The plain
MLP, treating the window as a flat 10-vector, is the most robust.

### 3. Per-component MSE — low frequencies are easier

![Per-component test MSE by model](docs/assets/test_mse_per_component.png)

Component 0 (2 Hz) is consistently the easiest target across all models;
components 1 – 3 (5 Hz, 11 Hz, 17 Hz) are 8 – 14 % harder. With a 10-sample
context at 1000 Hz sampling, a 2 Hz wave moves only ≈ 7° per sample, so
the window looks nearly linear and is well-fit. By 17 Hz the same window
spans ≈ 60° of phase — the sinusoid is no longer locally trivial and the
noise dominates what little curvature is visible.

### 4. Predicted vs. clean — models learn a bias, not the shape

![MLP predictions on component 0](docs/assets/predictions_mlp/mlp_component_0.png)

This is the most informative figure in the set. The orange dashed line
(predicted) is roughly constant inside each window while the blue line
(clean target) is clearly trending. The model is essentially predicting
a per-component, per-context-window offset rather than tracking the
sinusoid's shape. Equivalent plots for the RNN
(`docs/assets/predictions_rnn/`) and LSTM
(`docs/assets/predictions_lstm/`) tell the same story.

This is consistent with the headline numbers: an MSE of ≈ 0.37 against a
unit-amplitude sine is approximately what you get from predicting the
window mean. The 10-sample context window simply does not carry enough
information about a 17 Hz wave under σ = 0.5 noise to recover the
*shape* — only an amplitude estimate.

### Take-aways

- **Architecture matters less than information bandwidth.** The choice
  between MLP / RNN / LSTM moves the floor by < 5 % on this task.
  The dominant constraint is the 10-sample window length relative to
  the higher-frequency components.
- **Simpler is more robust.** When the test-time noise level exceeds
  what was seen in training, the MLP generalizes best; the recurrent
  models compound their errors.
- **Want a real win? Widen the window.** The clearest path to improving
  performance on this task is to feed the models a longer context (or
  Fourier features), not to swap one architecture for another.

To reproduce these figures locally:

```bash
uv run python scripts/generate_readme_assets.py
```
