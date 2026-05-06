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

Across all tested models, validation MSE converged to a consistent range of **0.36 – 0.37**. Analysis of training logs shows the MLP achieved peak performance most rapidly (best epoch **13**), followed by the RNN (best epoch **19**). The LSTM exhibited the most gradual improvement, reaching its optimal state at epoch **33** before early stopping. The parity in final loss values indicates that the gain from recurrent layers is negligible for this task, pointing to informational constraints rather than architectural limitations.

### 2. Robustness — MLP degrades the slowest under heavy noise

![Test MSE vs. noise sigma](docs/assets/sweep_sigma.png)

For low noise levels (σ = 0.0–0.5), the three curves remain tightly grouped, differing by no more than about 0.02 in MSE. The interesting regime is σ ≥ 1.0:

| σ_test | MLP    | RNN    | LSTM   |
|-------:|-------:|-------:|-------:|
| 0.0    | 0.351  | 0.362  | 0.358  |
| 0.5    | 0.370  | 0.384  | 0.381  |
| 1.0    | 0.402  | 0.431  | 0.434  |
| 2.0    | **0.499** | 0.578  | 0.620  |

At σ = 2.0 the LSTM is **24 %** worse than the MLP. Both the RNN and LSTM, which process the input as a 10-step sequence, tend to overfit to noise structures that fail to generalize during shifts in the noise distribution. Conversely, the plain MLP—treating the input window as a flat 10-vector—demonstrates the most robust performance.

### 3. Per-component MSE — low frequencies are easier

![Per-component test MSE by model](docs/assets/test_mse_per_component.png)

Component 0 (2 Hz) remains the most easily predicted across all models,
while components 1–3 (5 Hz, 11 Hz, 17 Hz) are about 8–14% more difficult. 
using a 10-sample window at a 1000 Hz sampling rate, a 2 Hz signal shifts only about 7° per sample, 
making the segment appear almost linear and easy to model. 
In contrast, at 17 Hz, the same window covers roughly 60° of phase, 
meaning the sinusoidal pattern is no longer locally simple, and noise tends to obscure the limited visible curvature.

### 4. Predicted vs. clean — models learn a bias, not the shape

![MLP predictions on component 0](docs/assets/predictions_mlp/mlp_component_0.png)

This figure is the most revealing in the set. The orange dashed curve (predictions) stays nearly flat within each window,
whereas the blue curve (clean target) shows a clear trend. In practice, the model is learning a per-component, 
per-window offset instead of following the actual sinusoidal pattern. Similar visualizations for the RNN (`docs/assets/predictions_rnn/`) and LSTM (`docs/assets/predictions_lstm/`) demonstrate the same behavior.

This aligns with the reported metrics: an MSE of about 0.37 for a unit-amplitude sine wave is roughly what you would expect when predicting the mean of each window. With a 10-sample context, there isn’t enough information to reconstruct the shape of a 17 Hz signal under σ = 0.5 noise—only a coarse estimate of its amplitude.

### Take-aways

- **Architecture matters less than information bandwidth.** Switching between MLP, RNN, and LSTM changes
  the performance floor by less than 5% on this task.
  The primary limitation comes from the 10-sample window size, especially in relation to the higher-frequency components.
- **Simplicity improves robustness.** When the noise level at test time exceeds what the model
  encountered during training, the MLP shows the strongest generalization,
  while recurrent architectures tend to accumulate and amplify their errors.
- **Want a real win? Widen the window.** The clearest path to improving
  performance on this task is to feed the models a longer context (or
  Fourier features), not to swap one architecture for another.

To reproduce these figures locally:

```bash
uv run python scripts/generate_readme_assets.py
```
