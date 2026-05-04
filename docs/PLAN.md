# PLAN — Architecture & Decisions

**Status:** Draft v0.1
**Last updated:** 2026-05-04

Companion to `PRD.md`. This document captures *how* the system is structured and the decisions behind it.

---

## 1. C4 — Context

```
+-------------------+        +---------------------------+
|   Researcher      | -----> |  sine-wave-denoising SDK  |
|   (CLI / Jupyter) |        |  (single Python package)  |
+-------------------+        +-------------+-------------+
                                           |
                                           v
                              +----------------------------+
                              | Local filesystem           |
                              | - config/*.json            |
                              | - checkpoints/             |
                              | - runs/  (metrics, plots)  |
                              +----------------------------+
```

No external services, no network calls. Everything runs locally with `uv`.

## 2. C4 — Container / Package

```
src/sine_denoiser/
├── shared/         # version.py, constants, type aliases, seeding utils
├── data/           # signal generation, dataset, dataloader factory
├── models/         # mlp.py, rnn.py, lstm.py, base.py, registry
├── training/       # train loop, optimizer factory, early stopping
├── evaluation/     # metrics, per-component breakdown, robustness sweep
├── plotting/       # matplotlib helpers (training curves, predictions)
├── sdk.py          # SDK class — the single public surface
└── cli.py          # thin argparse wrapper around SDK
```

Each module has its own `tests/unit/<module>/` mirror.

## 3. C4 — Component (SDK)

The `SDK` class is the only thing notebooks and the CLI talk to:

```python
class SDK:
    def __init__(self, config_path: Path): ...
    def generate_data(self) -> SineDataset: ...
    def train(self, model_name: str, *, seed: int) -> RunArtifacts: ...
    def evaluate(self, run: RunArtifacts) -> EvaluationReport: ...
    def sweep(self, model_name: str, axis: str, values: list) -> SweepReport: ...
    def predict(self, run: RunArtifacts, x_ctx, c) -> Tensor: ...
```

Internals (`models.registry.build`, `training.loop.train_one_epoch`, etc.) are not part of the public surface.

## 4. Data flow — single training run

```
config.json
   │
   ▼
SDK.train(model_name, seed)
   │
   ├── shared.seeding.seed_everything(seed)
   ├── data.build_datasets(cfg.data)         ──► (train, val, test)
   ├── models.registry.build(model_name, cfg.model[name])
   ├── training.loop.fit(model, train, val, cfg.training)
   │      └── early stopping on val MSE
   ├── evaluation.compute(model, test)        ──► EvaluationReport
   └── persist:
         checkpoints/<run_id>/best.pt
         runs/<run_id>/metrics.json
         runs/<run_id>/config_snapshot.json
         runs/<run_id>/training_curve.png
```

## 5. Data schemas

### 5.1 `config/default.json` (versioned)
```json
{
  "version": "1.00",
  "data": {
    "num_components": 4,
    "duration_s": 10.0,
    "sample_rate_hz": 1000,
    "noise_sigma": 0.5,
    "frequencies_hz": [2.0, 5.0, 11.0, 17.0],
    "phases_rad":     [0.0, 0.7, 1.5, 2.3],
    "amplitudes":     [1.0, 1.0, 1.0, 1.0],
    "context_window": 10,
    "split":          {"train": 0.8, "val": 0.1, "test": 0.1},
    "data_seed":      0
  },
  "model": {
    "mlp":  {"hidden_size": 64, "num_layers": 2, "activation": "relu", "dropout": 0.0},
    "rnn":  {"hidden_size": 64, "num_layers": 1, "nonlinearity": "tanh", "dropout": 0.0},
    "lstm": {"hidden_size": 64, "num_layers": 1, "dropout": 0.0, "bidirectional": false}
  },
  "training": {
    "optimizer": "adam",
    "lr": 1e-3,
    "batch_size": 256,
    "epochs": 50,
    "early_stopping_patience": 5,
    "seeds": [0, 1, 2]
  }
}
```

### 5.2 `runs/<run_id>/metrics.json`
```json
{
  "run_id": "lstm__seed0__2026-05-04T10-12-33",
  "model": "lstm",
  "seed": 0,
  "config_version": "1.00",
  "best_epoch": 23,
  "best_val_mse": 0.0142,
  "test_mse": 0.0151,
  "test_mse_per_component": [0.014, 0.016, 0.015, 0.015],
  "wall_clock_s": 187.4
}
```

## 6. Architecture Decision Records

### ADR-001 — PyTorch over TensorFlow / JAX
**Decision:** PyTorch.
**Why:** built-in `nn.RNN` / `nn.LSTM`, simplest API for the assignment, matches what the course examples use, and the small model sizes mean no need for JAX's compilation wins.
**Consequences:** locks us into the PyTorch ecosystem (lr schedulers, dataloader idioms).

### ADR-002 — Conditioning via concatenation, not FiLM / hypernetwork
**Decision:** concatenate the one-hot selector with the input (broadcast per timestep for sequence models).
**Why:** spec describes the input as "one-hot selector C + context window", so concatenation is the literal reading. FiLM is more powerful but adds parameters and complicates the cross-model comparison.
**Consequences:** noted as an open question to the instructor in `PRD.md` §9.

### ADR-003 — Sequence models receive the window as a length-10 sequence
**Decision:** for RNN/LSTM, treat the 10-sample window as 10 timesteps with a 5-dim per-step input (`[sample, one_hot_c]`).
**Why:** otherwise the recurrent models reduce to the MLP. The whole point of including them is to see if temporal structure helps.
**Consequences:** RNN/LSTM see one extra "trivial" copy of the selector at each step — fine.

### ADR-004 — Single SDK class as the public API
**Decision:** `sine_denoiser.SDK` is the only thing CLI and notebooks import.
**Why:** required by the submission guidelines; also keeps notebooks short and easy to grade.
**Consequences:** internal modules can refactor freely as long as the SDK signature is stable.

### ADR-005 — JSON for configs (not YAML / TOML)
**Decision:** JSON.
**Why:** zero extra dependencies, supports the `version` field cleanly, easy to diff in PRs.
**Consequences:** no comments in configs; live with it.

### ADR-006 — `uv` only, no `pip` / `python -m venv`
**Decision:** `uv` exclusively.
**Why:** required by the submission guidelines; faster, lockfile-first.
**Consequences:** docs and CI must use `uv` invocations everywhere.

### ADR-007 — Files capped at 150 lines (excluding blank/comment lines)
**Decision:** enforced via a small CI script.
**Why:** required by the submission guidelines.
**Consequences:** bias toward more, smaller modules; some helper splits will feel artificial.

## 7. Sequence — sweep run

```
SDK.sweep("lstm", axis="hidden_size", values=[32,64,128])
   │
   ├── for each value v:
   │     for each seed s in cfg.training.seeds:
   │         SDK.train("lstm", seed=s, override={"model.lstm.hidden_size": v})
   │         SDK.evaluate(run)
   │
   └── aggregate → SweepReport
        └── plotting.sweep_plot(report) → runs/sweeps/<axis>.png
```

## 8. Testing strategy

- **Unit:** small, fast, hit one module each. Examples: `data` produces correct shapes; `models.mlp` forward pass returns `(B, 10)`; `training.early_stopping` triggers when val MSE plateaus.
- **Integration:** one tiny end-to-end run (1 epoch, 32 batches, hidden=8) per model in `tests/integration/`. Asserts: the run produces a checkpoint and a metrics file, and test MSE beats the trivial baseline by some small margin.
- **Coverage gate:** `--cov-fail-under=85` in CI.

## 9. CI

GitHub Actions workflow `.github/workflows/ci.yml`:
1. `uv sync`
2. `uv run ruff check .`
3. `uv run pytest --cov=src --cov-fail-under=85`
4. (optional) line-count check for the 150-line rule.

Workflows are intentionally minimal — this is a research repo, not a production service.
