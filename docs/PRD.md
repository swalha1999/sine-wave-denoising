# PRD — Sine Wave Denoising (HW1)

**Course:** Orchestration of AI Agents
**Group code:** s82kma9e
**Owner:** swalha1999, Mhmdabad
**Status:** Draft v0.1
**Last updated:** 2026-05-04

---

## 1. Overview

A controlled study comparing three neural architectures — fully connected (MLP), vanilla RNN, and LSTM — on a denoising task. Given a one-hot selector and a short noisy context window from a mixture of four sine waves, each model learns to recover the corresponding clean window from the *selected* component sine wave.

The deliverable is reproducible code, trained checkpoints, and a written analysis of which architecture wins where, and why.

## 2. Problem Statement

Real signals are rarely seen in isolation; they arrive as noisy mixtures. We want to know how well three off-the-shelf architectures can perform conditional source separation under controlled, synthetic conditions where ground truth is exact and noise level is tunable.

## 3. Goals

**Measurable:**
- G1. Train all three models to convergence on the same dataset (same train/val/test split, seed-controlled).
- G2. Report test-set MSE per model with mean ± std over **≥3** random seeds.
- G3. Run a hyperparameter sweep on **≥3** axes per model (e.g. hidden size, depth, learning rate). Plot results.
- G4. Produce a results notebook with side-by-side prediction visualizations for each of the 4 sine components.
- G5. End-to-end reproducibility: `uv sync && uv run python -m sine_denoiser.train --config config/default.json` produces the reported numbers ±5 %.

**Quality bars (per submission guidelines):**
- ≥85 % test coverage; `ruff check` passes with zero violations.
- Each source file ≤150 lines (blank/comment lines excluded).
- All configuration in JSON files under `config/`; no hard-coded hyperparameters.

## 4. Non-Goals

- Real-world audio or biomedical signal denoising — synthetic only.
- Beating SOTA — we are *comparing* baselines, not pushing them.
- Live / streaming inference, GUI, deployment.
- Architectures beyond the three required (no Transformers, no diffusion, no GAN baselines).

## 5. Functional Requirements

### 5.1 Data generation
- F1. Generate **4** pure sine waves with distinct (frequency, phase, amplitude) tuples. Defaults seeded; configurable in `config/data.json`.
- F2. Each signal is sampled for **10 s at 1000 Hz** → 10,000 samples per signal.
- F3. Add per-sample Gaussian noise with configurable σ to each signal independently, then form the mixed signal as the sum of the four noisy signals.
- F4. Build training examples by:
  1. picking a target component `c ∈ {0,1,2,3}`,
  2. picking a random start index `t`,
  3. taking the 10-sample window `[t, t+10)` of the **mixed noisy** signal as `x_ctx`,
  4. taking the same window of the **chosen clean** signal as `y`.
- F5. Model input = `concat(one_hot(c, 4), x_ctx)` → vector of length 14 (or as a sequence for RNN/LSTM, with selector broadcast).
- F6. Train/val/test split by random start indices with no overlap; deterministic given a seed.

### 5.2 Models
Detailed in `docs/PRD_models.md`. At minimum:
- F7. **MLP**: ≥1 hidden layer, configurable width and depth.
- F8. **RNN**: vanilla `nn.RNN`, configurable hidden size and depth.
- F9. **LSTM**: `nn.LSTM`, configurable hidden size and depth.
- F10. All models output a 10-dim vector (the predicted clean window).

### 5.3 Training
- F11. Loss: MSE.
- F12. Optimizer: Adam by default; configurable.
- F13. Early stopping on validation MSE.
- F14. Per-run artifacts: best checkpoint, training-curve plot, JSON metrics file, config snapshot (with version field).
- F15. Seeding for `random`, `numpy`, `torch` (and CUDA if used).

### 5.4 Evaluation & analysis
- F16. Test-set MSE per model, mean ± std over ≥3 seeds.
- F17. Per-component breakdown: MSE conditioned on each `c ∈ {0..3}`.
- F18. Robustness curve: test MSE as a function of noise σ.
- F19. Qualitative plots: predicted vs. clean window for several test examples per component, per model.
- F20. Hyperparameter sensitivity: charts (line / heatmap) for each swept axis.

### 5.5 Code organization (per submission guidelines)
- F21. `src/sine_denoiser/` package with: `data/`, `models/`, `training/`, `evaluation/`, `shared/` (incl. `version.py` starting at `1.00`).
- F22. Single SDK entry class exposing all business logic — CLI and notebooks call only the SDK, not internals.
- F23. Configs in `config/*.json`, each carrying a `version` field.
- F24. `.env-example` provided; `.env` gitignored.

### 5.6 Testing
- F25. `tests/unit/<module>/test_<file>.py` mirrors `src/`.
- F26. `tests/integration/` covers an end-to-end small-scale training run.
- F27. Shared fixtures in `conftest.py`.
- F28. Coverage gate ≥85 % enforced in CI.

### 5.7 Documentation
- F29. `README.md`: install, usage, config guide, examples, license.
- F30. `docs/PLAN.md`: architecture (C4 / sequence diagrams), ADRs.
- F31. `docs/PRD_models.md`: per-model PRD (this PRD covers the project).
- F32. `docs/TODO.md`: phased task list with owners and DoD.
- F33. Results analysis Jupyter notebook under `notebooks/`.

## 6. Acceptance Criteria

The HW is "done" when **all** of the following hold:
- A1. `uv sync` installs the project cleanly on a fresh checkout.
- A2. `uv run pytest --cov=src --cov-fail-under=85` passes.
- A3. `uv run ruff check .` exits 0.
- A4. `uv run python -m sine_denoiser.train --config config/default.json` trains all three models end-to-end.
- A5. The results notebook renders without errors and contains: G2 numbers, G3 sweep plots, F17 per-component breakdown, F18 noise-robustness curve, F19 qualitative plots.
- A6. All five required docs (`README.md`, `PRD.md`, `PRD_models.md`, `PLAN.md`, `TODO.md`) exist and are non-trivial.
- A7. Git history is clean, with meaningful commit messages and a release tag for submission.

## 7. Deliverables

| # | Item | Path |
|---|------|------|
| D1 | Source package | `src/sine_denoiser/` |
| D2 | Config files | `config/` |
| D3 | Test suite | `tests/` |
| D4 | Results notebook | `notebooks/results_analysis.ipynb` |
| D5 | Trained checkpoints (small) | `checkpoints/` (gitignored; reproducible from configs) |
| D6 | All required docs | `docs/`, `README.md` |
| D7 | Submission tag | `git tag v1.0-submission` |

## 8. Milestones

| # | Milestone | Target |
|---|-----------|--------|
| M1 | Project scaffolding, uv setup, CI skeleton | done (this commit) |
| M2 | Data module + tests | week 1 |
| M3 | MLP model + training loop + tests | week 1 |
| M4 | RNN + LSTM models + tests | week 2 |
| M5 | Hyperparameter sweep + results notebook | week 2 |
| M6 | Docs polish, coverage gate, lint clean | week 3 |
| M7 | Submission tag | week 3 |

(Concrete dates to be filled in `docs/TODO.md`.)

## 9. Risks & Open Questions

- **R1.** Window of 10 samples is short — RNN/LSTM may underperform MLP because there is little temporal context to exploit. *Mitigation:* report it honestly; this is an interesting comparison finding, not a failure.
- **R2.** Noise σ choice strongly biases the comparison. *Mitigation:* sweep σ and report robustness curves rather than a single number.
- **R3.** The 150-line file cap may force more granular module splits than feel natural. *Mitigation:* design the package with that constraint upfront.
- **Q1.** Is conditioning by simple input concatenation acceptable, or is FiLM / hypernetwork conditioning expected? Defaulting to concat unless instructor clarifies.
- **Q2.** Are RNN/LSTM expected to consume the 10-sample window as a length-10 sequence, or as a flat vector with the selector? Defaulting to length-10 sequence with the one-hot broadcast at every timestep.
