# TODO — Phased Task List

**Status:** Draft v0.1
**Last updated:** 2026-05-04
**Owners:** `S` = swalha1999, `M` = Mhmdabad. Update as you claim items.

Status legend: ☐ pending · ◐ in progress · ☑ done

DoD = Definition of Done. Every task lists what "done" means concretely.

---

## Phase 0 — Scaffolding (target: today)

| Status | # | Owner | Task | DoD |
|---|---|---|---|---|
| ☑ | 0.1 | S | Create GitHub repo, add collaborator | Repo public at `swalha1999/sine-wave-denoising`, M added |
| ☑ | 0.2 | S | `uv init` + hello-world | `uv run main.py` prints hello |
| ☑ | 0.3 | S | Write `PRD.md`, `PRD_models.md`, `PLAN.md`, `TODO.md` | All four files committed under `docs/` |
| ☑ | 0.4 | S | Add `ruff` + `pytest` + `pytest-cov` as dev deps | `uv add --dev ruff pytest pytest-cov` runs cleanly |
| ☐ | 0.5 |   | Skeleton `src/sine_denoiser/` package + `shared/version.py` (`"1.00"`) | `uv run python -c "import sine_denoiser; print(sine_denoiser.__version__)"` works |
| ☐ | 0.6 |   | `.env-example`, update `.gitignore` for `.env`, `runs/`, `checkpoints/` | Files exist, `git status` clean after `cp .env-example .env` |
| ☐ | 0.7 |   | GitHub Actions CI: `uv sync` → `ruff` → `pytest --cov` | Green check on first push |

## Phase 1 — Data module (target: week 1)

| Status | # | Owner | Task | DoD |
|---|---|---|---|---|
| ☐ | 1.1 |   | `data/signals.py` — generate 4 sine waves from config | Unit test asserts shape `(4, 10000)` and known values for fixed seed |
| ☐ | 1.2 |   | `data/noise.py` — add per-signal Gaussian noise, build mixed signal | Unit test: σ=0 → mixed equals sum of pure; mean(noise)≈0 over many seeds |
| ☐ | 1.3 |   | `data/dataset.py` — `SineWindowDataset` produces `(x_ctx, c, y)` tuples | Unit test asserts shapes `(10,), int, (10,)`; respects split |
| ☐ | 1.4 |   | `data/loader.py` — DataLoader factory | Unit test: train / val / test loaders have disjoint indices |
| ☐ | 1.5 |   | Default `config/default.json` matches `PLAN.md` §5.1 | JSON loads, `version == "1.00"` |

## Phase 2 — Models (target: week 1–2)

| Status | # | Owner | Task | DoD |
|---|---|---|---|---|
| ☐ | 2.1 |   | `models/base.py` — `DenoiserModel` ABC | Unit test: subclass without `forward` fails to instantiate |
| ☐ | 2.2 |   | `models/mlp.py` — MLP per `PRD_models.md` §2 | Forward returns `(B, 10)`; supports all config keys |
| ☐ | 2.3 |   | `models/rnn.py` — vanilla RNN per §3 | Forward returns `(B, 10)`; sequence layout correct |
| ☐ | 2.4 |   | `models/lstm.py` — LSTM per §4 | Forward returns `(B, 10)`; sequence layout correct |
| ☐ | 2.5 |   | `models/registry.py` — `build(name, cfg)` factory | `build("mlp", cfg)` and friends return correct subclass |

## Phase 3 — Training & evaluation (target: week 2)

| Status | # | Owner | Task | DoD |
|---|---|---|---|---|
| ☐ | 3.1 |   | `training/loop.py` — fit loop + early stopping | Integration test: 1-epoch run on tiny dataset persists checkpoint and metrics |
| ☐ | 3.2 |   | `training/optim.py` — optimizer factory | Unit tests for "adam"/"sgd"; unknown name raises |
| ☐ | 3.3 |   | `evaluation/metrics.py` — MSE total + per-component | Unit test against hand-computed values |
| ☐ | 3.4 |   | `evaluation/robustness.py` — sweep test MSE over noise σ | Unit test: monotone-ish curve on toy data |
| ☐ | 3.5 |   | `sdk.py` — `SDK` class wiring data + model + training + eval | Notebook can run a full pipeline using only `SDK` |
| ☐ | 3.6 |   | `cli.py` — `python -m sine_denoiser.train --config ...` | Trains all 3 models end-to-end, exits 0 |

## Phase 4 — Analysis & plots (target: week 2–3)

| Status | # | Owner | Task | DoD |
|---|---|---|---|---|
| ☐ | 4.1 |   | `plotting/curves.py` — training curves | PNG produced per run under `runs/<id>/` |
| ☐ | 4.2 |   | `plotting/predictions.py` — predicted vs. clean window plots | PNG per (model, component) |
| ☐ | 4.3 |   | `plotting/sweep.py` — line / heatmap for sweep results | PNG per axis under `runs/sweeps/` |
| ☐ | 4.4 |   | `notebooks/results_analysis.ipynb` — final comparison notebook | Runs top-to-bottom without errors; contains all PRD §A5 items |
| ☐ | 4.5 |   | Run hyperparameter sweeps for all 3 models, ≥3 axes each | All sweep PNGs and JSONs committed (or reproducible from a single CLI command) |

## Phase 5 — Polish & submission (target: week 3)

| Status | # | Owner | Task | DoD |
|---|---|---|---|---|
| ☐ | 5.1 |   | `README.md` polished: install / usage / config / examples | Cold reader can run the project after reading only the README |
| ☐ | 5.2 |   | `ruff check` clean across the repo | CI passes |
| ☐ | 5.3 |   | Coverage ≥85 % | CI passes with `--cov-fail-under=85` |
| ☐ | 5.4 |   | 150-line/file rule satisfied | `scripts/check_line_count.sh` exits 0 in CI |
| ☐ | 5.5 |   | Prompt book — log significant prompts used to build the project | `docs/PROMPTS.md` exists and is non-trivial |
| ☐ | 5.6 |   | Tag submission release | `git tag v1.0-submission && git push --tags` |

---

## Backlog / nice-to-have (not blocking)

- Replace JSON configs with [pydantic](https://docs.pydantic.dev/) settings if validation gets painful.
- Add a Transformer baseline (out of scope per PRD §4 but interesting).
- Per-frequency analysis: does any model do better on low- vs. high-frequency components?
- Animated visualization of the denoising process across training epochs.
