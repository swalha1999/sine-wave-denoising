# PRD — Models (HW1)

**Scope:** the three neural architectures used in the sine wave denoising study.
**Status:** Draft v0.1
**Last updated:** 2026-05-04

This PRD specifies *what* each model must look like and how it is judged. Algorithmic and training-loop decisions live in `PLAN.md`; the dataset spec lives in `PRD.md` §5.1.

---

## 1. Shared contract

All three models implement the same interface so they are swappable in the training loop and the evaluation notebook.

### 1.1 Input
- `x_ctx`: float tensor of shape `(B, 10)` — the noisy mixed-signal context window.
- `c`: long tensor of shape `(B,)` — selector index ∈ `{0, 1, 2, 3}`.

The model is responsible for combining `x_ctx` and `c`. Default conditioning: concat the one-hot of `c` (4-dim) with the input. Sequence models broadcast the one-hot at every timestep.

### 1.2 Output
- `y_hat`: float tensor of shape `(B, 10)` — the predicted clean window of the chosen sine.

### 1.3 Loss
- `MSELoss(reduction='mean')` over the 10-sample window.

### 1.4 Python interface
```python
class DenoiserModel(nn.Module, abc.ABC):
    name: str  # "mlp" | "rnn" | "lstm"
    def forward(self, x_ctx: Tensor, c: Tensor) -> Tensor: ...
    @classmethod
    def from_config(cls, cfg: dict) -> "DenoiserModel": ...
```

Each concrete model lives in `src/sine_denoiser/models/<name>.py` and is registered in `models/__init__.py`.

---

## 2. Model 1 — MLP (`mlp`)

### 2.1 Architecture
- Input: flat vector of length 14 = `concat(one_hot(c, 4), x_ctx)`.
- `N` hidden layers of width `H`, each followed by a non-linearity (default `ReLU`) and optional dropout.
- Output linear layer → 10.

### 2.2 Required configurables
| Key | Type | Default | Notes |
|---|---|---|---|
| `hidden_size` | int | 64 | sweep: 32, 64, 128, 256 |
| `num_layers`  | int | 2  | sweep: 1, 2, 3 |
| `activation`  | str | "relu" | "relu" \| "tanh" \| "gelu" |
| `dropout`     | float | 0.0 | sweep: 0.0, 0.1, 0.2 |

### 2.3 Acceptance
- Trains end-to-end on the default config in <2 min on CPU.
- Test MSE strictly better than the trivial baseline `y_hat = x_ctx` (predicting the noisy mix as the clean signal).

---

## 3. Model 2 — Vanilla RNN (`rnn`)

### 3.1 Architecture
- Input is treated as a length-10 sequence. At each timestep `t`, the model consumes a 5-dim vector: `[x_ctx[t]] ++ one_hot(c, 4)` (selector broadcast to every step).
- `nn.RNN` with `hidden_size = H`, `num_layers = N`, non-linearity `tanh` or `relu`.
- Output projection: linear layer mapping the per-timestep hidden state to a scalar → stacked to give shape `(B, 10)`.

### 3.2 Required configurables
| Key | Type | Default | Notes |
|---|---|---|---|
| `hidden_size`   | int  | 64    | sweep: 32, 64, 128 |
| `num_layers`    | int  | 1     | sweep: 1, 2 |
| `nonlinearity`  | str  | "tanh"| "tanh" \| "relu" |
| `dropout`       | float| 0.0   | only meaningful when `num_layers > 1` |

### 3.3 Acceptance
- Trains end-to-end on the default config in <5 min on CPU.
- Test MSE strictly better than the trivial baseline.

---

## 4. Model 3 — LSTM (`lstm`)

### 4.1 Architecture
- Same input layout as RNN (length-10 sequence, selector broadcast per step).
- `nn.LSTM` with `hidden_size = H`, `num_layers = N`.
- Output projection identical to RNN.

### 4.2 Required configurables
| Key | Type | Default | Notes |
|---|---|---|---|
| `hidden_size` | int   | 64  | sweep: 32, 64, 128 |
| `num_layers`  | int   | 1   | sweep: 1, 2 |
| `dropout`     | float | 0.0 | only meaningful when `num_layers > 1` |
| `bidirectional` | bool | false | optional axis if time permits |

### 4.3 Acceptance
- Trains end-to-end on the default config in <5 min on CPU.
- Test MSE strictly better than the trivial baseline.

---

## 5. Training defaults (all models)

| Key | Default | Notes |
|---|---|---|
| `optimizer` | Adam | configurable |
| `lr` | 1e-3 | sweep: 1e-4, 3e-4, 1e-3, 3e-3 |
| `batch_size` | 256 | |
| `epochs` | 50 | with early stopping |
| `early_stopping_patience` | 5 | on val MSE |
| `seed` | 0, 1, 2 | three seeds for stats |
| `device` | auto (mps/cuda/cpu) | |

## 6. Cross-model success criteria

- **C1.** All three models hit the per-model acceptance bar (§2.3 / §3.3 / §4.3).
- **C2.** Numbers reported in the results notebook are reproducible from `config/default.json` to within ±5 %.
- **C3.** A clear, defensible answer to "which model wins, and under what conditions?" — even if the answer is "MLP, because the window is too short for recurrent models to add value."
- **C4.** Per-component MSE, noise-σ robustness curve, and qualitative prediction plots are all produced for every model.

## 7. Out of scope

- Transformers, state-space models, diffusion, GAN baselines.
- Pretraining / transfer learning.
- Hyperparameter optimization beyond manual / grid sweeps over the axes listed above.
