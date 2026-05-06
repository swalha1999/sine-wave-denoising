# Sine Wave Denoising

Comparing fully connected, RNN, and LSTM models on a denoising task: recover a chosen pure sine wave from a noisy mixture of four.

## Task

- Generate 4 sine waves (different frequencies/phases).
- Sample each for 10 s at 1000 Hz → 10,000 samples per signal.
- Add random noise to each, then sum the noisy signals to form a mixed signal.
- **Input:** one-hot selector `C` (which of the 4 to recover) + a 10-sample context window from the noisy mixture.
- **Target:** the aligned 10-sample window from the chosen *clean* sine wave.
- Train all three models with MSE loss, then compare.

## Models

1. Fully connected (MLP)
2. Vanilla RNN
3. LSTM

## Layout

```
docs/   assignment spec
```

Implementation directories (`src/`, `notebooks/`, etc.) will be added as work progresses.

## Hyperparameter sweeps

Run all four sweep axes (`lr`, `hidden_size`, `num_layers`, `noise_sigma`) across
all three models with one command:

```bash
uv run python -m sine_denoiser.sweeps \
  --config config/sweep.json \
  --out-dir runs/sweeps
```

For each axis the CLI writes `<axis>.json` (raw test/val MSEs per model) and
`<axis>.png` (multi-series line plot) under `--out-dir`. Pass `--axis` and/or
`--model` (repeatable) to run a subset.
