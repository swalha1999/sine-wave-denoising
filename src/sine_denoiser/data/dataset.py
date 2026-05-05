"""Window-level dataset producing (x_ctx, c, y) triples."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SplitIndices:
    """Disjoint (start_t, component) index arrays per split."""

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def make_split_indices(
    num_samples: int,
    num_components: int,
    context_window: int,
    split: dict,
    seed: int,
) -> SplitIndices:
    """Partition valid start indices into train/val/test, then pair each
    with every component to form ``(start, c)`` rows.
    """
    if context_window <= 0:
        raise ValueError("context_window must be > 0")
    if num_samples < context_window:
        raise ValueError("num_samples must be >= context_window")
    fractions = (split["train"], split["val"], split["test"])
    if abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError("split fractions must sum to 1")

    n_starts = num_samples - context_window + 1
    perm = np.random.default_rng(seed).permutation(n_starts)
    n_train = int(round(fractions[0] * n_starts))
    n_val = int(round(fractions[1] * n_starts))
    return SplitIndices(
        train=_expand(perm[:n_train], num_components),
        val=_expand(perm[n_train : n_train + n_val], num_components),
        test=_expand(perm[n_train + n_val :], num_components),
    )


def _expand(starts: np.ndarray, num_components: int) -> np.ndarray:
    if starts.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    ts = np.repeat(starts, num_components)
    cs = np.tile(np.arange(num_components), starts.size)
    return np.stack([ts, cs], axis=1).astype(np.int64)


class SineWindowDataset:
    """Maps a sample index to ``(x_ctx, c, y)``.

    - ``mixed``: ``(T,)`` noisy mixed signal.
    - ``pure``: ``(K, T)`` clean per-component signals.
    - ``indices``: ``(N, 2)`` array of ``(start_t, component)`` rows.
    """

    def __init__(
        self,
        mixed: np.ndarray,
        pure: np.ndarray,
        indices: np.ndarray,
        context_window: int = 10,
    ) -> None:
        if mixed.ndim != 1:
            raise ValueError("mixed must be 1-D")
        if pure.ndim != 2 or pure.shape[1] != mixed.shape[0]:
            raise ValueError("pure must be (K, T) matching mixed length")
        if indices.ndim != 2 or indices.shape[1] != 2:
            raise ValueError("indices must be (N, 2) of (start, component)")
        if context_window <= 0:
            raise ValueError("context_window must be > 0")
        self._mixed = mixed
        self._pure = pure
        self._indices = indices
        self._W = context_window

    def __len__(self) -> int:
        return self._indices.shape[0]

    def __getitem__(self, i: int) -> tuple[np.ndarray, int, np.ndarray]:
        start, c = self._indices[i]
        end = int(start) + self._W
        x_ctx = self._mixed[int(start) : end]
        y = self._pure[int(c), int(start) : end]
        return x_ctx, int(c), y
