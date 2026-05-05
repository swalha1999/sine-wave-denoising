"""DataLoader factory: batched train/val/test loaders over SineWindowDataset."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from sine_denoiser.data.dataset import (
    SineWindowDataset,
    SplitIndices,
    make_split_indices,
)


class SineWindowLoader:
    """Iterates a ``SineWindowDataset`` in batches of ``(x_ctx, c, y)``.

    Each batch stacks items into numpy arrays: ``x_ctx`` is ``(B, W)``,
    ``c`` is ``(B,)`` int64, ``y`` is ``(B, W)``. With ``shuffle=True`` a
    fresh permutation is drawn each epoch, seeded by ``seed + epoch`` so
    iteration order is deterministic given ``seed``.
    """

    def __init__(
        self,
        dataset: SineWindowDataset,
        batch_size: int,
        *,
        shuffle: bool = False,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        self._dataset = dataset
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._seed = seed
        self._drop_last = drop_last
        self._epoch = 0

    def __len__(self) -> int:
        n = len(self._dataset)
        if self._drop_last:
            return n // self._batch_size
        return (n + self._batch_size - 1) // self._batch_size

    def __iter__(self) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        n = len(self._dataset)
        if self._shuffle:
            rng = np.random.default_rng(self._seed + self._epoch)
            order = rng.permutation(n)
        else:
            order = np.arange(n)
        self._epoch += 1
        end = n - (n % self._batch_size) if self._drop_last else n
        for start in range(0, end, self._batch_size):
            idxs = order[start : start + self._batch_size]
            xs: list[np.ndarray] = []
            cs: list[int] = []
            ys: list[np.ndarray] = []
            for i in idxs:
                x, c, y = self._dataset[int(i)]
                xs.append(x)
                cs.append(c)
                ys.append(y)
            yield (
                np.stack(xs, axis=0),
                np.asarray(cs, dtype=np.int64),
                np.stack(ys, axis=0),
            )


@dataclass(frozen=True)
class Loaders:
    """Train/val/test loaders plus the split index arrays they came from."""

    train: SineWindowLoader
    val: SineWindowLoader
    test: SineWindowLoader
    splits: SplitIndices


def build_loaders(
    mixed: np.ndarray,
    pure: np.ndarray,
    *,
    context_window: int,
    split: dict,
    batch_size: int,
    seed: int,
    shuffle_train: bool = True,
    drop_last: bool = False,
) -> Loaders:
    """Partition the signal into disjoint train/val/test loaders.

    Splits are over window *start* indices, so no window straddles two
    splits. Each split is paired with every component to form
    ``(start, c)`` rows before being wrapped in a loader.
    """
    splits = make_split_indices(
        num_samples=mixed.shape[0],
        num_components=pure.shape[0],
        context_window=context_window,
        split=split,
        seed=seed,
    )
    train_ds = SineWindowDataset(mixed, pure, splits.train, context_window)
    val_ds = SineWindowDataset(mixed, pure, splits.val, context_window)
    test_ds = SineWindowDataset(mixed, pure, splits.test, context_window)
    return Loaders(
        train=SineWindowLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=shuffle_train,
            seed=seed,
            drop_last=drop_last,
        ),
        val=SineWindowLoader(val_ds, batch_size=batch_size, shuffle=False, seed=seed),
        test=SineWindowLoader(test_ds, batch_size=batch_size, shuffle=False, seed=seed),
        splits=splits,
    )
