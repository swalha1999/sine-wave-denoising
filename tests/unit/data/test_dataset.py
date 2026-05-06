import numpy as np
import pytest

from sine_denoiser.data.dataset import SineWindowDataset, make_split_indices


def _toy_signals(t: int = 64, k: int = 4) -> tuple[np.ndarray, np.ndarray]:
    pure = np.arange(k * t, dtype=np.float64).reshape(k, t)
    mixed = pure.sum(axis=0)
    return mixed, pure


def test_item_shapes_and_types():
    mixed, pure = _toy_signals()
    indices = np.array([[5, 2], [10, 0]], dtype=np.int64)
    ds = SineWindowDataset(mixed, pure, indices, context_window=10)
    x_ctx, c, y = ds[0]
    assert x_ctx.shape == (10,)
    assert isinstance(c, int)
    assert y.shape == (10,)


def test_item_window_matches_signals():
    mixed, pure = _toy_signals()
    indices = np.array([[5, 2]], dtype=np.int64)
    ds = SineWindowDataset(mixed, pure, indices, context_window=10)
    x_ctx, c, y = ds[0]
    np.testing.assert_array_equal(x_ctx, mixed[5:15])
    np.testing.assert_array_equal(y, pure[2, 5:15])
    assert c == 2


def test_len_matches_indices():
    mixed, pure = _toy_signals()
    indices = np.array([[0, 0], [1, 1], [2, 2]], dtype=np.int64)
    ds = SineWindowDataset(mixed, pure, indices, context_window=10)
    assert len(ds) == 3


def test_dataset_respects_split():
    mixed, pure = _toy_signals(t=200)
    splits = make_split_indices(
        num_samples=200,
        num_components=4,
        context_window=10,
        split={"train": 0.8, "val": 0.1, "test": 0.1},
        seed=0,
    )
    val_ds = SineWindowDataset(mixed, pure, splits.val, context_window=10)
    train_starts = set(splits.train[:, 0].tolist())
    for i in range(len(val_ds)):
        x_ctx, c, y = val_ds[i]
        start = int(splits.val[i, 0])
        assert start not in train_starts
        np.testing.assert_array_equal(x_ctx, mixed[start : start + 10])
        np.testing.assert_array_equal(y, pure[c, start : start + 10])


def test_dataset_rejects_bad_shapes():
    _mixed, pure = _toy_signals()
    with pytest.raises(ValueError):
        SineWindowDataset(
            np.zeros((4, 64)),
            pure,
            np.array([[0, 0]], dtype=np.int64),
            context_window=10,
        )
    with pytest.raises(ValueError):
        SineWindowDataset(
            np.zeros(64),
            np.zeros((4, 32)),
            np.array([[0, 0]], dtype=np.int64),
            context_window=10,
        )
    with pytest.raises(ValueError):
        SineWindowDataset(
            np.zeros(64),
            np.zeros((4, 64)),
            np.array([0, 0], dtype=np.int64),
            context_window=10,
        )
    with pytest.raises(ValueError):
        SineWindowDataset(
            np.zeros(64),
            np.zeros((4, 64)),
            np.zeros((1, 2), dtype=np.int64),
            context_window=0,
        )
