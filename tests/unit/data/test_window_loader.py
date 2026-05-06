import numpy as np
import pytest

from sine_denoiser.data.dataset import SineWindowDataset
from sine_denoiser.data.loader import SineWindowLoader


def _toy_signals(t: int = 200, k: int = 4) -> tuple[np.ndarray, np.ndarray]:
    pure = np.arange(k * t, dtype=np.float64).reshape(k, t)
    mixed = pure.sum(axis=0)
    return mixed, pure


def test_batch_shapes_and_dtypes():
    mixed, pure = _toy_signals()
    ds = SineWindowDataset(
        mixed,
        pure,
        np.array([[0, 0], [5, 1], [10, 2], [15, 3], [20, 0]], dtype=np.int64),
        context_window=10,
    )
    loader = SineWindowLoader(ds, batch_size=2, shuffle=False)
    batches = list(loader)
    assert [b[1].shape[0] for b in batches] == [2, 2, 1]
    x, c, y = batches[0]
    assert x.shape == (2, 10)
    assert c.shape == (2,)
    assert c.dtype == np.int64
    assert y.shape == (2, 10)


def test_drop_last_drops_partial_batch():
    mixed, pure = _toy_signals()
    ds = SineWindowDataset(
        mixed,
        pure,
        np.array([[i, 0] for i in range(7)], dtype=np.int64),
        context_window=10,
    )
    loader = SineWindowLoader(ds, batch_size=3, shuffle=False, drop_last=True)
    batches = list(loader)
    assert len(batches) == 2
    assert all(b[1].shape[0] == 3 for b in batches)
    assert len(loader) == 2


def test_len_without_drop_last_includes_partial_batch():
    mixed, pure = _toy_signals()
    ds = SineWindowDataset(
        mixed,
        pure,
        np.array([[i, 0] for i in range(7)], dtype=np.int64),
        context_window=10,
    )
    loader = SineWindowLoader(ds, batch_size=3, shuffle=False)
    assert len(loader) == 3


def test_shuffle_changes_order_across_epochs():
    mixed, pure = _toy_signals()
    ds = SineWindowDataset(
        mixed,
        pure,
        np.array([[i, 0] for i in range(20)], dtype=np.int64),
        context_window=10,
    )
    loader = SineWindowLoader(ds, batch_size=20, shuffle=True, seed=0)
    epoch1 = next(iter(loader))[0][:, 0].tolist()
    epoch2 = next(iter(loader))[0][:, 0].tolist()
    assert epoch1 != epoch2
    assert sorted(epoch1) == sorted(epoch2)


def test_shuffle_deterministic_given_seed():
    mixed, pure = _toy_signals()
    indices = np.array([[i, 0] for i in range(20)], dtype=np.int64)
    ds_a = SineWindowDataset(mixed, pure, indices, context_window=10)
    ds_b = SineWindowDataset(mixed, pure, indices, context_window=10)
    a = next(iter(SineWindowLoader(ds_a, batch_size=20, shuffle=True, seed=42)))[0]
    b = next(iter(SineWindowLoader(ds_b, batch_size=20, shuffle=True, seed=42)))[0]
    np.testing.assert_array_equal(a, b)


def test_no_shuffle_preserves_dataset_order():
    mixed, pure = _toy_signals()
    indices = np.array([[i, 0] for i in range(5)], dtype=np.int64)
    ds = SineWindowDataset(mixed, pure, indices, context_window=10)
    loader = SineWindowLoader(ds, batch_size=2, shuffle=False)
    seen: list[np.ndarray] = []
    for x, _c, _y in loader:
        seen.append(x)
    flat = np.concatenate(seen, axis=0)
    expected = np.stack([mixed[i : i + 10] for i in range(5)], axis=0)
    np.testing.assert_array_equal(flat, expected)


def test_invalid_batch_size_raises():
    mixed, pure = _toy_signals()
    ds = SineWindowDataset(
        mixed, pure, np.array([[0, 0]], dtype=np.int64), context_window=10
    )
    with pytest.raises(ValueError):
        SineWindowLoader(ds, batch_size=0)
