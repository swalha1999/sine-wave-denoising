import numpy as np
import pytest

from sine_denoiser.data.dataset import SineWindowDataset, make_split_indices
from sine_denoiser.data.loader import Loaders, SineWindowLoader, build_loaders


def _toy_signals(t: int = 200, k: int = 4) -> tuple[np.ndarray, np.ndarray]:
    pure = np.arange(k * t, dtype=np.float64).reshape(k, t)
    mixed = pure.sum(axis=0)
    return mixed, pure


def test_build_loaders_returns_loaders_struct():
    mixed, pure = _toy_signals()
    loaders = build_loaders(
        mixed,
        pure,
        context_window=10,
        split={"train": 0.8, "val": 0.1, "test": 0.1},
        batch_size=8,
        seed=0,
    )
    assert isinstance(loaders, Loaders)
    assert isinstance(loaders.train, SineWindowLoader)
    assert isinstance(loaders.val, SineWindowLoader)
    assert isinstance(loaders.test, SineWindowLoader)


def test_loaders_have_disjoint_indices():
    mixed, pure = _toy_signals()
    loaders = build_loaders(
        mixed,
        pure,
        context_window=10,
        split={"train": 0.8, "val": 0.1, "test": 0.1},
        batch_size=16,
        seed=0,
    )
    train_rows = {(int(s), int(c)) for s, c in loaders.splits.train}
    val_rows = {(int(s), int(c)) for s, c in loaders.splits.val}
    test_rows = {(int(s), int(c)) for s, c in loaders.splits.test}
    assert train_rows.isdisjoint(val_rows)
    assert train_rows.isdisjoint(test_rows)
    assert val_rows.isdisjoint(test_rows)
    train_starts = {s for s, _ in train_rows}
    val_starts = {s for s, _ in val_rows}
    test_starts = {s for s, _ in test_rows}
    assert train_starts.isdisjoint(val_starts)
    assert train_starts.isdisjoint(test_starts)
    assert val_starts.isdisjoint(test_starts)


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
    # 5 items, batch=2, no drop_last → 3 batches of sizes 2,2,1
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


def test_build_loaders_splits_match_make_split_indices():
    mixed, pure = _toy_signals()
    loaders = build_loaders(
        mixed,
        pure,
        context_window=10,
        split={"train": 0.8, "val": 0.1, "test": 0.1},
        batch_size=4,
        seed=123,
    )
    expected = make_split_indices(
        num_samples=mixed.shape[0],
        num_components=pure.shape[0],
        context_window=10,
        split={"train": 0.8, "val": 0.1, "test": 0.1},
        seed=123,
    )
    np.testing.assert_array_equal(loaders.splits.train, expected.train)
    np.testing.assert_array_equal(loaders.splits.val, expected.val)
    np.testing.assert_array_equal(loaders.splits.test, expected.test)


def test_total_items_iterated_matches_split_sizes():
    mixed, pure = _toy_signals()
    loaders = build_loaders(
        mixed,
        pure,
        context_window=10,
        split={"train": 0.8, "val": 0.1, "test": 0.1},
        batch_size=11,
        seed=0,
    )
    for name, loader in (
        ("train", loaders.train),
        ("val", loaders.val),
        ("test", loaders.test),
    ):
        total = sum(c.shape[0] for _x, c, _y in loader)
        expected = getattr(loaders.splits, name).shape[0]
        assert total == expected
