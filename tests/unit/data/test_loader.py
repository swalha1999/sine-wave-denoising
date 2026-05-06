import numpy as np

from sine_denoiser.data.dataset import make_split_indices
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
