import numpy as np
import pytest

from sine_denoiser.data.dataset import SplitIndices, make_split_indices


def test_make_split_indices_disjoint_starts():
    splits = make_split_indices(
        num_samples=100,
        num_components=4,
        context_window=10,
        split={"train": 0.8, "val": 0.1, "test": 0.1},
        seed=0,
    )
    assert isinstance(splits, SplitIndices)
    train_starts = set(splits.train[:, 0].tolist())
    val_starts = set(splits.val[:, 0].tolist())
    test_starts = set(splits.test[:, 0].tolist())
    assert train_starts.isdisjoint(val_starts)
    assert train_starts.isdisjoint(test_starts)
    assert val_starts.isdisjoint(test_starts)


def test_make_split_indices_covers_all_starts():
    splits = make_split_indices(
        num_samples=100,
        num_components=4,
        context_window=10,
        split={"train": 0.8, "val": 0.1, "test": 0.1},
        seed=0,
    )
    n_starts = 100 - 10 + 1
    total = splits.train.shape[0] + splits.val.shape[0] + splits.test.shape[0]
    assert total == n_starts * 4
    every_component = {0, 1, 2, 3}
    assert set(splits.train[:, 1].tolist()) == every_component


def test_make_split_indices_deterministic():
    args = dict(
        num_samples=100,
        num_components=4,
        context_window=10,
        split={"train": 0.8, "val": 0.1, "test": 0.1},
        seed=42,
    )
    a = make_split_indices(**args)
    b = make_split_indices(**args)
    np.testing.assert_array_equal(a.train, b.train)
    np.testing.assert_array_equal(a.val, b.val)
    np.testing.assert_array_equal(a.test, b.test)


def test_empty_split_section_yields_empty_indices():
    splits = make_split_indices(
        num_samples=50,
        num_components=4,
        context_window=10,
        split={"train": 1.0, "val": 0.0, "test": 0.0},
        seed=0,
    )
    assert splits.val.shape == (0, 2)
    assert splits.test.shape == (0, 2)
    assert splits.val.dtype == np.int64


def test_invalid_split_fractions_raise():
    with pytest.raises(ValueError):
        make_split_indices(
            num_samples=100,
            num_components=4,
            context_window=10,
            split={"train": 0.5, "val": 0.1, "test": 0.1},
            seed=0,
        )


def test_invalid_context_window_raises():
    with pytest.raises(ValueError):
        make_split_indices(
            num_samples=100,
            num_components=4,
            context_window=0,
            split={"train": 0.8, "val": 0.1, "test": 0.1},
            seed=0,
        )


def test_num_samples_smaller_than_window_raises():
    with pytest.raises(ValueError):
        make_split_indices(
            num_samples=5,
            num_components=4,
            context_window=10,
            split={"train": 0.8, "val": 0.1, "test": 0.1},
            seed=0,
        )
