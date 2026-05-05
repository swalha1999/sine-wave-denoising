import numpy as np
import pytest

from sine_denoiser.evaluation.metrics import (
    MseReport,
    compute,
    mse,
    mse_per_component,
)


def test_mse_hand_computed():
    y_hat = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([[2.0, 2.0], [3.0, 5.0]])
    # squared errors: [[1, 0], [0, 1]] -> mean 2/4 = 0.5
    assert mse(y_hat, y) == pytest.approx(0.5)


def test_mse_identical_arrays_is_zero():
    y = np.arange(20.0).reshape(4, 5)
    assert mse(y, y) == 0.0


def test_mse_accepts_lists():
    assert mse([1.0, 3.0], [2.0, 5.0]) == pytest.approx((1 + 4) / 2)


def test_mse_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape mismatch"):
        mse(np.zeros((2, 3)), np.zeros((2, 4)))


def test_mse_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        mse(np.zeros((0, 5)), np.zeros((0, 5)))


def test_mse_per_component_hand_computed():
    y_hat = np.array(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
    )
    y = np.array(
        [[2.0, 2.0], [3.0, 4.0], [5.0, 7.0], [9.0, 8.0]]
    )
    c = np.array([0, 0, 1, 1])
    # c=0 rows: sq=[[1,0],[0,0]] -> mean 1/4 = 0.25
    # c=1 rows: sq=[[0,1],[4,0]] -> mean 5/4 = 1.25
    out = mse_per_component(y_hat, y, c, num_components=2)
    np.testing.assert_allclose(out, [0.25, 1.25])


def test_mse_per_component_nan_for_missing_component():
    y_hat = np.array([[1.0], [2.0]])
    y = np.array([[0.0], [0.0]])
    c = np.array([0, 0])
    out = mse_per_component(y_hat, y, c, num_components=3)
    assert out[0] == pytest.approx((1.0 + 4.0) / 2)
    assert np.isnan(out[1])
    assert np.isnan(out[2])


def test_mse_per_component_returns_float64_array_of_correct_length():
    y_hat = np.zeros((4, 3))
    y = np.zeros((4, 3))
    c = np.array([0, 1, 2, 3])
    out = mse_per_component(y_hat, y, c, num_components=4)
    assert out.shape == (4,)
    assert out.dtype == np.float64


def test_mse_per_component_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape mismatch"):
        mse_per_component(
            np.zeros((2, 3)), np.zeros((2, 4)), np.array([0, 0]), 2
        )


def test_mse_per_component_batch_mismatch_raises():
    with pytest.raises(ValueError, match="batch mismatch"):
        mse_per_component(
            np.zeros((4, 3)), np.zeros((4, 3)), np.array([0, 1, 2]), 4
        )


def test_mse_per_component_c_must_be_1d():
    with pytest.raises(ValueError, match="1-D"):
        mse_per_component(
            np.zeros((2, 3)), np.zeros((2, 3)), np.zeros((2, 1)), 2
        )


def test_mse_per_component_invalid_num_components_raises():
    with pytest.raises(ValueError, match="num_components"):
        mse_per_component(
            np.zeros((2, 3)), np.zeros((2, 3)), np.array([0, 0]), 0
        )


def test_compute_returns_total_and_per_component():
    y_hat = np.array([[1.0, 0.0], [0.0, 2.0]])
    y = np.array([[0.0, 0.0], [0.0, 0.0]])
    c = np.array([0, 1])
    report = compute(y_hat, y, c, num_components=2)
    assert isinstance(report, MseReport)
    # total: (1+0+0+4)/4 = 1.25
    assert report.total == pytest.approx(1.25)
    # c=0 row: (1+0)/2 = 0.5; c=1 row: (0+4)/2 = 2.0
    np.testing.assert_allclose(report.per_component, [0.5, 2.0])
