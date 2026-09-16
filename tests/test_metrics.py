"""Unit tests for original-scale and per-horizon forecasting metrics."""

import numpy as np
import pytest

from src.evaluation.metrics import (
    compute_metrics,
    compute_per_horizon_metrics,
    evaluate_forecasts,
    inverse_scale_target,
    persistence_predictions,
)


class FakeScaler:
    mean_ = np.array([5.0, 10.0])
    scale_ = np.array([2.0, 4.0])


def test_metrics_match_hand_calculation() -> None:
    true = [[1.0, 2.0], [3.0, 4.0]]
    pred = [[2.0, 0.0], [3.0, 6.0]]
    metrics = compute_metrics(true, pred)
    assert metrics == pytest.approx({"mae": 1.25, "mse": 2.25, "rmse": 1.5})


def test_per_horizon_metrics() -> None:
    result = compute_per_horizon_metrics([[1, 2], [3, 4]], [[2, 0], [3, 6]])
    assert result[0] == pytest.approx({"horizon": 1, "mae": 0.5, "mse": 0.5,
                                      "rmse": np.sqrt(0.5)})
    assert result[1] == pytest.approx({"horizon": 2, "mae": 2.0, "mse": 4.0, "rmse": 2.0})


def test_inverse_target_uses_only_target_statistics() -> None:
    restored = inverse_scale_target([[0.0, 1.0]], FakeScaler(), 1)
    np.testing.assert_allclose(restored, [[10.0, 14.0]])


def test_evaluate_forecasts_reports_degrees_celsius() -> None:
    result = evaluate_forecasts([[0.0, 1.0]], [[1.0, 1.0]], scaler=FakeScaler(), target_index=1)
    assert result.unit == "degC"
    assert result.overall["mae"] == pytest.approx(2.0)
    assert result.sample_count == 1
    assert result.horizon == 2


def test_persistence_uses_same_population() -> None:
    result = persistence_predictions([1.0, 2.0], 3)
    np.testing.assert_array_equal(result, [[1, 1, 1], [2, 2, 2]])


@pytest.mark.parametrize(
    "true,pred",
    [([[1, 2]], [[1]]), ([[1, np.nan]], [[1, 2]]), ([], [])],
)
def test_metrics_reject_misaligned_or_invalid_arrays(true, pred) -> None:
    with pytest.raises(ValueError):
        compute_metrics(true, pred)
