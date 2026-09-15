"""Leakage-safe forecasting metrics for standardized or original-scale arrays."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _as_forecast_array(values: object, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim != 2:
        raise ValueError(f"{name} must have shape [N,H] or [N,H,1], got {array.shape}")
    if not array.size or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def validate_aligned_forecasts(y_true: object, y_pred: object) -> tuple[np.ndarray, np.ndarray]:
    """Return finite float64 arrays after enforcing sample/horizon alignment."""
    true = _as_forecast_array(y_true, "y_true")
    pred = _as_forecast_array(y_pred, "y_pred")
    if true.shape != pred.shape:
        raise ValueError(
            f"y_true and y_pred must have identical shape, got {true.shape} vs {pred.shape}"
        )
    return true, pred


def inverse_scale_target(
    values: object,
    scaler: Any | None = None,
    target_index: int | None = None,
    *,
    mean: float | None = None,
    scale: float | None = None,
) -> np.ndarray:
    """Restore a target-only array using a train-fitted StandardScaler."""
    array = np.asarray(values, dtype=np.float64)
    if scaler is not None:
        if target_index is None:
            raise ValueError("target_index is required when scaler is provided")
        if not hasattr(scaler, "mean_") or not hasattr(scaler, "scale_"):
            raise TypeError("scaler must expose fitted mean_ and scale_ arrays")
        if isinstance(target_index, bool) or not isinstance(target_index, int):
            raise TypeError("target_index must be an integer")
        if not 0 <= target_index < len(scaler.mean_):
            raise ValueError("target_index is outside the fitted scaler feature range")
        mean = float(scaler.mean_[target_index])
        scale = float(scaler.scale_[target_index])
    if mean is None or scale is None:
        raise ValueError("provide either scaler+target_index or explicit mean+scale")
    if not np.isfinite(mean) or not np.isfinite(scale) or scale <= 0:
        raise ValueError("inverse-scaling mean/scale must be finite and scale must be positive")
    return array * scale + mean


def compute_metrics(y_true: object, y_pred: object) -> dict[str, float]:
    """Compute deterministic MAE, MSE and RMSE for aligned forecast arrays."""
    true, pred = validate_aligned_forecasts(y_true, y_pred)
    residual = pred - true
    mse = float(np.mean(np.square(residual)))
    return {
        "mae": float(np.mean(np.abs(residual))),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
    }


def compute_per_horizon_metrics(y_true: object, y_pred: object) -> list[dict[str, float | int]]:
    """Compute MAE/MSE/RMSE separately for every forecast step."""
    true, pred = validate_aligned_forecasts(y_true, y_pred)
    residual = pred - true
    mse = np.mean(np.square(residual), axis=0)
    mae = np.mean(np.abs(residual), axis=0)
    return [
        {
            "horizon": step + 1,
            "mae": float(mae[step]),
            "mse": float(mse[step]),
            "rmse": float(np.sqrt(mse[step])),
        }
        for step in range(true.shape[1])
    ]


def persistence_predictions(last_observed_target: object, horizon: int) -> np.ndarray:
    """Build a persistence forecast on the same sample population as a model."""
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
        raise ValueError("horizon must be a positive integer")
    anchor = np.asarray(last_observed_target, dtype=np.float64)
    if anchor.ndim == 2 and anchor.shape[1] == 1:
        anchor = anchor[:, 0]
    if anchor.ndim != 1 or anchor.size == 0:
        raise ValueError("last_observed_target must have shape [N] or [N,1]")
    if not np.isfinite(anchor).all():
        raise ValueError("last_observed_target must contain only finite values")
    return np.repeat(anchor[:, None], horizon, axis=1)


@dataclass(frozen=True)
class EvaluationResult:
    """Overall and per-horizon metrics with explicit scale metadata."""

    overall: dict[str, float]
    per_horizon: list[dict[str, float | int]]
    unit: str
    sample_count: int
    horizon: int


def evaluate_forecasts(
    y_true: object,
    y_pred: object,
    *,
    scaler: Any | None = None,
    target_index: int | None = None,
    already_original_scale: bool = False,
) -> EvaluationResult:
    """Evaluate forecasts in degrees Celsius, inverse-scaling when required."""
    true, pred = validate_aligned_forecasts(y_true, y_pred)
    if not already_original_scale:
        if scaler is None:
            raise ValueError("a train-fitted scaler is required for standardized predictions")
        true = inverse_scale_target(true, scaler, target_index)
        pred = inverse_scale_target(pred, scaler, target_index)
    return EvaluationResult(
        overall=compute_metrics(true, pred),
        per_horizon=compute_per_horizon_metrics(true, pred),
        unit="degC",
        sample_count=int(true.shape[0]),
        horizon=int(true.shape[1]),
    )
