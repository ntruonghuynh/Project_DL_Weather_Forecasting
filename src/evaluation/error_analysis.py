"""Error analysis for aligned forecast records."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from .metrics import validate_aligned_forecasts


def build_error_frame(
    y_true: object,
    y_pred: object,
    *,
    target_timestamps: object | None = None,
    run_id: str,
    model_name: str,
    split: str,
) -> pd.DataFrame:
    """Create one traceable row per sample and forecast horizon."""
    if not run_id or not model_name or not split:
        raise ValueError("run_id, model_name and split are required")
    true, pred = validate_aligned_forecasts(y_true, y_pred)
    n_samples, horizon = true.shape
    if target_timestamps is None:
        timestamps = np.full((n_samples, horizon), None, dtype=object)
    else:
        timestamps = np.asarray(target_timestamps)
        if timestamps.shape != true.shape:
            raise ValueError("target_timestamps must align with [N,H] forecasts")

    frame = pd.DataFrame(
        {
            "sample_index": np.repeat(np.arange(n_samples), horizon),
            "horizon": np.tile(np.arange(1, horizon + 1), n_samples),
            "timestamp": timestamps.reshape(-1),
            "y_true": true.reshape(-1),
            "y_pred": pred.reshape(-1),
        }
    )
    frame["residual"] = frame["y_pred"] - frame["y_true"]
    frame["absolute_error"] = frame["residual"].abs()
    frame["squared_error"] = frame["residual"] ** 2
    frame["run_id"] = run_id
    frame["model_name"] = model_name
    frame["split"] = split
    frame["unit"] = "degC"
    return frame


def analyze_errors(predictions: object, top_n: int = 20) -> dict[str, Any]:
    """Return worst cases and optional hour/season summaries from a record frame."""
    if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n <= 0:
        raise ValueError("top_n must be a positive integer")
    frame = (
        predictions.copy() if isinstance(predictions, pd.DataFrame) else pd.DataFrame(predictions)
    )
    required = {"y_true", "y_pred", "horizon", "timestamp", "run_id", "model_name", "split"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"prediction records are missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("prediction records must not be empty")

    for column in ("y_true", "y_pred"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if not np.isfinite(frame[column]).all():
            raise ValueError(f"{column} must contain only finite values")
    frame["residual"] = frame["y_pred"] - frame["y_true"]
    frame["absolute_error"] = frame["residual"].abs()
    frame["squared_error"] = frame["residual"] ** 2

    worst = frame.nlargest(min(top_n, len(frame)), "absolute_error").reset_index(drop=True)
    summaries: dict[str, Any] = {"worst_cases": worst}
    parsed = pd.to_datetime(frame["timestamp"], errors="coerce")
    if parsed.notna().all():
        enriched = frame.assign(hour=parsed.dt.hour, month=parsed.dt.month)
        seasons = {
            12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring",
            5: "spring", 6: "summer", 7: "summer", 8: "summer", 9: "autumn",
            10: "autumn", 11: "autumn",
        }
        enriched["season"] = enriched["month"].map(seasons)
        summaries["by_hour"] = _group_summary(enriched, ["hour"])
        summaries["by_season"] = _group_summary(enriched, ["season"])
    summaries["by_horizon"] = _group_summary(frame, ["horizon"])
    return summaries


def _group_summary(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    grouped = frame.groupby(list(columns), observed=True, dropna=False)
    summary = grouped.agg(
        sample_count=("absolute_error", "size"),
        mae=("absolute_error", "mean"),
        mse=("squared_error", "mean"),
        mean_residual=("residual", "mean"),
    ).reset_index()
    summary["rmse"] = np.sqrt(summary["mse"])
    return summary
