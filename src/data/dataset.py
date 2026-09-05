"""Data Batch Contract v1, plus the TV2 sliding-window dataset implementation."""

from collections.abc import Mapping
from typing import Protocol, TypedDict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..config import load_data_config
from .validator import TIMESTAMP_COLUMN

# Single Source of Truth: configs/data.yaml (see src/config.py). Do not
# hard-code the window sizes elsewhere - import DEFAULT_INPUT_LENGTH /
# DEFAULT_HORIZON from this module instead.
_CONFIG = load_data_config()
_WINDOW_CFG = _CONFIG["window"]
_MISSING_CFG = _CONFIG["missing"]
_TIMESTAMP_CFG = _CONFIG["timestamp"]
DEFAULT_INPUT_LENGTH: int = _WINDOW_CFG["input_length_hours"]
DEFAULT_HORIZON: int = _WINDOW_CFG["horizon_hours"]
# Expected spacing between consecutive rows, in seconds (3600 for the "1h"
# resample_frequency this pipeline currently uses) - derived from config
# rather than hardcoded so a future change to the resample cadence does not
# silently desync this invariant check from the actual data.
EXPECTED_STEP_SECONDS: int = int(pd.Timedelta(_TIMESTAMP_CFG["resample_frequency"]).total_seconds())


class WeatherSample(TypedDict):
    """One dataset sample before collation.

    Shapes are x [168,n_features], y [72,1], input_timestamps [168], and
    target_timestamps [72]. A DataLoader collates multiple WeatherSample
    objects into one WeatherBatch by adding the leading batch dimension.
    """

    x: torch.Tensor
    y: torch.Tensor
    input_timestamps: torch.Tensor
    target_timestamps: torch.Tensor


class WeatherBatch(TypedDict):
    """Tensor batch with features, targets, and aligned Unix-like timestamps.

    Timestamp unit and timezone are defined by external schema metadata. The
    target feature index also comes from schema/configuration, never this type.
    """

    x: torch.Tensor
    y: torch.Tensor
    input_timestamps: torch.Tensor
    target_timestamps: torch.Tensor


def _require_tensor(batch: Mapping[str, object], key: str) -> torch.Tensor:
    """Return one tensor field or raise a field-specific contract error."""
    value = batch[key]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"WeatherBatch['{key}'] must be a torch.Tensor")
    return value


def validate_weather_batch(
    batch: Mapping[str, object],
    input_length: int = DEFAULT_INPUT_LENGTH,
    horizon: int = DEFAULT_HORIZON,
    target_dim: int = 1,
    n_features: int | None = None,
) -> None:
    """Validate keys, dtypes, shapes, alignment, and feature/target devices."""
    required_keys = {"x", "y", "input_timestamps", "target_timestamps"}
    actual_keys = set(batch)
    missing_keys = required_keys - actual_keys
    unexpected_keys = actual_keys - required_keys
    if missing_keys or unexpected_keys:
        raise KeyError(
            "WeatherBatch keys are invalid: "
            f"missing={sorted(missing_keys)}, unexpected={sorted(unexpected_keys)}"
        )

    if input_length <= 0 or horizon <= 0 or target_dim <= 0:
        raise ValueError("input_length, horizon, and target_dim must be positive")
    if n_features is not None and (
        isinstance(n_features, bool) or not isinstance(n_features, int) or n_features <= 0
    ):
        raise ValueError("n_features must be a positive integer when provided")

    x = _require_tensor(batch, "x")
    y = _require_tensor(batch, "y")
    input_timestamps = _require_tensor(batch, "input_timestamps")
    target_timestamps = _require_tensor(batch, "target_timestamps")

    if x.dtype != torch.float32:
        raise TypeError(f"WeatherBatch['x'] must have dtype torch.float32, got {x.dtype}")
    if y.dtype != torch.float32:
        raise TypeError(f"WeatherBatch['y'] must have dtype torch.float32, got {y.dtype}")
    if input_timestamps.dtype != torch.int64:
        raise TypeError(
            "WeatherBatch['input_timestamps'] must have dtype torch.int64, "
            f"got {input_timestamps.dtype}"
        )
    if target_timestamps.dtype != torch.int64:
        raise TypeError(
            "WeatherBatch['target_timestamps'] must have dtype torch.int64, "
            f"got {target_timestamps.dtype}"
        )

    if x.ndim != 3:
        raise ValueError(f"WeatherBatch['x'] must have rank 3, got rank {x.ndim}")
    if y.ndim != 3:
        raise ValueError(f"WeatherBatch['y'] must have rank 3, got rank {y.ndim}")
    if input_timestamps.ndim != 2:
        raise ValueError(
            "WeatherBatch['input_timestamps'] must have rank 2, "
            f"got rank {input_timestamps.ndim}"
        )
    if target_timestamps.ndim != 2:
        raise ValueError(
            "WeatherBatch['target_timestamps'] must have rank 2, "
            f"got rank {target_timestamps.ndim}"
        )

    batch_size = x.shape[0]
    if batch_size == 0:
        raise ValueError("WeatherBatch batch size must be greater than zero")
    if x.shape[1] != input_length:
        raise ValueError(
            f"WeatherBatch['x'] length must be {input_length}, got {x.shape[1]}"
        )
    if x.shape[2] <= 0:
        raise ValueError("WeatherBatch['x'] must contain at least one feature")
    if n_features is not None and x.shape[2] != n_features:
        raise ValueError(
            f"WeatherBatch['x'] feature count must be {n_features}, got {x.shape[2]}"
        )
    if y.shape[1:] != (horizon, target_dim):
        raise ValueError(
            f"WeatherBatch['y'] shape must be [B,{horizon},{target_dim}], "
            f"got {list(y.shape)}"
        )
    if y.shape[0] != batch_size:
        raise ValueError(
            "WeatherBatch batch alignment failed: "
            f"x batch={batch_size}, y batch={y.shape[0]}"
        )
    if input_timestamps.shape != (batch_size, input_length):
        raise ValueError(
            "WeatherBatch input timestamp alignment failed: expected "
            f"[{batch_size},{input_length}], got {list(input_timestamps.shape)}"
        )
    if target_timestamps.shape != (batch_size, horizon):
        raise ValueError(
            "WeatherBatch target timestamp alignment failed: expected "
            f"[{batch_size},{horizon}], got {list(target_timestamps.shape)}"
        )
    if x.device != y.device:
        raise ValueError(
            f"WeatherBatch['x'] and WeatherBatch['y'] must share a device, "
            f"got {x.device} and {y.device}"
        )


class WeatherDataset(Protocol):
    """Dataset returning samples that a DataLoader collates into WeatherBatch."""

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> WeatherSample: ...


# ---------------------------------------------------------------------------
# Sliding-window implementation (TV2)
# ---------------------------------------------------------------------------
#
# Preprocessing (src/data/preprocessing.py) only produces clean, split,
# imputed, scaled hourly rows - one row per hour, no sequences. This section
# turns those rows into the (168h history -> 72h forecast) samples a
# seq2seq model actually trains on: it slides a window one hour at a time
# over a single split's DataFrame, keeps only windows with no NaN/Inf
# anywhere in x or y, and returns each one as a WeatherSample.


class WeatherForecastDataset(Dataset):
    """Sliding-window PyTorch Dataset over one leakage-safe processed split.

    Construct one instance per split (train/val/test): windows are built
    only from the rows of the DataFrame passed in, so a window can never
    span two splits - pass `train_df`/`val_df`/`test_df` separately, never a
    concatenation of them.

    `T (degC)` is intentionally left unimputed by preprocessing, so a gap in
    the raw data can still leave NaN in the feature matrix (input side) or
    the target column (forecast side). Any window touching such a gap -
    anywhere in its 168 input hours or 72 forecast hours - is dropped at
    construction time rather than being fed to the model, so every returned
    sample is guaranteed NaN/Inf-free (see `_find_valid_starts`).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        features: list[str],
        target: str,
        input_window: int = DEFAULT_INPUT_LENGTH,
        horizon: int = DEFAULT_HORIZON,
    ) -> None:
        if target not in features:
            raise ValueError(f"target={target!r} must be included in features")
        if input_window <= 0 or horizon <= 0:
            raise ValueError("input_window and horizon must be positive")
        expected_rule = "drop_window_if_any_nan_or_inf_in_input_or_target"
        if _MISSING_CFG["window_rejection_rule"] != expected_rule:
            raise NotImplementedError(
                "configs/data.yaml missing.window_rejection_rule="
                f"{_MISSING_CFG['window_rejection_rule']!r} is not implemented; "
                "_find_valid_starts always drops a window on any NaN/Inf in x or y."
            )

        df = df.reset_index(drop=True)
        timestamps = pd.to_datetime(df[TIMESTAMP_COLUMN])
        if not timestamps.is_monotonic_increasing:
            raise ValueError(
                "df must be sorted by timestamp ascending (no leakage guarantee otherwise)"
            )
        if len(timestamps) > 1:
            step_seconds = timestamps.diff().dropna().dt.total_seconds().to_numpy()
            irregular = step_seconds != EXPECTED_STEP_SECONDS
            if irregular.any():
                raise ValueError(
                    f"df must be on a strictly {EXPECTED_STEP_SECONDS}s-spaced grid "
                    "(configs/data.yaml timestamp.resample_frequency): found "
                    f"{int(irregular.sum())} irregular step(s). WeatherForecastDataset assumes "
                    "preprocessing already resampled to a regular grid (resample_hourly fills "
                    "every hour bin, so a real gap is a NaN row, never a missing timestamp)."
                )

        self.features = list(features)
        self.target = target
        self.input_window = input_window
        self.horizon = horizon
        # Read from the caller-supplied feature list, never hardcoded, per
        # docs/WORKFLOW.md: "target_feature_index phải lấy từ schema hoặc cấu hình".
        self.target_index = self.features.index(target)

        self._feature_arr = df[self.features].to_numpy(dtype=np.float32)
        self._target_arr = df[self.target].to_numpy(dtype=np.float32)
        # Timestamps as int64 Unix epoch seconds (source clock is
        # "unverified_source_local_time" per feature_schema.json - no
        # timezone conversion is applied, only encoded as seconds-since-epoch
        # for the tensor contract's int64 dtype).
        #
        # Cast the datetime64 array to datetime64[s] before viewing it as
        # int64: pandas datetime64 columns are not guaranteed to be in "ns"
        # resolution (pandas >=2.0 infers the resolution from the source
        # data, e.g. "us"), so dividing a raw int64 view by a fixed 10**9
        # silently corrupts the epoch value whenever the resolution isn't
        # nanoseconds.
        self._timestamps_s = timestamps.to_numpy().astype("datetime64[s]").astype(np.int64)

        self._valid_starts = self._find_valid_starts()

    def _find_valid_starts(self) -> np.ndarray:
        """Vectorized scan for window start indices with no NaN/Inf in x or y.

        For each candidate start `s`, x covers rows [s, s+input_window) and y
        covers rows [s+input_window, s+input_window+horizon). Prefix sums of
        a per-row "is bad" flag let every window's bad-row count be read off
        in O(1), so the full scan is O(n) instead of O(n * window_len).
        """
        n = len(self._feature_arr)
        window_len = self.input_window + self.horizon
        if n < window_len:
            return np.array([], dtype=np.int64)

        row_x_bad = ~np.isfinite(self._feature_arr).all(axis=1)
        row_y_bad = ~np.isfinite(self._target_arr)
        prefix_x = np.concatenate([[0], np.cumsum(row_x_bad.astype(np.int64))])
        prefix_y = np.concatenate([[0], np.cumsum(row_y_bad.astype(np.int64))])

        max_start = n - window_len
        starts = np.arange(0, max_start + 1)
        x_bad_counts = prefix_x[starts + self.input_window] - prefix_x[starts]
        y_bad_counts = prefix_y[starts + window_len] - prefix_y[starts + self.input_window]
        valid_mask = (x_bad_counts == 0) & (y_bad_counts == 0)
        return starts[valid_mask]

    def __len__(self) -> int:
        return len(self._valid_starts)

    def __getitem__(self, index: int) -> WeatherSample:
        start = int(self._valid_starts[index])
        input_end = start + self.input_window
        target_end = input_end + self.horizon

        return {
            "x": torch.from_numpy(self._feature_arr[start:input_end].copy()),
            "y": torch.from_numpy(self._target_arr[input_end:target_end].copy()).unsqueeze(-1),
            "input_timestamps": torch.from_numpy(self._timestamps_s[start:input_end].copy()),
            "target_timestamps": torch.from_numpy(self._timestamps_s[input_end:target_end].copy()),
        }

    def last_observed_target(self, index: int) -> float:
        """Scaled T(degC) at the final input hour - a persistence-baseline anchor.

        Kept off the WeatherSample/WeatherBatch contract on purpose: every
        model-facing batch must carry exactly the four contract keys (see
        validate_weather_batch), so this convenience value lives on the
        dataset instead, for evaluation/visualization/persistence-baseline
        code that reads it explicitly. It is in the same standardized units
        as `x`/`y` - inverse-transform with the fitted
        artifacts/preprocessing/scaler.joblib to recover degrees Celsius.
        """
        start = int(self._valid_starts[index])
        input_end = start + self.input_window
        return float(self._feature_arr[input_end - 1, self.target_index])

    def window_bounds(self, index: int) -> dict[str, pd.Timestamp]:
        """Human-readable input/forecast period boundaries for one sample."""
        start = int(self._valid_starts[index])
        input_end = start + self.input_window
        target_end = input_end + self.horizon

        def _to_timestamp(epoch_seconds: int) -> pd.Timestamp:
            return pd.Timestamp(int(epoch_seconds), unit="s")

        return {
            "input_start": _to_timestamp(self._timestamps_s[start]),
            "input_end": _to_timestamp(self._timestamps_s[input_end - 1]),
            "forecast_start": _to_timestamp(self._timestamps_s[input_end]),
            "forecast_end": _to_timestamp(self._timestamps_s[target_end - 1]),
        }
