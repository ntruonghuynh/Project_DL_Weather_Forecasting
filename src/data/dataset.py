"""Data Batch Contract v1; dataset implementation belongs to TV2."""

from collections.abc import Mapping
from typing import Protocol, TypedDict

import torch


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
    input_length: int = 168,
    horizon: int = 72,
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
