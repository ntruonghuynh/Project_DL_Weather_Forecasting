from __future__ import annotations

import torch

HOURS_PER_DAY = 24
HOURS_PER_WEEK = 7 * HOURS_PER_DAY

def _validate_positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer; bool is not accepted")
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _validate_forecast_inputs(
    x: torch.Tensor,
    horizon: int,
    target_feature_index: int,
) -> None:
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch.Tensor")
    if x.ndim != 3:
        raise ValueError(
            f"x must have shape [B, input_len, n_features], got {tuple(x.shape)}"
        )
    if any(size <= 0 for size in x.shape):
        raise ValueError("x batch size, input length and feature count must be positive")
    if not x.is_floating_point():
        raise TypeError("x must be a floating-point tensor")

    _validate_positive_integer(horizon, "horizon")
    if isinstance(target_feature_index, bool) or not isinstance(target_feature_index, int):
        raise TypeError("target_feature_index must be an integer, not bool")
    if not 0 <= target_feature_index < x.shape[-1]:
        raise ValueError(f"Invalid target_feature_index={target_feature_index}")


def persistence_forecast(
    x: torch.Tensor,
    horizon: int,
    target_feature_index: int,
) -> torch.Tensor:
    """
    Persistence baseline.

    Dùng giá trị target cuối cùng trong chuỗi input
    để dự báo cho toàn bộ các bước tương lai.

    Args:
        x: Tensor đầu vào có shape [B, input_len, n_features].
        horizon: Số bước cần dự báo.
        target_feature_index: Vị trí của target trong feature list.

    Returns:
        Tensor dự báo có shape [B, horizon, 1].
    """

    _validate_forecast_inputs(x, horizon, target_feature_index)

    last_target = x[:, -1, target_feature_index].reshape(-1, 1, 1)

    forecast = last_target.repeat(1, horizon, 1)

    return forecast


def seasonal_naive_forecast(
    x: torch.Tensor,
    horizon: int,
    season_length: int,
    target_feature_index: int,
) -> torch.Tensor:
    """
    Seasonal naive baseline.

    Dự báo tương lai bằng cách lặp lại pattern target
    của chu kỳ gần nhất.

    Args:
        x: Input tensor [B, input_len, n_features].
        horizon: Số bước dự báo.
        season_length: Độ dài chu kỳ, ví dụ 24 hoặc 168 giờ.
        target_feature_index: Vị trí target trong feature list.

    Returns:
        Forecast tensor [B, horizon, 1].
    """

    _validate_forecast_inputs(x, horizon, target_feature_index)
    _validate_positive_integer(season_length, "season_length")

    if x.shape[1] < season_length:
        raise ValueError(
            f"input length {x.shape[1]} is smaller than "
            f"season_length={season_length}"
        )

    seasonal_pattern = x[
        :, -season_length:, target_feature_index
    ]

    repeats = (horizon + season_length - 1) // season_length

    forecast = seasonal_pattern.repeat(1, repeats)
    forecast = forecast[:, :horizon]

    return forecast.unsqueeze(-1)


def seasonal_naive_24h(
    x: torch.Tensor,
    horizon: int,
    target_feature_index: int,
) -> torch.Tensor:
    return seasonal_naive_forecast(
        x=x,
        horizon=horizon,
        season_length=HOURS_PER_DAY,
        target_feature_index=target_feature_index,
    )


def seasonal_naive_168h(
    x: torch.Tensor,
    horizon: int,
    target_feature_index: int,
) -> torch.Tensor:
    return seasonal_naive_forecast(
        x=x,
        horizon=horizon,
        season_length=HOURS_PER_WEEK,
        target_feature_index=target_feature_index,
    )