import torch

from src.models.baselines import (
    persistence_forecast,
    seasonal_naive_24h,
    seasonal_naive_168h,
)


def test_persistence_output_shape():
    x = torch.randn(4, 168, 18)

    prediction = persistence_forecast(
        x=x,
        horizon=72,
        target_feature_index=1,
    )

    assert prediction.shape == (4, 72, 1)


def test_persistence_uses_last_target():
    x = torch.zeros(2, 168, 18)

    x[0, -1, 1] = 5.0
    x[1, -1, 1] = 10.0

    prediction = persistence_forecast(
        x=x,
        horizon=72,
        target_feature_index=1,
    )

    assert torch.all(prediction[0] == 5.0)
    assert torch.all(prediction[1] == 10.0)


def test_persistence_batch_size_one():
    x = torch.randn(1, 168, 18)

    prediction = persistence_forecast(
        x=x,
        horizon=72,
        target_feature_index=1,
    )

    assert prediction.shape == (1, 72, 1)
def test_seasonal_naive_24h_output_shape():
    x = torch.randn(4, 168, 18)

    prediction = seasonal_naive_24h(
        x=x,
        horizon=72,
        target_feature_index=1,
    )

    assert prediction.shape == (4, 72, 1)


def test_seasonal_naive_24h_repeats_last_24_hours():
    x = torch.zeros(1, 168, 18)

    pattern = torch.arange(24, dtype=torch.float32)
    x[0, -24:, 1] = pattern

    prediction = seasonal_naive_24h(
        x=x,
        horizon=72,
        target_feature_index=1,
    )

    expected = pattern.repeat(3)

    assert torch.equal(prediction[0, :, 0], expected)


def test_seasonal_naive_168h_output_shape():
    x = torch.randn(4, 168, 18)

    prediction = seasonal_naive_168h(
        x=x,
        horizon=72,
        target_feature_index=1,
    )

    assert prediction.shape == (4, 72, 1)


def test_seasonal_naive_168h_uses_week_ago_values():
    x = torch.zeros(1, 168, 18)

    pattern = torch.arange(168, dtype=torch.float32)
    x[0, :, 1] = pattern

    prediction = seasonal_naive_168h(
        x=x,
        horizon=72,
        target_feature_index=1,
    )

    expected = pattern[:72]

    assert torch.equal(prediction[0, :, 0], expected)