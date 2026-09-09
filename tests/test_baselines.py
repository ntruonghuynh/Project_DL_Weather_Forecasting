from functools import partial

import pytest
import torch

from src.models.baselines import (
    persistence_forecast,
    seasonal_naive_24h,
    seasonal_naive_168h,
    seasonal_naive_forecast,
)

BASELINES = [
    pytest.param(persistence_forecast, id="persistence"),
    pytest.param(partial(seasonal_naive_forecast, season_length=3), id="seasonal"),
    pytest.param(seasonal_naive_24h, id="seasonal_24h"),
    pytest.param(seasonal_naive_168h, id="seasonal_168h"),
]


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


@pytest.mark.parametrize("forecast", BASELINES)
@pytest.mark.parametrize("x", [None, [1.0], 1.0])
def test_baselines_reject_non_tensor(forecast, x):
    with pytest.raises(TypeError, match="x must be a torch.Tensor"):
        forecast(x=x, horizon=5, target_feature_index=0)


@pytest.mark.parametrize("forecast", BASELINES)
@pytest.mark.parametrize("shape", [(), (3,), (2, 3), (1, 2, 3, 4)])
def test_baselines_reject_invalid_rank(forecast, shape):
    with pytest.raises(ValueError, match="x must have shape"):
        forecast(x=torch.zeros(shape), horizon=5, target_feature_index=0)


@pytest.mark.parametrize("forecast", BASELINES)
@pytest.mark.parametrize("dtype", [torch.int64, torch.bool, torch.complex64])
def test_baselines_reject_non_floating_tensor(forecast, dtype):
    with pytest.raises(TypeError, match="floating-point"):
        forecast(x=torch.zeros(1, 168, 3, dtype=dtype), horizon=5, target_feature_index=0)


@pytest.mark.parametrize("forecast", BASELINES)
@pytest.mark.parametrize("shape", [(0, 168, 3), (1, 0, 3), (1, 168, 0)])
def test_baselines_reject_empty_dimensions(forecast, shape):
    with pytest.raises(ValueError, match="must be positive"):
        forecast(x=torch.zeros(shape), horizon=5, target_feature_index=0)


INVALID_INTEGERS = [True, False, 1.0, "1", None, float("nan"), float("inf")]


@pytest.mark.parametrize("forecast", BASELINES)
@pytest.mark.parametrize("horizon", INVALID_INTEGERS)
def test_baselines_reject_non_integer_horizon(forecast, horizon):
    with pytest.raises(TypeError, match="horizon"):
        forecast(x=torch.zeros(1, 168, 3), horizon=horizon, target_feature_index=0)


@pytest.mark.parametrize("forecast", BASELINES)
@pytest.mark.parametrize("horizon", [0, -1])
def test_baselines_reject_non_positive_horizon(forecast, horizon):
    with pytest.raises(ValueError, match="horizon"):
        forecast(x=torch.zeros(1, 168, 3), horizon=horizon, target_feature_index=0)


@pytest.mark.parametrize("forecast", BASELINES)
@pytest.mark.parametrize("index", INVALID_INTEGERS)
def test_baselines_reject_non_integer_target_index(forecast, index):
    with pytest.raises(TypeError, match="target_feature_index"):
        forecast(x=torch.zeros(1, 168, 3), horizon=5, target_feature_index=index)


@pytest.mark.parametrize("forecast", BASELINES)
@pytest.mark.parametrize("index", [-1, 3, 4])
def test_baselines_reject_out_of_bounds_target_index(forecast, index):
    with pytest.raises(ValueError, match="target_feature_index"):
        forecast(x=torch.zeros(1, 168, 3), horizon=5, target_feature_index=index)


@pytest.mark.parametrize("season_length", INVALID_INTEGERS)
def test_seasonal_rejects_non_integer_season_length(season_length):
    with pytest.raises(TypeError, match="season_length"):
        seasonal_naive_forecast(torch.zeros(1, 6, 3), 5, season_length, 0)


@pytest.mark.parametrize("season_length", [0, -1])
def test_seasonal_rejects_non_positive_season_length(season_length):
    with pytest.raises(ValueError, match="season_length"):
        seasonal_naive_forecast(torch.zeros(1, 6, 3), 5, season_length, 0)


@pytest.mark.parametrize("forecast,input_length", [
    (partial(seasonal_naive_forecast, season_length=3), 2),
    (seasonal_naive_24h, 23),
    (seasonal_naive_168h, 167),
])
def test_seasonal_rejects_input_shorter_than_season(forecast, input_length):
    with pytest.raises(ValueError, match="input length.*smaller than"):
        forecast(x=torch.zeros(1, input_length, 3), horizon=5, target_feature_index=0)


@pytest.mark.parametrize("season_length,horizon", [(1, 1), (3, 2), (3, 3), (3, 8)])
def test_seasonal_repeats_last_cycle_and_truncates(season_length, horizon):
    x = torch.arange(7 * 4, dtype=torch.float64).reshape(1, 7, 4)
    prediction = seasonal_naive_forecast(x, horizon, season_length, 2)
    expected = torch.stack([
        x[:, 7 - season_length + step % season_length, 2]
        for step in range(horizon)
    ], dim=1).unsqueeze(-1)
    torch.testing.assert_close(prediction, expected, rtol=0, atol=0)


@pytest.mark.parametrize("forecast", BASELINES)
@pytest.mark.parametrize("n_features,index", [(1, 0), (5, 4)])
@pytest.mark.parametrize("device,dtype", [
    ("cpu", torch.float32),
    ("cpu", torch.float64),
    pytest.param("cuda", torch.float32, marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA is unavailable"
    )),
    pytest.param("cuda", torch.float64, marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA is unavailable"
    )),
    pytest.param("mps", torch.float32, marks=pytest.mark.skipif(
        not torch.backends.mps.is_available(), reason="MPS is unavailable"
    )),
])
def test_baselines_preserve_batch_one_shape_dtype_device(
    forecast, n_features, index, device, dtype
):
    x = torch.arange(175 * n_features, device=device, dtype=dtype).reshape(1, 175, n_features)
    original = x.clone()
    prediction = forecast(x=x, horizon=11, target_feature_index=index)

    assert prediction.shape == (1, 11, 1)
    assert prediction.dtype == x.dtype
    assert prediction.device == x.device
    torch.testing.assert_close(x, original, rtol=0, atol=0)
    if forecast is persistence_forecast:
        expected = x[:, -1:, index:index + 1].expand(-1, 11, -1)
    else:
        season_length = 24 if forecast is seasonal_naive_24h else 168
        if isinstance(forecast, partial):
            season_length = forecast.keywords["season_length"]
        expected = torch.stack([
            x[:, x.shape[1] - season_length + step % season_length, index]
            for step in range(11)
        ], dim=1).unsqueeze(-1)
    torch.testing.assert_close(prediction, expected, rtol=0, atol=0)


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
