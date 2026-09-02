"""Executable tests for Data Batch Contract v1 and the TV2 sliding-window dataset."""

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.dataloader import build_dataloaders
from src.data.dataset import (
    WeatherBatch,
    WeatherDataset,
    WeatherForecastDataset,
    WeatherSample,
    validate_weather_batch,
)


def make_batch(batch_size: int = 2) -> WeatherBatch:
    """Build a synthetic batch without reading a dataset."""
    return {
        "x": torch.zeros(batch_size, 168, 4, dtype=torch.float32),
        "y": torch.zeros(batch_size, 72, 1, dtype=torch.float32),
        "input_timestamps": torch.zeros(batch_size, 168, dtype=torch.int64),
        "target_timestamps": torch.zeros(batch_size, 72, dtype=torch.int64),
    }


def test_valid_batch_contract() -> None:
    """Accept a correctly shaped and typed synthetic batch."""
    validate_weather_batch(make_batch())


def test_dataset_returns_samples_before_dataloader_collation() -> None:
    """Keep the dataset item contract distinct from the collated batch contract."""
    return_type = WeatherDataset.__getitem__.__annotations__["return"]

    assert return_type is WeatherSample
    assert "DataLoader collates" in (WeatherSample.__doc__ or "")
    assert "DataLoader collates" in (WeatherDataset.__doc__ or "")


def test_expected_feature_count_is_accepted() -> None:
    """Accept the schema-provided feature count when it matches x."""
    validate_weather_batch(make_batch(), n_features=4)


@pytest.mark.parametrize("n_features", [3, 5])
def test_wrong_feature_count_is_rejected(n_features: int) -> None:
    """Reject x when its feature count differs from schema/configuration."""
    with pytest.raises(ValueError, match="feature count"):
        validate_weather_batch(make_batch(), n_features=n_features)


@pytest.mark.parametrize("n_features", [0, -1, 1.5, True])
def test_invalid_expected_feature_count_is_rejected(n_features: object) -> None:
    """Require n_features to be a positive integer when supplied."""
    with pytest.raises(ValueError, match="positive integer"):
        validate_weather_batch(make_batch(), n_features=n_features)  # type: ignore[arg-type]


def test_empty_batch_is_rejected() -> None:
    """Reject batch_size=0 even when all remaining dimensions are valid."""
    validate_target = {
        "x": torch.zeros(0, 168, 4, dtype=torch.float32),
        "y": torch.zeros(0, 72, 1, dtype=torch.float32),
        "input_timestamps": torch.zeros(0, 168, dtype=torch.int64),
        "target_timestamps": torch.zeros(0, 72, dtype=torch.int64),
    }

    with pytest.raises(ValueError, match="greater than zero"):
        validate_weather_batch(validate_target)


def test_missing_key_is_rejected() -> None:
    """Reject a batch that omits a required contract key."""
    batch = dict(make_batch())
    del batch["target_timestamps"]

    with pytest.raises(KeyError, match="target_timestamps"):
        validate_weather_batch(batch)


def test_wrong_feature_shape_is_rejected() -> None:
    """Reject an incorrect input sequence length."""
    batch = make_batch()
    batch["x"] = torch.zeros(2, 167, 4, dtype=torch.float32)

    with pytest.raises(ValueError, match="length must be 168"):
        validate_weather_batch(batch)


def test_wrong_target_dtype_is_rejected() -> None:
    """Reject targets that are not float32."""
    batch = make_batch()
    batch["y"] = batch["y"].to(torch.float64)

    with pytest.raises(TypeError, match="torch.float32"):
        validate_weather_batch(batch)


def test_wrong_timestamp_dtype_is_rejected() -> None:
    """Reject timestamps that are not int64."""
    batch = make_batch()
    batch["input_timestamps"] = batch["input_timestamps"].to(torch.int32)

    with pytest.raises(TypeError, match="torch.int64"):
        validate_weather_batch(batch)


def test_timestamp_alignment_is_enforced() -> None:
    """Reject timestamps that do not align with their sequence."""
    batch = make_batch()
    batch["target_timestamps"] = torch.zeros(2, 71, dtype=torch.int64)

    with pytest.raises(ValueError, match="target timestamp alignment"):
        validate_weather_batch(batch)


def test_feature_and_target_devices_must_match() -> None:
    """Reject an x/y device mismatch using PyTorch's allocation-free meta device."""
    batch = make_batch()
    batch["y"] = torch.empty(2, 72, 1, dtype=torch.float32, device="meta")

    with pytest.raises(ValueError, match="must share a device"):
        validate_weather_batch(batch)


# ---------------------------------------------------------------------------
# WeatherForecastDataset - sliding-window implementation (TV2)
# ---------------------------------------------------------------------------

FEATURES = ["p", "T", "hour_sin"]
TARGET = "T"
WINDOW = 168
HORIZON = 72
SAMPLE_SPAN = WINDOW + HORIZON  # 240 hours needed for one sample


def make_hourly_df(
    n_hours: int, start: str = "2020-01-01", nan_hours: set[int] | None = None
) -> pd.DataFrame:
    """Build a small synthetic processed-style hourly DataFrame for dataset tests."""
    nan_hours = nan_hours or set()
    timestamps = pd.date_range(start, periods=n_hours, freq="1h")
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "Date Time": timestamps,
            "p": rng.normal(size=n_hours).astype("float32"),
            "T": rng.normal(size=n_hours).astype("float32"),
            "hour_sin": np.sin(2 * np.pi * timestamps.hour / 24).astype("float32"),
        }
    )
    for hour_index in nan_hours:
        df.loc[hour_index, "T"] = np.nan
    return df


def test_dataset_is_created_with_expected_length() -> None:
    """Dataset length equals the number of valid 240-hour windows."""
    n_hours = SAMPLE_SPAN + 10
    df = make_hourly_df(n_hours)
    dataset = WeatherForecastDataset(df, FEATURES, TARGET, input_window=WINDOW, horizon=HORIZON)
    assert len(dataset) == n_hours - SAMPLE_SPAN + 1


def test_sample_shapes_and_dtypes() -> None:
    """x, y, and both timestamp tensors match the WeatherSample contract."""
    df = make_hourly_df(SAMPLE_SPAN + 5)
    dataset = WeatherForecastDataset(df, FEATURES, TARGET, input_window=WINDOW, horizon=HORIZON)
    sample = dataset[0]

    assert sample["x"].shape == (WINDOW, len(FEATURES))
    assert sample["y"].shape == (HORIZON, 1)
    assert sample["input_timestamps"].shape == (WINDOW,)
    assert sample["target_timestamps"].shape == (HORIZON,)
    assert sample["x"].dtype == torch.float32
    assert sample["y"].dtype == torch.float32
    assert sample["input_timestamps"].dtype == torch.int64
    assert sample["target_timestamps"].dtype == torch.int64


def test_no_nan_or_inf_in_any_sample() -> None:
    """Every emitted sample is finite, even when the source data has gaps."""
    df = make_hourly_df(SAMPLE_SPAN * 3, nan_hours={5, 300})
    df.loc[100, "p"] = np.inf
    dataset = WeatherForecastDataset(df, FEATURES, TARGET, input_window=WINDOW, horizon=HORIZON)

    assert len(dataset) > 0
    for index in range(len(dataset)):
        sample = dataset[index]
        assert torch.isfinite(sample["x"]).all()
        assert torch.isfinite(sample["y"]).all()


def test_windows_touching_a_gap_are_dropped() -> None:
    """A single missing hour removes exactly the windows whose span covers it."""
    n_hours = 600
    gap_hour = 300
    clean_dataset = WeatherForecastDataset(
        make_hourly_df(n_hours), FEATURES, TARGET, input_window=WINDOW, horizon=HORIZON
    )
    gapped_dataset = WeatherForecastDataset(
        make_hourly_df(n_hours, nan_hours={gap_hour}),
        FEATURES,
        TARGET,
        input_window=WINDOW,
        horizon=HORIZON,
    )

    total_windows = n_hours - SAMPLE_SPAN + 1
    expected_dropped = min(SAMPLE_SPAN, total_windows)
    assert len(clean_dataset) == total_windows
    assert len(gapped_dataset) == total_windows - expected_dropped

    for index in range(len(gapped_dataset)):
        assert torch.isfinite(gapped_dataset[index]["x"]).all()
        assert torch.isfinite(gapped_dataset[index]["y"]).all()


def test_timestamps_are_hourly_and_forecast_follows_input() -> None:
    """Timestamps step by exactly one hour, and forecast starts right after input ends."""
    df = make_hourly_df(SAMPLE_SPAN + 5)
    dataset = WeatherForecastDataset(df, FEATURES, TARGET, input_window=WINDOW, horizon=HORIZON)
    sample = dataset[0]

    input_ts = sample["input_timestamps"].numpy()
    target_ts = sample["target_timestamps"].numpy()
    assert np.all(np.diff(input_ts) == 3600)
    assert np.all(np.diff(target_ts) == 3600)
    assert target_ts[0] == input_ts[-1] + 3600


def test_split_boundary_never_crossed() -> None:
    """A dataset built from one split's rows can never emit a timestamp outside that split."""
    full_df = make_hourly_df(SAMPLE_SPAN * 4)
    split_point = len(full_df) // 2
    train_df = full_df.iloc[:split_point].reset_index(drop=True)
    val_df = full_df.iloc[split_point:].reset_index(drop=True)

    train_dataset = WeatherForecastDataset(
        train_df, FEATURES, TARGET, input_window=WINDOW, horizon=HORIZON
    )
    val_dataset = WeatherForecastDataset(
        val_df, FEATURES, TARGET, input_window=WINDOW, horizon=HORIZON
    )

    train_end = int(train_df["Date Time"].iloc[-1].timestamp())
    val_start = int(val_df["Date Time"].iloc[0].timestamp())

    for index in range(len(train_dataset)):
        assert train_dataset[index]["target_timestamps"][-1].item() <= train_end
    for index in range(len(val_dataset)):
        assert val_dataset[index]["input_timestamps"][0].item() >= val_start


def test_unsorted_timestamps_are_rejected() -> None:
    """Refuse to build windows over a DataFrame that is not time-sorted."""
    df = make_hourly_df(SAMPLE_SPAN + 5)
    shuffled = df.sample(frac=1.0, random_state=0).reset_index(drop=True)

    with pytest.raises(ValueError, match="sorted"):
        WeatherForecastDataset(shuffled, FEATURES, TARGET, input_window=WINDOW, horizon=HORIZON)


def test_target_must_be_in_features() -> None:
    """The target column must be one of the feature columns (its index is derived from it)."""
    df = make_hourly_df(SAMPLE_SPAN + 5)

    with pytest.raises(ValueError, match="features"):
        WeatherForecastDataset(df, FEATURES, "not_a_feature", input_window=WINDOW, horizon=HORIZON)


def test_dataloader_batches_have_expected_dimensions() -> None:
    """build_dataloaders yields WeatherBatch-shaped, contract-valid batches."""
    df = make_hourly_df(SAMPLE_SPAN * 3)
    train_dataset = WeatherForecastDataset(
        df, FEATURES, TARGET, input_window=WINDOW, horizon=HORIZON
    )
    val_dataset = WeatherForecastDataset(
        df, FEATURES, TARGET, input_window=WINDOW, horizon=HORIZON
    )
    test_dataset = WeatherForecastDataset(
        df, FEATURES, TARGET, input_window=WINDOW, horizon=HORIZON
    )

    loaders = build_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=4)
    batch = next(iter(loaders["train"]))

    assert batch["x"].shape == (4, WINDOW, len(FEATURES))
    assert batch["y"].shape == (4, HORIZON, 1)
    assert batch["input_timestamps"].shape == (4, WINDOW)
    assert batch["target_timestamps"].shape == (4, HORIZON)
    validate_weather_batch(batch, n_features=len(FEATURES))


def test_last_observed_target_matches_final_input_hour() -> None:
    """last_observed_target reads the target value at the last input timestep, not beyond."""
    df = make_hourly_df(SAMPLE_SPAN + 5)
    dataset = WeatherForecastDataset(df, FEATURES, TARGET, input_window=WINDOW, horizon=HORIZON)

    expected = float(df[TARGET].iloc[WINDOW - 1])
    assert dataset.last_observed_target(0) == pytest.approx(expected)
