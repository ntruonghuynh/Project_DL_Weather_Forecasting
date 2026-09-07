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
    missing_indicator_diagnostics,
    validate_weather_batch,
    window_rejection_diagnostics,
)
from src.data.preprocessing import FEATURE_COLUMNS as MODEL_FEATURE_COLUMNS


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


def test_long_gap_survives_impute_and_rejects_touching_windows_end_to_end() -> None:
    """Integration: preprocessing.impute_missing's output feeds dataset window rejection.

    Unlike test_windows_touching_a_gap_are_dropped (which injects NaN
    directly into the dataset), this drives two real gaps through
    src.data.preprocessing.impute_missing first: a SHORT gap (fully closed by
    forward-fill, zero remainder) placed far enough from a LONG gap (only its
    first forward_fill_max_hours closed, remainder left NaN) that no single
    240-hour window can touch both. This proves, end to end across both
    modules, that a fully-closed short gap never rejects a window while a
    long gap's leftover NaN rejects every window touching it.
    """
    from src.data.preprocessing import (
        RAW_NUMERIC_COLUMNS,
        impute_missing,
    )
    from src.data.preprocessing import (
        TARGET_COLUMN as PP_TARGET_COLUMN,
    )
    from src.data.preprocessing import (
        TIMESTAMP_COLUMN as PP_TIMESTAMP_COLUMN,
    )

    n_hours = 500
    short_gap_start = 50
    short_len = 3  # matches configs/data.yaml missing.forward_fill_max_hours - fully closable
    long_gap_start = 350  # SAMPLE_SPAN (240h) away from the short gap - no window can span both
    long_len = 10  # beyond forward_fill_max_hours - remainder must stay NaN
    gap_column = next(c for c in RAW_NUMERIC_COLUMNS if c != PP_TARGET_COLUMN)

    timestamps = pd.date_range("2020-01-01", periods=n_hours, freq="1h")
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {PP_TIMESTAMP_COLUMN: timestamps}
        | {column: rng.normal(size=n_hours) for column in RAW_NUMERIC_COLUMNS}
    )
    df.loc[short_gap_start : short_gap_start + short_len - 1, gap_column] = np.nan
    df.loc[long_gap_start : long_gap_start + long_len - 1, gap_column] = np.nan
    empty = df.iloc[0:0].copy()

    imputed_df, _, _, _ = impute_missing(df, empty, empty, forward_fill_limit_hours=short_len)

    short_gap_slice = imputed_df[gap_column].iloc[short_gap_start : short_gap_start + short_len]
    assert short_gap_slice.notna().all(), (
        "a gap no longer than forward_fill_max_hours must be fully forward-filled"
    )
    long_filled = imputed_df[gap_column].iloc[long_gap_start : long_gap_start + short_len]
    long_remainder_end = long_gap_start + long_len
    long_remainder = imputed_df[gap_column].iloc[long_gap_start + short_len : long_remainder_end]
    assert long_filled.notna().all(), "the SHORT portion of the long gap must be forward-filled"
    assert long_remainder.isna().all(), "the LONG remainder must stay NaN, never fabricated"

    dataset = WeatherForecastDataset(
        imputed_df, RAW_NUMERIC_COLUMNS, PP_TARGET_COLUMN, input_window=WINDOW, horizon=HORIZON
    )
    unfilled_timestamps = set(imputed_df.loc[imputed_df[gap_column].isna(), PP_TIMESTAMP_COLUMN])
    short_gap_timestamps = set(timestamps[short_gap_start : short_gap_start + short_len])

    # gap_column is an INPUT feature, never part of y - so only each window's INPUT
    # span (not its forecast span) can legitimately be rejected by a NaN in it.
    covers_unfilled_input = 0
    covers_closed_short_gap_input = 0
    for index in range(len(dataset)):
        bounds = dataset.window_bounds(index)
        input_span = set(pd.date_range(bounds["input_start"], bounds["input_end"], freq="1h"))
        if unfilled_timestamps & input_span:
            covers_unfilled_input += 1
        if short_gap_timestamps & input_span:
            covers_closed_short_gap_input += 1

    assert covers_unfilled_input == 0, (
        "no window's input period may touch a timestamp impute_missing left as NaN"
    )
    assert covers_closed_short_gap_input > 0, (
        "a fully forward-filled short gap is all-finite and must not blanket-reject every "
        "window whose input period merely passes through it"
    )


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


def test_irregular_timestamp_spacing_is_rejected() -> None:
    """Refuse to build windows when a row is missing from the hourly grid itself.

    resample_hourly always fills every hour bin (a real gap becomes a NaN row,
    not a missing timestamp), so a jump other than exactly 3600s means the
    caller passed data that never went through preprocessing correctly.
    """
    df = make_hourly_df(SAMPLE_SPAN + 5)
    df = df.drop(index=50).reset_index(drop=True)

    with pytest.raises(ValueError, match="3600"):
        WeatherForecastDataset(df, FEATURES, TARGET, input_window=WINDOW, horizon=HORIZON)


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


def test_model_feature_batch_has_18_inputs() -> None:
    n_hours = SAMPLE_SPAN + 40
    timestamps = pd.date_range("2020-01-01", periods=n_hours, freq="1h")
    frame = pd.DataFrame({"Date Time": timestamps})
    for index, column in enumerate(MODEL_FEATURE_COLUMNS):
        frame[column] = np.arange(n_hours, dtype=np.float32) + index
    dataset = WeatherForecastDataset(
        frame, MODEL_FEATURE_COLUMNS, "T (degC)", input_window=WINDOW, horizon=HORIZON
    )
    loader = build_dataloaders(dataset, dataset, dataset, batch_size=4)["train"]
    batch = next(iter(loader))
    assert batch["x"].shape == (4, 168, 18)
    assert batch["y"].shape == (4, 72, 1)
    validate_weather_batch(batch, n_features=18)


def test_last_observed_target_matches_final_input_hour() -> None:
    """last_observed_target reads the target value at the last input timestep, not beyond."""
    df = make_hourly_df(SAMPLE_SPAN + 5)
    dataset = WeatherForecastDataset(df, FEATURES, TARGET, input_window=WINDOW, horizon=HORIZON)

    expected = float(df[TARGET].iloc[WINDOW - 1])
    assert dataset.last_observed_target(0) == pytest.approx(expected)


def test_window_rejection_diagnostic_matches_dataset_length() -> None:
    df = make_hourly_df(SAMPLE_SPAN + 5)
    df.loc[1, FEATURES[0]] = np.nan
    dataset = WeatherForecastDataset(df, FEATURES, TARGET, input_window=WINDOW, horizon=HORIZON)
    diagnostic = window_rejection_diagnostics(df, FEATURES, TARGET, WINDOW, HORIZON)
    assert diagnostic["valid_windows"] == len(dataset)
    rejected = sum(value for key, value in diagnostic.items() if key.startswith("rejected_"))
    assert diagnostic["valid_windows"] + rejected == diagnostic["candidate_windows"]


def test_missing_indicator_diagnostic_counts_rows_and_valid_windows() -> None:
    df = make_hourly_df(SAMPLE_SPAN + 2)
    indicator = "missing_demo"
    df[indicator] = 0.0
    df.loc[0, indicator] = 1.0
    features = [*FEATURES, indicator]
    dataset = WeatherForecastDataset(df, features, TARGET, input_window=WINDOW, horizon=HORIZON)
    diagnostic = missing_indicator_diagnostics(df, dataset, [indicator])
    assert diagnostic["processed_rows_with_indicator"] == 1
    assert diagnostic["valid_windows_with_indicator"] == 1
