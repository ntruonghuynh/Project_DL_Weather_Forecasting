"""Executable tests for the leakage-safe preprocessing pipeline (src.data.preprocessing).

Every test builds a small, in-memory synthetic DataFrame - never reads
data/raw/jena_climate_2009_2016.csv - so the whole file runs in well under
a second and has no dependency on the large raw dataset being present.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from src.data.preprocessing import (
    RAW_NUMERIC_COLUMNS,
    TARGET_COLUMN,
    TARGET_INDEX,
    TIMESTAMP_COLUMN,
    _circular_mean_deg,
    handle_sentinels,
    impute_missing,
    inverse_transform_target,
    resample_hourly,
    scale_features,
    temporal_split,
)


def make_toy_raw_df(timestamps: pd.DatetimeIndex, **column_overrides: list[float]) -> pd.DataFrame:
    """Small synthetic raw-schema DataFrame.

    All RAW_NUMERIC_COLUMNS are present; unset ones default to 0.0.
    """
    n = len(timestamps)
    data = {TIMESTAMP_COLUMN: pd.DatetimeIndex(timestamps)}
    for column in RAW_NUMERIC_COLUMNS:
        data[column] = list(column_overrides.get(column, [0.0] * n))
    return pd.DataFrame(data, columns=[TIMESTAMP_COLUMN, *RAW_NUMERIC_COLUMNS])


def make_toy_hourly_df(
    timestamps: pd.DatetimeIndex, **column_overrides: list[float]
) -> pd.DataFrame:
    """Small synthetic post-resample DataFrame (hourly grid, RAW_NUMERIC_COLUMNS present)."""
    n = len(timestamps)
    data = {TIMESTAMP_COLUMN: pd.DatetimeIndex(timestamps)}
    for column in RAW_NUMERIC_COLUMNS:
        data[column] = list(column_overrides.get(column, np.arange(n, dtype=float)))
    return pd.DataFrame(data, columns=[TIMESTAMP_COLUMN, *RAW_NUMERIC_COLUMNS])


# ---------------------------------------------------------------------------
# 1. Sentinel handling
# ---------------------------------------------------------------------------


def test_sentinel_values_replaced_with_nan() -> None:
    """-9999 sentinel codes become NaN; non-sentinel values and other columns are untouched."""
    timestamps = pd.date_range("2020-01-01", periods=4, freq="10min")
    df = make_toy_raw_df(
        timestamps,
        **{"wv (m/s)": [1.0, -9999.0, 3.0, -9999.0], "max. wv (m/s)": [2.0, 2.0, 2.0, 2.0]},
    )
    result, report = handle_sentinels(
        df, sentinel_values=[-9999, -9999.0], sentinel_columns=["wv (m/s)", "max. wv (m/s)"]
    )

    assert result["wv (m/s)"].isna().sum() == 2
    assert result["wv (m/s)"].tolist()[0] == 1.0
    assert result["wv (m/s)"].tolist()[2] == 3.0
    assert result["max. wv (m/s)"].isna().sum() == 0
    assert report["total_replaced"] == 2
    assert report["backward_fill"] is False
    assert report["interpolate"] is False


def test_sentinel_handling_does_not_mutate_input_df() -> None:
    """handle_sentinels returns a modified copy; the caller's original DataFrame is unchanged."""
    timestamps = pd.date_range("2020-01-01", periods=2, freq="10min")
    df = make_toy_raw_df(timestamps, **{"wv (m/s)": [-9999.0, 1.0]})

    handle_sentinels(df, sentinel_values=[-9999.0], sentinel_columns=["wv (m/s)"])

    assert df["wv (m/s)"].isna().sum() == 0, "input DataFrame must not be mutated in place"


def test_sentinel_column_missing_from_df_is_skipped() -> None:
    """A configured sentinel column absent from the DataFrame is skipped, not a KeyError."""
    timestamps = pd.date_range("2020-01-01", periods=2, freq="10min")
    df = make_toy_raw_df(timestamps)

    result, report = handle_sentinels(
        df, sentinel_values=[-9999.0], sentinel_columns=["not_a_real_column"]
    )

    assert report["total_replaced"] == 0
    assert len(result) == len(df)


# ---------------------------------------------------------------------------
# 2. Resampling aggregation
# ---------------------------------------------------------------------------


def test_resample_hourly_mean_aggregation() -> None:
    """A mean-aggregated feature (T (degC)) is hourly-averaged from six 10-minute readings."""
    timestamps = pd.date_range("2020-01-01 00:00", periods=6, freq="10min")
    df = make_toy_raw_df(timestamps, **{"T (degC)": [10.0, 12.0, 14.0, 16.0, 18.0, 20.0]})

    hourly = resample_hourly(df)

    assert len(hourly) == 1
    assert hourly["T (degC)"].iloc[0] == pytest.approx(15.0)


def test_resample_hourly_max_aggregation() -> None:
    """max. wv (m/s) is hourly-maxed, not averaged - a wind gust must not be smoothed away."""
    timestamps = pd.date_range("2020-01-01 00:00", periods=6, freq="10min")
    df = make_toy_raw_df(timestamps, **{"max. wv (m/s)": [1.0, 5.0, 2.0, 9.0, 3.0, 4.0]})

    hourly = resample_hourly(df)

    assert hourly["max. wv (m/s)"].iloc[0] == pytest.approx(9.0)


def test_resample_hourly_produces_regular_grid_with_empty_hour_as_nan() -> None:
    """An hour with zero raw readings still appears as a row, with NaN.

    It is never a skipped timestamp.
    """
    # hour 01:00 has no readings
    timestamps = pd.DatetimeIndex(["2020-01-01 00:05", "2020-01-01 02:05"])
    df = make_toy_raw_df(timestamps, **{"T (degC)": [10.0, 20.0]})

    hourly = resample_hourly(df)

    assert len(hourly) == 3
    expected_index = pd.date_range("2020-01-01 00:00", periods=3, freq="1h")
    assert list(hourly[TIMESTAMP_COLUMN]) == list(expected_index)
    assert hourly["T (degC)"].isna().tolist() == [False, True, False]


# ---------------------------------------------------------------------------
# 3. Circular mean wind direction
# ---------------------------------------------------------------------------


def test_circular_mean_helper_wraps_around_360() -> None:
    """359deg/1deg are 2deg apart in reality; must average to ~0deg, not the arithmetic 180deg."""
    result = _circular_mean_deg(pd.Series([359.0, 1.0]))

    # 0 and 360 are the same angle; _circular_mean_deg's `% 360` can land on
    # either edge depending on floating-point sign, so compare circularly.
    assert min(result, 360.0 - result) == pytest.approx(0.0, abs=1e-6)
    assert result != pytest.approx(180.0)


def test_circular_mean_helper_ignores_nan_and_handles_all_nan() -> None:
    """NaN readings are dropped before averaging; an all-NaN group returns NaN, not an error."""
    assert _circular_mean_deg(pd.Series([10.0, np.nan, 30.0])) == pytest.approx(20.0, abs=1e-6)
    assert np.isnan(_circular_mean_deg(pd.Series([np.nan, np.nan])))


def test_resample_hourly_wind_direction_uses_circular_mean() -> None:
    """wd (deg) resampling matches _circular_mean_deg exactly, not a plain arithmetic mean."""
    timestamps = pd.date_range("2020-01-01 00:00", periods=2, freq="10min")
    df = make_toy_raw_df(timestamps, **{"wd (deg)": [359.0, 1.0]})

    hourly = resample_hourly(df)

    result = hourly["wd (deg)"].iloc[0]
    assert min(result, 360.0 - result) == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 4. Chronological split
# ---------------------------------------------------------------------------


def test_temporal_split_is_chronological_non_overlapping_by_ratio() -> None:
    """train/val/test are contiguous, time-ordered ranges sized by the given fractions.

    No shuffling.
    """
    timestamps = pd.date_range("2020-01-01", periods=100, freq="1h")
    df = pd.DataFrame({TIMESTAMP_COLUMN: timestamps, TARGET_COLUMN: range(100)})

    train_df, val_df, test_df, metadata = temporal_split(df, train_fraction=0.7, val_fraction=0.15)

    assert len(train_df) == 70
    assert len(val_df) == 15
    assert len(test_df) == 15
    assert train_df[TIMESTAMP_COLUMN].max() < val_df[TIMESTAMP_COLUMN].min()
    assert val_df[TIMESTAMP_COLUMN].max() < test_df[TIMESTAMP_COLUMN].min()
    assert metadata["method"] == "chronological_no_shuffle"
    assert list(train_df[TARGET_COLUMN]) == list(range(70)), (
        "train must be the OLDEST rows, not a random sample"
    )


def test_temporal_split_rejects_fractions_leaving_no_test_share() -> None:
    """train_fraction + val_fraction >= 1 would leave zero rows for test - must fail loudly."""
    timestamps = pd.date_range("2020-01-01", periods=10, freq="1h")
    df = pd.DataFrame({TIMESTAMP_COLUMN: timestamps, TARGET_COLUMN: range(10)})

    with pytest.raises(ValueError, match="positive test share"):
        temporal_split(df, train_fraction=0.9, val_fraction=0.2)


# ---------------------------------------------------------------------------
# 5. Train-only scaler
# ---------------------------------------------------------------------------


def test_scale_features_fits_train_only() -> None:
    """Scaler statistics come from train rows only.

    Wildly different val/test values must never shift them.
    """
    rng = np.random.default_rng(0)
    feature_columns = ["a", "b"]
    train_df = pd.DataFrame({"a": rng.normal(0, 1, 50), "b": rng.normal(5, 2, 50)})
    val_df = pd.DataFrame({"a": rng.normal(1000, 1, 20), "b": rng.normal(5000, 2, 20)})
    test_df = pd.DataFrame({"a": rng.normal(-1000, 1, 20), "b": rng.normal(-5000, 2, 20)})

    train_scaled, val_scaled, test_scaled, scaler = scale_features(
        train_df, val_df, test_df, feature_columns
    )

    assert scaler.mean_ == pytest.approx(train_df[feature_columns].mean().to_numpy(), abs=1e-6)
    assert scaler.n_samples_seen_ == len(train_df)
    assert train_scaled["a"].mean() == pytest.approx(0.0, abs=1e-4)
    assert train_scaled["a"].std(ddof=0) == pytest.approx(1.0, abs=1e-4)
    # val/test are shifted by +-1000/5000 relative to train: if the scaler had
    # leaked their statistics into its fit, this would collapse toward 0.
    assert abs(val_scaled["a"].mean()) > 100
    assert abs(test_scaled["a"].mean()) > 100


def test_scale_features_output_dtype_is_float32() -> None:
    """Scaled columns are float32, matching the tensor contract downstream."""
    train_df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    val_df = pd.DataFrame({"a": [1.5]})
    test_df = pd.DataFrame({"a": [2.5]})

    train_scaled, _, _, _ = scale_features(train_df, val_df, test_df, ["a"])

    assert train_scaled["a"].dtype == np.float32


# ---------------------------------------------------------------------------
# 6. Target never imputed
# ---------------------------------------------------------------------------


def test_impute_missing_never_fills_target() -> None:
    """T (degC) keeps every NaN as-is.

    A parallel input feature with the same gap pattern gets forward-filled.
    """
    timestamps = pd.date_range("2020-01-01", periods=10, freq="1h")
    target_values = [1.0, np.nan, np.nan, 4.0, 5.0, np.nan, 7.0, 8.0, 9.0, 10.0]
    df = make_toy_hourly_df(
        timestamps, **{TARGET_COLUMN: target_values, "p (mbar)": list(target_values)}
    )
    empty = df.iloc[0:0].copy()

    train_df, _, _, report = impute_missing(df, empty, empty, forward_fill_limit_hours=3)

    assert np.array_equal(
        train_df[TARGET_COLUMN].to_numpy(), np.array(target_values), equal_nan=True
    ), "target must be returned completely unchanged, NaNs included"
    assert train_df["p (mbar)"].isna().sum() == 0, (
        "the same NaN pattern on an input feature must be closed"
    )
    assert train_df["p (mbar)"].tolist() == [1.0, 1.0, 1.0, 4.0, 5.0, 5.0, 7.0, 8.0, 9.0, 10.0]
    assert report["target_column_imputed"] is False


def test_impute_missing_leaves_long_gap_unfilled_beyond_limit() -> None:
    """A gap longer than forward_fill_limit_hours is only partially closed; the rest stays NaN."""
    timestamps = pd.date_range("2020-01-01", periods=8, freq="1h")
    input_values = [1.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 8.0]  # 6-hour gap, limit=3
    df = make_toy_hourly_df(timestamps, **{"p (mbar)": input_values})
    empty = df.iloc[0:0].copy()

    train_df, _, _, _ = impute_missing(df, empty, empty, forward_fill_limit_hours=3)

    assert train_df["p (mbar)"].isna().sum() == 3
    assert train_df["p (mbar)"].tolist()[1:4] == [1.0, 1.0, 1.0]
    assert train_df["p (mbar)"].iloc[4:7].isna().all()
    assert train_df["p (mbar)"].iloc[7] == 8.0


def test_impute_missing_never_fills_backward() -> None:
    """A gap at the very start of a split (no prior value) stays NaN.

    It is never filled from the future.
    """
    timestamps = pd.date_range("2020-01-01", periods=4, freq="1h")
    input_values = [np.nan, np.nan, 3.0, 4.0]
    df = make_toy_hourly_df(timestamps, **{"p (mbar)": input_values})
    empty = df.iloc[0:0].copy()

    train_df, _, _, _ = impute_missing(df, empty, empty, forward_fill_limit_hours=3)

    assert train_df["p (mbar)"].iloc[0:2].isna().all(), (
        "no earlier value exists to forward-fill from"
    )


# ---------------------------------------------------------------------------
# 10. Inverse transform target
# ---------------------------------------------------------------------------


def test_inverse_transform_target_round_trips_scale_features() -> None:
    """inverse_transform_target undoes scale_features' transform exactly, for the target alone."""
    rng = np.random.default_rng(0)
    feature_columns = ["a", TARGET_COLUMN, "c"]
    target_index = feature_columns.index(TARGET_COLUMN)
    train_df = pd.DataFrame(
        {
            "a": rng.normal(size=40),
            TARGET_COLUMN: rng.normal(loc=15.0, scale=8.0, size=40),
            "c": rng.normal(size=40),
        }
    )
    val_df = train_df.iloc[:5].copy()
    test_df = train_df.iloc[:5].copy()

    train_scaled, _, _, scaler = scale_features(train_df, val_df, test_df, feature_columns)

    recovered = inverse_transform_target(
        train_scaled[TARGET_COLUMN].to_numpy(dtype="float64"), scaler, target_index=target_index
    )

    assert recovered == pytest.approx(train_df[TARGET_COLUMN].to_numpy(), abs=1e-3)


def test_inverse_transform_target_matches_sklearn_inverse_transform() -> None:
    """Cross-checked against StandardScaler.inverse_transform on a full-width row.

    Not just re-derived math.
    """
    scaler = StandardScaler().fit(np.array([[0.0, 10.0], [2.0, 30.0], [4.0, 50.0]]))
    scaled_target_only = np.array([-1.0, 0.0, 1.0])

    manual = inverse_transform_target(scaled_target_only, scaler, target_index=1)

    full_width = np.zeros((3, 2))
    full_width[:, 1] = scaled_target_only
    sklearn_result = scaler.inverse_transform(full_width)[:, 1]

    assert manual == pytest.approx(sklearn_result)


def test_inverse_transform_target_default_index_is_target_index() -> None:
    """The default target_index argument matches the module's own TARGET_INDEX constant."""
    scaler = StandardScaler()
    scaler.mean_ = np.zeros(32)
    scaler.scale_ = np.ones(32)
    scaler.mean_[TARGET_INDEX] = 5.0
    scaler.scale_[TARGET_INDEX] = 2.0

    result = inverse_transform_target(np.array([1.0]), scaler)

    assert result == pytest.approx([7.0])
