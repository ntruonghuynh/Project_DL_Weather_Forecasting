"""Executable tests for the boundary-based chronological split utility (src.data.splits)."""

import numpy as np
import pandas as pd
import pytest

from src.data.splits import SplitBoundaries, build_splits


def make_hourly_timestamps(n: int, start: str = "2020-01-01") -> pd.Series:
    return pd.Series(pd.date_range(start, periods=n, freq="1h"))


def test_build_splits_is_chronological_and_non_overlapping() -> None:
    """Every index lands in exactly one split, and splits stay in time order."""
    timestamps = make_hourly_timestamps(100)
    boundaries = SplitBoundaries(
        train_end="2020-01-03 00:00:00",  # 48 hourly rows: index 0..47
        validation_end="2020-01-04 00:00:00",  # next 24 rows: index 48..71
        test_end="2020-01-05 12:00:00",  # next 36 rows: index 72..99 (capped by data)
    )

    result = build_splits(timestamps, boundaries)

    assert list(result["train"]) == list(range(0, 48))
    assert list(result["validation"]) == list(range(48, 72))
    assert list(result["test"]) == list(range(72, 100))

    all_indices = np.concatenate([result["train"], result["validation"], result["test"]])
    assert len(all_indices) == len(set(all_indices.tolist())), "no index assigned to two splits"
    assert timestamps.iloc[result["train"]].max() < timestamps.iloc[result["validation"]].min()
    assert timestamps.iloc[result["validation"]].max() < timestamps.iloc[result["test"]].min()


def test_build_splits_rejects_non_increasing_boundaries() -> None:
    """Boundaries that are not strictly increasing must fail loudly, not silently misassign rows."""
    timestamps = make_hourly_timestamps(10)
    boundaries = SplitBoundaries(
        train_end="2020-01-02 00:00:00",
        validation_end="2020-01-01 12:00:00",  # before train_end - invalid
        test_end="2020-01-03 00:00:00",
    )

    with pytest.raises(ValueError, match="strictly increasing"):
        build_splits(timestamps, boundaries)


def test_build_splits_excludes_timestamps_outside_all_boundaries() -> None:
    """Timestamps at/after test_end belong to no split - they are dropped, never mis-bucketed."""
    timestamps = make_hourly_timestamps(10)
    boundaries = SplitBoundaries(
        train_end="2020-01-01 03:00:00",
        validation_end="2020-01-01 06:00:00",
        test_end="2020-01-01 08:00:00",
    )

    result = build_splits(timestamps, boundaries)

    covered = len(result["train"]) + len(result["validation"]) + len(result["test"])
    assert covered < len(timestamps), "rows at/after test_end must be excluded, not force-assigned"
