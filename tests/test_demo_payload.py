"""Tests for exporting traceable real-data demo windows without fabricated values."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.generate_demo_payload import _select_real_window, generate_payload


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date Time": pd.date_range("2015-01-01", periods=10, freq="h"),
            "feature": np.arange(10, dtype=float),
            "T (degC)": np.arange(10, dtype=float) + 10,
        }
    )


def test_select_real_window_preserves_observed_values() -> None:
    history, future, start = _select_real_window(
        _frame(), feature_names=["feature", "T (degC)"],
        target_name="T (degC)", timestamp_column="Date Time",
        input_length=5, horizon=2, start_index=2,
    )
    assert start == 2
    assert history["feature"].tolist() == [2, 3, 4, 5, 6]
    assert future["T (degC)"].tolist() == [17, 18]


def test_select_real_window_skips_missing_values() -> None:
    frame = _frame()
    frame.loc[0, "feature"] = np.nan
    _, _, start = _select_real_window(
        frame, feature_names=["feature", "T (degC)"],
        target_name="T (degC)", timestamp_column="Date Time",
        input_length=3, horizon=2, start_index=None,
    )
    assert start == 1


def test_demo_export_refuses_test_split_before_reading_data(tmp_path) -> None:
    with pytest.raises(ValueError, match="test remains sealed"):
        generate_payload(
            tmp_path / "missing.csv", tmp_path / "missing-schema.json",
            tmp_path / "payload.json", split="test",
        )
