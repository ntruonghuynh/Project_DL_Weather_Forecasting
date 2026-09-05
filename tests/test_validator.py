"""Executable tests for the raw-data quality validator (src.data.validator)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.validator import (
    _CONFIG,
    EXPECTED_COLUMNS,
    EXPECTED_FREQUENCY_MINUTES,
    TARGET_COLUMN,
    TIMESTAMP_COLUMN,
    TIMESTAMP_FORMAT,
    _assign_split,
    _build_gap_table,
    _classify_gap_severity,
    _split_boundaries,
    generate_data_quality_report,
    validate_csv,
)

SHORT_THRESHOLD = _CONFIG["missing"]["short_gap_threshold_hours"]
LONG_THRESHOLD = _CONFIG["missing"]["long_gap_threshold_hours"]
SENTINEL_COLUMN = _CONFIG["missing"]["sentinel_columns"][0]
SENTINEL_VALUE = _CONFIG["missing"]["sentinel_values"][0]


def make_raw_df(
    n_rows: int, start: str = "2020-01-01", freq_minutes: int | None = None
) -> pd.DataFrame:
    """Build a synthetic, schema-valid raw-format DataFrame (no NaN/sentinel/dupes)."""
    freq_minutes = freq_minutes or EXPECTED_FREQUENCY_MINUTES
    timestamps = pd.date_range(start=start, periods=n_rows, freq=f"{freq_minutes}min")
    rng = np.random.default_rng(0)
    data = {TIMESTAMP_COLUMN: timestamps.strftime(TIMESTAMP_FORMAT)}
    for column in EXPECTED_COLUMNS:
        if column == TIMESTAMP_COLUMN:
            continue
        data[column] = rng.normal(loc=10.0, scale=1.0, size=n_rows)
    return pd.DataFrame(data, columns=EXPECTED_COLUMNS)


def write_csv(tmp_path: Path, df: pd.DataFrame, name: str = "raw.csv") -> Path:
    """Write a DataFrame to a temp CSV and return its path."""
    path = tmp_path / name
    df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# 1. Schema validation
# ---------------------------------------------------------------------------


def test_valid_csv_passes_all_checks(tmp_path: Path) -> None:
    """A clean, config-conformant CSV passes schema/timestamp/missing checks."""
    report = validate_csv(write_csv(tmp_path, make_raw_df(50)))

    assert report["schema_valid"] is True
    assert report["target_column_exists"] is True
    assert report["timestamp"]["parse_errors"] == 0
    assert report["timestamp"]["monotonic_in_file"] is True
    assert report["timestamp"]["duplicate_rows_after_first"] == 0
    assert report["timestamp"]["gap_count"] == 0
    assert report["total_missing_values"] == 0
    assert report["validation_status"] == "PASS"


def test_missing_expected_column_detected(tmp_path: Path) -> None:
    """A dropped expected column is reported and fails the schema check."""
    df = make_raw_df(20).drop(columns=["wd (deg)"])
    report = validate_csv(write_csv(tmp_path, df))

    assert "wd (deg)" in report["missing_columns"]
    assert report["schema_valid"] is False


def test_unexpected_column_detected(tmp_path: Path) -> None:
    """An extra, unexpected column is flagged without hiding the rest of the schema."""
    df = make_raw_df(20)
    df["extra_sensor"] = 1.0
    report = validate_csv(write_csv(tmp_path, df))

    assert "extra_sensor" in report["unexpected_columns"]


def test_target_column_missing_is_flagged(tmp_path: Path) -> None:
    """target_column_exists turns False when the configured target column is absent."""
    df = make_raw_df(20).drop(columns=[TARGET_COLUMN])
    report = validate_csv(write_csv(tmp_path, df))

    assert report["target_column_exists"] is False


# ---------------------------------------------------------------------------
# 2. Timestamp validation
# ---------------------------------------------------------------------------


def test_timestamp_parse_errors_are_counted(tmp_path: Path) -> None:
    """Unparseable timestamp strings are coerced to NaT and counted, not silently dropped."""
    df = make_raw_df(20)
    df.loc[[3, 7], TIMESTAMP_COLUMN] = "not-a-timestamp"
    report = validate_csv(write_csv(tmp_path, df))

    assert report["timestamp"]["parse_errors"] == 2


def test_non_monotonic_timestamps_detected(tmp_path: Path) -> None:
    """Reversing row order is caught as non-monotonic even though the file still parses."""
    df = make_raw_df(20).iloc[::-1].reset_index(drop=True)
    report = validate_csv(write_csv(tmp_path, df))

    assert report["timestamp"]["monotonic_in_file"] is False


def test_duplicate_timestamps_are_counted(tmp_path: Path) -> None:
    """A repeated timestamp value is counted once, regardless of where it appears in the file."""
    df = make_raw_df(20)
    df = pd.concat([df, df.iloc[[5]]], ignore_index=True)
    report = validate_csv(write_csv(tmp_path, df))

    assert report["timestamp"]["duplicate_rows_after_first"] == 1


def test_frequency_gap_detected(tmp_path: Path) -> None:
    """Removing one row opens exactly one gap, sized at twice the expected interval."""
    df = make_raw_df(20).drop(index=10).reset_index(drop=True)
    report = validate_csv(write_csv(tmp_path, df))

    assert report["timestamp"]["gap_count"] == 1
    assert report["timestamp"]["gaps"][0]["interval_minutes"] == 2 * EXPECTED_FREQUENCY_MINUTES


def test_missing_csv_raises_file_not_found() -> None:
    """A nonexistent source path fails loudly instead of returning an empty report."""
    with pytest.raises(FileNotFoundError):
        validate_csv(Path("does/not/exist.csv"))


# ---------------------------------------------------------------------------
# 3. Missing analysis
# ---------------------------------------------------------------------------


def test_missing_summary_counts_rate_and_affected_columns(tmp_path: Path) -> None:
    """missing_summary reports the exact count, rate, and affected-column list per column."""
    df = make_raw_df(50)
    df.loc[[1, 2, 3], "p (mbar)"] = np.nan
    report = generate_data_quality_report(write_csv(tmp_path, df))

    column_block = next(
        row for row in report["missing_summary"]["by_column"] if row["column"] == "p (mbar)"
    )
    assert column_block["missing_count"] == 3
    assert column_block["missing_rate"] == pytest.approx(3 / 50)
    assert "p (mbar)" in report["missing_summary"]["columns_affected"]
    assert report["missing_summary"]["total_missing_cells"] == 3


# ---------------------------------------------------------------------------
# 4. Sentinel detection
# ---------------------------------------------------------------------------


def test_sentinel_values_are_counted_and_located(tmp_path: Path) -> None:
    """Sentinel codes from configs/data.yaml are counted and their row positions recorded."""
    df = make_raw_df(30)
    sentinel_rows = [2, 15]
    df.loc[sentinel_rows, SENTINEL_COLUMN] = SENTINEL_VALUE
    report = generate_data_quality_report(write_csv(tmp_path, df))

    column_block = next(
        row for row in report["sentinel_summary"]["by_column"] if row["column"] == SENTINEL_COLUMN
    )
    assert column_block["sentinel_count"] == 2
    assert {p["row_index"] for p in column_block["positions"]} == set(sentinel_rows)
    assert report["sentinel_summary"]["total_sentinel_count"] == 2


def test_sentinel_column_missing_from_dataframe_is_reported(tmp_path: Path) -> None:
    """A configured sentinel column absent from the CSV is flagged, not silently skipped."""
    custom_missing = {**_CONFIG["missing"], "sentinel_columns": ["not_a_real_column"]}
    custom_cfg = {**_CONFIG, "missing": custom_missing}
    df = make_raw_df(10)
    report = generate_data_quality_report(write_csv(tmp_path, df), config=custom_cfg)

    assert report["sentinel_summary"]["by_column"][0]["status"] == "COLUMN_NOT_FOUND"


# ---------------------------------------------------------------------------
# 5. Gap detection (severity classification + split assignment)
# ---------------------------------------------------------------------------


def test_gap_severity_classification_boundaries() -> None:
    """SHORT <= short threshold; LONG between short and long threshold; CRITICAL beyond it."""
    assert _classify_gap_severity(SHORT_THRESHOLD, SHORT_THRESHOLD, LONG_THRESHOLD) == "SHORT"
    midpoint = (SHORT_THRESHOLD + LONG_THRESHOLD) / 2
    assert _classify_gap_severity(midpoint, SHORT_THRESHOLD, LONG_THRESHOLD) == "LONG"
    assert _classify_gap_severity(LONG_THRESHOLD, SHORT_THRESHOLD, LONG_THRESHOLD) == "LONG"
    assert _classify_gap_severity(LONG_THRESHOLD + 1, SHORT_THRESHOLD, LONG_THRESHOLD) == "CRITICAL"


def test_gap_table_has_required_columns_and_severities() -> None:
    """_build_gap_table emits start/end/missing_hours/severity/split for each gap."""
    durations_hours = {
        "SHORT": SHORT_THRESHOLD / 2,
        "LONG": (SHORT_THRESHOLD + LONG_THRESHOLD) / 2,
        "CRITICAL": LONG_THRESHOLD + 1,
    }
    gap_records = []
    cursor = pd.Timestamp("2020-01-01")
    for hours in durations_hours.values():
        end = cursor + pd.Timedelta(hours=hours)
        gap_records.append(
            {"after": cursor.isoformat(), "before": end.isoformat(), "interval_minutes": hours * 60}
        )
        cursor = end + pd.Timedelta(hours=1)

    split_boundaries = [
        {"split": "train", "start": "2020-01-01T00:00:00", "end": "2100-01-01T00:00:00"}
    ]
    rows = _build_gap_table(gap_records, SHORT_THRESHOLD, LONG_THRESHOLD, split_boundaries)

    assert [row["severity"] for row in rows] == ["SHORT", "LONG", "CRITICAL"]
    for row, expected_hours in zip(rows, durations_hours.values()):
        assert set(row.keys()) == {"start", "end", "missing_hours", "severity", "split"}
        assert row["missing_hours"] == pytest.approx(expected_hours)
        assert row["split"] == "train"


def test_split_boundaries_partition_the_full_date_range() -> None:
    """train/validation/test boundaries tile [date_min, date_max] with no gap, by config ratios."""
    date_min = pd.Timestamp("2020-01-01")
    date_max = pd.Timestamp("2020-01-11")
    boundaries = _split_boundaries(date_min, date_max, _CONFIG["split"])

    assert [b["split"] for b in boundaries] == ["train", "validation", "test"]
    assert pd.Timestamp(boundaries[0]["start"]) == date_min
    assert pd.Timestamp(boundaries[-1]["end"]) == date_max
    for previous, current in zip(boundaries, boundaries[1:]):
        assert pd.Timestamp(previous["end"]) == pd.Timestamp(current["start"])


def test_assign_split_maps_timestamp_to_its_segment() -> None:
    """A timestamp resolves to whichever boundary segment actually contains it."""
    boundaries = _split_boundaries(
        pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-11"), _CONFIG["split"]
    )

    train_point = pd.Timestamp(boundaries[0]["start"]) + (
        pd.Timestamp(boundaries[0]["end"]) - pd.Timestamp(boundaries[0]["start"])
    ) / 2
    validation_point = pd.Timestamp(boundaries[1]["start"]) + (
        pd.Timestamp(boundaries[1]["end"]) - pd.Timestamp(boundaries[1]["start"])
    ) / 2
    test_point = pd.Timestamp(boundaries[2]["start"]) + (
        pd.Timestamp(boundaries[2]["end"]) - pd.Timestamp(boundaries[2]["start"])
    ) / 2

    assert _assign_split(train_point, boundaries) == "train"
    assert _assign_split(validation_point, boundaries) == "validation"
    assert _assign_split(test_point, boundaries) == "test"


# ---------------------------------------------------------------------------
# Full report: structure, persistence, and quality_status roll-up
# ---------------------------------------------------------------------------


def test_report_has_all_required_sections(tmp_path: Path) -> None:
    """generate_data_quality_report always emits the five required top-level sections."""
    report = generate_data_quality_report(write_csv(tmp_path, make_raw_df(60)))

    required_keys = (
        "schema",
        "timestamp",
        "missing_summary",
        "sentinel_summary",
        "gap_summary",
        "quality_status",
    )
    for key in required_keys:
        assert key in report
    assert report["quality_status"]["overall_status"] == "PASS"


def test_report_is_written_to_output_path(tmp_path: Path) -> None:
    """output_path receives the exact same JSON the function returns."""
    output_path = tmp_path / "report.json"
    report = generate_data_quality_report(
        write_csv(tmp_path, make_raw_df(30)), output_path=output_path
    )

    on_disk = json.loads(output_path.read_text(encoding="utf-8"))
    assert on_disk["row_count"] == report["row_count"]
    assert on_disk["quality_status"] == report["quality_status"]


def test_schema_failure_sets_overall_status_fail(tmp_path: Path) -> None:
    """Losing the target column fails schema and propagates to overall_status."""
    df = make_raw_df(20).drop(columns=[TARGET_COLUMN])
    report = generate_data_quality_report(write_csv(tmp_path, df))

    assert report["schema"]["status"] == "FAIL"
    assert report["quality_status"]["overall_status"] == "FAIL"


def test_missing_values_trigger_review_not_fail(tmp_path: Path) -> None:
    """Missing cells alone downgrade to REVIEW_REQUIRED, not a hard FAIL."""
    df = make_raw_df(30)
    df.loc[0, "p (mbar)"] = np.nan
    report = generate_data_quality_report(write_csv(tmp_path, df))

    assert report["quality_status"]["missing_status"] == "REVIEW"
    assert report["quality_status"]["overall_status"] == "REVIEW_REQUIRED"
