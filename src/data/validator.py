from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import load_data_config

EXPECTED_COLUMNS = [
    "Date Time",
    "p (mbar)",
    "T (degC)",
    "Tpot (K)",
    "Tdew (degC)",
    "rh (%)",
    "VPmax (mbar)",
    "VPact (mbar)",
    "VPdef (mbar)",
    "sh (g/kg)",
    "H2OC (mmol/mol)",
    "rho (g/m**3)",
    "wv (m/s)",
    "max. wv (m/s)",
    "wd (deg)",
]

_CONFIG = load_data_config()
_TIMESTAMP_CFG = _CONFIG["timestamp"]

# Single Source of Truth: configs/data.yaml. Do not hard-code these values
# elsewhere - import them from this module instead.
TIMESTAMP_COLUMN = _TIMESTAMP_CFG["column"]
TARGET_COLUMN = _CONFIG["target"]["column"]
EXPECTED_FREQUENCY_MINUTES = _TIMESTAMP_CFG["raw_frequency_minutes"]
TIMESTAMP_FORMAT = _TIMESTAMP_CFG["format"]


def validate_csv(csv_path: Path) -> dict:
    """Đọc và kiểm tra dữ liệu CSV, không ghi đè hay chỉnh sửa file nguồn."""

    if not csv_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {csv_path}")

    # Chỉ đọc dữ liệu
    df = pd.read_csv(csv_path)

    report = {
        "source_file": str(csv_path),
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "columns": list(df.columns),
        "expected_columns": EXPECTED_COLUMNS,
    }

    # ---------------------------------------------------------
    # 1. Kiểm tra tên và số lượng cột
    # ---------------------------------------------------------
    missing_columns = [
        column for column in EXPECTED_COLUMNS
        if column not in df.columns
    ]

    unexpected_columns = [
        column for column in df.columns
        if column not in EXPECTED_COLUMNS
    ]

    report["missing_columns"] = missing_columns
    report["unexpected_columns"] = unexpected_columns
    report["schema_valid"] = (
        list(df.columns) == EXPECTED_COLUMNS
    )
    report["target_column_exists"] = TARGET_COLUMN in df.columns

    # ---------------------------------------------------------
    # 2. Parse timestamp
    # ---------------------------------------------------------
    if TIMESTAMP_COLUMN not in df.columns:
        report["timestamp"] = {
            "parse_errors": None,
            "status": "FAIL_MISSING_TIMESTAMP_COLUMN",
        }
    else:
        timestamps = pd.to_datetime(
            df[TIMESTAMP_COLUMN],
            format=TIMESTAMP_FORMAT,
            errors="coerce",
        )

        parse_errors = int(timestamps.isna().sum())

        report["timestamp"] = {
            "format": TIMESTAMP_FORMAT,
            "parse_errors": parse_errors,
            "date_min": (
                timestamps.min().isoformat()
                if parse_errors < len(timestamps)
                else None
            ),
            "date_max": (
                timestamps.max().isoformat()
                if parse_errors < len(timestamps)
                else None
            ),
            "monotonic_in_file": bool(
                timestamps.dropna().is_monotonic_increasing
            ),
            "duplicate_rows_after_first": int(
                timestamps.dropna().duplicated().sum()
            ),
        }

        # -----------------------------------------------------
        # 3. Kiểm tra frequency và khoảng thời gian bị thiếu
        # -----------------------------------------------------
        unique_timestamps = (
            timestamps.dropna()
            .drop_duplicates()
            .sort_values()
            .reset_index(drop=True)
        )

        differences = unique_timestamps.diff().dropna()

        if len(differences) > 0:
            mode_difference = differences.mode().iloc[0]
            mode_minutes = mode_difference.total_seconds() / 60
        else:
            mode_minutes = None

        gap_records = []

        for index in range(1, len(unique_timestamps)):
            previous_time = unique_timestamps.iloc[index - 1]
            current_time = unique_timestamps.iloc[index]
            difference = current_time - previous_time
            difference_minutes = difference.total_seconds() / 60

            if difference_minutes > EXPECTED_FREQUENCY_MINUTES:
                gap_records.append(
                    {
                        "after": previous_time.isoformat(),
                        "before": current_time.isoformat(),
                        "interval_minutes": int(difference_minutes),
                    }
                )

        report["timestamp"]["expected_frequency_minutes"] = (
            EXPECTED_FREQUENCY_MINUTES
        )
        report["timestamp"]["mode_interval_minutes"] = mode_minutes
        report["timestamp"]["gap_count"] = len(gap_records)
        report["timestamp"]["gaps"] = gap_records

    # ---------------------------------------------------------
    # 4. Kiểm tra giá trị thiếu
    # ---------------------------------------------------------
    missing_by_column = {
        column: int(value)
        for column, value in df.isna().sum().items()
    }

    report["missing_by_column"] = missing_by_column
    report["total_missing_values"] = sum(missing_by_column.values())

    # ---------------------------------------------------------
    # 5. Kiểm tra kiểu số
    # ---------------------------------------------------------
    numeric_columns = [
        column
        for column in EXPECTED_COLUMNS
        if column != TIMESTAMP_COLUMN and column in df.columns
    ]

    numeric_df = df[numeric_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    non_empty_original = df[numeric_columns].notna()

    non_numeric_by_column = {
        column: int(
            ((numeric_df[column].isna()) & non_empty_original[column]).sum()
        )
        for column in numeric_columns
    }

    report["non_numeric_by_column"] = non_numeric_by_column

    # ---------------------------------------------------------
    # 6. Kiểm tra Inf
    # ---------------------------------------------------------
    infinite_by_column = {
        column: int(np.isinf(numeric_df[column].to_numpy()).sum())
        for column in numeric_columns
    }

    report["infinite_by_column"] = infinite_by_column

    # ---------------------------------------------------------
    # 7. Kiểm tra cột hằng
    # ---------------------------------------------------------
    constant_columns = [
        column
        for column in numeric_columns
        if numeric_df[column].nunique(dropna=False) <= 1
    ]

    report["constant_columns"] = constant_columns

    # ---------------------------------------------------------
    # 8. Kết luận tổng quát
    # ---------------------------------------------------------
    timestamp_info = report["timestamp"]

    has_error = (
        not report["schema_valid"]
        or not report["target_column_exists"]
        or timestamp_info.get("parse_errors", 0) > 0
        or timestamp_info.get("duplicate_rows_after_first", 0) > 0
        or timestamp_info.get("gap_count", 0) > 0
        or report["total_missing_values"] > 0
        or sum(report["non_numeric_by_column"].values()) > 0
        or sum(report["infinite_by_column"].values()) > 0
        or len(report["constant_columns"]) > 0
        or not timestamp_info.get("monotonic_in_file", False)
    )

    report["validation_status"] = (
        "REVIEW_REQUIRED" if has_error else "PASS"
    )

    return report


# ---------------------------------------------------------------------------
# Full data-quality report (schema + timestamp + missing + sentinel + gap)
# ---------------------------------------------------------------------------
#
# validate_csv() above answers "is this file structurally usable" with a
# single PASS/REVIEW_REQUIRED verdict. generate_data_quality_report() below
# builds on it to answer the TV2 pre-preprocessing gate: schema status,
# missing summary, sentinel summary, and a gap table with severity/split -
# every field here is a count or a value read directly off the raw CSV, not
# an editorial judgment about whether the data is fit to use.


def _split_boundaries(
    date_min: pd.Timestamp, date_max: pd.Timestamp, split_cfg: dict
) -> list[dict]:
    """Chronological train/validation/test boundaries by elapsed time.

    This mirrors preprocessing.temporal_split's row-fraction split closely
    enough for reporting purposes, but is computed directly on raw
    timestamps (elapsed time, not row count) since this report runs before
    any resampling exists. It is only used to label which split a gap in
    the raw file falls into, not to actually split the data.
    """
    total_span = date_max - date_min
    train_end = date_min + total_span * split_cfg["train_fraction"]
    validation_end = train_end + total_span * split_cfg["validation_fraction"]
    return [
        {"split": "train", "start": date_min.isoformat(), "end": train_end.isoformat()},
        {"split": "validation", "start": train_end.isoformat(), "end": validation_end.isoformat()},
        {"split": "test", "start": validation_end.isoformat(), "end": date_max.isoformat()},
    ]


def _assign_split(timestamp: pd.Timestamp, boundaries: list[dict]) -> str:
    """Which split boundary (from _split_boundaries) a timestamp falls into."""
    for boundary in boundaries:
        if pd.Timestamp(boundary["start"]) <= timestamp <= pd.Timestamp(boundary["end"]):
            return boundary["split"]
    return "unknown"


def _classify_gap_severity(
    missing_hours: float, short_gap_threshold_hours: float, long_gap_threshold_hours: float
) -> str:
    """SHORT <= short threshold; LONG <= long threshold; CRITICAL beyond it."""
    if missing_hours <= short_gap_threshold_hours:
        return "SHORT"
    if missing_hours <= long_gap_threshold_hours:
        return "LONG"
    return "CRITICAL"


def _detect_sentinels(
    df: pd.DataFrame, sentinel_values: list, sentinel_columns: list[str]
) -> dict:
    """Count and locate known invalid-reading sentinel codes (e.g. -9999)."""
    by_column = []
    total_count = 0
    row_count = len(df)

    for column in sentinel_columns:
        if column not in df.columns:
            by_column.append({"column": column, "status": "COLUMN_NOT_FOUND"})
            continue

        series = pd.to_numeric(df[column], errors="coerce")
        is_sentinel = series.isin(sentinel_values)
        count = int(is_sentinel.sum())
        total_count += count

        positions = [
            {
                "row_index": int(row_index),
                TIMESTAMP_COLUMN: str(df.loc[row_index, TIMESTAMP_COLUMN]),
            }
            for row_index in df.index[is_sentinel]
        ]

        by_column.append(
            {
                "column": column,
                "sentinel_count": count,
                "sentinel_rate": round(count / row_count, 6) if row_count else 0.0,
                "positions": positions,
            }
        )

    return {
        "sentinel_values": sentinel_values,
        "sentinel_columns": sentinel_columns,
        "total_sentinel_count": total_count,
        "by_column": by_column,
    }


def _build_gap_table(
    gap_records: list[dict],
    short_gap_threshold_hours: float,
    long_gap_threshold_hours: float,
    split_boundaries: list[dict],
) -> list[dict]:
    """|start|end|missing_hours|severity|split| rows from validate_csv's gap_records.

    validate_csv's own "after"/"before" fields hold the last reading before
    the gap and the first reading after it, respectively - relabeled here to
    the more direct start/end the task asks for.
    """
    rows = []
    for gap in gap_records:
        start = pd.Timestamp(gap["after"])
        end = pd.Timestamp(gap["before"])
        missing_hours = round(gap["interval_minutes"] / 60, 4)
        rows.append(
            {
                "start": start.isoformat(),
                "end": end.isoformat(),
                "missing_hours": missing_hours,
                "severity": _classify_gap_severity(
                    missing_hours, short_gap_threshold_hours, long_gap_threshold_hours
                ),
                "split": _assign_split(start, split_boundaries) if split_boundaries else "unknown",
            }
        )
    return rows


def generate_data_quality_report(
    csv_path: Path,
    config: dict | None = None,
    output_path: Path | None = None,
) -> dict:
    """Build the raw-data quality report and optionally write it to disk.

    Reuses validate_csv() for schema/timestamp/missing/dtype checks instead
    of re-deriving them, then adds the sentinel and gap-severity/split
    sections that validate_csv does not cover. Every `*_status` field is a
    mechanical PASS/FAIL/REVIEW derived from a count above it - this
    function reports evidence, it does not decide whether the data is fit
    for preprocessing.
    """
    cfg = config if config is not None else _CONFIG
    csv_path = Path(csv_path)

    report = validate_csv(csv_path)
    df = pd.read_csv(csv_path)

    # --- 1. Schema (columns + datatype + target) ---
    dtype_check = {
        TIMESTAMP_COLUMN: {
            "expected_type": "datetime",
            "parse_errors": report["timestamp"].get("parse_errors"),
            "valid": report["timestamp"].get("parse_errors") == 0,
        }
    }
    for column, non_numeric_count in report["non_numeric_by_column"].items():
        dtype_check[column] = {
            "expected_type": "numeric",
            "non_numeric_count": non_numeric_count,
            "infinite_count": report["infinite_by_column"].get(column, 0),
            "valid": non_numeric_count == 0 and report["infinite_by_column"].get(column, 0) == 0,
        }

    schema_status = (
        "PASS"
        if (
            report["schema_valid"]
            and report["target_column_exists"]
            and all(entry["valid"] for entry in dtype_check.values())
        )
        else "FAIL"
    )

    schema_block = {
        "expected_columns": report["expected_columns"],
        "actual_columns": report["columns"],
        "missing_columns": report["missing_columns"],
        "unexpected_columns": report["unexpected_columns"],
        "columns_match_expected": report["schema_valid"],
        "target_column": TARGET_COLUMN,
        "target_column_exists": report["target_column_exists"],
        "dtype_check": dtype_check,
        "status": schema_status,
    }

    # --- 2. Timestamp (parse / monotonic / duplicate / frequency) ---
    timestamp_block = {k: v for k, v in report["timestamp"].items() if k != "gaps"}

    # --- 3. Missing analysis ---
    total_cells = report["row_count"] * report["column_count"]
    missing_by_column = [
        {
            "column": column,
            "missing_count": count,
            "missing_rate": round(count / report["row_count"], 6) if report["row_count"] else 0.0,
        }
        for column, count in report["missing_by_column"].items()
    ]
    missing_block = {
        "total_missing_cells": report["total_missing_values"],
        "total_cells": total_cells,
        "missing_rate_overall": (
            round(report["total_missing_values"] / total_cells, 6) if total_cells else 0.0
        ),
        "by_column": missing_by_column,
        "columns_affected": [
            row["column"] for row in missing_by_column if row["missing_count"] > 0
        ],
    }

    # --- 4. Sentinel detection ---
    sentinel_block = _detect_sentinels(
        df, cfg["missing"]["sentinel_values"], cfg["missing"]["sentinel_columns"]
    )

    # --- 5. Gap detection ---
    short_gap_threshold_hours = cfg["missing"]["short_gap_threshold_hours"]
    long_gap_threshold_hours = cfg["missing"]["long_gap_threshold_hours"]
    date_min = (
        pd.Timestamp(report["timestamp"]["date_min"])
        if report["timestamp"].get("date_min")
        else None
    )
    date_max = (
        pd.Timestamp(report["timestamp"]["date_max"])
        if report["timestamp"].get("date_max")
        else None
    )
    split_boundaries = (
        _split_boundaries(date_min, date_max, cfg["split"])
        if date_min is not None and date_max is not None
        else []
    )
    gap_rows = _build_gap_table(
        report["timestamp"].get("gaps", []),
        short_gap_threshold_hours,
        long_gap_threshold_hours,
        split_boundaries,
    )
    gap_block = {
        "short_gap_threshold_hours": short_gap_threshold_hours,
        "long_gap_threshold_hours": long_gap_threshold_hours,
        "critical_gap_threshold_hours": long_gap_threshold_hours,
        "split_boundaries": split_boundaries,
        "gap_count": len(gap_rows),
        "gaps": gap_rows,
    }

    # --- Quality status: mechanical PASS/FAIL/REVIEW per section, no narrative ---
    quality_status = {
        "schema_status": schema_status,
        "timestamp_parse_status": "PASS" if timestamp_block.get("parse_errors", 0) == 0 else "FAIL",
        "monotonic_status": "PASS" if timestamp_block.get("monotonic_in_file") else "FAIL",
        "duplicate_status": (
            "PASS" if timestamp_block.get("duplicate_rows_after_first", 0) == 0 else "REVIEW"
        ),
        "missing_status": "PASS" if missing_block["total_missing_cells"] == 0 else "REVIEW",
        "sentinel_status": "PASS" if sentinel_block["total_sentinel_count"] == 0 else "REVIEW",
        "gap_status": "PASS" if gap_block["gap_count"] == 0 else "REVIEW",
    }
    quality_status["overall_status"] = (
        "FAIL"
        if any(status == "FAIL" for status in quality_status.values())
        else "REVIEW_REQUIRED"
        if any(status == "REVIEW" for status in quality_status.values())
        else "PASS"
    )

    full_report = {
        "report_type": "raw_data_quality_report",
        "generated_by": "src.data.validator.generate_data_quality_report",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_file": str(csv_path),
        "config_source": "configs/data.yaml",
        "row_count": report["row_count"],
        "column_count": report["column_count"],
        "schema": schema_block,
        "timestamp": timestamp_block,
        "missing_summary": missing_block,
        "sentinel_summary": sentinel_block,
        "gap_summary": gap_block,
        "quality_status": quality_status,
    }

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(full_report, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    return full_report


def main() -> None:
    """Sinh artifacts/data_quality_report.json và in ra JSON.

    Nếu bỏ qua đối số dòng lệnh thì dùng CSV raw mặc định.
    """

    project_root = Path(__file__).resolve().parents[2]
    default_csv = project_root / "data" / "raw" / "jena_climate_2009_2016.csv"
    default_output = project_root / "artifacts" / "data_quality_report.json"

    if len(sys.argv) > 2:
        print(
            "Cách dùng:\n"
            "  python .\\src\\data\\validator.py [duong_dan_den_file_csv]\n"
            f"Mặc định: {default_csv}\n"
            f"Output:   {default_output}"
        )
        raise SystemExit(2)

    csv_path = Path(sys.argv[1]) if len(sys.argv) == 2 else default_csv

    try:
        report = generate_data_quality_report(csv_path, output_path=default_output)
    except FileNotFoundError as error:
        print(f"Lỗi: {error}", file=sys.stderr)
        raise SystemExit(2) from error

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
