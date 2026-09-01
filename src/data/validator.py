from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


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

TIMESTAMP_COLUMN = "Date Time"
TARGET_COLUMN = "T (degC)"
EXPECTED_FREQUENCY_MINUTES = 10


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
            format="%d.%m.%Y %H:%M:%S",
            errors="coerce",
        )

        parse_errors = int(timestamps.isna().sum())

        report["timestamp"] = {
            "format": "%d.%m.%Y %H:%M:%S",
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


def main() -> None:
    """In báo cáo JSON; nếu bỏ qua đối số thì dùng CSV raw mặc định."""

    project_root = Path(__file__).resolve().parents[2]
    default_csv = project_root / "data" / "raw" / "jena_climate_2009_2016.csv"

    if len(sys.argv) > 2:
        print(
            "Cách dùng:\n"
            "  python .\\src\\data\\validator.py [duong_dan_den_file_csv]\n"
            f"Mặc định: {default_csv}"
        )
        raise SystemExit(2)

    csv_path = Path(sys.argv[1]) if len(sys.argv) == 2 else default_csv

    try:
        report = validate_csv(csv_path)
    except FileNotFoundError as error:
        print(f"Lỗi: {error}", file=sys.stderr)
        raise SystemExit(2) from error

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
