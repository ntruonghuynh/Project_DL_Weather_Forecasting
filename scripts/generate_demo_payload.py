"""Export a real, leakage-safe 168h/72h demo payload from the Jena source CSV."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import (  # noqa: E402
    add_missing_indicators,
    add_time_features,
    clean_timestamps,
    handle_sentinels,
    impute_missing,
    load_raw_data,
    resample_hourly,
    temporal_split,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _select_real_window(
    frame: pd.DataFrame,
    *,
    feature_names: list[str],
    target_name: str,
    timestamp_column: str,
    input_length: int,
    horizon: int,
    start_index: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Select one contiguous finite window; never synthesize or interpolate values."""
    required = [timestamp_column, *feature_names]
    missing = [name for name in required if name not in frame]
    if missing:
        raise ValueError(f"source frame is missing columns: {missing}")
    total = input_length + horizon
    if len(frame) < total:
        raise ValueError(f"source split needs at least {total} hourly rows")

    candidates = [start_index] if start_index is not None else range(len(frame) - total + 1)
    for start in candidates:
        if start is None or start < 0 or start + total > len(frame):
            raise ValueError("start_index is outside the selected split")
        window = frame.iloc[start : start + total]
        timestamps = pd.to_datetime(window[timestamp_column], errors="raise")
        cadence = timestamps.diff().dropna().dt.total_seconds().to_numpy()
        values = window[feature_names].to_numpy(dtype=np.float64)
        future_target = window.iloc[input_length:][target_name].to_numpy(dtype=np.float64)
        if (
            np.isfinite(values).all()
            and np.isfinite(future_target).all()
            and np.all(cadence == 3600)
        ):
            return window.iloc[:input_length], window.iloc[input_length:], int(start)
    requested = f" at start_index={start_index}" if start_index is not None else ""
    raise ValueError(f"no complete finite {input_length}h/{horizon}h real window found{requested}")


def generate_payload(
    raw_csv_path: Path,
    schema_path: Path,
    output_path: Path,
    *,
    split: str = "validation",
    start_index: int | None = None,
) -> Path:
    """Run the canonical pre-scaling pipeline and export one real demo window."""
    if split not in {"train", "validation"}:
        raise ValueError("demo export permits only train or validation; test remains sealed")
    raw_csv_path = Path(raw_csv_path)
    schema_path = Path(schema_path)
    output_path = Path(output_path)
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    feature_names = list(schema["ordered_features"])
    target_name = str(schema["target"]["name"])
    timestamp_column = str(schema["timestamp"]["column"])
    input_length = int(schema["window_contract"]["input_len_hours"])
    horizon = int(schema["window_contract"]["horizon_hours"])

    raw = load_raw_data(raw_csv_path)
    cleaned, _ = clean_timestamps(raw)
    cleaned, _ = handle_sentinels(cleaned)
    hourly = add_missing_indicators(add_time_features(resample_hourly(cleaned)))
    train, validation, test, _ = temporal_split(hourly)
    train, validation, _, _ = impute_missing(train, validation, test)
    selected_split = train if split == "train" else validation
    history, future, selected_index = _select_real_window(
        selected_split,
        feature_names=feature_names,
        target_name=target_name,
        timestamp_column=timestamp_column,
        input_length=input_length,
        horizon=horizon,
        start_index=start_index,
    )

    def _iso(values: pd.Series) -> list[str]:
        return [pd.Timestamp(value).isoformat() for value in values]

    payload = {
        "timestamps": _iso(history[timestamp_column]),
        "observations": [
            {name: float(row[name]) for name in feature_names}
            for _, row in history.iterrows()
        ],
        "future_timestamps": _iso(future[timestamp_column]),
        "future_actual": future[target_name].astype(float).tolist(),
        "provenance": {
            "source_file": raw_csv_path.name,
            "source_sha256": _sha256(raw_csv_path),
            "schema_sha256": _sha256(schema_path),
            "split": split,
            "start_index_within_split": selected_index,
            "input_start": pd.Timestamp(history[timestamp_column].iloc[0]).isoformat(),
            "forecast_end": pd.Timestamp(future[timestamp_column].iloc[-1]).isoformat(),
            "synthetic": False,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Exported real {split} payload to {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-csv",
        type=Path,
        default=Path("data/raw/jena_climate_2009_2016.csv"),
        help="Verified, unmodified Jena source CSV.",
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=Path("artifacts/preprocessing/feature_schema.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument("--start-index", type=int)
    args = parser.parse_args()
    generate_payload(
        args.raw_csv,
        args.schema,
        args.output,
        split=args.split,
        start_index=args.start_index,
    )


if __name__ == "__main__":
    main()
