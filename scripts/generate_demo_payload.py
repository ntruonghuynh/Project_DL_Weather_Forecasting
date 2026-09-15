"""Generate a verified 168-hour sample payload for Streamlit and API demonstrations."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import numpy as np


def generate_payload(
    schema_path: Path,
    scaler_path: Path,
    output_path: Path,
    *,
    start_time: datetime | None = None,
    seed: int = 42,
) -> Path:
    """Generate a valid 168-hour input payload with 72-hour future actuals."""
    schema_path = Path(schema_path)
    scaler_path = Path(scaler_path)
    output_path = Path(output_path)

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    scaler = joblib.load(scaler_path)
    feature_names = schema["ordered_features"]
    target_idx = schema["target"]["index"]
    input_length = schema["window_contract"]["input_len_hours"]
    horizon = schema["window_contract"]["horizon_hours"]

    if start_time is None:
        start_time = datetime(2024, 5, 1, 0, 0, tzinfo=timezone.utc)

    np.random.seed(seed)
    total_hours = input_length + horizon

    # Base realistic time series using mean values + smooth diurnal temperature cycle
    means = scaler.mean_
    scales = scaler.scale_

    # Create synthetic series anchored near the dataset statistics
    series = np.zeros((total_hours, len(feature_names)), dtype=np.float32)
    timestamps = [
        (start_time + timedelta(hours=h)).isoformat().replace("+00:00", "Z")
        for h in range(total_hours)
    ]

    for h in range(total_hours):
        hour_of_day = (start_time + timedelta(hours=h)).hour
        day_of_year = (start_time + timedelta(hours=h)).timetuple().tm_yday

        # Cyclic features
        hour_sin = math.sin(2 * math.pi * hour_of_day / 24.0)
        hour_cos = math.cos(2 * math.pi * hour_of_day / 24.0)
        doy_sin = math.sin(2 * math.pi * day_of_year / 365.25)
        doy_cos = math.cos(2 * math.pi * day_of_year / 365.25)

        # Diurnal temperature cycle: peak around 15:00, low around 05:00
        temp_variation = 5.0 * math.sin(2 * math.pi * (hour_of_day - 9) / 24.0)
        noise = np.random.normal(0, 0.5)
        temp = float(means[target_idx] + temp_variation + noise)

        # Fill observation row
        for f_idx, f_name in enumerate(feature_names):
            if f_name == "T (degC)":
                series[h, f_idx] = temp
            elif f_name == "hour_sin":
                series[h, f_idx] = hour_sin
            elif f_name == "hour_cos":
                series[h, f_idx] = hour_cos
            elif f_name == "dayofyear_sin":
                series[h, f_idx] = doy_sin
            elif f_name == "dayofyear_cos":
                series[h, f_idx] = doy_cos
            elif f_name == "Tpot (K)":
                series[h, f_idx] = temp + 273.15
            elif f_name == "Tdew (degC)":
                series[h, f_idx] = temp - 4.0
            else:
                # Add mild perturbation around the mean within 0.2 of standard deviation
                jitter = np.random.normal(0, 0.1) * scales[f_idx]
                series[h, f_idx] = float(means[f_idx] + jitter)

    input_timestamps = timestamps[:input_length]
    future_timestamps = timestamps[input_length:]
    future_actual = series[input_length:, target_idx].tolist()

    observations = [
        {name: float(series[step, idx]) for idx, name in enumerate(feature_names)}
        for step in range(input_length)
    ]

    payload = {
        "timestamps": input_timestamps,
        "observations": observations,
        "future_timestamps": future_timestamps,
        "future_actual": future_actual,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"Generated sample payload at {output_path} "
        f"({len(observations)} hours input, {len(future_actual)} hours future)"
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate demo payload JSON for Streamlit.")
    parser.add_argument(
        "--schema",
        type=Path,
        default=Path("artifacts/preprocessing/feature_schema.json"),
        help="Path to feature schema.",
    )
    parser.add_argument(
        "--scaler",
        type=Path,
        default=Path("artifacts/preprocessing/scaler.joblib"),
        help="Path to fitted scaler.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/demo/sample_168h.json"),
        help="Destination JSON file path.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    generate_payload(args.schema, args.scaler, args.output, seed=args.seed)


if __name__ == "__main__":
    main()
