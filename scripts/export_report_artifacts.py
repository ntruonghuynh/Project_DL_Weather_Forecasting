"""Repackage scripts/evaluate.py output into the flat CSV pair used in reports.

Takes the metrics.json that scripts/evaluate.py already produced (real numbers,
not recomputed here) plus the raw prediction NPZ from
scripts/generate_predictions.py, and writes:
  - <metrics-csv-out>: Metric,Value (MAE/RMSE/MSE, degC / degC^2)
  - <prediction-csv-out>: timestamp,actual_temperature,predicted_temperature
    for one representative 72h window (sample_index=0), inverse-scaled to degC

No metric is recomputed here; only reformatted for the report artifact layout.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.error_analysis import build_error_frame  # noqa: E402
from src.evaluation.metrics import inverse_scale_target  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-json", type=Path, required=True)
    parser.add_argument("--predictions-npz", type=Path, required=True)
    parser.add_argument("--scaler", type=Path, required=True)
    parser.add_argument("--target-index", type=int, required=True)
    parser.add_argument("--metrics-csv-out", type=Path, required=True)
    parser.add_argument("--prediction-csv-out", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    metrics_payload = json.loads(args.metrics_json.read_text(encoding="utf-8"))
    overall = metrics_payload["overall"]

    metrics_table = pd.DataFrame({
        "Metric": ["MAE (°C)", "RMSE (°C)", "MSE (°C²)"],
        "Value": [overall["mae"], overall["rmse"], overall["mse"]],
    })
    args.metrics_csv_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_table.to_csv(args.metrics_csv_out, index=False)

    arrays = np.load(args.predictions_npz, allow_pickle=False)
    scaler = joblib.load(args.scaler)
    y_true = inverse_scale_target(arrays["y_true"][..., 0], scaler, args.target_index)
    y_pred = inverse_scale_target(arrays["y_pred"][..., 0], scaler, args.target_index)

    frame = build_error_frame(
        y_true, y_pred, target_timestamps=arrays["target_timestamps"],
        run_id=metrics_payload["run_id"], model_name=metrics_payload["model_name"],
        split=metrics_payload["split"],
    )
    first_sample = frame["sample_index"].iloc[0]
    representative = (
        frame.loc[frame["sample_index"] == first_sample, ["timestamp", "y_true", "y_pred"]]
        .rename(columns={"y_true": "actual_temperature", "y_pred": "predicted_temperature"})
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    args.prediction_csv_out.parent.mkdir(parents=True, exist_ok=True)
    representative.to_csv(args.prediction_csv_out, index=False)

    print(f"Saved: {args.metrics_csv_out}")
    print(f"Saved: {args.prediction_csv_out} ({len(representative)} rows)")


if __name__ == "__main__":
    main()
