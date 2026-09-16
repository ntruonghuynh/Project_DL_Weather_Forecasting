"""Evaluate immutable prediction arrays without training or target leakage."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.error_analysis import analyze_errors, build_error_frame  # noqa: E402
from src.evaluation.metrics import (  # noqa: E402
    evaluate_forecasts,
    inverse_scale_target,
    persistence_predictions,
)
from src.evaluation.registry import authorize_final_test  # noqa: E402
from src.evaluation.visualization import (  # noqa: E402
    register_figure,
    save_forecast_figure,
    save_residual_figure,
)

REQUIRED_METADATA = {
    "run_id", "model_name", "model_version", "split", "population_id",
    "checkpoint", "resolved_config", "schema_hash", "scaler_sha256",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True, help="NPZ with y_true/y_pred")
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scale", choices=("standardized", "degC"), required=True)
    parser.add_argument("--scaler", type=Path)
    parser.add_argument("--target-index", type=int)
    parser.add_argument("--selection-manifest", type=Path)
    parser.add_argument("--final-test-audit", type=Path)
    parser.add_argument("--top-n", type=int, default=20)
    return parser


def _load_metadata(path: Path) -> dict:
    metadata = json.loads(path.read_text(encoding="utf-8"))
    missing = REQUIRED_METADATA - set(metadata)
    if missing:
        raise ValueError(f"evaluation metadata is missing fields: {sorted(missing)}")
    if metadata["split"] not in {"validation", "test"}:
        raise ValueError("evaluation split must be validation or test")
    if any(not metadata[key] for key in REQUIRED_METADATA):
        raise ValueError("evaluation provenance fields must not be empty")
    return metadata


def main() -> None:
    """Evaluate an existing prediction artifact and write traceable outputs."""
    args = _parser().parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty evaluation directory: {args.output_dir}"
        )
    metadata = _load_metadata(args.metadata)
    audit: dict | None = None
    if metadata["split"] == "test":
        if args.selection_manifest is None or args.final_test_audit is None:
            raise ValueError("test evaluation requires --selection-manifest and --final-test-audit")
        audit = authorize_final_test(
            args.selection_manifest, metadata["run_id"], args.final_test_audit
        )

    arrays = np.load(args.predictions, allow_pickle=False)
    if "y_true" not in arrays or "y_pred" not in arrays:
        raise ValueError("prediction NPZ must contain y_true and y_pred")
    scaler = None
    if args.scale == "standardized":
        if args.scaler is None or args.target_index is None:
            raise ValueError("standardized evaluation requires --scaler and --target-index")
        scaler = joblib.load(args.scaler)
    result = evaluate_forecasts(
        arrays["y_true"], arrays["y_pred"], scaler=scaler,
        target_index=args.target_index, already_original_scale=args.scale == "degC",
    )
    y_true = np.asarray(arrays["y_true"])
    y_pred = np.asarray(arrays["y_pred"])
    if y_true.ndim == 3:
        y_true, y_pred = y_true[..., 0], y_pred[..., 0]
    if scaler is not None:
        y_true = inverse_scale_target(y_true, scaler, args.target_index)
        y_pred = inverse_scale_target(y_pred, scaler, args.target_index)
    timestamps = arrays["target_timestamps"] if "target_timestamps" in arrays else None
    frame = build_error_frame(
        y_true, y_pred, target_timestamps=timestamps,
        run_id=metadata["run_id"], model_name=metadata["model_name"], split=metadata["split"],
    )
    analysis = analyze_errors(frame, top_n=args.top_n)
    baseline_metrics = None
    if "last_observed_target" in arrays:
        anchors = arrays["last_observed_target"]
        if scaler is not None:
            anchors = inverse_scale_target(anchors, scaler, args.target_index)
        baseline = persistence_predictions(anchors, result.horizon)
        baseline_metrics = evaluate_forecasts(
            y_true, baseline, already_original_scale=True
        ).overall

    args.output_dir.mkdir(parents=True, exist_ok=True)
    analysis["worst_cases"].to_csv(args.output_dir / "worst_cases.csv", index=False)
    analysis["by_horizon"].to_csv(args.output_dir / "error_by_horizon.csv", index=False)
    if "by_hour" in analysis:
        analysis["by_hour"].to_csv(args.output_dir / "error_by_hour.csv", index=False)
        analysis["by_season"].to_csv(args.output_dir / "error_by_season.csv", index=False)
    import pandas as pd

    pd.DataFrame(result.per_horizon).to_csv(
        args.output_dir / "metrics_per_horizon.csv", index=False
    )
    metrics_payload = {
        **metadata, **asdict(result), "baseline": baseline_metrics,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    forecast_path = save_forecast_figure(
        frame, args.output_dir / "forecast.png", run_id=metadata["run_id"],
        model_id=metadata["model_name"], split=metadata["split"],
    )
    residual_path = save_residual_figure(
        frame, args.output_dir / "residuals.png", run_id=metadata["run_id"],
        model_id=metadata["model_name"], split=metadata["split"],
    )
    for figure_id, path, description in (
        ("forecast", forecast_path, "Actual and predicted temperature"),
        ("residuals", residual_path, "Residual by forecast horizon"),
    ):
        register_figure(
            args.output_dir / "figure_manifest.csv", figure_id=figure_id, path=path,
            run_id=metadata["run_id"], model_id=metadata["model_name"],
            split=metadata["split"], description=description,
            finding=f"RMSE={result.overall['rmse']:.4f} °C on {result.sample_count} samples",
            impact="Quantifies forecast error on the declared evaluation population",
            decision="Use validation for selection; test is final reporting only",
            created_at=metrics_payload["evaluated_at"],
        )
    if audit is not None:
        audit.update({"status": "completed", "completed_at": metrics_payload["evaluated_at"],
                      "metrics_path": str(args.output_dir / "metrics.json")})
        args.final_test_audit.write_text(
            json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(args.output_dir / "metrics.json")


if __name__ == "__main__":
    main()
