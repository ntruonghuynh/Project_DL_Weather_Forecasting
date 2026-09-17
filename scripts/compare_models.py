"""Compare validation results and optionally lock a promoted candidate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.registry import (  # noqa: E402
    CandidateMetric,
    select_validation_candidate,
    write_selection_manifest,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save_comparison_figure(frame: pd.DataFrame, output_path: Path) -> None:
    """Grouped MAE/RMSE bar chart, one bar pair per model, sorted by RMSE."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    positions = np.arange(len(frame))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(positions - width / 2, frame["mae_deg_c"], width, label="MAE (°C)")
    ax.bar(positions + width / 2, frame["rmse_deg_c"], width, label="RMSE (°C)")
    ax.set_xticks(positions)
    ax.set_xticklabels(frame["model_name"], rotation=10, ha="right")
    ax.set_ylabel("Error (°C)")
    ax.set_title("Model comparison — MAE and RMSE (validation)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-rmse", type=float, required=True)
    parser.add_argument("--population-id", required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--scaler", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path)
    parser.add_argument("--approved-by", action="append", default=[])
    parser.add_argument("--figure-output", type=Path)
    args = parser.parse_args()

    with args.registry.open(encoding="utf-8", newline="") as handle:
        registry_rows = {row["run_id"]: row for row in csv.DictReader(handle)}
    schema_payload = json.loads(args.schema.read_text(encoding="utf-8"))
    schema_sha256 = _sha256_file(args.schema)
    scaler_sha256 = _sha256_file(args.scaler)

    candidates: list[CandidateMetric] = []
    rows = []
    comparison_values: dict[str, set[object]] = {
        "population_id": set(),
        "sample_count": set(),
        "horizon": set(),
        "schema_hash": set(),
        "scaler_sha256": set(),
        "schema_sha256": set(),
        "dataset_sha256": set(),
        "input_length": set(),
        "unit": set(),
    }
    for path in args.metrics:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("split") != "validation":
            raise ValueError("comparison and selection only accept validation metrics")
        if payload.get("population_id") != args.population_id:
            raise ValueError("all models must use the same comparison population")
        for field in comparison_values:
            if field not in payload:
                raise ValueError(f"comparison metric is missing {field}: {path}")
            value = payload[field]
            if isinstance(value, (dict, list, set)):
                raise ValueError(f"comparison field must be scalar: {field}")
            comparison_values[field].add(value)
        overall = payload.get("overall")
        if not isinstance(overall, dict) or any(
            key not in overall
            or not math.isfinite(float(overall[key]))
            or float(overall[key]) < 0
            for key in ("mae", "mse", "rmse")
        ):
            raise ValueError(f"comparison metric has invalid MAE/MSE/RMSE: {path}")
        baseline = payload.get("baseline", {})
        if "rmse" not in baseline or not math.isfinite(float(baseline["rmse"])):
            raise ValueError(f"comparison metric has invalid persistence baseline: {path}")
        if not math.isclose(float(baseline["rmse"]), args.baseline_rmse, abs_tol=1e-12):
            raise ValueError("--baseline-rmse does not match validation evaluation artifacts")

        run_id = payload["run_id"]
        if run_id not in registry_rows:
            raise ValueError(f"run is missing from experiment registry: {run_id}")
        registry = registry_rows[run_id]
        if (
            registry["status"] != "completed"
            or registry["smoke"].lower() != "false"
            or registry["split"] != "validation"
        ):
            raise ValueError(f"registry run is not an eligible real validation run: {run_id}")
        exact_registry_values = {
            "model_name": payload["model_name"],
            "population_id": payload["population_id"],
            "schema_hash": payload["schema_hash"],
            "scaler_sha256": payload["scaler_sha256"],
            "checkpoint_path": payload["checkpoint"],
            "config_path": payload["resolved_config"],
            "predictions_path": payload["prediction_artifact"],
            "metrics_path": str(path),
        }
        for field, expected in exact_registry_values.items():
            if registry[field] != str(expected):
                raise ValueError(f"registry/evaluation mismatch for {field}: {run_id}")
        numeric_registry_values = {
            "mae_deg_c": overall["mae"],
            "mse_deg_c2": overall["mse"],
            "rmse_deg_c": overall["rmse"],
            "baseline_rmse_deg_c": baseline["rmse"],
        }
        for field, expected in numeric_registry_values.items():
            if not math.isclose(float(registry[field]), float(expected), abs_tol=1e-12):
                raise ValueError(f"registry/evaluation mismatch for {field}: {run_id}")
        if schema_payload.get("schema_hash") != payload["schema_hash"]:
            raise ValueError("feature schema identity does not match validation metrics")
        if schema_sha256 != payload["schema_sha256"]:
            raise ValueError("feature schema checksum does not match validation metrics")
        if scaler_sha256 != payload["scaler_sha256"]:
            raise ValueError("scaler checksum does not match validation metrics")

        checkpoint_path = Path(payload["checkpoint"])
        config_path = Path(payload["resolved_config"])
        prediction_path = Path(payload["prediction_artifact"])
        artifact_checks = {
            checkpoint_path: registry["checkpoint_sha256"],
            config_path: registry["config_sha256"],
            prediction_path: registry["predictions_sha256"],
            path: registry["metrics_sha256"],
        }
        for artifact_path, expected_sha256 in artifact_checks.items():
            if not artifact_path.is_file() or _sha256_file(artifact_path) != expected_sha256:
                raise ValueError(f"artifact checksum mismatch: {artifact_path}")
        candidate = CandidateMetric(
            run_id=run_id, model_name=payload["model_name"],
            checkpoint=str(checkpoint_path),
            checkpoint_sha256=registry["checkpoint_sha256"],
            config=str(config_path), config_sha256=registry["config_sha256"],
            schema_hash=payload["schema_hash"], schema_path=str(args.schema),
            schema_sha256=schema_sha256, scaler_path=str(args.scaler),
            scaler_sha256=scaler_sha256, prediction_path=str(prediction_path),
            prediction_sha256=registry["predictions_sha256"], metrics_path=str(path),
            metrics_sha256=registry["metrics_sha256"],
            split="validation", rmse_deg_c=float(payload["overall"]["rmse"]),
        )
        candidates.append(candidate)
        rows.append({
            "run_id": candidate.run_id, "model_name": candidate.model_name,
            "mae_deg_c": payload["overall"]["mae"],
            "mse_deg_c2": payload["overall"]["mse"],
            "rmse_deg_c": candidate.rmse_deg_c,
            "sample_count": payload["sample_count"], "horizon": payload["horizon"],
            "population_id": payload["population_id"],
            "persistence_rmse_deg_c": baseline["rmse"],
            "improvement_vs_persistence": (
                float(baseline["rmse"]) - candidate.rmse_deg_c
            ) / float(baseline["rmse"]),
        })
    if len(candidates) != 3 or len({item.model_name for item in candidates}) != 3:
        raise ValueError("final comparison requires exactly three distinct model architectures")
    if len({item.run_id for item in candidates}) != len(candidates):
        raise ValueError("comparison run_id values must be unique")
    for field, values in comparison_values.items():
        if len(values) != 1:
            raise ValueError(f"fair-comparison mismatch for {field}: {sorted(values, key=str)}")
    if comparison_values["unit"] != {"degC"} or comparison_values["horizon"] != {72}:
        raise ValueError("final comparison requires degree-Celsius metrics at 72 horizons")
    best, improvement = select_validation_candidate(candidates, args.baseline_rmse)
    frame = pd.DataFrame(rows).sort_values("rmse_deg_c")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    if args.figure_output is not None:
        _save_comparison_figure(frame, args.figure_output)
    if args.selection_manifest is not None:
        if not args.approved_by:
            raise ValueError("--approved-by is required when locking a selection manifest")
        write_selection_manifest(
            args.selection_manifest, candidate=best,
            baseline_rmse_deg_c=args.baseline_rmse,
            improvement_fraction=improvement, comparison_population_id=args.population_id,
            approved_by=args.approved_by, comparison_path=args.output,
        )
    print(f"selected {best.model_name} run={best.run_id}; improvement={improvement:.2%}")


if __name__ == "__main__":
    main()
