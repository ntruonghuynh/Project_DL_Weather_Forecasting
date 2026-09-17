"""Compare validation results and optionally lock a promoted candidate."""

from __future__ import annotations

import argparse
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
    parser.add_argument("--selection-manifest", type=Path)
    parser.add_argument("--approved-by", action="append", default=[])
    parser.add_argument("--figure-output", type=Path)
    args = parser.parse_args()

    candidates: list[CandidateMetric] = []
    rows = []
    comparison_values: dict[str, set[object]] = {
        "population_id": set(),
        "sample_count": set(),
        "horizon": set(),
        "schema_hash": set(),
        "scaler_sha256": set(),
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
        candidate = CandidateMetric(
            run_id=payload["run_id"], model_name=payload["model_name"],
            checkpoint=payload["checkpoint"], config=payload["resolved_config"],
            schema_hash=payload["schema_hash"], scaler_sha256=payload["scaler_sha256"],
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
            approved_by=args.approved_by,
        )
    print(f"selected {best.model_name} run={best.run_id}; improvement={improvement:.2%}")


if __name__ == "__main__":
    main()
