"""Compare validation results and optionally lock a promoted candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.evaluation.registry import (
    CandidateMetric,
    select_validation_candidate,
    write_selection_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-rmse", type=float, required=True)
    parser.add_argument("--population-id", required=True)
    parser.add_argument("--selection-manifest", type=Path)
    parser.add_argument("--approved-by", action="append", default=[])
    args = parser.parse_args()

    candidates: list[CandidateMetric] = []
    rows = []
    for path in args.metrics:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("split") != "validation":
            raise ValueError("comparison and selection only accept validation metrics")
        if payload.get("population_id") != args.population_id:
            raise ValueError("all models must use the same comparison population")
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
    best, improvement = select_validation_candidate(candidates, args.baseline_rmse)
    frame = pd.DataFrame(rows).sort_values("rmse_deg_c")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
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
