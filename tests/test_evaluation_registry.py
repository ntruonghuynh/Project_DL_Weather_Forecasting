import json

import pytest

from src.evaluation.registry import (
    CandidateMetric,
    append_registry_row,
    authorize_final_test,
    select_validation_candidate,
    write_selection_manifest,
)


def candidate(run_id="run-a", rmse=2.0, split="validation"):
    return CandidateMetric(
        run_id=run_id, model_name="lstm", checkpoint="best.pt", config="config.json",
        schema_hash="schema-hash", scaler_sha256="scaler-hash", split=split,
        rmse_deg_c=rmse,
    )


def test_selection_uses_validation_and_baseline_gate() -> None:
    best, improvement = select_validation_candidate(
        [candidate("a", 2.0), candidate("b", 1.5)], baseline_rmse_deg_c=2.5
    )
    assert best.run_id == "b"
    assert improvement == pytest.approx(0.4)
    with pytest.raises(ValueError, match="validation"):
        select_validation_candidate([candidate(split="test")], 2.5)
    with pytest.raises(ValueError, match="below"):
        select_validation_candidate([candidate(rmse=2.49)], 2.5)


def test_selection_manifest_is_immutable_and_test_is_one_shot(tmp_path) -> None:
    manifest = tmp_path / "selection.json"
    write_selection_manifest(
        manifest, candidate=candidate(), baseline_rmse_deg_c=3.0,
        improvement_fraction=1 / 3, comparison_population_id="population-v1",
        approved_by=["TV6", "TV1"],
    )
    assert json.loads(manifest.read_text())["locked"] is True
    with pytest.raises(FileExistsError):
        write_selection_manifest(
            manifest, candidate=candidate(), baseline_rmse_deg_c=3.0,
            improvement_fraction=1 / 3, comparison_population_id="population-v1",
            approved_by=["TV6"],
        )
    audit = tmp_path / "final-test.json"
    authorize_final_test(manifest, "run-a", audit)
    with pytest.raises(RuntimeError, match="already"):
        authorize_final_test(manifest, "run-a", audit)


def test_registry_is_append_only(tmp_path) -> None:
    row = {
        "run_id": "r1", "parent_run_id": "", "status": "completed",
        "model_name": "lstm", "model_version": "1", "config_path": "config.json",
        "seed": 42, "split": "validation", "started_at": "start",
        "completed_at": "end", "notes": "",
    }
    path = tmp_path / "registry.csv"
    append_registry_row(path, row)
    with pytest.raises(ValueError, match="already exists"):
        append_registry_row(path, row)
