import hashlib
import json
from pathlib import Path

import pytest

from src.evaluation.registry import (
    CandidateMetric,
    append_registry_row,
    authorize_final_test,
    canonical_final_test_audit_path,
    select_validation_candidate,
    write_selection_approval,
    write_selection_manifest,
)


def candidate(run_id="run-a", rmse=2.0, split="validation"):
    artifact = Path(__file__)
    sha256 = hashlib.sha256(artifact.read_bytes()).hexdigest()
    return CandidateMetric(
        run_id=run_id, model_name="lstm", checkpoint=str(artifact),
        checkpoint_sha256=sha256, config=str(artifact), config_sha256=sha256,
        schema_hash="schema-hash", schema_path=str(artifact), schema_sha256=sha256,
        scaler_path=str(artifact), scaler_sha256=sha256,
        prediction_path=str(artifact), prediction_sha256=sha256,
        metrics_path=str(artifact), metrics_sha256=sha256,
        split=split, rmse_deg_c=rmse,
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
    with pytest.raises(ValueError, match="positive"):
        select_validation_candidate([candidate()], float("nan"))
    with pytest.raises(ValueError, match="non-negative"):
        select_validation_candidate([candidate(rmse=float("nan"))], 2.5)


def test_selection_manifest_is_immutable_and_test_is_one_shot(tmp_path) -> None:
    manifest = tmp_path / "selection.json"
    write_selection_manifest(
        manifest, candidate=candidate(), baseline_rmse_deg_c=3.0,
        improvement_fraction=1 / 3, comparison_population_id="population-v1",
        approved_by=["TV6", "TV1"], comparison_path=Path(__file__),
    )
    assert json.loads(manifest.read_text())["locked"] is True
    with pytest.raises(FileExistsError):
        write_selection_manifest(
            manifest, candidate=candidate(), baseline_rmse_deg_c=3.0,
            improvement_fraction=1 / 3, comparison_population_id="population-v1",
            approved_by=["TV6"], comparison_path=Path(__file__),
        )
    write_selection_approval(manifest)
    authorize_final_test(manifest, "run-a")
    assert canonical_final_test_audit_path(manifest).is_file()
    with pytest.raises(RuntimeError, match="already"):
        authorize_final_test(manifest, "run-a")
    with pytest.raises(ValueError, match="canonical"):
        authorize_final_test(manifest, "run-a", tmp_path / "bypass-audit.json")


def test_selection_manifest_recomputes_gate_instead_of_trusting_caller(tmp_path) -> None:
    with pytest.raises(ValueError, match="does not match"):
        write_selection_manifest(
            tmp_path / "forged.json", candidate=candidate(rmse=2.9),
            baseline_rmse_deg_c=3.0, improvement_fraction=0.5,
            comparison_population_id="population-v1", approved_by=["TV6"],
            comparison_path=Path(__file__),
        )
    with pytest.raises(ValueError, match="does not clear"):
        write_selection_manifest(
            tmp_path / "below-gate.json", candidate=candidate(rmse=2.95),
            baseline_rmse_deg_c=3.0, improvement_fraction=1 / 60,
            comparison_population_id="population-v1", approved_by=["TV6"],
            comparison_path=Path(__file__),
        )


def test_final_test_rejects_tampered_selection_manifest(tmp_path) -> None:
    manifest = tmp_path / "selection.json"
    write_selection_manifest(
        manifest, candidate=candidate(), baseline_rmse_deg_c=3.0,
        improvement_fraction=1 / 3, comparison_population_id="population-v1",
        approved_by=["TV6", "TV1"], comparison_path=Path(__file__),
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["candidate"]["run_id"] = "tampered-run"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        authorize_final_test(manifest, "tampered-run", tmp_path / "audit.json")


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


def test_registry_rejects_header_drift(tmp_path) -> None:
    path = tmp_path / "registry.csv"
    path.write_text("run_id,status\n", encoding="utf-8")
    row = {
        "run_id": "r1", "parent_run_id": "", "status": "completed",
        "model_name": "lstm", "model_version": "1", "config_path": "config.json",
        "seed": 42, "split": "validation", "started_at": "start",
        "completed_at": "end", "notes": "",
    }
    with pytest.raises(ValueError, match="header"):
        append_registry_row(path, row)
