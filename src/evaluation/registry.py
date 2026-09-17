"""Append-only experiment registry and validation-only model selection."""

from __future__ import annotations

import csv
import hashlib
import hmac
import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REGISTRY_COLUMNS = [
    "run_id", "parent_run_id", "status", "model_name", "model_version",
    "config_path", "config_sha256", "seed", "split", "started_at", "completed_at",
    "duration_seconds", "epochs_completed", "best_epoch", "global_step", "smoke",
    "train_dataset_sha256", "validation_dataset_sha256", "population_id",
    "sample_count", "input_length", "horizon", "schema_hash", "schema_sha256",
    "scaler_sha256", "checkpoint_path", "checkpoint_sha256", "predictions_path",
    "predictions_sha256", "metrics_path", "metrics_sha256", "mae_deg_c",
    "mse_deg_c2", "rmse_deg_c", "baseline_rmse_deg_c", "notes",
]
REQUIRED_REGISTRY_COLUMNS = {
    "run_id", "parent_run_id", "status", "model_name", "model_version",
    "config_path", "seed", "split", "started_at", "completed_at", "notes",
}
VALID_RUN_STATUSES = {"created", "running", "completed", "failed", "not_promoted"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_registry_row(path: Path, row: dict[str, Any]) -> None:
    """Append a run without rewriting or hiding prior records."""
    missing = REQUIRED_REGISTRY_COLUMNS - set(row)
    if missing:
        raise ValueError(f"registry row is missing fields: {sorted(missing)}")
    if row["status"] not in VALID_RUN_STATUSES:
        raise ValueError(f"invalid run status: {row['status']}")
    if row["split"] not in {"train", "validation", "test"}:
        raise ValueError("registry split must be train, validation or test")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    if exists:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != REGISTRY_COLUMNS:
                raise ValueError("registry header does not match the current registry schema")
            existing = list(reader)
        if any(item["run_id"] == str(row["run_id"]) for item in existing):
            raise ValueError(f"run_id already exists in append-only registry: {row['run_id']}")
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REGISTRY_COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in REGISTRY_COLUMNS})


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_validation_registry_row(
    run_dir: Path,
    metrics_path: Path,
    predictions_path: Path,
) -> dict[str, Any]:
    """Build one registry row from verified real validation artifacts."""
    run_dir = Path(run_dir)
    metrics_path = Path(metrics_path)
    predictions_path = Path(predictions_path)
    record = json.loads((run_dir / "run_record.json").read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    if record.get("status") != "completed" or record.get("smoke") is not False:
        raise ValueError("registry requires a completed real run with smoke=false")
    if metrics.get("split") != "validation":
        raise ValueError("registry model-selection evidence must use validation")
    if metrics.get("run_id") != record.get("run_id"):
        raise ValueError("metrics run_id does not match the run record")
    if Path(record["best_checkpoint"]).resolve() != Path(metrics["checkpoint"]).resolve():
        raise ValueError("metrics checkpoint does not match the run's best checkpoint")
    if Path(record["predictions"]).resolve() != predictions_path.resolve():
        raise ValueError("prediction artifact does not match the run record")

    overall = metrics.get("overall", {})
    if any(key not in overall or not math.isfinite(float(overall[key])) for key in (
        "mae", "mse", "rmse"
    )):
        raise ValueError("metrics must contain finite MAE/MSE/RMSE")
    baseline = metrics.get("baseline", {})
    if "rmse" not in baseline or not math.isfinite(float(baseline["rmse"])):
        raise ValueError("metrics must contain a finite validation baseline RMSE")

    config_path = run_dir / "resolved_config.yaml"
    checkpoint_path = Path(record["best_checkpoint"])
    dataset_identity = record["dataset_identity"]
    schema_identity = record["schema_identity"]
    return {
        "run_id": record["run_id"],
        "parent_run_id": record.get("parent_run_id") or "",
        "status": record["status"],
        "model_name": record["model_name"],
        "model_version": metrics["model_version"],
        "config_path": str(config_path),
        "config_sha256": _sha256_file(config_path),
        "seed": record["seed"],
        "split": "validation",
        "started_at": record["start_time"],
        "completed_at": record["end_time"],
        "duration_seconds": record["duration_seconds"],
        "epochs_completed": record["epochs_completed"],
        "best_epoch": record["best_epoch"],
        "global_step": record["global_step"],
        "smoke": record["smoke"],
        "train_dataset_sha256": dataset_identity["train"]["sha256"],
        "validation_dataset_sha256": dataset_identity["validation"]["sha256"],
        "population_id": metrics["population_id"],
        "sample_count": metrics["sample_count"],
        "input_length": metrics["input_length"],
        "horizon": metrics["horizon"],
        "schema_hash": metrics["schema_hash"],
        "schema_sha256": schema_identity["sha256"],
        "scaler_sha256": metrics["scaler_sha256"],
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "predictions_path": str(predictions_path),
        "predictions_sha256": _sha256_file(predictions_path),
        "metrics_path": str(metrics_path),
        "metrics_sha256": _sha256_file(metrics_path),
        "mae_deg_c": overall["mae"],
        "mse_deg_c2": overall["mse"],
        "rmse_deg_c": overall["rmse"],
        "baseline_rmse_deg_c": baseline["rmse"],
        "notes": "real Jena validation; no test metrics used",
    }


@dataclass(frozen=True)
class CandidateMetric:
    """Validation metric plus all provenance needed to promote one run."""

    run_id: str
    model_name: str
    checkpoint: str
    checkpoint_sha256: str
    config: str
    config_sha256: str
    schema_hash: str
    schema_path: str
    schema_sha256: str
    scaler_path: str
    scaler_sha256: str
    prediction_path: str
    prediction_sha256: str
    metrics_path: str
    metrics_sha256: str
    split: str
    rmse_deg_c: float


def select_validation_candidate(
    candidates: list[CandidateMetric],
    baseline_rmse_deg_c: float,
    minimum_improvement: float = 0.03,
) -> tuple[CandidateMetric, float]:
    """Select the best validation run only when it clears the declared baseline gate."""
    if not candidates:
        raise ValueError("at least one candidate is required")
    if not math.isfinite(baseline_rmse_deg_c) or baseline_rmse_deg_c <= 0:
        raise ValueError("baseline_rmse_deg_c must be positive")
    if not math.isfinite(minimum_improvement) or not 0 <= minimum_improvement < 1:
        raise ValueError("minimum_improvement must be within [0,1)")
    for candidate in candidates:
        if candidate.split != "validation":
            raise ValueError("model selection may only consume validation metrics")
        if not math.isfinite(candidate.rmse_deg_c) or candidate.rmse_deg_c < 0:
            raise ValueError("candidate RMSE must be non-negative")
        provenance = (
            candidate.run_id, candidate.model_name, candidate.checkpoint,
            candidate.checkpoint_sha256, candidate.config, candidate.config_sha256,
            candidate.schema_hash, candidate.schema_path, candidate.schema_sha256,
            candidate.scaler_path, candidate.scaler_sha256, candidate.prediction_path,
            candidate.prediction_sha256, candidate.metrics_path, candidate.metrics_sha256,
        )
        if any(not field for field in provenance):
            raise ValueError("candidate provenance fields must not be empty")
    best = min(candidates, key=lambda item: item.rmse_deg_c)
    improvement = (baseline_rmse_deg_c - best.rmse_deg_c) / baseline_rmse_deg_c
    if improvement < minimum_improvement:
        raise ValueError(
            f"best candidate improves baseline by {improvement:.2%}, "
            f"below {minimum_improvement:.2%} gate"
        )
    return best, float(improvement)


def write_selection_manifest(
    path: Path,
    *,
    candidate: CandidateMetric,
    baseline_rmse_deg_c: float,
    improvement_fraction: float,
    comparison_population_id: str,
    approved_by: list[str],
    comparison_path: Path,
    minimum_improvement: float = 0.03,
) -> Path:
    """Create an immutable selection manifest before any final-test evaluation."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"selection manifest is immutable and already exists: {path}")
    if candidate.split != "validation":
        raise ValueError("selection manifest must be based on validation")
    if (
        not math.isfinite(baseline_rmse_deg_c)
        or not math.isfinite(candidate.rmse_deg_c)
        or baseline_rmse_deg_c <= 0
        or candidate.rmse_deg_c < 0
    ):
        raise ValueError("baseline RMSE must be positive and candidate RMSE non-negative")
    if not math.isfinite(minimum_improvement) or not 0 <= minimum_improvement < 1:
        raise ValueError("minimum_improvement must be within [0,1)")
    if not math.isfinite(improvement_fraction):
        raise ValueError("improvement_fraction must be finite")
    expected_improvement = (
        baseline_rmse_deg_c - candidate.rmse_deg_c
    ) / baseline_rmse_deg_c
    if not abs(improvement_fraction - expected_improvement) <= 1e-12:
        raise ValueError("improvement_fraction does not match baseline and candidate RMSE")
    if improvement_fraction < minimum_improvement:
        raise ValueError("candidate does not clear the declared baseline gate")
    if not comparison_population_id or not approved_by or any(not name for name in approved_by):
        raise ValueError("comparison_population_id and approved_by are required")
    artifact_bindings = {
        "selected_checkpoint": {
            "path": candidate.checkpoint,
            "sha256": candidate.checkpoint_sha256,
        },
        "resolved_config": {
            "path": candidate.config,
            "sha256": candidate.config_sha256,
        },
        "feature_schema": {
            "path": candidate.schema_path,
            "sha256": candidate.schema_sha256,
            "schema_hash": candidate.schema_hash,
        },
        "scaler": {
            "path": candidate.scaler_path,
            "sha256": candidate.scaler_sha256,
        },
        "validation_prediction": {
            "path": candidate.prediction_path,
            "sha256": candidate.prediction_sha256,
        },
        "validation_metrics": {
            "path": candidate.metrics_path,
            "sha256": candidate.metrics_sha256,
        },
        "comparison": {
            "path": str(comparison_path),
            "sha256": _sha256_file(Path(comparison_path)),
        },
    }
    for name, binding in artifact_bindings.items():
        artifact_path = Path(binding["path"])
        if not artifact_path.is_file():
            raise FileNotFoundError(f"selection binding is missing {name}: {artifact_path}")
        if _sha256_file(artifact_path) != binding["sha256"]:
            raise ValueError(f"selection binding checksum mismatch for {name}")
    payload = {
        "schema_version": "2.0",
        "locked": True,
        "locked_at": _utc_now(),
        "selection_split": "validation",
        "selection_metric": "rmse_deg_c",
        "baseline_rmse_deg_c": baseline_rmse_deg_c,
        "minimum_improvement_fraction": minimum_improvement,
        "observed_improvement_fraction": improvement_fraction,
        "comparison_population_id": comparison_population_id,
        "approved_by": approved_by,
        "candidate": asdict(candidate),
        "artifact_bindings": artifact_bindings,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["manifest_sha256"] = hashlib.sha256(canonical).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _verified_manifest(selection_manifest: Path) -> tuple[dict[str, Any], str]:
    path = Path(selection_manifest)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    stored_digest = manifest.pop("manifest_sha256", None)
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    expected_digest = hashlib.sha256(canonical).hexdigest()
    if not isinstance(stored_digest, str) or not hmac.compare_digest(
        stored_digest, expected_digest
    ):
        raise ValueError("selection manifest checksum is missing or invalid")
    if manifest.get("locked") is not True or manifest.get("selection_split") != "validation":
        raise ValueError("selection manifest must be locked from validation")
    if manifest.get("selection_metric") != "rmse_deg_c":
        raise ValueError("selection manifest metric must be rmse_deg_c")
    for name, binding in manifest.get("artifact_bindings", {}).items():
        artifact_path = Path(binding["path"])
        if not artifact_path.is_file() or _sha256_file(artifact_path) != binding["sha256"]:
            raise ValueError(f"selection manifest binding mismatch for {name}")
    return manifest, stored_digest


def canonical_final_test_audit_path(selection_manifest: Path) -> Path:
    """Return the only permitted audit location for a selection manifest."""
    return Path(selection_manifest).resolve().parent / "final_test_audit.json"


def canonical_selection_approval_path(selection_manifest: Path) -> Path:
    """Return the immutable team-approval artifact location."""
    return Path(selection_manifest).resolve().parent / "selection_approval.json"


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise RuntimeError(f"immutable artifact already exists: {path}") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_selection_approval(
    selection_manifest: Path,
    *,
    approval: str = "team_lead_review",
) -> Path:
    """Record immutable human/team approval without mutating the locked manifest."""
    manifest, manifest_sha256 = _verified_manifest(selection_manifest)
    if approval != "team_lead_review":
        raise ValueError("approval must be team_lead_review")
    payload = {
        "schema_version": "1.0",
        "manifest_sha256": manifest_sha256,
        "selected_run_id": manifest["candidate"]["run_id"],
        "approval": approval,
        "approved_at": _utc_now(),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["approval_sha256"] = hashlib.sha256(canonical).hexdigest()
    path = canonical_selection_approval_path(selection_manifest)
    _write_json_exclusive(path, payload)
    return path


def _verify_selection_approval(selection_manifest: Path, manifest_sha256: str) -> None:
    path = canonical_selection_approval_path(selection_manifest)
    if not path.is_file():
        raise ValueError(f"team approval artifact is missing: {path}")
    approval = json.loads(path.read_text(encoding="utf-8"))
    stored_digest = approval.pop("approval_sha256", None)
    canonical = json.dumps(approval, sort_keys=True, separators=(",", ":")).encode()
    expected_digest = hashlib.sha256(canonical).hexdigest()
    if not isinstance(stored_digest, str) or not hmac.compare_digest(
        stored_digest, expected_digest
    ):
        raise ValueError("selection approval checksum is missing or invalid")
    if (
        approval.get("manifest_sha256") != manifest_sha256
        or approval.get("approval") != "team_lead_review"
    ):
        raise ValueError("selection approval does not bind the locked manifest")


def authorize_final_test(
    selection_manifest: Path,
    run_id: str,
    audit_path: Path | None = None,
) -> dict[str, Any]:
    """Atomically authorize one final test at the canonical audit path only."""
    manifest, manifest_sha256 = _verified_manifest(selection_manifest)
    _verify_selection_approval(selection_manifest, manifest_sha256)
    candidate = manifest.get("candidate", {})
    if candidate.get("run_id") != run_id:
        raise ValueError("final test is authorized only for the locked candidate run")
    canonical_path = canonical_final_test_audit_path(selection_manifest)
    if audit_path is not None and Path(audit_path).resolve() != canonical_path:
        raise ValueError(f"final-test audit path is canonical and must be {canonical_path}")
    bindings = manifest["artifact_bindings"]
    audit = {
        "schema_version": "1.0",
        "manifest_sha256": manifest_sha256,
        "selected_run_id": run_id,
        "checkpoint_sha256": bindings["selected_checkpoint"]["sha256"],
        "config_sha256": bindings["resolved_config"]["sha256"],
        "schema_sha256": bindings["feature_schema"]["sha256"],
        "schema_hash": bindings["feature_schema"]["schema_hash"],
        "scaler_sha256": bindings["scaler"]["sha256"],
        "test_holdout_status": "previously_exposed_during_development",
        "selection_used_test_metrics": False,
        "candidate_selected_from": "validation",
        "limitation": (
            "Current final candidate was selected exclusively on validation. "
            "The existing test split had been accessed in an earlier invalid development "
            "iteration and therefore is not treated as a pristine unseen holdout."
        ),
        "started_at": _utc_now(),
        "completed_at": None,
        "status": "authorized",
        "test_population_id": None,
        "metrics_path": None,
        "prediction_path": None,
        "metrics_sha256": None,
        "prediction_sha256": None,
    }
    try:
        _write_json_exclusive(canonical_path, audit)
    except RuntimeError as error:
        raise RuntimeError(
            f"final test has already been authorized for this selection: {canonical_path}"
        ) from error
    return audit


def record_final_test_prediction(
    selection_manifest: Path,
    *,
    prediction_path: Path,
    test_population_id: str,
) -> dict[str, Any]:
    """Bind the one-shot inference artifact to the canonical audit."""
    audit_path = canonical_final_test_audit_path(selection_manifest)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "authorized":
        raise RuntimeError("final test is not in the authorized inference state")
    prediction_path = Path(prediction_path)
    audit.update({
        "status": "inference_completed",
        "test_population_id": test_population_id,
        "prediction_path": str(prediction_path),
        "prediction_sha256": _sha256_file(prediction_path),
    })
    _write_json_atomic(audit_path, audit)
    return audit


def require_final_test_evaluation(
    selection_manifest: Path,
    *,
    run_id: str,
    prediction_path: Path,
) -> dict[str, Any]:
    """Verify one-shot inference completed before allowing final evaluation."""
    manifest, manifest_sha256 = _verified_manifest(selection_manifest)
    audit_path = canonical_final_test_audit_path(selection_manifest)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (
        audit.get("status") != "inference_completed"
        or audit.get("manifest_sha256") != manifest_sha256
        or audit.get("selected_run_id") != run_id
        or manifest["candidate"]["run_id"] != run_id
    ):
        raise RuntimeError("final-test evaluation is not authorized for this artifact")
    if (
        Path(audit["prediction_path"]).resolve() != Path(prediction_path).resolve()
        or audit["prediction_sha256"] != _sha256_file(Path(prediction_path))
    ):
        raise ValueError("final-test prediction does not match the canonical audit")
    return audit


def complete_final_test(
    selection_manifest: Path,
    *,
    metrics_path: Path,
    completed_at: str,
) -> dict[str, Any]:
    """Atomically complete the canonical audit after final metrics are persisted."""
    audit_path = canonical_final_test_audit_path(selection_manifest)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "inference_completed":
        raise RuntimeError("final-test audit cannot be completed from its current state")
    metrics_path = Path(metrics_path)
    audit.update({
        "status": "completed",
        "completed_at": completed_at,
        "metrics_path": str(metrics_path),
        "metrics_sha256": _sha256_file(metrics_path),
    })
    canonical = json.dumps(audit, sort_keys=True, separators=(",", ":")).encode()
    audit["audit_sha256"] = hashlib.sha256(canonical).hexdigest()
    _write_json_atomic(audit_path, audit)
    return audit
