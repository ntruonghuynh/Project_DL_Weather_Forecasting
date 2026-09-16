"""Append-only experiment registry and validation-only model selection."""

from __future__ import annotations

import csv
import hashlib
import hmac
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REGISTRY_COLUMNS = [
    "run_id", "parent_run_id", "status", "model_name", "model_version",
    "config_path", "seed", "split", "started_at", "completed_at", "notes",
]
VALID_RUN_STATUSES = {"created", "running", "completed", "failed", "not_promoted"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_registry_row(path: Path, row: dict[str, Any]) -> None:
    """Append a run without rewriting or hiding prior records."""
    missing = set(REGISTRY_COLUMNS) - set(row)
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
            existing = list(csv.DictReader(handle))
        if any(item["run_id"] == str(row["run_id"]) for item in existing):
            raise ValueError(f"run_id already exists in append-only registry: {row['run_id']}")
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REGISTRY_COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row[key] for key in REGISTRY_COLUMNS})


@dataclass(frozen=True)
class CandidateMetric:
    """Validation metric plus all provenance needed to promote one run."""

    run_id: str
    model_name: str
    checkpoint: str
    config: str
    schema_hash: str
    scaler_sha256: str
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
            candidate.config, candidate.schema_hash, candidate.scaler_sha256,
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
    payload = {
        "schema_version": "1.0",
        "locked": True,
        "locked_at": _utc_now(),
        "selection_metric": "rmse_deg_c",
        "baseline_rmse_deg_c": baseline_rmse_deg_c,
        "minimum_improvement_fraction": minimum_improvement,
        "observed_improvement_fraction": improvement_fraction,
        "comparison_population_id": comparison_population_id,
        "approved_by": approved_by,
        "candidate": asdict(candidate),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["manifest_sha256"] = hashlib.sha256(canonical).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def authorize_final_test(
    selection_manifest: Path, run_id: str, audit_path: Path
) -> dict[str, Any]:
    """Refuse test access unless selection is locked and was not tested before."""
    manifest = json.loads(Path(selection_manifest).read_text(encoding="utf-8"))
    stored_digest = manifest.pop("manifest_sha256", None)
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    expected_digest = hashlib.sha256(canonical).hexdigest()
    if not isinstance(stored_digest, str) or not hmac.compare_digest(
        stored_digest, expected_digest
    ):
        raise ValueError("selection manifest checksum is missing or invalid")
    if manifest.get("locked") is not True or manifest.get("candidate", {}).get("run_id") != run_id:
        raise ValueError("final test is authorized only for the locked candidate run")
    audit_path = Path(audit_path)
    if audit_path.exists():
        prior = json.loads(audit_path.read_text(encoding="utf-8"))
        if prior.get("final_test_started_at"):
            raise RuntimeError("final test has already been started for this selection")
    audit = {
        "run_id": run_id,
        "selection_manifest": str(selection_manifest),
        "final_test_started_at": _utc_now(),
        "status": "started",
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return audit
