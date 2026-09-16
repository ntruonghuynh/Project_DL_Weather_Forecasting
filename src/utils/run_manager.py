"""Immutable run metadata contracts; persistence belongs to TV1."""

from __future__ import annotations

import platform
import subprocess
import sys
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import torch
import yaml

from .io import read_json, write_json

RunStatus = Literal["created", "running", "completed", "failed"]


@dataclass(frozen=True)
class RunRecord:
    """Immutable identity and required metadata/artifact locations for one run."""

    run_id: str
    status: RunStatus
    parent_run_id: str | None
    resolved_config: dict[str, Any]
    seed: int
    metrics: dict[str, float] = field(default_factory=dict)
    best_checkpoint: Path | None = None
    last_checkpoint: Path | None = None
    predictions: Path | None = None
    run_dir: Path | None = None


def get_git_commit() -> str:
    """Safely obtain the current Git commit hash if available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def get_environment_metadata() -> dict[str, Any]:
    """Capture environment details for reproducibility audit."""
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "git_commit": get_git_commit(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def create_run(
    config: dict[str, Any],
    seed: int,
    parent_run_id: str | None = None,
    runs_dir: Path | str = "runs",
) -> RunRecord:
    """Create a persisted run with a new immutable run ID and directory structure."""
    model_name = config.get("model", {}).get("name", "forecast_model")
    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    unique_suffix = uuid.uuid4().hex[:6]
    run_id = f"{model_name}_{timestamp_str}_{unique_suffix}"

    run_dir_root = Path(runs_dir)
    run_dir = run_dir_root / model_name / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save resolved config
    config_path = run_dir / "resolved_config.yaml"
    with config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    # 2. Save environment metadata
    env_path = run_dir / "environment.json"
    write_json(env_path, get_environment_metadata())

    # 3. Create initial RunRecord
    record = RunRecord(
        run_id=run_id,
        status="created",
        parent_run_id=parent_run_id,
        resolved_config=config,
        seed=seed,
        metrics={},
        best_checkpoint=None,
        last_checkpoint=None,
        predictions=None,
        run_dir=run_dir,
    )

    record_path = run_dir / "run_record.json"
    record_dict = asdict(record)
    write_json(record_path, record_dict)

    return record


def update_run_status(
    run_dir: Path | str,
    status: RunStatus,
    metrics: dict[str, float] | None = None,
    best_checkpoint: Path | None = None,
    last_checkpoint: Path | None = None,
    predictions: Path | None = None,
) -> RunRecord:
    """Update a run's status and metadata, persisting final metrics atomically."""
    run_dir = Path(run_dir)
    record_path = run_dir / "run_record.json"
    if not record_path.is_file():
        raise FileNotFoundError(f"Run record not found in {run_dir}")

    existing_data = read_json(record_path)
    merged_metrics = dict(existing_data.get("metrics", {}))
    if metrics:
        merged_metrics.update(metrics)

    updated_record = RunRecord(
        run_id=existing_data["run_id"],
        status=status,
        parent_run_id=existing_data.get("parent_run_id"),
        resolved_config=existing_data["resolved_config"],
        seed=existing_data["seed"],
        metrics=merged_metrics,
        best_checkpoint=best_checkpoint
        if best_checkpoint is not None
        else (
            Path(existing_data["best_checkpoint"])
            if existing_data.get("best_checkpoint")
            else None
        ),
        last_checkpoint=last_checkpoint
        if last_checkpoint is not None
        else (
            Path(existing_data["last_checkpoint"])
            if existing_data.get("last_checkpoint")
            else None
        ),
        predictions=predictions
        if predictions is not None
        else (
            Path(existing_data["predictions"])
            if existing_data.get("predictions")
            else None
        ),
        run_dir=run_dir,
    )

    write_json(record_path, asdict(updated_record))

    if metrics:
        write_json(run_dir / "final_metrics.json", merged_metrics)

    return updated_record
