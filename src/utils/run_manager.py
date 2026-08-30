"""Immutable run metadata contracts; persistence belongs to TV1."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

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


def create_run(config: dict[str, Any], seed: int, parent_run_id: str | None = None) -> RunRecord:
    """Create a persisted run with a new immutable run ID."""
    raise NotImplementedError("TV1 must implement run persistence")
