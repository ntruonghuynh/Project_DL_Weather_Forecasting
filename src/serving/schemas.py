"""Framework-neutral prediction and model-bundle schemas."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PredictionRecord:
    """One prediction row shared by evaluation and serving."""

    timestamp: str
    y_true: float | None
    y_pred: float
    model_name: str
    model_version: str
    split: str
    run_id: str
    horizon: int
    unit: str = "degC"


@dataclass(frozen=True)
class ModelBundle:
    """Loaded inference resources identified by model and immutable run."""

    model: object
    model_name: str
    model_version: str
    run_id: str
    scaler: object
    feature_schema: dict[str, Any]
    resolved_config: dict[str, Any]
    metadata: dict[str, Any]
    bundle_path: Path | None = None
    checksums: dict[str, str] = field(default_factory=dict)
