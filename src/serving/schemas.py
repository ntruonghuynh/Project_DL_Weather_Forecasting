"""Framework-neutral prediction schemas."""

from dataclasses import dataclass


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


@dataclass(frozen=True)
class ModelBundle:
    """Loaded inference resources identified by model and immutable run."""

    model: object
    model_name: str
    model_version: str
    run_id: str
