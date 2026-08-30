"""Predictor interface; loading/inference implementation belongs to serving owner."""

from .schemas import ModelBundle, PredictionRecord


class Predictor:
    """Run inference from an already-loaded bundle; never train models."""

    def __init__(self, bundle: ModelBundle) -> None:
        self.bundle = bundle

    def predict(self, inputs: object, timestamps: object) -> list[PredictionRecord]:
        """Return records satisfying the shared prediction contract."""
        raise NotImplementedError("Serving owner must implement inference")
