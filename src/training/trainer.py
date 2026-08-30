"""Training orchestration interface; implementation belongs to TV3-TV5."""

from typing import Any


class Trainer:
    """Coordinate training without defining model-specific algorithms."""

    def fit(self, model: object, data: object, config: dict[str, Any]) -> dict[str, float]:
        """Train and return validation metrics when implemented."""
        raise NotImplementedError("Training owners must implement Trainer.fit")
