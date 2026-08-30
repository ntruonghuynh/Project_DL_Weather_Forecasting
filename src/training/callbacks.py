"""Training callback contracts; implementation belongs to training owners."""

from typing import Protocol


class TrainingCallback(Protocol):
    """Receive immutable run-aware training lifecycle events."""

    def on_epoch_end(self, epoch: int, metrics: dict[str, float]) -> None: ...
