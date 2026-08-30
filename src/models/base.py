"""Shared model contract."""

from typing import Protocol


class ForecastModel(Protocol):
    """Model producing [B,72,1]; decoder starts at the final input target.

    Teacher forcing is permitted only while training.
    """

    training: bool

    def forward(
        self, x: object, y: object | None = None, teacher_forcing_ratio: float = 0.0
    ) -> object:
        """Forecast 72 hours from inputs shaped [B,168,n_features]."""
        ...
