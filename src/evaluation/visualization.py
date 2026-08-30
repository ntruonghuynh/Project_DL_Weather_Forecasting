"""Evaluation visualization interface; implementation belongs to TV6."""

from pathlib import Path


def save_forecast_figure(predictions: object, output_path: Path) -> Path:
    """Save a registered forecast figure when implemented."""
    raise NotImplementedError("TV6 must implement evaluation visualization")
