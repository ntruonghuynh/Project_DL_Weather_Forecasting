"""Preprocessing interface; implementation belongs to TV2."""

from pathlib import Path
from typing import Any


def preprocess(raw_path: Path, output_dir: Path, config: dict[str, Any]) -> Path:
    """Create processed data without fitting transformations on validation/test data."""
    raise NotImplementedError("TV2 must implement leakage-safe preprocessing")
