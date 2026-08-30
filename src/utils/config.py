"""Configuration loading interface; implementation belongs to TV1."""

from pathlib import Path
from typing import Any


def load_config(path: Path) -> dict[str, Any]:
    """Load and resolve a YAML configuration without mutating it."""
    raise NotImplementedError("TV1 must implement config resolution")
