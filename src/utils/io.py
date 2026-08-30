"""Safe project I/O interfaces; implementation belongs to TV1."""

from pathlib import Path
from typing import Any


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON atomically when implemented."""
    raise NotImplementedError("TV1 must implement atomic JSON output")
