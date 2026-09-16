"""Safe project I/O interfaces; implementation belongs to TV1."""

from __future__ import annotations

import json
import os
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np


class ProjectJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder handling datetimes, Path objects, and numpy types."""

    def default(self, o: Any) -> Any:
        if isinstance(o, (datetime, date)):
            return o.isoformat()
        if isinstance(o, Path):
            return str(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def write_json(path: Path | str, payload: Any, indent: int = 2) -> None:
    """Write JSON atomically using a temporary file to avoid partial writes.

    Args:
        path: Target file path.
        payload: Python data structure to serialize.
        indent: Indentation level for pretty-printing.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")

    try:
        with temp_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=indent, ensure_ascii=False, cls=ProjectJSONEncoder)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


def read_json(path: Path | str) -> Any:
    """Read and deserialize JSON safely from disk."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
