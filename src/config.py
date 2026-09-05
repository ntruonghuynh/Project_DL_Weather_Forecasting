"""Single Source of Truth loader for configs/data.yaml.

Every TV2 data-pipeline policy value (timestamp column/frequency, target
column, sliding-window sizes, missing-data thresholds, split ratios,
scaling rule) is defined once in configs/data.yaml. validator.py,
preprocessing.py, dataset.py, and the notebooks all call
`load_data_config()` instead of hard-coding their own copy, so changing a
policy value in one place changes it everywhere.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_CONFIG_PATH = PROJECT_ROOT / "configs" / "data.yaml"


@lru_cache(maxsize=None)
def load_data_config(config_path: Path | str | None = None) -> dict[str, Any]:
    """Load and cache configs/data.yaml (or an explicit override path).

    Cached by path so every caller in the same process shares one parsed
    dict instead of re-reading the file, while still allowing tests to pass
    a different `config_path` to exercise alternate policies.
    """
    path = Path(config_path) if config_path is not None else DEFAULT_DATA_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Data policy config not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"{path} did not parse to a mapping")

    required_sections = ("timestamp", "target", "window", "missing", "split", "scaling")
    missing_sections = [s for s in required_sections if s not in config]
    if missing_sections:
        raise ValueError(f"{path} is missing required section(s): {missing_sections}")

    return config
