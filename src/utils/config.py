"""Configuration loading interface; implementation belongs to TV1."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge update into base without mutating either dict."""
    result = copy.deepcopy(base)
    for key, value in update.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path: Path | str) -> dict[str, Any]:
    """Load and resolve a YAML configuration without mutating it.

    Supports recursive resolution of base configurations specified by 'defaults'.
    Resolves relative paths against the config directory or project root.
    """
    path = Path(path)
    if not path.is_file():
        # Try resolving relative to configs/ or project root
        candidate_configs = PROJECT_ROOT / "configs" / path.name
        candidate_root = PROJECT_ROOT / path
        if candidate_configs.is_file():
            path = candidate_configs
        elif candidate_root.is_file():
            path = candidate_root
        else:
            raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Configuration at {path} did not parse to a dictionary")

    # Resolve parent default configuration if present
    defaults_ref = config.get("defaults")
    if defaults_ref:
        defaults_path = path.parent / defaults_ref
        if not defaults_path.is_file():
            defaults_path = PROJECT_ROOT / "configs" / defaults_ref
        if not defaults_path.is_file():
            raise FileNotFoundError(
                f"Parent defaults configuration '{defaults_ref}' referenced by {path} not found"
            )

        parent_config = load_config(defaults_path)
        child_config = {k: v for k, v in config.items() if k != "defaults"}
        resolved_config = _deep_merge(parent_config, child_config)
    else:
        resolved_config = copy.deepcopy(config)

    return resolved_config
