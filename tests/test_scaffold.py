"""Smoke tests for the repository scaffold."""

from importlib import import_module
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_packages_are_importable() -> None:
    """The top-level source and API packages can be imported."""
    assert import_module("src") is not None
    assert import_module("api") is not None


def test_required_scaffold_files_exist() -> None:
    """Required configuration, documentation, and entry-point files exist."""
    required_paths = (
        "configs/base.yaml",
        "configs/lstm.yaml",
        "configs/attention.yaml",
        "configs/transformer.yaml",
        "docs/WORKFLOW.md",
        "docs/EXPERIMENT_RULES.md",
        "docs/CODEBASE_AUDIT.md",
        "scripts/prepare_data.py",
        "scripts/train.py",
        "scripts/evaluate.py",
        "scripts/compare_models.py",
    )

    missing_paths = [path for path in required_paths if not (REPOSITORY_ROOT / path).is_file()]

    assert not missing_paths, f"Missing required scaffold files: {missing_paths}"
