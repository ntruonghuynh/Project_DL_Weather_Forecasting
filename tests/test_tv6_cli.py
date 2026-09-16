"""Smoke tests that documented TV6 command-line entry points start from the repo root."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "script",
    [
        "scripts/evaluate.py",
        "scripts/compare_models.py",
        "scripts/export_bundle.py",
        "scripts/generate_demo_payload.py",
        "scripts/run_stack.py",
    ],
)
def test_documented_cli_help_starts(script: str) -> None:
    completed = subprocess.run(
        [sys.executable, script, "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout.lower()
