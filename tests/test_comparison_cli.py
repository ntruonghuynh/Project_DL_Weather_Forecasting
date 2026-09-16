"""End-to-end checks for fair three-model validation comparison."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _metric(path: Path, model: str, rmse: float, *, scaler: str = "scaler") -> Path:
    payload = {
        "run_id": f"run-{model}",
        "model_name": model,
        "model_version": "1.0",
        "split": "validation",
        "population_id": "population-v1",
        "checkpoint": f"runs/run-{model}/best.pt",
        "resolved_config": f"runs/run-{model}/resolved_config.json",
        "schema_hash": "schema",
        "scaler_sha256": scaler,
        "unit": "degC",
        "sample_count": 100,
        "horizon": 72,
        "overall": {"mae": rmse - 0.5, "mse": rmse**2, "rmse": rmse},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _run(paths: list[Path], output: Path, manifest: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "scripts/compare_models.py",
            *map(str, paths),
            "--output",
            str(output),
            "--baseline-rmse",
            "3.0",
            "--population-id",
            "population-v1",
            "--selection-manifest",
            str(manifest),
            "--approved-by",
            "TV6",
            "--approved-by",
            "TV1",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def test_compare_cli_locks_best_of_three_fair_models(tmp_path: Path) -> None:
    paths = [
        _metric(tmp_path / "lstm.json", "lstm", 2.2),
        _metric(tmp_path / "attention.json", "attention", 2.0),
        _metric(tmp_path / "transformer.json", "transformer", 2.1),
    ]
    manifest = tmp_path / "selection.json"
    completed = _run(paths, tmp_path / "comparison.csv", manifest)
    assert completed.returncode == 0, completed.stderr
    selection = json.loads(manifest.read_text(encoding="utf-8"))
    assert selection["candidate"]["model_name"] == "attention"
    assert selection["locked"] is True


def test_compare_cli_rejects_scaler_mismatch(tmp_path: Path) -> None:
    paths = [
        _metric(tmp_path / "lstm.json", "lstm", 2.2),
        _metric(tmp_path / "attention.json", "attention", 2.0),
        _metric(tmp_path / "transformer.json", "transformer", 2.1, scaler="other"),
    ]
    completed = _run(paths, tmp_path / "comparison.csv", tmp_path / "selection.json")
    assert completed.returncode != 0
    assert "fair-comparison mismatch for scaler_sha256" in completed.stderr
