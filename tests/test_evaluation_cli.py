"""End-to-end smoke test for the documented evaluation CLI."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_evaluate_cli_writes_traceable_validation_outputs(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.npz"
    base = 1_767_225_600
    timestamps = np.vstack(
        [base + np.arange(72) * 3600, base + 3600 + np.arange(72) * 3600]
    )
    np.savez(
        predictions,
        y_true=np.zeros((2, 72), dtype=np.float32),
        y_pred=np.ones((2, 72), dtype=np.float32),
        target_timestamps=timestamps,
        last_observed_target=np.array([0.0, 0.0], dtype=np.float32),
    )
    metadata = tmp_path / "metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "run_id": "validation-run",
                "model_name": "model",
                "model_version": "1.0",
                "split": "validation",
                "population_id": "population-v1",
                "checkpoint": "runs/validation-run/best.pt",
                "resolved_config": "runs/validation-run/resolved_config.json",
                "schema_hash": "schema-sha256",
                "scaler_sha256": "scaler-sha256",
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "evaluation"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate.py",
            "--predictions",
            str(predictions),
            "--metadata",
            str(metadata),
            "--output-dir",
            str(output),
            "--scale",
            "degC",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["overall"] == {"mae": 1.0, "mse": 1.0, "rmse": 1.0}
    assert metrics["baseline"] == {"mae": 0.0, "mse": 0.0, "rmse": 0.0}
    assert metrics["unit"] == "degC"
    assert len((output / "metrics_per_horizon.csv").read_text().splitlines()) == 73
    assert len((output / "figure_manifest.csv").read_text().splitlines()) == 3
    assert (output / "forecast.png").is_file()
    assert (output / "residuals.png").is_file()
