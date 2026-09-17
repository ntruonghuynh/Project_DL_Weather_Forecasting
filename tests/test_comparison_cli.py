"""End-to-end checks for fair three-model validation comparison."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from src.evaluation.registry import REGISTRY_COLUMNS

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _metric(path: Path, model: str, rmse: float, *, scaler: str = "scaler") -> Path:
    checkpoint = path.with_suffix(".pt")
    config = path.with_suffix(".yaml")
    prediction = path.with_suffix(".npz")
    checkpoint.write_bytes(f"checkpoint-{model}".encode())
    config.write_text(f"model: {model}\n", encoding="utf-8")
    prediction.write_bytes(f"prediction-{model}".encode())
    payload = {
        "run_id": f"run-{model}",
        "model_name": model,
        "model_version": "1.0",
        "split": "validation",
        "population_id": "population-v1",
        "checkpoint": str(checkpoint),
        "resolved_config": str(config),
        "schema_hash": "schema",
        "scaler_sha256": scaler,
        "schema_sha256": "AUTO",
        "dataset_sha256": "dataset",
        "input_length": 168,
        "unit": "degC",
        "sample_count": 100,
        "horizon": 72,
        "overall": {"mae": rmse - 0.5, "mse": rmse**2, "rmse": rmse},
        "baseline": {"mae": 2.0, "mse": 9.0, "rmse": 3.0},
        "prediction_artifact": str(prediction),
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _run(paths: list[Path], output: Path, manifest: Path) -> subprocess.CompletedProcess[str]:
    schema = output.parent / "schema.json"
    scaler = output.parent / "scaler.joblib"
    registry = output.parent / "registry.csv"
    schema.write_text(json.dumps({"schema_hash": "schema"}), encoding="utf-8")
    scaler.write_bytes(b"scaler")

    def sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    rows = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["schema_sha256"] = sha256(schema)
        if payload["scaler_sha256"] == "scaler":
            payload["scaler_sha256"] = sha256(scaler)
        path.write_text(json.dumps(payload), encoding="utf-8")
        row = {column: "" for column in REGISTRY_COLUMNS}
        row.update({
            "run_id": payload["run_id"],
            "status": "completed",
            "model_name": payload["model_name"],
            "split": "validation",
            "smoke": "False",
            "population_id": payload["population_id"],
            "schema_hash": payload["schema_hash"],
            "scaler_sha256": payload["scaler_sha256"],
            "checkpoint_path": payload["checkpoint"],
            "checkpoint_sha256": sha256(Path(payload["checkpoint"])),
            "config_path": payload["resolved_config"],
            "config_sha256": sha256(Path(payload["resolved_config"])),
            "predictions_path": payload["prediction_artifact"],
            "predictions_sha256": sha256(Path(payload["prediction_artifact"])),
            "metrics_path": str(path),
            "metrics_sha256": sha256(path),
            "mae_deg_c": payload["overall"]["mae"],
            "mse_deg_c2": payload["overall"]["mse"],
            "rmse_deg_c": payload["overall"]["rmse"],
            "baseline_rmse_deg_c": payload["baseline"]["rmse"],
        })
        rows.append(row)
    with registry.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REGISTRY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

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
            "--registry",
            str(registry),
            "--schema",
            str(schema),
            "--scaler",
            str(scaler),
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
    assert "scaler checksum does not match validation metrics" in completed.stderr
