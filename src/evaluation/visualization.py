"""Traceable evaluation plots and figure-manifest persistence."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

FIGURE_MANIFEST_COLUMNS = [
    "figure_id", "path", "run_id", "model_id", "split", "description",
    "finding", "impact", "decision", "created_at",
]


def _require_metadata(**values: str) -> None:
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise ValueError(f"figure metadata is required: {missing}")


def save_forecast_figure(
    predictions: object,
    output_path: Path,
    *,
    run_id: str,
    model_id: str,
    split: str,
    max_points: int = 500,
) -> Path:
    """Plot actual and predicted temperatures from traceable record rows."""
    _require_metadata(run_id=run_id, model_id=model_id, split=split)
    frame = predictions if isinstance(predictions, pd.DataFrame) else pd.DataFrame(predictions)
    required = {"timestamp", "y_true", "y_pred"}
    if not required.issubset(frame.columns) or frame.empty:
        raise ValueError("predictions must contain non-empty timestamp/y_true/y_pred columns")
    view = frame.iloc[:max_points]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(view["timestamp"], view["y_true"], label="Actual", linewidth=1.5)
    ax.plot(view["timestamp"], view["y_pred"], label="Prediction", linewidth=1.2)
    ax.set_title(f"Forecast — model={model_id} run={run_id} split={split}")
    ax.set_xlabel("Target timestamp")
    ax.set_ylabel("Temperature (°C)")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_residual_figure(
    predictions: object, output_path: Path, *, run_id: str, model_id: str, split: str
) -> Path:
    """Plot residual against horizon in degrees Celsius."""
    _require_metadata(run_id=run_id, model_id=model_id, split=split)
    frame = predictions if isinstance(predictions, pd.DataFrame) else pd.DataFrame(predictions)
    required = {"horizon", "y_true", "y_pred"}
    if not required.issubset(frame.columns) or frame.empty:
        raise ValueError("predictions must contain horizon/y_true/y_pred")
    residual = frame["y_pred"] - frame["y_true"]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(frame["horizon"], residual, s=8, alpha=0.25, label="Residual")
    ax.axhline(0, color="black", linewidth=1, label="Zero error")
    ax.set_title(f"Residuals — model={model_id} run={run_id} split={split}")
    ax.set_xlabel("Forecast horizon (hours ahead)")
    ax.set_ylabel("Prediction − actual (°C)")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def register_figure(
    manifest_path: Path,
    *,
    figure_id: str,
    path: Path,
    run_id: str,
    model_id: str,
    split: str,
    description: str,
    finding: str,
    impact: str,
    decision: str,
    created_at: str,
) -> None:
    """Append one fully traceable figure row using the canonical schema."""
    _require_metadata(
        figure_id=figure_id, run_id=run_id, model_id=model_id, split=split,
        description=description, finding=finding, impact=impact, decision=decision,
        created_at=created_at,
    )
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    exists = manifest_path.exists() and manifest_path.stat().st_size > 0
    with manifest_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIGURE_MANIFEST_COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow(
            {
                "figure_id": figure_id, "path": str(path), "run_id": run_id,
                "model_id": model_id, "split": split, "description": description,
                "finding": finding, "impact": impact, "decision": decision,
                "created_at": created_at,
            }
        )
