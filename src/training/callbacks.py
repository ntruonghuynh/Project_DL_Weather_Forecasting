"""Training callback contracts and implementations; belongs to training owners."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Protocol

import torch
from torch import nn


class TrainingCallback(Protocol):
    """Receive immutable run-aware training lifecycle events."""

    def on_epoch_end(self, epoch: int, metrics: dict[str, float]) -> None: ...


class EarlyStoppingCallback:
    """Stop training when a monitored metric stops improving."""

    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 5,
        min_delta: float = 1e-4,
        mode: str = "min",
    ) -> None:
        if patience <= 0:
            raise ValueError("patience must be a positive integer")
        if mode not in ("min", "max"):
            raise ValueError("mode must be either 'min' or 'max'")

        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.wait = 0
        self.should_stop = False

    def on_epoch_end(self, epoch: int, metrics: dict[str, float]) -> None:
        if self.monitor not in metrics:
            return

        current = metrics[self.monitor]
        improved = (
            (current < self.best_value - self.min_delta)
            if self.mode == "min"
            else (current > self.best_value + self.min_delta)
        )

        if improved:
            self.best_value = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.should_stop = True


class ModelCheckpointCallback:
    """Save best and last checkpoints based on a validation criterion."""

    def __init__(
        self,
        save_dir: Path | str,
        monitor: str = "val_loss",
        mode: str = "min",
        save_last: bool = True,
        min_delta: float = 1e-4,
    ) -> None:
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.save_last = save_last
        self.min_delta = min_delta

        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.best_epoch = -1
        self.best_checkpoint_path: Path | None = None
        self.last_checkpoint_path: Path | None = None

        self._model_ref: nn.Module | None = None
        self._optimizer_ref: torch.optim.Optimizer | None = None
        self._extra_metadata: dict[str, Any] = {}

    def attach(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Attach references to model, optimizer and metadata for checkpointing."""
        self._model_ref = model
        self._optimizer_ref = optimizer
        self._extra_metadata = dict(extra_metadata or {})

    def on_epoch_end(self, epoch: int, metrics: dict[str, float]) -> None:
        if self._model_ref is None:
            return

        checkpoint_data: dict[str, Any] = {
            "epoch": epoch,
            "metrics": metrics,
            "model_state_dict": self._model_ref.state_dict(),
            "metadata": self._extra_metadata,
        }
        if self._optimizer_ref is not None:
            checkpoint_data["optimizer_state_dict"] = self._optimizer_ref.state_dict()

        # Save last checkpoint
        if self.save_last:
            last_path = self.save_dir / "last_checkpoint.pt"
            torch.save(checkpoint_data, last_path)
            self.last_checkpoint_path = last_path

        # Check for improvement
        if self.monitor in metrics:
            current = metrics[self.monitor]
            improved = (
                (current < self.best_value - self.min_delta)
                if self.mode == "min"
                else (current > self.best_value + self.min_delta)
            )

            if improved:
                self.best_value = current
                self.best_epoch = epoch
                best_path = self.save_dir / "best_checkpoint.pt"
                checkpoint_data["best_value"] = self.best_value
                torch.save(checkpoint_data, best_path)
                self.best_checkpoint_path = best_path


class MetricLoggerCallback:
    """Log training and validation metrics to a structured CSV file."""

    def __init__(self, log_dir: Path | str, filename: str = "metrics.csv") -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.filepath = self.log_dir / filename
        self._headers_written = False

    def on_epoch_end(self, epoch: int, metrics: dict[str, float]) -> None:
        row_data = {"epoch": epoch, **metrics}
        fieldnames = list(row_data.keys())

        if not self._headers_written:
            write_header = not self.filepath.exists() or self.filepath.stat().st_size == 0
            with self.filepath.open("a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if write_header:
                    writer.writeheader()
                writer.writerow(row_data)
            self._headers_written = True
        else:
            with self.filepath.open("a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow(row_data)
