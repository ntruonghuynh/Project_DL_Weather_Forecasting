"""Training orchestration and execution engine; owned by training contributors."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader

from .callbacks import (
    EarlyStoppingCallback,
    ModelCheckpointCallback,
    TrainingCallback,
)
from .losses import build_loss

logger = logging.getLogger(__name__)


def _extract_batch_tensors(batch: Any, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract x and y tensors from DataLoader batch and send them to device."""
    if isinstance(batch, dict):
        x = batch["x"].to(device)
        y = batch["y"].to(device)
    elif hasattr(batch, "x") and hasattr(batch, "y"):
        x = batch.x.to(device)
        y = batch.y.to(device)
    elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
        x = batch[0].to(device)
        y = batch[1].to(device)
    else:
        raise TypeError(f"Unrecognized batch format: {type(batch).__name__}")
    return x, y


class Trainer:
    """Coordinate model training and autoregressive validation in a reproducible loop."""

    def __init__(
        self,
        device: str | torch.device = "auto",
        loss_name: str = "mse",
        loss_kwargs: dict[str, Any] | None = None,
        callbacks: Sequence[TrainingCallback] | None = None,
    ) -> None:
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.loss_fn = build_loss(loss_name, **(loss_kwargs or {}))
        self.callbacks: list[TrainingCallback] = list(callbacks or [])

    def fit(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict[str, Any],
        run_dir: Path | str | None = None,
        checkpoint_metadata: dict[str, Any] | None = None,
    ) -> dict[str, float | int]:
        """Execute the training and autoregressive validation loop.

        Args:
            model: PyTorch model conforming to Model Contract v1.
            train_loader: DataLoader yielding training batches.
            val_loader: DataLoader yielding validation batches.
            config: Resolved configuration dictionary.
            run_dir: Optional path to the run directory for checkpoints and logs.

        Returns:
            Dictionary containing final and best validation metrics.
        """
        training_cfg = config.get("training", {})

        # Training hyperparameters
        epochs = int(training_cfg.get("epochs", 10))
        learning_rate = float(training_cfg.get("learning_rate", 1e-3))
        weight_decay = float(training_cfg.get("weight_decay", 1e-4))
        grad_clip = training_cfg.get("grad_clip", 1.0)
        grad_clip_val = float(grad_clip) if grad_clip is not None else None
        teacher_forcing_ratio = float(training_cfg.get("teacher_forcing_ratio", 0.0))

        # Teacher forcing decay schedule
        tf_decay = float(training_cfg.get("teacher_forcing_decay", 0.0))

        # Move model to device
        model = model.to(self.device)

        # Optimizer
        optimizer_name = training_cfg.get("optimizer", "adam").lower()
        if optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=learning_rate, weight_decay=weight_decay
            )
        else:
            optimizer = torch.optim.Adam(
                model.parameters(), lr=learning_rate, weight_decay=weight_decay
            )

        # Scheduler
        scheduler_name = training_cfg.get("scheduler")
        scheduler = None
        if scheduler_name == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=2
            )
        elif scheduler_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        # Attach callbacks if ModelCheckpointCallback is present
        for cb in self.callbacks:
            if isinstance(cb, ModelCheckpointCallback):
                cb.attach(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    extra_metadata={
                        "config": config,
                        "device": str(self.device),
                        **(checkpoint_metadata or {}),
                    },
                )

        best_val_loss = float("inf")
        best_epoch: int | None = None
        global_step = 0
        history: list[dict[str, float | int]] = []

        for epoch in range(1, epochs + 1):
            # Compute current teacher forcing ratio
            current_tf_ratio = max(0.0, teacher_forcing_ratio - (epoch - 1) * tf_decay)

            # 1. Training epoch
            model.train()
            train_loss_accum = 0.0
            train_batches = 0

            for batch in train_loader:
                x, y = _extract_batch_tensors(batch, self.device)
                optimizer.zero_grad()

                # In training: y can be passed for teacher forcing
                predictions = model(x, y=y, teacher_forcing_ratio=current_tf_ratio)
                loss = self.loss_fn(predictions, y)

                loss.backward()

                if grad_clip_val is not None and grad_clip_val > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_val)

                optimizer.step()
                global_step += 1

                train_loss_accum += loss.item()
                train_batches += 1

            avg_train_loss = train_loss_accum / max(train_batches, 1)

            # 2. Validation epoch (AUTOREGRESSIVE ONLY - NEVER PASS TARGET Y)
            model.eval()
            val_loss_accum = 0.0
            val_mae_accum = 0.0
            val_batches = 0

            with torch.no_grad():
                for batch in val_loader:
                    x, y = _extract_batch_tensors(batch, self.device)

                    # CRITICAL: Model Contract v1 forbids y in validation
                    predictions = model(x, y=None, teacher_forcing_ratio=0.0)
                    val_loss = self.loss_fn(predictions, y)

                    val_loss_accum += val_loss.item()
                    val_mae_accum += torch.mean(torch.abs(predictions - y)).item()
                    val_batches += 1

            avg_val_loss = val_loss_accum / max(val_batches, 1)
            avg_val_mae = val_mae_accum / max(val_batches, 1)

            # Learning rate step
            current_lr = optimizer.param_groups[0]["lr"]
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(avg_val_loss)
                else:
                    scheduler.step()

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch

            epoch_metrics = {
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "val_mae": avg_val_mae,
                "learning_rate": current_lr,
                "teacher_forcing_ratio": current_tf_ratio,
                "global_step": global_step,
            }
            history.append(epoch_metrics)

            logger.info(
                "Epoch %d/%d - train_loss: %.4f - val_loss: %.4f - val_mae: %.4f",
                epoch,
                epochs,
                avg_train_loss,
                avg_val_loss,
                avg_val_mae,
            )

            # Invoke callbacks
            stop_early = False
            for cb in self.callbacks:
                cb.on_epoch_end(epoch, epoch_metrics)
                if isinstance(cb, EarlyStoppingCallback) and cb.should_stop:
                    stop_early = True

            if stop_early:
                logger.info("Early stopping triggered at epoch %d", epoch)
                break

        return {
            "final_train_loss": history[-1]["train_loss"] if history else float("nan"),
            "final_val_loss": history[-1]["val_loss"] if history else float("nan"),
            "final_val_mae": history[-1]["val_mae"] if history else float("nan"),
            "best_val_loss": best_val_loss,
            "epochs_completed": len(history),
            "best_epoch": best_epoch if best_epoch is not None else 0,
            "global_step": global_step,
        }
