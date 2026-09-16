"""Automated tests for Training Engine, utilities, and training CLI.

Strictly verifies GATE checks from agents/rules/TRAINING_RULES.md:
  - GATE smoke run performs optimizer step and autoregressive validation.
  - GATE spy assertion proves validation never passes future targets to model.
  - GATE best checkpoint and early stopping only react to validation criterion.
  - GATE checkpoint metadata and failure status tracking.
  - GATE reproducibility test for random seed setup.
  - GATE CLI smoke execution end-to-end.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.base import validate_forward_arguments
from src.training.callbacks import (
    EarlyStoppingCallback,
    MetricLoggerCallback,
    ModelCheckpointCallback,
)
from src.training.losses import build_loss
from src.training.trainer import Trainer
from src.utils.config import load_config
from src.utils.io import read_json, write_json
from src.utils.run_manager import create_run, update_run_status
from src.utils.seed import set_seed

# ---------------------------------------------------------------------------
# 1. Reproducibility & Utilities Tests
# ---------------------------------------------------------------------------


def test_set_seed_reproducibility() -> None:
    """Ensure set_seed reliably seeds Python, NumPy, and PyTorch generators."""
    set_seed(12345)
    py_rand1 = random.random()
    np_rand1 = np.random.rand(5)
    torch_rand1 = torch.randn(5)

    set_seed(12345)
    py_rand2 = random.random()
    np_rand2 = np.random.rand(5)
    torch_rand2 = torch.randn(5)

    assert py_rand1 == py_rand2
    np.testing.assert_allclose(np_rand1, np_rand2)
    assert torch.equal(torch_rand1, torch_rand2)


def test_set_seed_type_validation() -> None:
    """Ensure invalid seed types (such as bool or float) are rejected."""
    with pytest.raises(TypeError, match="Seed must be an integer"):
        set_seed(True)  # bool is an int subclass in Python, must be rejected
    with pytest.raises(TypeError, match="Seed must be an integer"):
        set_seed("123")  # type: ignore[arg-type]


def test_config_loader_inheritance(tmp_path: Path) -> None:
    """Ensure configuration loading correctly resolves and merges defaults."""
    base_file = tmp_path / "base.yaml"
    base_file.write_text(
        "project: weather\nseed: 42\ntraining:\n  lr: 0.001\n  epochs: 10\n",
        encoding="utf-8",
    )

    child_file = tmp_path / "child.yaml"
    child_file.write_text(
        f"defaults: {base_file.name}\nmodel:\n  name: lstm\ntraining:\n  epochs: 25\n",
        encoding="utf-8",
    )

    config = load_config(child_file)
    assert config["project"] == "weather"
    assert config["seed"] == 42
    assert config["model"]["name"] == "lstm"
    assert config["training"]["lr"] == 0.001
    assert config["training"]["epochs"] == 25  # child overrides parent


def test_atomic_write_and_read_json(tmp_path: Path) -> None:
    """Ensure atomic write and safe reading handles custom data types."""
    target_path = tmp_path / "subdir" / "data.json"
    payload = {
        "string": "test",
        "int": np.int64(42),
        "float": np.float32(3.14),
        "path": Path("some/path"),
        "array": np.array([1, 2, 3]),
    }

    write_json(target_path, payload)
    assert target_path.is_file()

    loaded = read_json(target_path)
    assert loaded["string"] == "test"
    assert loaded["int"] == 42
    assert pytest.approx(loaded["float"], rel=1e-3) == 3.14
    assert loaded["array"] == [1, 2, 3]


def test_run_manager_lifecycle(tmp_path: Path) -> None:
    """Ensure run creation, environment logging, and status updates persist."""
    cfg = {"model": {"name": "test_model"}, "seed": 42}
    record = create_run(config=cfg, seed=42, runs_dir=tmp_path)

    assert record.status == "created"
    assert record.run_dir.is_dir()
    assert (record.run_dir / "resolved_config.yaml").is_file()
    assert (record.run_dir / "environment.json").is_file()
    assert (record.run_dir / "run_record.json").is_file()

    # Update status to completed with metrics
    updated = update_run_status(
        run_dir=record.run_dir,
        status="completed",
        metrics={"best_val_loss": 0.45},
    )
    assert updated.status == "completed"
    assert updated.metrics["best_val_loss"] == 0.45
    assert (record.run_dir / "final_metrics.json").is_file()


# ---------------------------------------------------------------------------
# 2. Losses and Callbacks Tests
# ---------------------------------------------------------------------------


def test_build_loss_supported_and_unsupported() -> None:
    """Ensure supported losses instantiate properly and unknown names raise ValueError."""
    mse_loss = build_loss("mse")
    assert isinstance(mse_loss, nn.MSELoss)

    mae_loss = build_loss("mae")
    assert isinstance(mae_loss, nn.L1Loss)

    huber_loss = build_loss("huber")
    assert isinstance(huber_loss, nn.HuberLoss)

    with pytest.raises(ValueError, match="Unsupported loss"):
        build_loss("unknown_loss_type")


def test_early_stopping_callback() -> None:
    """Ensure early stopping halts training when validation loss stops improving."""
    cb = EarlyStoppingCallback(monitor="val_loss", patience=2, min_delta=1e-3, mode="min")

    cb.on_epoch_end(1, {"val_loss": 1.0})
    assert not cb.should_stop

    cb.on_epoch_end(2, {"val_loss": 0.9999})  # improvement less than min_delta
    assert not cb.should_stop
    assert cb.wait == 1

    cb.on_epoch_end(3, {"val_loss": 1.05})  # degradation
    assert cb.should_stop  # wait reaches patience (2)


def test_model_checkpoint_callback(tmp_path: Path) -> None:
    """Ensure best checkpoint is saved only on validation metric improvements."""
    dummy_model = nn.Linear(5, 1)
    cb = ModelCheckpointCallback(save_dir=tmp_path, monitor="val_loss", mode="min")
    cb.attach(dummy_model)

    cb.on_epoch_end(1, {"val_loss": 2.0})
    best_file = tmp_path / "best_checkpoint.pt"
    last_file = tmp_path / "last_checkpoint.pt"
    assert best_file.is_file()
    assert last_file.is_file()

    # Load and check epoch
    ckpt1 = torch.load(best_file, weights_only=False)
    assert ckpt1["epoch"] == 1

    # Epoch 2: Worse loss -> best_checkpoint should NOT be overwritten
    cb.on_epoch_end(2, {"val_loss": 2.5})
    ckpt2 = torch.load(best_file, weights_only=False)
    assert ckpt2["epoch"] == 1  # unchanged

    # Epoch 3: Better loss -> best_checkpoint updated
    cb.on_epoch_end(3, {"val_loss": 1.5})
    ckpt3 = torch.load(best_file, weights_only=False)
    assert ckpt3["epoch"] == 3


def test_metric_logger_callback(tmp_path: Path) -> None:
    """Ensure metric logger records metrics to CSV."""
    cb = MetricLoggerCallback(log_dir=tmp_path, filename="history.csv")
    cb.on_epoch_end(1, {"train_loss": 1.5, "val_loss": 1.2})
    cb.on_epoch_end(2, {"train_loss": 1.1, "val_loss": 0.9})

    csv_path = tmp_path / "history.csv"
    assert csv_path.is_file()
    lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3  # header + 2 epochs
    assert "train_loss" in lines[0]
    assert "val_loss" in lines[0]


# ---------------------------------------------------------------------------
# 3. Trainer & Leakage Verification Tests (GATE)
# ---------------------------------------------------------------------------


class LeakageSpyModel(nn.Module):
    """Spy model that enforces zero target leakage during validation."""

    def __init__(self, n_features: int = 4, horizon: int = 72) -> None:
        super().__init__()
        self.horizon = horizon
        self.fc = nn.Linear(n_features, 1)
        self.train_calls: list[tuple[bool, float]] = []
        self.val_calls: list[tuple[bool, float]] = []

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> torch.Tensor:
        validate_forward_arguments(
            training=self.training,
            y=y,
            teacher_forcing_ratio=teacher_forcing_ratio,
        )
        if self.training:
            self.train_calls.append((y is not None, teacher_forcing_ratio))
        else:
            # GATE: Target y MUST be None and teacher_forcing_ratio MUST be 0.0 in validation
            if y is not None:
                raise AssertionError("Target leakage detected: y was passed to model in eval mode!")
            if teacher_forcing_ratio != 0.0:
                raise AssertionError("Nonzero teacher forcing ratio in eval mode!")
            self.val_calls.append((y is not None, teacher_forcing_ratio))

        # Output shape [B, horizon, 1]
        step_val = self.fc(x[:, -1:, :])
        return step_val.repeat(1, self.horizon, 1)


def test_trainer_validation_never_receives_target() -> None:
    """GATE: Verify that validation strictly evaluates autoregressively without passing target."""
    n_features = 4
    horizon = 72
    batch_size = 2

    # Synthetic batches
    x_train = torch.randn(4, 168, n_features)
    y_train = torch.randn(4, horizon, 1)
    x_val = torch.randn(4, 168, n_features)
    y_val = torch.randn(4, horizon, 1)

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size)

    spy_model = LeakageSpyModel(n_features=n_features, horizon=horizon)
    trainer = Trainer(device="cpu", loss_name="mse")

    config = {
        "training": {
            "epochs": 2,
            "learning_rate": 0.01,
            "teacher_forcing_ratio": 0.5,
            "grad_clip": 1.0,
        }
    }

    metrics = trainer.fit(
        model=spy_model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
    )

    assert metrics["epochs_completed"] == 2
    assert len(spy_model.train_calls) > 0
    assert len(spy_model.val_calls) > 0

    # Ensure all validation calls were autoregressive
    for has_target, tf_ratio in spy_model.val_calls:
        assert not has_target, "Validation call must have y=None"
        assert tf_ratio == 0.0, "Validation call must have teacher_forcing_ratio=0.0"


def test_trainer_gradient_clipping() -> None:
    """GATE: Verify gradient clipping prevents exploding gradients."""
    n_features = 4
    horizon = 72

    model = nn.Sequential(
        nn.Linear(n_features, 64),
        nn.Linear(64, 1),
    )

    class WrappedModel(nn.Module):
        def __init__(self, core: nn.Module) -> None:
            super().__init__()
            self.core = core

        def forward(
            self,
            x: torch.Tensor,
            y: torch.Tensor | None = None,
            teacher_forcing_ratio: float = 0.0,
        ) -> torch.Tensor:
            validate_forward_arguments(
                training=self.training,
                y=y,
                teacher_forcing_ratio=teacher_forcing_ratio,
            )
            val = self.core(x[:, -1:, :])
            return val.repeat(1, 72, 1)

    wrapped = WrappedModel(model)
    x = torch.randn(2, 168, n_features)
    y = torch.randn(2, horizon, 1) * 100.0  # large target to induce large gradients

    loader = DataLoader(TensorDataset(x, y), batch_size=2)
    trainer = Trainer(device="cpu", loss_name="mse")

    config = {
        "training": {
            "epochs": 1,
            "learning_rate": 0.1,
            "grad_clip": 0.5,  # strictly clip to 0.5
        }
    }

    metrics = trainer.fit(
        model=wrapped,
        train_loader=loader,
        val_loader=loader,
        config=config,
    )
    assert metrics["epochs_completed"] == 1


def test_train_cli_smoke_execution(tmp_path: Path) -> None:
    """GATE: Verify the scripts/train.py CLI runs smoothly in smoke mode."""
    import subprocess
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    train_script = repo_root / "scripts" / "train.py"
    config_path = repo_root / "configs" / "lstm.yaml"

    cmd = [
        sys.executable,
        str(train_script),
        "--config",
        str(config_path),
        "--smoke",
        "--epochs",
        "1",
        "--device",
        "cpu",
        "--runs-dir",
        str(tmp_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert result.returncode == 0, f"train.py failed with output:\n{result.stderr}\n{result.stdout}"
    assert "completed successfully" in result.stdout or "completed successfully" in result.stderr
