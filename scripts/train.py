"""Training CLI for weather forecasting sequence-to-sequence models."""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEFAULT_DATA_CONFIG_PATH, get_processed_split_path  # noqa: E402
from src.data.dataloader import build_dataloader  # noqa: E402
from src.data.dataset import WeatherForecastDataset  # noqa: E402
from src.models.attention_lstm_seq2seq import AttentionLSTMSeq2Seq  # noqa: E402
from src.models.seq2seq_lstm import Seq2SeqLSTM  # noqa: E402
from src.models.transformer import WeatherTransformer  # noqa: E402
from src.training.callbacks import (  # noqa: E402
    EarlyStoppingCallback,
    MetricLoggerCallback,
    ModelCheckpointCallback,
)
from src.training.trainer import Trainer  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.io import read_json  # noqa: E402
from src.utils.run_manager import create_run, update_run_status  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

DEFAULT_SCHEMA_PATH = PROJECT_ROOT / "artifacts" / "preprocessing" / "feature_schema.json"
DEFAULT_SCALER_PATH = PROJECT_ROOT / "artifacts" / "preprocessing" / "scaler.joblib"
DEFAULT_SPLIT_METADATA_PATH = (
    PROJECT_ROOT / "artifacts" / "preprocessing" / "split_metadata.json"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_cli")


def _resolve_schema(schema_path: Path | str | None = None) -> dict[str, Any]:
    path = _resolve_schema_path(schema_path)
    return read_json(path)


def _resolve_schema_path(schema_path: Path | str | None = None) -> Path:
    path = Path(schema_path) if schema_path else DEFAULT_SCHEMA_PATH
    if not path.is_file():
        alt_path = PROJECT_ROOT / "artifacts" / "feature_schema.json"
        if alt_path.is_file():
            path = alt_path
        else:
            raise FileNotFoundError(f"Feature schema not found at {path} or {alt_path}")
    return path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_identity(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _build_model(
    config: dict[str, Any],
    n_features: int,
    target_feature_index: int,
    input_length: int = 168,
    horizon: int = 72,
) -> nn.Module:
    model_cfg = config.get("model", {})
    name = model_cfg.get("name", "").lower()

    if "attention" in name:
        return AttentionLSTMSeq2Seq(
            n_features=n_features,
            target_feature_index=target_feature_index,
            hidden_size=int(model_cfg.get("hidden_size", 128)),
            num_layers=int(model_cfg.get("num_layers", 1)),
            dropout=float(model_cfg.get("dropout", 0.0)),
            horizon=horizon,
            attention_dim=model_cfg.get("attention_dim"),
        )
    elif "transformer" in name:
        return WeatherTransformer(
            n_features=n_features,
            target_feature_index=target_feature_index,
            d_model=int(model_cfg.get("d_model", 128)),
            nhead=int(model_cfg.get("nhead", 8)),
            num_encoder_layers=int(model_cfg.get("num_encoder_layers", 3)),
            num_decoder_layers=int(model_cfg.get("num_decoder_layers", 3)),
            dim_feedforward=int(model_cfg.get("dim_feedforward", 512)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            horizon=horizon,
            input_length=input_length,
        )
    elif "lstm" in name:
        return Seq2SeqLSTM(
            n_features=n_features,
            target_feature_index=target_feature_index,
            hidden_size=int(model_cfg.get("hidden_size", 128)),
            num_layers=int(model_cfg.get("num_layers", 1)),
            dropout=float(model_cfg.get("dropout", 0.0)),
            horizon=horizon,
        )
    else:
        raise ValueError(f"Unsupported model name in config: '{name}'")


def _build_synthetic_loaders(
    n_features: int, input_len: int, horizon: int
) -> tuple[DataLoader, DataLoader]:
    """Build a tiny synthetic dataset for smoke runs."""
    x = torch.randn(8, input_len, n_features, dtype=torch.float32)
    y = torch.randn(8, horizon, 1, dtype=torch.float32)
    ds = TensorDataset(x, y)
    train_loader = DataLoader(ds, batch_size=4, shuffle=True)
    val_loader = DataLoader(ds, batch_size=4, shuffle=False)
    return train_loader, val_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Jena Climate Seq2Seq model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run quick smoke test with synthetic batches",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Compute device",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size",
    )
    parser.add_argument(
        "--runs-dir",
        type=str,
        default="runs",
        help="Directory to store training runs",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed",
        help="Directory containing the canonical processed split files",
    )
    parser.add_argument(
        "--schema",
        type=str,
        default=str(DEFAULT_SCHEMA_PATH),
        help="Feature schema artifact used to construct the model and datasets",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Subsampling stride for dataset windows (default from config or 1)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 1. Load resolved configuration
    config = load_config(args.config)
    training_cfg = config.setdefault("training", {})

    if args.epochs is not None:
        training_cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        training_cfg["batch_size"] = args.batch_size
    if args.smoke and "epochs" not in training_cfg:
        training_cfg["epochs"] = 1

    seed = int(config.get("seed", 42))
    set_seed(seed)

    # 2. Extract schema information
    schema_path = _resolve_schema_path(args.schema)
    schema = _resolve_schema(schema_path)
    ordered_features = schema["ordered_features"]
    n_features = schema.get("n_features", len(ordered_features))
    target_info = schema.get("target", {})
    target_name = target_info.get("name", "T (degC)")
    target_index = target_info.get("index", ordered_features.index(target_name))
    window_cfg = schema.get("window_contract", {})
    input_length = int(window_cfg.get("input_len_hours", 168))
    horizon = int(window_cfg.get("horizon_hours", 72))

    configured_input_length = int(config.get("data", {}).get("input_hours", input_length))
    configured_horizon = int(config.get("data", {}).get("forecast_hours", horizon))
    if (configured_input_length, configured_horizon) != (input_length, horizon):
        raise ValueError(
            "Model config and feature schema disagree on the locked window contract: "
            f"config=({configured_input_length}, {configured_horizon}), "
            f"schema=({input_length}, {horizon})"
        )

    train_stride = int(
        args.stride if args.stride is not None else training_cfg.get("data_stride", 1)
    )
    if train_stride <= 0:
        raise ValueError("training data stride must be positive")
    training_cfg["data_stride"] = train_stride
    config["runtime"] = {
        "smoke": bool(args.smoke),
        "data_dir": str(args.data_dir),
        "schema_path": str(schema_path),
    }

    schema_identity = {
        **_artifact_identity(schema_path),
        "schema_hash": schema["schema_hash"],
        "n_features": n_features,
        "target_name": target_name,
        "target_index": target_index,
        "input_length": input_length,
        "horizon": horizon,
    }
    if args.smoke:
        dataset_identity = {"kind": "synthetic_smoke", "smoke": True}
    else:
        data_dir = Path(args.data_dir)
        if not data_dir.is_absolute():
            data_dir = PROJECT_ROOT / data_dir
        train_csv = get_processed_split_path(data_dir, "train")
        val_csv = get_processed_split_path(data_dir, "validation")
        required_artifacts = {
            "train": train_csv,
            "validation": val_csv,
            "scaler": DEFAULT_SCALER_PATH,
            "split_metadata": DEFAULT_SPLIT_METADATA_PATH,
            "data_config": DEFAULT_DATA_CONFIG_PATH,
        }
        missing = [str(path) for path in required_artifacts.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Required real-data artifacts are missing: " + ", ".join(missing)
            )
        dataset_identity = {
            "kind": "jena_processed",
            "smoke": False,
            "training_stride": train_stride,
            **{
                name: _artifact_identity(path)
                for name, path in required_artifacts.items()
            },
        }

    # 3. Create immutable run record
    record = create_run(
        config=config,
        seed=seed,
        runs_dir=args.runs_dir,
        smoke=args.smoke,
        dataset_identity=dataset_identity,
        schema_identity=schema_identity,
    )
    run_dir = record.run_dir
    logger.info("Initialized run %s in %s", record.run_id, run_dir)
    update_run_status(run_dir=run_dir, status="running")

    try:
        # 4. Build Model
        model = _build_model(
            config=config,
            n_features=n_features,
            target_feature_index=target_index,
            input_length=input_length,
            horizon=horizon,
        )
        logger.info("Built model: %s", type(model).__name__)

        # 5. Build DataLoaders
        if args.smoke:
            logger.info("Smoke mode enabled: generating synthetic DataLoaders")
            train_loader, val_loader = _build_synthetic_loaders(
                n_features=n_features, input_len=input_length, horizon=horizon
            )
        else:
            logger.info("Loading preprocessed training and validation CSV files...")
            train_df = pd.read_csv(train_csv)
            val_df = pd.read_csv(val_csv)

            train_ds = WeatherForecastDataset(
                df=train_df,
                features=ordered_features,
                target=target_name,
                input_window=input_length,
                horizon=horizon,
                stride=train_stride,
            )
            val_ds = WeatherForecastDataset(
                df=val_df,
                features=ordered_features,
                target=target_name,
                input_window=input_length,
                horizon=horizon,
            )

            if len(train_ds) == 0 or len(val_ds) == 0:
                raise ValueError(
                    "Real processed data produced an empty training or validation dataset"
                )
            logger.info(
                "Real data windows: train=%d (stride=%d), validation=%d (stride=1)",
                len(train_ds),
                train_stride,
                len(val_ds),
            )

            batch_size = int(training_cfg.get("batch_size", 64))
            train_loader = build_dataloader(train_ds, batch_size=batch_size, shuffle=True)
            val_loader = build_dataloader(val_ds, batch_size=batch_size, shuffle=False)

        # 6. Configure Callbacks
        patience = int(training_cfg.get("patience", 5))
        checkpoint_cb = ModelCheckpointCallback(
            save_dir=run_dir,
            monitor="val_loss",
            mode="min",
            save_last=True,
        )
        early_stopping_cb = EarlyStoppingCallback(
            monitor="val_loss",
            patience=patience,
            mode="min",
        )
        logger_cb = MetricLoggerCallback(log_dir=run_dir)

        # 7. Initialize Trainer and Fit
        trainer = Trainer(
            device=args.device,
            loss_name=str(training_cfg.get("loss", "mse")),
            callbacks=[checkpoint_cb, early_stopping_cb, logger_cb],
        )

        logger.info("Starting model training loop...")
        metrics = trainer.fit(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            run_dir=run_dir,
            checkpoint_metadata={
                "run_id": record.run_id,
                "smoke": bool(args.smoke),
                "dataset_identity": dataset_identity,
                "schema_identity": schema_identity,
            },
        )

        # 8. Mark Run Completed
        update_run_status(
            run_dir=run_dir,
            status="completed",
            metrics=metrics,
            best_checkpoint=checkpoint_cb.best_checkpoint_path,
            last_checkpoint=checkpoint_cb.last_checkpoint_path,
        )
        logger.info("Run %s completed successfully. Best val loss: %.4f",
                    record.run_id, metrics.get("best_val_loss", float("nan")))

    except Exception as exc:
        logger.error("Run %s failed: %s", record.run_id, exc, exc_info=True)
        update_run_status(run_dir=run_dir, status="failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
