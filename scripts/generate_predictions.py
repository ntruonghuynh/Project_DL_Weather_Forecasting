"""Bridge a training run to the prediction artifact scripts/evaluate.py expects.

Loads a checkpoint's model (via scripts/train.py::_build_model, not duplicated
here), runs inference-only forward passes (y=None, teacher_forcing_ratio=0.0)
over one dataset split, and writes:
  - a prediction NPZ (y_true, y_pred, target_timestamps, last_observed_target)
  - an evaluation metadata JSON satisfying scripts/evaluate.py's REQUIRED_METADATA

Never trains and never inverse-scales; both stay standardized/scaled, matching
the ``--scale standardized`` contract of scripts/evaluate.py.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train import _build_model  # noqa: E402
from src.config import get_processed_split_path  # noqa: E402
from src.data.dataloader import build_dataloader  # noqa: E402
from src.data.dataset import WeatherForecastDataset  # noqa: E402
from src.evaluation.registry import (  # noqa: E402
    authorize_final_test,
    record_final_test_prediction,
)
from src.serving.bundle import sha256_file  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.io import read_json, write_json  # noqa: E402
from src.utils.run_manager import update_run_status  # noqa: E402

DEFAULT_SCHEMA_PATH = PROJECT_ROOT / "artifacts" / "preprocessing" / "feature_schema.json"
DEFAULT_SCALER_PATH = PROJECT_ROOT / "artifacts" / "preprocessing" / "scaler.joblib"
SUPPORTED_SPLITS = ("validation", "test")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--split", choices=SUPPORTED_SPLITS, required=True)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "processed")
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA_PATH)
    parser.add_argument("--scaler", type=Path, default=DEFAULT_SCALER_PATH)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--output-metadata", type=Path, required=True)
    parser.add_argument("--model-version", type=str, default="1.0.0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--selection-manifest", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    checkpoint_path = args.run_dir / "best_checkpoint.pt"
    config_path = args.run_dir / "resolved_config.yaml"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    run_record = read_json(args.run_dir / "run_record.json")
    if run_record.get("status") != "completed" or run_record.get("smoke") is not False:
        raise ValueError("predictions require a completed real run with smoke=false")
    resolved_config = load_config(config_path)
    schema = read_json(args.schema)
    ordered_features = schema["ordered_features"]
    n_features = schema["n_features"]
    target_name = schema["target"]["name"]
    target_index = schema["target"]["index"]
    input_length = schema["window_contract"]["input_len_hours"]
    horizon = schema["window_contract"]["horizon_hours"]

    if args.output_npz.exists() or args.output_metadata.exists():
        raise FileExistsError("refusing to overwrite an existing prediction artifact")
    manifest_sha256 = None

    device = torch.device("cpu")
    model = _build_model(
        config=resolved_config,
        n_features=n_features,
        target_feature_index=target_index,
        input_length=input_length,
        horizon=horizon,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    checkpoint_run_id = checkpoint.get("metadata", {}).get("run_id")
    if checkpoint_run_id != run_record["run_id"]:
        raise ValueError(
            "checkpoint run_id does not match run_record: "
            f"{checkpoint_run_id!r} != {run_record['run_id']!r}"
        )
    schema_sha256 = sha256_file(args.schema)
    if schema_sha256 != run_record["schema_identity"].get("sha256"):
        raise ValueError("feature schema checksum does not match the training run")
    scaler_sha256 = sha256_file(args.scaler)
    if scaler_sha256 != run_record["dataset_identity"].get("scaler", {}).get("sha256"):
        raise ValueError("scaler checksum does not match the training run")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()

    split_path = get_processed_split_path(args.data_dir, args.split)
    if not split_path.is_file():
        raise FileNotFoundError(f"Processed {args.split} dataset not found: {split_path}")
    split_sha256 = sha256_file(split_path)
    split_identity = run_record["dataset_identity"].get(args.split, {})
    if args.split != "test" and split_sha256 != split_identity.get("sha256"):
        raise ValueError(
            f"processed {args.split} checksum does not match the training run"
        )
    df = pd.read_csv(split_path)
    dataset = WeatherForecastDataset(
        df=df, features=ordered_features, target=target_name,
        input_window=input_length, horizon=horizon,
    )
    loader = build_dataloader(dataset, batch_size=args.batch_size, shuffle=False)

    if args.split == "test":
        if args.selection_manifest is None:
            raise ValueError("test inference requires --selection-manifest")
        audit = authorize_final_test(args.selection_manifest, run_record["run_id"])
        manifest_sha256 = audit["manifest_sha256"]

    y_true_chunks: list[np.ndarray] = []
    y_pred_chunks: list[np.ndarray] = []
    timestamp_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            prediction = model(x, y=None, teacher_forcing_ratio=0.0)
            y_true_chunks.append(y.cpu().numpy())
            y_pred_chunks.append(prediction.cpu().numpy())
            timestamp_chunks.append(batch["target_timestamps"].numpy())

    y_true = np.concatenate(y_true_chunks, axis=0)
    y_pred = np.concatenate(y_pred_chunks, axis=0)
    target_timestamps = np.concatenate(timestamp_chunks, axis=0)
    last_observed_target = np.array(
        [dataset.last_observed_target(i) for i in range(len(dataset))], dtype=np.float32
    )

    expected_shape = (len(dataset), horizon, 1)
    if y_true.shape != expected_shape or y_pred.shape != expected_shape:
        raise ValueError(
            f"prediction contract violation: expected {expected_shape}, "
            f"got y_true={y_true.shape}, y_pred={y_pred.shape}"
        )
    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ValueError("prediction artifact contains NaN or infinite values")

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output_npz,
        y_true=y_true,
        y_pred=y_pred,
        target_timestamps=target_timestamps,
        last_observed_target=last_observed_target,
    )

    schema_hash = schema["schema_hash"]
    population_id = (
        f"jena_{args.split}_{split_sha256[:16]}_in{input_length}_out{horizon}"
    )
    metadata = {
        "run_id": run_record["run_id"],
        "model_name": resolved_config["model"]["name"],
        "model_version": args.model_version,
        "split": args.split,
        "population_id": population_id,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "resolved_config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "schema_hash": schema_hash,
        "scaler_sha256": scaler_sha256,
        "schema_sha256": schema_sha256,
        "dataset_sha256": split_sha256,
        "sample_count": len(dataset),
        "input_length": input_length,
        "horizon": horizon,
        "n_features": n_features,
        "manifest_sha256": manifest_sha256,
        "test_holdout_status": (
            "previously_exposed_during_development" if args.split == "test" else None
        ),
        "selection_used_test_metrics": False if args.split == "test" else None,
        "candidate_selected_from": "validation" if args.split == "test" else None,
        "prediction_schema": {
            "y_true": f"float32 [N,{horizon},1]",
            "y_pred": f"float32 [N,{horizon},1]",
            "target_timestamps": f"int64 [N,{horizon}]",
            "last_observed_target": "float32 [N]",
        },
    }
    write_json(args.output_metadata, metadata)
    if args.split == "test":
        record_final_test_prediction(
            args.selection_manifest,
            prediction_path=args.output_npz,
            test_population_id=population_id,
        )
    else:
        update_run_status(
            run_dir=args.run_dir,
            status="completed",
            predictions=args.output_npz,
        )

    print(f"N samples: {len(dataset)}")
    print(f"Saved predictions: {args.output_npz}")
    print(f"Saved metadata: {args.output_metadata}")


if __name__ == "__main__":
    main()
