"""CLI to export a validated, versioned ModelBundle from training run artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from src.serving.bundle import build_bundle, verify_bundle


def parse_arguments(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package run artifacts into an immutable release ModelBundle."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Path to the training run directory containing checkpoint, scaler, config and schema.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Destination directory for the exported bundle.",
    )
    parser.add_argument(
        "--model-class",
        type=str,
        required=True,
        help=(
            "Fully-qualified model class name "
            "(e.g., 'src.models.attention_lstm_seq2seq.AttentionLSTMSeq2Seq')."
        ),
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Public model identifier (e.g., 'attention_lstm_seq2seq').",
    )
    parser.add_argument(
        "--model-version",
        type=str,
        default="1.0.0",
        help="Model semantic version (default: '1.0.0').",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        required=True,
        help="Immutable run identifier matching the run metadata.",
    )
    parser.add_argument(
        "--model-kwargs",
        type=str,
        default=None,
        help="Optional JSON string of keyword arguments passed to model constructor.",
    )
    return parser.parse_args(arguments)


def _find_file(directory: Path, candidates: list[str]) -> Path:
    for candidate in candidates:
        path = directory / candidate
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"Could not find any of {candidates} in {directory}"
    )


def export_bundle_from_run(
    run_dir: Path,
    output_dir: Path,
    *,
    model_class: str,
    model_name: str,
    model_version: str,
    run_id: str,
    model_kwargs: dict[str, Any] | None = None,
) -> Path:
    """Collect artifacts from a run directory and export a verified ModelBundle."""
    run_dir = Path(run_dir)
    output_dir = Path(output_dir)

    checkpoint = _find_file(run_dir, ["best.pt", "model.pt", "checkpoint.pt"])
    scaler = _find_file(run_dir, ["scaler.joblib", "scaler.pkl"])
    feature_schema = _find_file(run_dir, ["feature_schema.json"])
    resolved_config = _find_file(run_dir, ["resolved_config.json", "config.json"])
    metadata = _find_file(run_dir, ["metadata.json", "run_metadata.json"])

    if model_kwargs is None:
        config_data = json.loads(resolved_config.read_text(encoding="utf-8"))
        model_kwargs = config_data.get("model", {}).get("kwargs", {})

    print(f"Building ModelBundle for '{model_name}' (v{model_version}) from {run_dir}...")
    destination = build_bundle(
        output_dir,
        checkpoint=checkpoint,
        scaler=scaler,
        feature_schema=feature_schema,
        resolved_config=resolved_config,
        metadata=metadata,
        model_class=model_class,
        model_kwargs=model_kwargs,
        model_name=model_name,
        model_version=model_version,
        run_id=run_id,
    )

    manifest = verify_bundle(destination)
    print(f"Bundle successfully created and verified at: {destination}")
    print(f"Files included: {list(manifest['files'].values())}")
    return destination


def main(arguments: list[str] | None = None) -> None:
    args = parse_arguments(arguments)
    kwargs = json.loads(args.model_kwargs) if args.model_kwargs else None
    try:
        export_bundle_from_run(
            args.run_dir,
            args.output_dir,
            model_class=args.model_class,
            model_name=args.model_name,
            model_version=args.model_version,
            run_id=args.run_id,
            model_kwargs=kwargs,
        )
    except Exception as error:
        print(f"Error exporting bundle: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
