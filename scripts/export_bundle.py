"""CLI to export a validated, versioned ModelBundle from training run artifacts."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.serving.bundle import build_bundle, load_bundle, verify_bundle  # noqa: E402
from src.utils.config import load_config  # noqa: E402

DEFAULT_SCALER = PROJECT_ROOT / "artifacts" / "preprocessing" / "scaler.joblib"
DEFAULT_FEATURE_SCHEMA = PROJECT_ROOT / "artifacts" / "preprocessing" / "feature_schema.json"


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
        required=True,
        help="JSON object of constructor arguments from the locked run config/schema.",
    )
    parser.add_argument(
        "--scaler",
        type=Path,
        default=DEFAULT_SCALER,
        help="Path to the fitted scaler (shared preprocessing artifact, not per-run).",
    )
    parser.add_argument(
        "--feature-schema",
        type=Path,
        default=DEFAULT_FEATURE_SCHEMA,
        help="Path to the locked feature schema (shared preprocessing artifact, not per-run).",
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
    model_kwargs: dict[str, Any],
    scaler: Path = DEFAULT_SCALER,
    feature_schema: Path = DEFAULT_FEATURE_SCHEMA,
) -> Path:
    """Collect artifacts from a run directory and export a verified ModelBundle.

    ``scaler``/``feature_schema`` are shared preprocessing artifacts (not
    written per-run by scripts/train.py), so they are passed in explicitly
    rather than discovered inside ``run_dir``. ``resolved_config`` is written
    by run_manager.py as ``resolved_config.yaml`` (not JSON); it is converted
    to a temporary JSON file here because build_bundle's destination file
    (``resolved_config.json``) is later parsed as JSON by load_bundle().
    """
    run_dir = Path(run_dir)
    output_dir = Path(output_dir)

    checkpoint = _find_file(run_dir, ["best_checkpoint.pt", "best.pt", "model.pt", "checkpoint.pt"])
    resolved_config_source = _find_file(run_dir, ["resolved_config.yaml", "resolved_config.json", "config.json"])
    metadata = _find_file(run_dir, ["run_record.json", "metadata.json", "run_metadata.json"])
    if not Path(scaler).is_file():
        raise FileNotFoundError(f"scaler artifact not found: {scaler}")
    if not Path(feature_schema).is_file():
        raise FileNotFoundError(f"feature schema artifact not found: {feature_schema}")

    print(f"Building ModelBundle for '{model_name}' (v{model_version}) from {run_dir}...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        if resolved_config_source.suffix in {".yaml", ".yml"}:
            resolved_config = Path(tmp_dir) / "resolved_config.json"
            resolved_config.write_text(
                json.dumps(load_config(resolved_config_source), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        else:
            resolved_config = resolved_config_source

        destination = build_bundle(
            output_dir,
            checkpoint=checkpoint,
            scaler=Path(scaler),
            feature_schema=Path(feature_schema),
            resolved_config=resolved_config,
            metadata=metadata,
            model_class=model_class,
            model_kwargs=model_kwargs,
            model_name=model_name,
            model_version=model_version,
            run_id=run_id,
        )

    manifest = verify_bundle(destination)
    load_bundle(destination)
    print(f"Bundle successfully created and verified at: {destination}")
    print(f"Files included: {list(manifest['files'].values())}")
    return destination


def main(arguments: list[str] | None = None) -> None:
    args = parse_arguments(arguments)
    try:
        kwargs = json.loads(args.model_kwargs)
        if not isinstance(kwargs, dict):
            raise ValueError("--model-kwargs must decode to a JSON object")
        export_bundle_from_run(
            args.run_dir,
            args.output_dir,
            model_class=args.model_class,
            model_name=args.model_name,
            model_version=args.model_version,
            run_id=args.run_id,
            model_kwargs=kwargs,
            scaler=args.scaler,
            feature_schema=args.feature_schema,
        )
    except Exception as error:
        print(f"Error exporting bundle: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
