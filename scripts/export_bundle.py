"""Export the locked validation-selected candidate as a production ModelBundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.serving.bundle import build_bundle, load_bundle, sha256_file, verify_bundle  # noqa: E402
from src.utils.config import load_config  # noqa: E402

DEFAULT_SELECTION = PROJECT_ROOT / "experiments" / "selection_manifest.json"
DEFAULT_FINAL_AUDIT = PROJECT_ROOT / "experiments" / "final_test_audit.json"
DEFAULT_PREPROCESSING_CONFIG = PROJECT_ROOT / "configs" / "data.yaml"


def _canonical_json_hash(payload: dict[str, Any], checksum_field: str) -> str:
    value = dict(payload)
    stored = value.pop(checksum_field, None)
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    actual = hashlib.sha256(canonical).hexdigest()
    if stored != actual:
        raise ValueError(f"invalid {checksum_field}: expected {stored}, recomputed {actual}")
    return actual


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _model_spec(config: dict[str, Any], schema: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    model = config["model"]
    name = model["name"]
    common = {
        "n_features": int(schema["n_features"]),
        "target_feature_index": int(schema["target"]["index"]),
        "horizon": int(schema["window_contract"]["horizon_hours"]),
    }
    if name == "seq2seq_attention":
        return "src.models.attention_lstm_seq2seq.AttentionLSTMSeq2Seq", {
            **common,
            "hidden_size": int(model.get("hidden_size", 128)),
            "num_layers": int(model.get("num_layers", 1)),
            "dropout": float(model.get("dropout", 0.0)),
            "attention_dim": model.get("attention_dim"),
        }
    raise ValueError(f"locked candidate model is unsupported by exporter: {name}")


def parse_arguments(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--final-test-audit", type=Path, default=DEFAULT_FINAL_AUDIT)
    parser.add_argument(
        "--preprocessing-config", type=Path, default=DEFAULT_PREPROCESSING_CONFIG
    )
    parser.add_argument("--model-version", default="1.0.0")
    return parser.parse_args(arguments)


def export_locked_bundle(
    output_dir: Path,
    *,
    selection_manifest: Path = DEFAULT_SELECTION,
    final_test_audit: Path = DEFAULT_FINAL_AUDIT,
    preprocessing_config: Path = DEFAULT_PREPROCESSING_CONFIG,
    model_version: str = "1.0.0",
) -> Path:
    """Verify locked provenance, build one immutable bundle, and strict-load it."""
    selection_manifest = Path(selection_manifest)
    final_test_audit = Path(final_test_audit)
    manifest = _load_json(selection_manifest)
    manifest_sha256 = _canonical_json_hash(manifest, "manifest_sha256")
    if (
        manifest.get("locked") is not True
        or manifest.get("selection_split") != "validation"
        or manifest.get("selection_metric") != "rmse_deg_c"
    ):
        raise ValueError("selection manifest is not a locked validation RMSE selection")

    audit = _load_json(final_test_audit)
    audit_sha256 = _canonical_json_hash(audit, "audit_sha256")
    candidate = manifest["candidate"]
    if (
        audit.get("status") != "completed"
        or audit.get("manifest_sha256") != manifest_sha256
        or audit.get("selected_run_id") != candidate["run_id"]
        or audit.get("selection_used_test_metrics") is not False
        or audit.get("candidate_selected_from") != "validation"
    ):
        raise ValueError("final-test audit does not authorize the locked candidate release")

    bindings = manifest["artifact_bindings"]
    for name, binding in bindings.items():
        path = Path(binding["path"])
        if not path.is_file() or sha256_file(path) != binding["sha256"]:
            raise ValueError(f"selection binding mismatch for {name}")

    checkpoint = Path(candidate["checkpoint"])
    resolved_config = Path(candidate["config"])
    feature_schema = Path(candidate["schema_path"])
    scaler = Path(candidate["scaler_path"])
    if audit["checkpoint_sha256"] != sha256_file(checkpoint):
        raise ValueError("final audit checkpoint checksum mismatch")
    if audit["config_sha256"] != sha256_file(resolved_config):
        raise ValueError("final audit config checksum mismatch")
    if audit["schema_sha256"] != sha256_file(feature_schema):
        raise ValueError("final audit schema checksum mismatch")
    if audit["scaler_sha256"] != sha256_file(scaler):
        raise ValueError("final audit scaler checksum mismatch")

    validation_metrics_path = Path(bindings["validation_metrics"]["path"])
    final_metrics_path = Path(audit["metrics_path"])
    if sha256_file(final_metrics_path) != audit["metrics_sha256"]:
        raise ValueError("final-test metrics checksum mismatch")
    validation_metrics = _load_json(validation_metrics_path)
    final_metrics = _load_json(final_metrics_path)
    schema = _load_json(feature_schema)
    config = load_config(resolved_config)
    model_class, model_kwargs = _model_spec(config, schema)

    metadata = {
        "release_stage": "production",
        "model_name": candidate["model_name"],
        "model_version": model_version,
        "run_id": candidate["run_id"],
        "input_length": int(schema["window_contract"]["input_len_hours"]),
        "horizon": int(schema["window_contract"]["horizon_hours"]),
        "target": schema["target"]["name"],
        "target_unit": schema["target"]["unit"],
        "checkpoint_sha256": sha256_file(checkpoint),
        "config_sha256": sha256_file(resolved_config),
        "schema_sha256": sha256_file(feature_schema),
        "schema_hash": schema["schema_hash"],
        "scaler_sha256": sha256_file(scaler),
        "selection_manifest_sha256": manifest_sha256,
        "final_test_audit_sha256": audit_sha256,
        "validation_metrics": validation_metrics["overall"],
        "final_test_metrics": final_metrics["overall"],
        "metric_units": final_metrics["metric_units"],
        "test_holdout_status": audit["test_holdout_status"],
        "selection_used_test_metrics": False,
        "candidate_selected_from": "validation",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    with tempfile.TemporaryDirectory() as temporary_directory:
        metadata_path = Path(temporary_directory) / "metadata.json"
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        destination = build_bundle(
            Path(output_dir),
            checkpoint=checkpoint,
            scaler=scaler,
            feature_schema=feature_schema,
            resolved_config=resolved_config,
            preprocessing_config=Path(preprocessing_config),
            selection_manifest=selection_manifest,
            final_test_audit=final_test_audit,
            metadata=metadata_path,
            model_class=model_class,
            model_kwargs=model_kwargs,
            model_name=candidate["model_name"],
            model_version=model_version,
            run_id=candidate["run_id"],
        )

    verify_bundle(destination)
    load_bundle(destination)
    return destination


def main(arguments: list[str] | None = None) -> None:
    args = parse_arguments(arguments)
    try:
        destination = export_locked_bundle(
            args.output_dir,
            selection_manifest=args.selection_manifest,
            final_test_audit=args.final_test_audit,
            preprocessing_config=args.preprocessing_config,
            model_version=args.model_version,
        )
    except Exception as error:
        print(f"Error exporting bundle: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    print(f"Production bundle created and verified: {destination}")


if __name__ == "__main__":
    main()
