"""Versioned, checksummed ModelBundle creation and loading."""

from __future__ import annotations

import hashlib
import importlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .schemas import ModelBundle

BUNDLE_FILES = {
    "checkpoint": "model.pt",
    "scaler": "scaler.joblib",
    "feature_schema": "feature_schema.json",
    "resolved_config": "resolved_config.json",
    "metadata": "metadata.json",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def build_bundle(
    destination: Path,
    *,
    checkpoint: Path,
    scaler: Path,
    feature_schema: Path,
    resolved_config: Path,
    metadata: Path,
    model_class: str,
    model_kwargs: dict[str, Any],
    model_name: str,
    model_version: str,
    run_id: str,
) -> Path:
    """Copy verified run artifacts into a new immutable release directory."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing bundle: {destination}")
    if not all((model_class, model_name, model_version, run_id)):
        raise ValueError("model_class, model_name, model_version and run_id are required")
    sources = {
        "checkpoint": Path(checkpoint), "scaler": Path(scaler),
        "feature_schema": Path(feature_schema), "resolved_config": Path(resolved_config),
        "metadata": Path(metadata),
    }
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"bundle source artifacts are missing: {missing}")
    source_metadata = _read_json(sources["metadata"])
    if source_metadata.get("run_id") not in {None, run_id}:
        raise ValueError("metadata run_id does not match requested bundle run_id")
    destination.mkdir(parents=True)
    for key, source in sources.items():
        shutil.copy2(source, destination / BUNDLE_FILES[key])
    checksums = {name: sha256_file(destination / name) for name in BUNDLE_FILES.values()}
    manifest = {
        "bundle_schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id, "model_name": model_name, "model_version": model_version,
        "model_class": model_class, "model_kwargs": model_kwargs,
        "files": BUNDLE_FILES, "sha256": checksums,
    }
    (destination / "bundle_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return destination


def verify_bundle(bundle_path: Path) -> dict[str, Any]:
    """Validate required files, checksums and cross-artifact run identity."""
    bundle_path = Path(bundle_path)
    manifest_path = bundle_path / "bundle_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"bundle manifest not found: {manifest_path}")
    manifest = _read_json(manifest_path)
    if manifest.get("bundle_schema_version") != "1.0":
        raise ValueError("unsupported bundle schema version")
    files, checksums = manifest.get("files"), manifest.get("sha256")
    if not isinstance(files, dict) or not isinstance(checksums, dict):
        raise ValueError("bundle manifest files/sha256 sections are invalid")
    if files != BUNDLE_FILES:
        raise ValueError("bundle manifest file mapping does not match schema 1.0")
    if not isinstance(manifest.get("model_kwargs"), dict):
        raise ValueError("bundle manifest model_kwargs must be an object")
    for key in BUNDLE_FILES:
        filename = files.get(key)
        if not filename or Path(filename).name != filename:
            raise ValueError(f"invalid bundle filename for {key}")
        path = bundle_path / filename
        if not path.is_file():
            raise FileNotFoundError(f"bundle artifact not found: {path}")
        if checksums.get(filename) != sha256_file(path):
            raise ValueError(f"checksum mismatch for bundle artifact: {filename}")
    metadata = _read_json(bundle_path / files["metadata"])
    if metadata.get("run_id") not in {None, manifest.get("run_id")}:
        raise ValueError("bundle metadata and manifest have different run_id values")
    return manifest


def _resolve_model_class(path: str) -> type:
    module_name, separator, class_name = path.rpartition(".")
    if not separator:
        raise ValueError("model_class must be a fully qualified import path")
    model_class = getattr(importlib.import_module(module_name), class_name, None)
    if not isinstance(model_class, type):
        raise TypeError(f"model_class did not resolve to a type: {path}")
    return model_class


def load_bundle(
    bundle_path: Path,
    *,
    device: str = "cpu",
    model_factory: Callable[[dict[str, Any]], object] | None = None,
) -> ModelBundle:
    """Verify a bundle, restore its model and return ready inference resources."""
    import joblib
    import torch

    bundle_path = Path(bundle_path)
    manifest = verify_bundle(bundle_path)
    files = manifest["files"]
    schema = _read_json(bundle_path / files["feature_schema"])
    config = _read_json(bundle_path / files["resolved_config"])
    metadata = _read_json(bundle_path / files["metadata"])
    scaler = joblib.load(bundle_path / files["scaler"])
    model = (
        model_factory(manifest["model_kwargs"])
        if model_factory is not None
        else _resolve_model_class(manifest["model_class"])(**manifest["model_kwargs"])
    )
    checkpoint = torch.load(
        bundle_path / files["checkpoint"], map_location=device, weights_only=True
    )
    state = checkpoint
    if isinstance(checkpoint, dict):
        state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    if not isinstance(state, dict):
        raise TypeError("checkpoint does not contain a model state dictionary")
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return ModelBundle(
        model=model, model_name=manifest["model_name"],
        model_version=manifest["model_version"], run_id=manifest["run_id"],
        scaler=scaler, feature_schema=schema, resolved_config=config,
        metadata=metadata, bundle_path=bundle_path, checksums=manifest["sha256"],
    )
