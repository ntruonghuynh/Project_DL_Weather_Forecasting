import json

import joblib
import numpy as np
import pytest
import torch
from sklearn.preprocessing import StandardScaler

from src.serving.bundle import build_bundle, load_bundle, verify_bundle


def _artifacts(tmp_path):
    model = torch.nn.Linear(2, 1)
    checkpoint = tmp_path / "source.pt"
    torch.save(model.state_dict(), checkpoint)
    scaler = tmp_path / "source.joblib"
    joblib.dump(StandardScaler().fit(np.array([[0.0, 1.0], [2.0, 3.0]])), scaler)
    schema = tmp_path / "schema.json"
    schema.write_text(json.dumps({"ordered_features": ["a", "b"]}))
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"model": "linear"}))
    metadata = tmp_path / "metadata.json"
    metadata.write_text(json.dumps({"run_id": "run-1"}))
    return model, checkpoint, scaler, schema, config, metadata


def test_bundle_checksum_load_and_model_parity(tmp_path) -> None:
    model, checkpoint, scaler, schema, config, metadata = _artifacts(tmp_path)
    destination = tmp_path / "bundle"
    build_bundle(
        destination, checkpoint=checkpoint, scaler=scaler, feature_schema=schema,
        resolved_config=config, metadata=metadata, model_class="torch.nn.Linear",
        model_kwargs={"in_features": 2, "out_features": 1}, model_name="linear",
        model_version="1", run_id="run-1",
    )
    verify_bundle(destination)
    loaded = load_bundle(destination)
    x = torch.tensor([[1.0, 2.0]])
    torch.testing.assert_close(model(x), loaded.model(x))


def test_bundle_detects_tampering_and_refuses_overwrite(tmp_path) -> None:
    _, checkpoint, scaler, schema, config, metadata = _artifacts(tmp_path)
    destination = tmp_path / "bundle"
    kwargs = dict(
        checkpoint=checkpoint, scaler=scaler, feature_schema=schema,
        resolved_config=config, metadata=metadata, model_class="torch.nn.Linear",
        model_kwargs={"in_features": 2, "out_features": 1}, model_name="linear",
        model_version="1", run_id="run-1",
    )
    build_bundle(destination, **kwargs)
    with pytest.raises(FileExistsError):
        build_bundle(destination, **kwargs)
    (destination / "metadata.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        verify_bundle(destination)


def test_bundle_rejects_manifest_file_remapping(tmp_path) -> None:
    _, checkpoint, scaler, schema, config, metadata = _artifacts(tmp_path)
    destination = tmp_path / "bundle"
    build_bundle(
        destination, checkpoint=checkpoint, scaler=scaler, feature_schema=schema,
        resolved_config=config, metadata=metadata, model_class="torch.nn.Linear",
        model_kwargs={"in_features": 2, "out_features": 1}, model_name="linear",
        model_version="1", run_id="run-1",
    )
    manifest_path = destination / "bundle_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["checkpoint"] = "metadata.json"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="file mapping"):
        verify_bundle(destination)
