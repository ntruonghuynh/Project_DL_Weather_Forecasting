"""Tests for model bundle packaging, predictor inference and serving API (owner: TV6)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pytest
import torch
from torch import nn

from api.main import PredictRequest, create_app
from src.serving.bundle import build_bundle, load_bundle, sha256_file, verify_bundle
from src.serving.predictor import Predictor
from src.serving.schemas import ModelBundle


class DummyServingModel(nn.Module):
    """Simple linear forecast model conforming to Model Contract v1."""

    def __init__(self, n_features: int = 4, horizon: int = 6, with_attention: bool = False) -> None:
        super().__init__()
        self.n_features = n_features
        self.horizon = horizon
        self.with_attention = with_attention
        self.linear = nn.Linear(n_features, horizon)
        self._last_attention: torch.Tensor | None = None

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> torch.Tensor:
        if not self.training and (y is not None or teacher_forcing_ratio != 0.0):
            raise ValueError("Inference requires y=None and teacher_forcing_ratio=0.0")
        batch_size = x.shape[0]
        # Use mean over time sequence for simple dummy prediction: [B, H, 1]
        pooled = x.mean(dim=1)
        pred = self.linear(pooled).unsqueeze(-1)
        if self.with_attention:
            self._last_attention = torch.ones(batch_size, self.horizon, x.shape[1]) / x.shape[1]
        return pred

    def get_last_attention_weights(self) -> torch.Tensor:
        if self._last_attention is None:
            raise RuntimeError("No attention weights recorded")
        return self._last_attention


class DummyScaler:
    """Mock fitted StandardScaler with transform and inverse parameters."""

    def __init__(self, n_features: int = 4) -> None:
        self.mean_ = np.array([10.0 + i for i in range(n_features)], dtype=np.float64)
        self.scale_ = np.array([2.0 for _ in range(n_features)], dtype=np.float64)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean_) / self.scale_


def _create_sample_run_files(
    directory: Path,
    n_features: int = 4,
    input_length: int = 12,
    horizon: int = 6,
    target_index: int = 1,
    run_id: str = "run_test_001",
) -> dict[str, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = directory / "checkpoint.pt"
    scaler_path = directory / "scaler.joblib"
    schema_path = directory / "feature_schema.json"
    config_path = directory / "resolved_config.json"
    metadata_path = directory / "metadata.json"

    model = DummyServingModel(n_features=n_features, horizon=horizon, with_attention=True)
    torch.save(model.state_dict(), checkpoint_path)

    scaler = DummyScaler(n_features=n_features)
    joblib.dump(scaler, scaler_path)

    feature_names = [f"feat_{i}" for i in range(n_features)]
    feature_names[target_index] = "T (degC)"
    schema_data = {
        "ordered_features": feature_names,
        "target": {"name": "T (degC)", "index": target_index},
        "window_contract": {"input_len_hours": input_length, "horizon_hours": horizon},
    }
    schema_path.write_text(json.dumps(schema_data, indent=2), encoding="utf-8")

    config_data = {"model": {"name": "dummy_serving", "horizon": horizon}, "seed": 42}
    config_path.write_text(json.dumps(config_data, indent=2), encoding="utf-8")

    metadata_data = {
        "run_id": run_id,
        "model_name": "dummy_model",
        "model_version": "v1.0.0",
        "created_at": "2026-09-15T00:00:00Z",
    }
    metadata_path.write_text(json.dumps(metadata_data, indent=2), encoding="utf-8")

    return {
        "checkpoint": checkpoint_path,
        "scaler": scaler_path,
        "feature_schema": schema_path,
        "resolved_config": config_path,
        "metadata": metadata_path,
    }


# ============================================================================
# 1. ModelBundle & Checksum Tests
# ============================================================================


def test_sha256_file(tmp_path: Path) -> None:
    sample_file = tmp_path / "hello.txt"
    sample_file.write_text("hello world", encoding="utf-8")
    digest = sha256_file(sample_file)
    assert isinstance(digest, str)
    assert len(digest) == 64


def test_build_and_verify_bundle(tmp_path: Path) -> None:
    src_dir = tmp_path / "run_source"
    bundle_dest = tmp_path / "exported_bundle"
    files = _create_sample_run_files(src_dir)

    built = build_bundle(
        bundle_dest,
        checkpoint=files["checkpoint"],
        scaler=files["scaler"],
        feature_schema=files["feature_schema"],
        resolved_config=files["resolved_config"],
        metadata=files["metadata"],
        model_class="tests.test_serving.DummyServingModel",
        model_kwargs={"n_features": 4, "horizon": 6, "with_attention": True},
        model_name="dummy_model",
        model_version="v1.0.0",
        run_id="run_test_001",
    )
    assert built == bundle_dest
    manifest = verify_bundle(bundle_dest)
    assert manifest["run_id"] == "run_test_001"
    assert manifest["model_name"] == "dummy_model"
    assert len(manifest["sha256"]) == 5


def test_verify_bundle_detects_tampered_artifact(tmp_path: Path) -> None:
    src_dir = tmp_path / "run_source"
    bundle_dest = tmp_path / "exported_bundle"
    files = _create_sample_run_files(src_dir)

    build_bundle(
        bundle_dest,
        checkpoint=files["checkpoint"],
        scaler=files["scaler"],
        feature_schema=files["feature_schema"],
        resolved_config=files["resolved_config"],
        metadata=files["metadata"],
        model_class="tests.test_serving.DummyServingModel",
        model_kwargs={"n_features": 4, "horizon": 6, "with_attention": True},
        model_name="dummy_model",
        model_version="v1.0.0",
        run_id="run_test_001",
    )

    # Tamper with resolved_config.json
    config_file = bundle_dest / "resolved_config.json"
    config_file.write_text('{"tampered": true}', encoding="utf-8")

    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_bundle(bundle_dest)


def test_build_bundle_refuses_overwrite(tmp_path: Path) -> None:
    src_dir = tmp_path / "run_source"
    bundle_dest = tmp_path / "exported_bundle"
    files = _create_sample_run_files(src_dir)

    build_bundle(
        bundle_dest,
        checkpoint=files["checkpoint"],
        scaler=files["scaler"],
        feature_schema=files["feature_schema"],
        resolved_config=files["resolved_config"],
        metadata=files["metadata"],
        model_class="tests.test_serving.DummyServingModel",
        model_kwargs={"n_features": 4, "horizon": 6},
        model_name="dummy_model",
        model_version="v1.0.0",
        run_id="run_test_001",
    )
    with pytest.raises(FileExistsError):
        build_bundle(
            bundle_dest,
            checkpoint=files["checkpoint"],
            scaler=files["scaler"],
            feature_schema=files["feature_schema"],
            resolved_config=files["resolved_config"],
            metadata=files["metadata"],
            model_class="tests.test_serving.DummyServingModel",
            model_kwargs={"n_features": 4, "horizon": 6},
            model_name="dummy_model",
            model_version="v1.0.0",
            run_id="run_test_001",
        )


def test_load_bundle(tmp_path: Path) -> None:
    src_dir = tmp_path / "run_source"
    bundle_dest = tmp_path / "exported_bundle"
    files = _create_sample_run_files(src_dir)

    build_bundle(
        bundle_dest,
        checkpoint=files["checkpoint"],
        scaler=files["scaler"],
        feature_schema=files["feature_schema"],
        resolved_config=files["resolved_config"],
        metadata=files["metadata"],
        model_class="tests.test_serving.DummyServingModel",
        model_kwargs={"n_features": 4, "horizon": 6, "with_attention": True},
        model_name="dummy_model",
        model_version="v1.0.0",
        run_id="run_test_001",
    )

    loaded = load_bundle(bundle_dest)
    assert isinstance(loaded, ModelBundle)
    assert loaded.model_name == "dummy_model"
    assert loaded.run_id == "run_test_001"
    assert hasattr(loaded.model, "forward")


# ============================================================================
# 2. Predictor & Inference Tests
# ============================================================================


@pytest.fixture
def sample_predictor(tmp_path: Path) -> Predictor:
    src_dir = tmp_path / "sample_predictor_run"
    bundle_dest = tmp_path / "sample_predictor_bundle"
    files = _create_sample_run_files(src_dir, n_features=4, input_length=12, horizon=6)

    build_bundle(
        bundle_dest,
        checkpoint=files["checkpoint"],
        scaler=files["scaler"],
        feature_schema=files["feature_schema"],
        resolved_config=files["resolved_config"],
        metadata=files["metadata"],
        model_class="tests.test_serving.DummyServingModel",
        model_kwargs={"n_features": 4, "horizon": 6, "with_attention": True},
        model_name="dummy_serving",
        model_version="1.0.0",
        run_id="run_test_001",
    )
    bundle = load_bundle(bundle_dest)
    return Predictor(bundle)


def test_predictor_model_info(sample_predictor: Predictor) -> None:
    info = sample_predictor.model_info()
    assert info["model_name"] == "dummy_serving"
    assert info["input_length_hours"] == 12
    assert info["forecast_horizon_hours"] == 6
    assert info["n_features"] == 4
    assert info["target_feature"] == "T (degC)"
    assert info["unit"] == "degC"
    assert info["attention_available"] is True


def test_predictor_predict_with_feature_dicts(sample_predictor: Predictor) -> None:
    start_time = datetime(2026, 1, 1, 0, 0)
    timestamps = [(start_time + timedelta(hours=i)).isoformat() for i in range(12)]
    features = sample_predictor.features

    observations = [
        {name: float(10 + j * 0.5) for j, name in enumerate(features)}
        for _ in range(12)
    ]

    records = sample_predictor.predict(observations, timestamps)
    assert len(records) == 6
    assert records[0].horizon == 1
    assert records[-1].horizon == 6
    assert all(r.unit == "degC" for r in records)
    assert all(isinstance(r.y_pred, float) for r in records)

    # Check attention weights retrieval
    attn = sample_predictor.last_attention_weights()
    assert attn is not None
    assert len(attn) == 6  # horizon
    assert len(attn[0]) == 12  # input_length


def test_predictor_predict_with_numpy_matrix(sample_predictor: Predictor) -> None:
    start_time = datetime(2026, 1, 1, 0, 0)
    timestamps = [(start_time + timedelta(hours=i)).isoformat() for i in range(12)]
    matrix = np.full((12, 4), 15.0, dtype=np.float32)

    records = sample_predictor.predict(matrix, timestamps)
    assert len(records) == 6


def test_predictor_persistence_baseline(sample_predictor: Predictor) -> None:
    matrix = np.zeros((12, 4), dtype=np.float32)
    # target index is 1
    matrix[-1, 1] = 23.5
    baseline = sample_predictor.persistence_baseline(matrix)
    assert len(baseline) == 6
    assert baseline == [23.5] * 6


def test_predictor_rejects_invalid_timestamps(sample_predictor: Predictor) -> None:
    matrix = np.zeros((12, 4), dtype=np.float32)
    # Non-1-hour cadence (gap of 2 hours)
    start_time = datetime(2026, 1, 1, 0, 0)
    bad_cadence = [
        (start_time + timedelta(hours=i if i < 5 else i + 1)).isoformat() for i in range(12)
    ]
    with pytest.raises(ValueError, match="one-hour cadence"):
        sample_predictor.predict(matrix, bad_cadence)

    # Wrong length (only 10 instead of 12)
    too_short = [(start_time + timedelta(hours=i)).isoformat() for i in range(10)]
    with pytest.raises(ValueError, match="expected 12 timestamps"):
        sample_predictor.predict(matrix, too_short)


def test_predictor_rejects_mismatched_features(sample_predictor: Predictor) -> None:
    start_time = datetime(2026, 1, 1, 0, 0)
    timestamps = [(start_time + timedelta(hours=i)).isoformat() for i in range(12)]
    # Missing 'T (degC)'
    bad_observations = [{"feat_0": 1.0, "feat_2": 2.0, "feat_3": 3.0} for _ in range(12)]
    with pytest.raises(ValueError, match="feature mismatch"):
        sample_predictor.predict(bad_observations, timestamps)


# ============================================================================
# 3. FastAPI Endpoint Tests
# ============================================================================


def test_api_health_not_ready() -> None:
    application: Any = create_app(predictor=None)
    health_endpoint = [
        r.endpoint for r in application.routes if getattr(r, "path", None) == "/health"
    ][0]
    response = health_endpoint()
    assert response["status"] == "not_ready"


def test_api_ready_flow(sample_predictor: Predictor) -> None:
    application: Any = create_app(predictor=sample_predictor)
    routes = {r.path: r.endpoint for r in application.routes if hasattr(r, "path")}

    # Health check
    health_resp = routes["/health"]()
    assert health_resp["status"] == "ready"

    # Model info
    info_resp = routes["/model-info"]()
    assert info_resp["model_name"] == "dummy_serving"

    # Predict request
    start_time = datetime(2026, 1, 1, 0, 0)
    timestamps = [(start_time + timedelta(hours=i)).isoformat() for i in range(12)]
    observations = [
        {name: 12.0 for name in sample_predictor.features} for _ in range(12)
    ]
    req = PredictRequest(timestamps=timestamps, observations=observations)

    predict_resp = routes["/predict"](req)
    assert len(predict_resp.predictions) == 6
    assert predict_resp.latency_ms >= 0
    assert len(predict_resp.persistence_baseline) == 6
