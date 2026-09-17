"""Validated preprocessing, autoregressive inference and postprocessing."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from numbers import Real

import numpy as np

from src.evaluation.metrics import inverse_scale_target

from .schemas import ModelBundle, PredictionRecord, SchemaBoundArray


class Predictor:
    """Run inference from an already-loaded bundle; never train models."""

    def __init__(self, bundle: ModelBundle, device: str = "cpu") -> None:
        self.bundle = bundle
        self.device = device
        self.features, self.target_index, self.input_length, self.horizon = self._schema_values()
        schema_hash = bundle.feature_schema.get("schema_hash")
        self.schema_hash = str(schema_hash) if schema_hash else "unversioned-schema"

    def _schema_values(self) -> tuple[list[str], int, int, int]:
        schema = self.bundle.feature_schema
        features = schema.get("ordered_features")
        if features is None:
            features = [item["name"] for item in schema.get("features", [])]
        if not isinstance(features, list) or not features or any(
            not isinstance(item, str) for item in features
        ):
            raise ValueError("feature schema has no valid ordered feature list")
        target = schema.get("target", {})
        target_index = target.get("index", schema.get("target_index"))
        if not isinstance(target_index, int) or not 0 <= target_index < len(features):
            raise ValueError("feature schema has an invalid target index")
        contract = schema.get("window_contract", {})
        input_length = contract.get("input_len_hours", contract.get("input_length_hours"))
        horizon = contract.get("horizon_hours")
        if not isinstance(input_length, int) or not isinstance(horizon, int):
            raise ValueError("feature schema is missing integer input/horizon lengths")
        return features, target_index, input_length, horizon

    def _validate_timestamps(self, timestamps: Sequence[object]) -> list[datetime]:
        if len(timestamps) != self.input_length:
            raise ValueError(f"expected {self.input_length} timestamps, got {len(timestamps)}")
        parsed: list[datetime] = []
        for value in timestamps:
            if isinstance(value, datetime):
                parsed.append(value)
            elif isinstance(value, str):
                try:
                    parsed.append(datetime.fromisoformat(value.replace("Z", "+00:00")))
                except ValueError as error:
                    raise ValueError(f"invalid ISO timestamp: {value}") from error
            else:
                raise TypeError("timestamps must be datetime objects or ISO strings")
        if any(parsed[index] - parsed[index - 1] != timedelta(hours=1)
               for index in range(1, len(parsed))):
            raise ValueError("timestamps must be strictly increasing at one-hour cadence")
        return parsed

    def _feature_matrix(self, inputs: object) -> np.ndarray:
        try:
            import pandas as pd
        except ImportError:  # pragma: no cover - pandas is a declared runtime dependency
            pd = None  # type: ignore[assignment]

        if isinstance(inputs, np.ndarray):
            raise TypeError(
                "bare ndarray input is ambiguous; use SchemaBoundArray with explicit "
                "feature_order and schema_hash"
            )
        if pd is not None and isinstance(inputs, pd.DataFrame):
            if list(inputs.columns) != self.features:
                raise ValueError("DataFrame columns must exactly match schema feature order")
            matrix = inputs.to_numpy(dtype=np.float32)
            if matrix.shape != (self.input_length, len(self.features)):
                raise ValueError(
                    f"input matrix must have shape [{self.input_length},{len(self.features)}]"
                )
        elif isinstance(inputs, Sequence) and not isinstance(inputs, (str, bytes)):
            if len(inputs) != self.input_length:
                raise ValueError(f"expected {self.input_length} observations, got {len(inputs)}")
            rows: list[list[float]] = []
            expected = set(self.features)
            for index, item in enumerate(inputs):
                if not isinstance(item, Mapping):
                    raise TypeError("each observation must be a feature mapping")
                actual = set(item)
                if actual != expected:
                    raise ValueError(
                        f"observation {index} feature mismatch: "
                        f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
                    )
                if list(item.keys()) != self.features:
                    raise ValueError(
                        f"observation {index} feature order must exactly match the schema"
                    )
                ordered: list[float] = []
                for name in self.features:
                    value = item[name]
                    if isinstance(value, bool) or not isinstance(value, Real):
                        raise TypeError(
                            f"observation {index} feature {name!r} must be numeric"
                        )
                    ordered.append(float(value))
                rows.append(ordered)
            matrix = np.asarray(rows, dtype=np.float32)
        else:
            raise TypeError("inputs must be a numpy matrix or sequence of feature mappings")
        if not np.isfinite(matrix).all():
            raise ValueError("inputs must contain only finite values")
        return matrix

    def _validate_schema_bound_array(self, inputs: SchemaBoundArray) -> np.ndarray:
        if tuple(inputs.feature_order) != tuple(self.features):
            raise ValueError("array feature_order does not match the bundle schema")
        if inputs.schema_hash != self.schema_hash:
            raise ValueError("array schema_hash does not match the bundle schema")
        values = np.asarray(inputs.values, dtype=np.float32)
        if values.ndim == 2:
            values = values[None, ...]
        expected_tail = (self.input_length, len(self.features))
        if values.ndim != 3 or values.shape[1:] != expected_tail or values.shape[0] <= 0:
            raise ValueError(f"array must have shape [B,{expected_tail[0]},{expected_tail[1]}]")
        if not np.isfinite(values).all():
            raise ValueError("inputs must contain only finite values")
        return values

    def predict_batch(self, inputs: SchemaBoundArray) -> np.ndarray:
        """Return degree-Celsius predictions with shape ``[B,horizon,1]``."""
        import torch

        raw = self._validate_schema_bound_array(inputs)
        flat = raw.reshape(-1, raw.shape[-1])
        scaled = np.asarray(self.bundle.scaler.transform(flat), dtype=np.float32).reshape(raw.shape)
        if not np.isfinite(scaled).all():
            raise ValueError("scaler returned invalid features")
        x = torch.from_numpy(scaled).to(self.device)
        model = self.bundle.model
        model.eval()
        with torch.no_grad():
            prediction = model(x, y=None, teacher_forcing_ratio=0.0)
        expected_shape = (raw.shape[0], self.horizon, 1)
        if not isinstance(prediction, torch.Tensor) or prediction.shape != expected_shape:
            shape = getattr(prediction, "shape", None)
            raise ValueError(f"model output must have shape {expected_shape}, got {shape}")
        if prediction.device != x.device or not prediction.is_floating_point():
            raise ValueError("model output must be floating-point and on the input device")
        restored = inverse_scale_target(
            prediction.detach().cpu().numpy(), self.bundle.scaler, self.target_index
        )
        if not np.isfinite(restored).all():
            raise ValueError("model output contains non-finite values")
        return np.asarray(restored, dtype=np.float64)

    def predict(self, inputs: object, timestamps: object) -> list[PredictionRecord]:
        """Return records satisfying the shared prediction contract."""
        if not isinstance(timestamps, Sequence) or isinstance(timestamps, (str, bytes)):
            raise TypeError("timestamps must be a sequence")
        parsed_timestamps = self._validate_timestamps(timestamps)
        raw_matrix = self._feature_matrix(inputs)
        if not hasattr(self.bundle.scaler, "transform"):
            raise TypeError("bundle scaler must expose transform")
        target = self.predict_batch(
            SchemaBoundArray(
                values=raw_matrix,
                feature_order=tuple(self.features),
                schema_hash=self.schema_hash,
            )
        )[0, :, 0]
        start = parsed_timestamps[-1]
        return [
            PredictionRecord(
                timestamp=(start + timedelta(hours=step)).isoformat(),
                y_true=None, y_pred=float(target[step - 1]),
                model_name=self.bundle.model_name,
                model_version=self.bundle.model_version,
                split="inference", run_id=self.bundle.run_id, horizon=step,
            )
            for step in range(1, self.horizon + 1)
        ]

    def model_info(self) -> dict[str, object]:
        """Return public, non-secret model provenance for API/demo display."""
        return {
            "model_name": self.bundle.model_name,
            "model_version": self.bundle.model_version,
            "run_id": self.bundle.run_id,
            "input_length_hours": self.input_length,
            "forecast_horizon_hours": self.horizon,
            "n_features": len(self.features),
            "target_index": self.target_index,
            "target_feature": self.features[self.target_index],
            "unit": "degC",
            "schema_version": self.bundle.feature_schema.get("schema_version", "unknown"),
            "schema_hash": self.schema_hash,
            "attention_available": hasattr(self.bundle.model, "get_last_attention_weights"),
        }

    def persistence_baseline(self, inputs: object) -> list[float]:
        """Return a same-population persistence baseline in original °C."""
        raw_matrix = self._feature_matrix(inputs)
        anchor = float(raw_matrix[-1, self.target_index])
        return [anchor] * self.horizon

    def last_attention_weights(self) -> list[list[float]] | None:
        """Return the latest single-sample attention matrix when supported."""
        getter = getattr(self.bundle.model, "get_last_attention_weights", None)
        if getter is None:
            return None
        weights = getter()
        array = weights.detach().cpu().numpy()
        if array.shape != (1, self.horizon, self.input_length):
            raise ValueError("attention weights do not match [1,horizon,input_length]")
        if not np.isfinite(array).all():
            raise ValueError("attention weights contain non-finite values")
        if (array < -1e-7).any() or not np.allclose(array.sum(axis=-1), 1.0, atol=1e-5):
            raise ValueError("attention weights must be non-negative and sum to one")
        return array[0].tolist()
