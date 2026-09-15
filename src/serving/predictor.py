"""Validated preprocessing, autoregressive inference and postprocessing."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta

import numpy as np

from src.evaluation.metrics import inverse_scale_target

from .schemas import ModelBundle, PredictionRecord


class Predictor:
    """Run inference from an already-loaded bundle; never train models."""

    def __init__(self, bundle: ModelBundle, device: str = "cpu") -> None:
        self.bundle = bundle
        self.device = device
        self.features, self.target_index, self.input_length, self.horizon = self._schema_values()

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
        if isinstance(inputs, np.ndarray):
            matrix = np.asarray(inputs, dtype=np.float32)
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
                rows.append([float(item[name]) for name in self.features])
            matrix = np.asarray(rows, dtype=np.float32)
        else:
            raise TypeError("inputs must be a numpy matrix or sequence of feature mappings")
        if not np.isfinite(matrix).all():
            raise ValueError("inputs must contain only finite values")
        return matrix

    def predict(self, inputs: object, timestamps: object) -> list[PredictionRecord]:
        """Return records satisfying the shared prediction contract."""
        import torch

        if not isinstance(timestamps, Sequence) or isinstance(timestamps, (str, bytes)):
            raise TypeError("timestamps must be a sequence")
        parsed_timestamps = self._validate_timestamps(timestamps)
        raw_matrix = self._feature_matrix(inputs)
        scaler = self.bundle.scaler
        if not hasattr(scaler, "transform"):
            raise TypeError("bundle scaler must expose transform")
        scaled = np.asarray(scaler.transform(raw_matrix), dtype=np.float32)
        if scaled.shape != raw_matrix.shape or not np.isfinite(scaled).all():
            raise ValueError("scaler returned invalid features")
        x = torch.from_numpy(scaled).unsqueeze(0).to(self.device)
        model = self.bundle.model
        model.eval()
        with torch.no_grad():
            prediction = model(x, y=None, teacher_forcing_ratio=0.0)
        if not isinstance(prediction, torch.Tensor) or prediction.shape != (1, self.horizon, 1):
            shape = getattr(prediction, "shape", None)
            raise ValueError(f"model output must have shape [1,{self.horizon},1], got {shape}")
        if prediction.device != x.device or not prediction.is_floating_point():
            raise ValueError("model output must be floating-point and on the input device")
        scaled_target = prediction.detach().cpu().numpy()[0, :, 0]
        target = inverse_scale_target(scaled_target, scaler, self.target_index)
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
        return array[0].tolist()
