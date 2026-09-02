"""Executable tests for Data Batch Contract v1."""

import pytest
import torch

from src.data.dataset import WeatherBatch, WeatherDataset, WeatherSample, validate_weather_batch


def make_batch(batch_size: int = 2) -> WeatherBatch:
    """Build a synthetic batch without reading a dataset."""
    return {
        "x": torch.zeros(batch_size, 168, 4, dtype=torch.float32),
        "y": torch.zeros(batch_size, 72, 1, dtype=torch.float32),
        "input_timestamps": torch.zeros(batch_size, 168, dtype=torch.int64),
        "target_timestamps": torch.zeros(batch_size, 72, dtype=torch.int64),
    }


def test_valid_batch_contract() -> None:
    """Accept a correctly shaped and typed synthetic batch."""
    validate_weather_batch(make_batch())


def test_dataset_returns_samples_before_dataloader_collation() -> None:
    """Keep the dataset item contract distinct from the collated batch contract."""
    return_type = WeatherDataset.__getitem__.__annotations__["return"]

    assert return_type is WeatherSample
    assert "DataLoader collates" in (WeatherSample.__doc__ or "")
    assert "DataLoader collates" in (WeatherDataset.__doc__ or "")


def test_expected_feature_count_is_accepted() -> None:
    """Accept the schema-provided feature count when it matches x."""
    validate_weather_batch(make_batch(), n_features=4)


@pytest.mark.parametrize("n_features", [3, 5])
def test_wrong_feature_count_is_rejected(n_features: int) -> None:
    """Reject x when its feature count differs from schema/configuration."""
    with pytest.raises(ValueError, match="feature count"):
        validate_weather_batch(make_batch(), n_features=n_features)


@pytest.mark.parametrize("n_features", [0, -1, 1.5, True])
def test_invalid_expected_feature_count_is_rejected(n_features: object) -> None:
    """Require n_features to be a positive integer when supplied."""
    with pytest.raises(ValueError, match="positive integer"):
        validate_weather_batch(make_batch(), n_features=n_features)  # type: ignore[arg-type]


def test_empty_batch_is_rejected() -> None:
    """Reject batch_size=0 even when all remaining dimensions are valid."""
    validate_target = {
        "x": torch.zeros(0, 168, 4, dtype=torch.float32),
        "y": torch.zeros(0, 72, 1, dtype=torch.float32),
        "input_timestamps": torch.zeros(0, 168, dtype=torch.int64),
        "target_timestamps": torch.zeros(0, 72, dtype=torch.int64),
    }

    with pytest.raises(ValueError, match="greater than zero"):
        validate_weather_batch(validate_target)


def test_missing_key_is_rejected() -> None:
    """Reject a batch that omits a required contract key."""
    batch = dict(make_batch())
    del batch["target_timestamps"]

    with pytest.raises(KeyError, match="target_timestamps"):
        validate_weather_batch(batch)


def test_wrong_feature_shape_is_rejected() -> None:
    """Reject an incorrect input sequence length."""
    batch = make_batch()
    batch["x"] = torch.zeros(2, 167, 4, dtype=torch.float32)

    with pytest.raises(ValueError, match="length must be 168"):
        validate_weather_batch(batch)


def test_wrong_target_dtype_is_rejected() -> None:
    """Reject targets that are not float32."""
    batch = make_batch()
    batch["y"] = batch["y"].to(torch.float64)

    with pytest.raises(TypeError, match="torch.float32"):
        validate_weather_batch(batch)


def test_wrong_timestamp_dtype_is_rejected() -> None:
    """Reject timestamps that are not int64."""
    batch = make_batch()
    batch["input_timestamps"] = batch["input_timestamps"].to(torch.int32)

    with pytest.raises(TypeError, match="torch.int64"):
        validate_weather_batch(batch)


def test_timestamp_alignment_is_enforced() -> None:
    """Reject timestamps that do not align with their sequence."""
    batch = make_batch()
    batch["target_timestamps"] = torch.zeros(2, 71, dtype=torch.int64)

    with pytest.raises(ValueError, match="target timestamp alignment"):
        validate_weather_batch(batch)


def test_feature_and_target_devices_must_match() -> None:
    """Reject an x/y device mismatch using PyTorch's allocation-free meta device."""
    batch = make_batch()
    batch["y"] = torch.empty(2, 72, 1, dtype=torch.float32, device="meta")

    with pytest.raises(ValueError, match="must share a device"):
        validate_weather_batch(batch)
