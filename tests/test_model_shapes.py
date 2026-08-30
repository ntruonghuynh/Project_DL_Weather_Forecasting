"""Executable tests for Model Contract v1."""

import pytest
import torch

from src.models.base import validate_forward_arguments, validate_model_output


class DummyForecastModel(torch.nn.Module):
    """Minimal contract fixture; it is not a forecasting implementation."""

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> torch.Tensor:
        """Return zeros after enforcing the shared forward-argument rules."""
        validate_forward_arguments(
            training=self.training,
            y=y,
            teacher_forcing_ratio=teacher_forcing_ratio,
        )
        return torch.zeros(x.shape[0], 72, 1, dtype=x.dtype, device=x.device)


def test_model_output_shape() -> None:
    """Accept output shaped [B,72,1]."""
    model = DummyForecastModel().eval()
    x = torch.zeros(3, 168, 4, dtype=torch.float32)
    output = model(x)

    validate_model_output(output, x)


def test_batch_size_one_is_supported() -> None:
    """Keep the batch dimension when B=1."""
    model = DummyForecastModel().eval()
    x = torch.zeros(1, 168, 4, dtype=torch.float32)
    output = model(x)

    assert output.shape == (1, 72, 1)
    validate_model_output(output, x)


def test_inference_rejects_future_targets() -> None:
    """Reject y during validation, test, and inference mode."""
    model = DummyForecastModel().eval()
    x = torch.zeros(2, 168, 4, dtype=torch.float32)
    y = torch.zeros(2, 72, 1, dtype=torch.float32)

    with pytest.raises(ValueError, match="requires y=None"):
        model(x, y=y)


def test_inference_rejects_teacher_forcing() -> None:
    """Reject a positive teacher-forcing ratio outside training."""
    model = DummyForecastModel().eval()
    x = torch.zeros(2, 168, 4, dtype=torch.float32)

    with pytest.raises(ValueError, match="must be 0.0"):
        model(x, teacher_forcing_ratio=0.5)


def test_wrong_output_shape_is_rejected() -> None:
    """Detect output that violates [B,72,1]."""
    x = torch.zeros(2, 168, 4, dtype=torch.float32)
    wrong_output = torch.zeros(2, 71, 1, dtype=torch.float32)

    with pytest.raises(ValueError, match="output shape"):
        validate_model_output(wrong_output, x)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_floating_output_dtypes_are_accepted(dtype: torch.dtype) -> None:
    """Accept floating outputs used by regular and mixed-precision execution."""
    x = torch.zeros(2, 168, 4, dtype=torch.float32)
    output = torch.zeros(2, 72, 1, dtype=dtype)

    validate_model_output(output, x)


def test_integer_output_is_rejected() -> None:
    """Reject non-floating model output."""
    x = torch.zeros(2, 168, 4, dtype=torch.float32)
    output = torch.zeros(2, 72, 1, dtype=torch.int64)

    with pytest.raises(TypeError, match="floating-point dtype"):
        validate_model_output(output, x)


@pytest.mark.parametrize("ratio", [-0.1, 1.1])
def test_teacher_forcing_ratio_outside_range_is_rejected(ratio: float) -> None:
    """Require teacher forcing ratio to stay within its closed unit interval."""
    with pytest.raises(ValueError, match=r"\[0.0, 1.0\]"):
        validate_forward_arguments(training=True, y=None, teacher_forcing_ratio=ratio)


@pytest.mark.parametrize("ratio", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_teacher_forcing_ratio_is_rejected(ratio: float) -> None:
    """Reject NaN and infinite teacher-forcing ratios."""
    with pytest.raises(ValueError, match="finite"):
        validate_forward_arguments(training=True, y=None, teacher_forcing_ratio=ratio)


def test_boolean_teacher_forcing_ratio_is_rejected() -> None:
    """Do not accept bool even though it is an int subclass in Python."""
    with pytest.raises(TypeError, match="not bool"):
        validate_forward_arguments(training=True, y=None, teacher_forcing_ratio=True)
