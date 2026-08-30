"""Model Contract v1 shared by all forecasting model families."""

import math
from numbers import Real
from typing import Protocol

import torch


def validate_forward_arguments(
    *, training: bool, y: torch.Tensor | None, teacher_forcing_ratio: float
) -> None:
    """Reject future targets and teacher forcing outside training mode."""
    if isinstance(teacher_forcing_ratio, bool) or not isinstance(teacher_forcing_ratio, Real):
        raise TypeError("teacher_forcing_ratio must be a number, not bool")
    if not math.isfinite(float(teacher_forcing_ratio)):
        raise ValueError("teacher_forcing_ratio must be finite")
    if not 0.0 <= teacher_forcing_ratio <= 1.0:
        raise ValueError("teacher_forcing_ratio must be within [0.0, 1.0]")
    if not training and y is not None:
        raise ValueError("Validation/test/inference requires y=None to prevent target leakage")
    if not training and teacher_forcing_ratio != 0.0:
        raise ValueError(
            "teacher_forcing_ratio must be 0.0 when model.training is False"
        )


def validate_model_output(
    output: torch.Tensor,
    x: torch.Tensor,
    horizon: int = 72,
    target_dim: int = 1,
) -> None:
    """Validate model output type, shape, floating dtype, and input device.

    Evaluation/serving is responsible for casting predictions and restoring
    their original scale; this validator performs neither operation.
    """
    if not isinstance(output, torch.Tensor):
        raise TypeError("Model output must be a torch.Tensor")
    if not isinstance(x, torch.Tensor):
        raise TypeError("Model input x must be a torch.Tensor")
    if x.ndim < 1:
        raise ValueError("Model input x must include a batch dimension")
    if horizon <= 0 or target_dim <= 0:
        raise ValueError("horizon and target_dim must be positive")
    expected_shape = (x.shape[0], horizon, target_dim)
    if output.shape != expected_shape:
        raise ValueError(
            f"Model output shape must be {list(expected_shape)}, got {list(output.shape)}"
        )
    if not output.is_floating_point():
        raise TypeError(f"Model output must have a floating-point dtype, got {output.dtype}")
    if output.device != x.device:
        raise ValueError(
            f"Model output device must match x ({x.device}), got {output.device}"
        )


class ForecastModel(Protocol):
    """Model producing [B,72,1]; decoder starts at the final input target.

    Teacher forcing is permitted only while training. The target feature index
    comes from schema/configuration and must not be hard-coded by this contract.
    """

    training: bool

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> torch.Tensor:
        """Forecast 72 hours from inputs shaped [B,168,n_features]."""
        ...

    def __call__(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> torch.Tensor:
        """Invoke forward through the callable model interface used by Trainer."""
        ...
