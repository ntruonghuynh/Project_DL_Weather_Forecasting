"""Loss factory interface; implementation belongs to training owners."""

from __future__ import annotations

from typing import Any

from torch import nn

_LOSS_FACTORIES: dict[str, type[nn.Module]] = {
    "mse": nn.MSELoss,
    "mae": nn.L1Loss,
    "l1": nn.L1Loss,
    "huber": nn.HuberLoss,
    "smooth_l1": nn.HuberLoss,
}


def build_loss(name: str, **kwargs: Any) -> nn.Module:
    """Build a configured forecasting loss module.

    Args:
        name: Name of the loss function (e.g. 'mse', 'mae', 'l1', 'huber', 'smooth_l1').
        **kwargs: Optional arguments passed to the PyTorch loss constructor.

    Returns:
        An instance of torch.nn.Module computing the loss.

    Raises:
        ValueError: If the requested loss name is unsupported.
    """
    if not isinstance(name, str):
        raise TypeError(f"Loss name must be a string, got {type(name).__name__}")

    normalized_name = name.strip().lower()
    if normalized_name not in _LOSS_FACTORIES:
        available = ", ".join(sorted(_LOSS_FACTORIES.keys()))
        raise ValueError(
            f"Unsupported loss: '{name}'. Supported loss functions are: {available}"
        )

    loss_cls = _LOSS_FACTORIES[normalized_name]
    return loss_cls(**kwargs)
