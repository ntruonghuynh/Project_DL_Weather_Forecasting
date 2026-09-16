"""Reproducibility interface; implementation belongs to TV1."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed all supported random number generators for reproducible runs.

    Args:
        seed: Integer random seed.
        deterministic: If True, configure PyTorch and cuDNN for deterministic execution.
    """
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError(f"Seed must be an integer, got {type(seed).__name__}")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
