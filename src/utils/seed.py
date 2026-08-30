"""Reproducibility interface; implementation belongs to TV1."""


def set_seed(seed: int) -> None:
    """Seed supported random number generators when implemented."""
    raise NotImplementedError("TV1 must implement deterministic seed setup")
