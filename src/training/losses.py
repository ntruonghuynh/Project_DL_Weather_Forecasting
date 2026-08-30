"""Loss factory interface; implementation belongs to training owners."""


def build_loss(name: str) -> object:
    """Build a configured forecasting loss."""
    raise NotImplementedError(f"Training owners must implement loss: {name}")
