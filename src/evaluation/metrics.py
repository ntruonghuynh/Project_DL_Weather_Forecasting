"""Metric interfaces; implementation belongs to TV6."""


def compute_metrics(y_true: object, y_pred: object) -> dict[str, float]:
    """Compute agreed forecasting metrics without changing model selection."""
    raise NotImplementedError("TV6 must implement evaluation metrics")
