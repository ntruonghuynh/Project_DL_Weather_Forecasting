"""Metric tests pending TV6 implementation."""

import pytest


@pytest.mark.skip(reason="TV6 metric implementation is pending")
def test_metrics_contract() -> None:
    """Require deterministic scalar metrics for aligned inputs."""
