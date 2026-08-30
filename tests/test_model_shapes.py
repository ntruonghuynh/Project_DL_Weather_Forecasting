"""Model shape tests pending TV3-TV5 implementations."""

import pytest


@pytest.mark.skip(reason="Forecast model implementations are pending")
def test_output_shape() -> None:
    """Require every model output to have shape [B,72,1]."""
