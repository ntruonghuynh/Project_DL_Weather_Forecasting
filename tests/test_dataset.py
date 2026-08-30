"""Dataset contract tests pending TV2 implementation."""

import pytest


@pytest.mark.skip(reason="TV2 dataset implementation is pending")
def test_batch_contract() -> None:
    """Require x [B,168,F], y [B,72,1], and timestamp alignment."""
