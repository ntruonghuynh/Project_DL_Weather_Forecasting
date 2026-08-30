"""Attention Seq2Seq placeholder; model implementation belongs to TV4."""


class Seq2SeqAttention:
    """Unimplemented model conforming to the shared forecast contract."""

    def forward(
        self, x: object, y: object | None = None, teacher_forcing_ratio: float = 0.0
    ) -> object:
        """Return a forecast shaped [B,72,1] when implemented."""
        raise NotImplementedError("TV4 must implement attention Seq2Seq")
