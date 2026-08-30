"""Seq2Seq LSTM placeholder; model implementation belongs to TV3."""


class Seq2SeqLSTM:
    """Unimplemented model conforming to the shared forecast contract."""

    def forward(
        self, x: object, y: object | None = None, teacher_forcing_ratio: float = 0.0
    ) -> object:
        """Return a forecast shaped [B,72,1] when implemented."""
        raise NotImplementedError("TV3 must implement Seq2Seq LSTM")
