"""Transformer placeholder; model implementation belongs to TV5."""


class WeatherTransformer:
    """Unimplemented model conforming to the shared forecast contract."""

    def forward(
        self, x: object, y: object | None = None, teacher_forcing_ratio: float = 0.0
    ) -> object:
        """Return a forecast shaped [B,72,1] when implemented."""
        raise NotImplementedError("TV5 must implement the Transformer")
