"""Dataset and batch contracts; implementation belongs to TV2."""

from typing import Protocol, TypedDict


class WeatherBatch(TypedDict):
    """Batch with x [B,168,n_features], y [B,72,1], and aligned timestamps."""

    x: object
    y: object
    input_timestamps: object
    target_timestamps: object


class WeatherDataset(Protocol):
    """Minimal dataset protocol returning the shared batch contract."""

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> WeatherBatch: ...
