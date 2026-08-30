"""Chronological split interface; implementation belongs to TV2."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SplitBoundaries:
    """Exclusive chronological boundaries for train, validation, and test."""

    train_end: str
    validation_end: str
    test_end: str


def build_splits(timestamps: object, boundaries: SplitBoundaries) -> dict[str, object]:
    """Return leakage-safe chronological split indices."""
    raise NotImplementedError("TV2 must implement chronological splitting")
