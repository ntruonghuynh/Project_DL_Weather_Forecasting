"""Boundary-based chronological split utility (TV2).

`src.data.preprocessing.temporal_split` is the split implementation the
production pipeline actually runs (see `preprocess()`): it partitions one
contiguous, already-hourly-resampled DataFrame into train/validation/test by
row *fraction* (configs/data.yaml split.*), and its output already drives
`data/processed/*.csv` and `split_metadata.json`.

This module covers the complementary case that `temporal_split` does not:
assigning arbitrary timestamps to train/validation/test using fixed
*boundary dates* instead of fractions - e.g. reproducing one specific
backtest window, or labeling timestamps that were never part of the
DataFrame passed to `temporal_split` in the first place. It intentionally
does not re-implement fraction-based splitting; that logic stays owned by
`temporal_split` alone so the two never drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SplitBoundaries:
    """Exclusive chronological boundaries for train, validation, and test.

    A timestamp belongs to train if it is earlier than `train_end`, to
    validation if it is in [train_end, validation_end), and to test if it is
    in [validation_end, test_end). Anything at or after `test_end`, or
    before the earliest timestamp, belongs to none of the three splits.
    """

    train_end: str
    validation_end: str
    test_end: str


def build_splits(timestamps: pd.Series, boundaries: SplitBoundaries) -> dict[str, np.ndarray]:
    """Return leakage-safe chronological split indices for arbitrary timestamps.

    The three ranges are contiguous and half-open ([start, end)), so they
    cannot overlap by construction; `boundaries` must be strictly increasing
    (train_end < validation_end < test_end) or this raises ValueError rather
    than silently producing an empty or overlapping split.

    Returns positional indices (0-based, matching `timestamps`'s row order),
    one array per split key ("train", "validation", "test") - not boolean
    masks - so callers can use them directly with `DataFrame.iloc`.
    """
    train_end = pd.Timestamp(boundaries.train_end)
    validation_end = pd.Timestamp(boundaries.validation_end)
    test_end = pd.Timestamp(boundaries.test_end)

    if not train_end < validation_end < test_end:
        raise ValueError(
            "SplitBoundaries must be strictly increasing: "
            f"train_end={train_end} < validation_end={validation_end} < test_end={test_end} "
            "does not hold"
        )

    ts = pd.to_datetime(pd.Series(timestamps).reset_index(drop=True))

    train_mask = ts < train_end
    validation_mask = (ts >= train_end) & (ts < validation_end)
    test_mask = (ts >= validation_end) & (ts < test_end)

    return {
        "train": np.flatnonzero(train_mask.to_numpy()),
        "validation": np.flatnonzero(validation_mask.to_numpy()),
        "test": np.flatnonzero(test_mask.to_numpy()),
    }
