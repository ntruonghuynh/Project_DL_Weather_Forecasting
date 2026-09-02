"""DataLoader construction for WeatherForecastDataset (TV2).

Wraps WeatherForecastDataset instances (src/data/dataset.py) into PyTorch
DataLoaders. Default collation already produces WeatherBatch-shaped tensors
(x [B,168,F], y [B,72,1], input_timestamps [B,168], target_timestamps [B,72])
since every WeatherSample dict has the same tensor shapes and keys - no
custom collate_fn is needed.
"""

from __future__ import annotations

from torch.utils.data import DataLoader

from .dataset import WeatherForecastDataset

DEFAULT_BATCH_SIZE = 32


def build_dataloader(
    dataset: WeatherForecastDataset,
    batch_size: int = DEFAULT_BATCH_SIZE,
    shuffle: bool = False,
    num_workers: int = 0,
    drop_last: bool = False,
) -> DataLoader:
    """Wrap one dataset split into a DataLoader."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
    )


def build_dataloaders(
    train_dataset: WeatherForecastDataset,
    val_dataset: WeatherForecastDataset,
    test_dataset: WeatherForecastDataset,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = 0,
) -> dict[str, DataLoader]:
    """Build the standard train/val/test DataLoader trio for model training.

    Train is shuffled at the sample (window) level: shuffling which
    240-hour window comes next in an epoch is safe because each window is
    still built only from contiguous rows of its own split (see
    WeatherForecastDataset) - this never mixes timesteps across splits or
    leaks the future within a sample. Validation and test stay in
    chronological order for reproducible, comparable evaluation.
    """
    return {
        "train": build_dataloader(train_dataset, batch_size, shuffle=True, num_workers=num_workers),
        "val": build_dataloader(val_dataset, batch_size, shuffle=False, num_workers=num_workers),
        "test": build_dataloader(test_dataset, batch_size, shuffle=False, num_workers=num_workers),
    }
