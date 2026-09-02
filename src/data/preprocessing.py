"""Leakage-safe preprocessing pipeline for the Jena Climate dataset (TV2).

Raw CSV (10-minute) -> timestamp cleaning -> hourly resampling -> time
feature engineering -> missingness indicators -> chronological split ->
input imputation -> feature scaling -> processed CSV + artifacts.

This module implements the contract already approved and locked in:
  - artifacts/preprocessing/tv2_data_cleaning_policy.json (cleaning/aggregation/
    imputation/split policy)
  - artifacts/preprocessing/feature_schema.json (ordered feature list, target
    index, window contract - consumed by TV3-TV5 models, must not drift)

It intentionally reuses ``src.data.validator`` for the raw column list, the
target column name, and the timestamp column name instead of redefining
them, and it never writes to the raw CSV. No model training happens here;
the sliding-window dataset (168h in -> 72h out) is a later step.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .validator import EXPECTED_COLUMNS, TARGET_COLUMN, TIMESTAMP_COLUMN

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths and locked contract
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCKED_SCHEMA_PATH = PROJECT_ROOT / "artifacts" / "preprocessing" / "feature_schema.json"

DEFAULT_CONFIG: dict[str, Any] = {
    "train_fraction": 0.70,
    "val_fraction": 0.15,
    "forward_fill_limit_hours": 3,
    "artifacts_dir": str(PROJECT_ROOT / "artifacts" / "preprocessing"),
}

TIMESTAMP_FORMAT = "%d.%m.%Y %H:%M:%S"  # kept in sync with validator.py's parser

# ---------------------------------------------------------------------------
# Column policy (mirrors artifacts/preprocessing/tv2_data_cleaning_policy.json)
# ---------------------------------------------------------------------------

RAW_NUMERIC_COLUMNS: list[str] = [c for c in EXPECTED_COLUMNS if c != TIMESTAMP_COLUMN]

MEAN_FEATURES: list[str] = [
    "p (mbar)",
    "T (degC)",
    "Tpot (K)",
    "Tdew (degC)",
    "rh (%)",
    "VPmax (mbar)",
    "VPact (mbar)",
    "VPdef (mbar)",
    "sh (g/kg)",
    "H2OC (mmol/mol)",
    "rho (g/m**3)",
    "wv (m/s)",
]
MAX_FEATURES: list[str] = ["max. wv (m/s)"]
CIRCULAR_FEATURES: list[str] = ["wd (deg)"]  # wind direction: circular, see _circular_mean_deg

assert set(MEAN_FEATURES) | set(MAX_FEATURES) | set(CIRCULAR_FEATURES) == set(RAW_NUMERIC_COLUMNS)

TIME_FEATURE_COLUMNS: list[str] = ["hour_sin", "hour_cos", "dayofyear_sin", "dayofyear_cos"]

# Order matches RAW_NUMERIC_COLUMNS so the derived indicator block lines up
# with the locked feature_schema.json ordered_features tail.
INDICATOR_NAME_MAP: dict[str, str] = {
    "p (mbar)": "missing_p_mbar",
    "T (degC)": "missing_T_degC",
    "Tpot (K)": "missing_Tpot_K",
    "Tdew (degC)": "missing_Tdew_degC",
    "rh (%)": "missing_rh_pct",
    "VPmax (mbar)": "missing_VPmax_mbar",
    "VPact (mbar)": "missing_VPact_mbar",
    "VPdef (mbar)": "missing_VPdef_mbar",
    "sh (g/kg)": "missing_sh_gkg",
    "H2OC (mmol/mol)": "missing_H2OC_mmolmol",
    "rho (g/m**3)": "missing_rho_gm3",
    "wv (m/s)": "missing_wv_ms",
    "max. wv (m/s)": "missing_max_wv_ms",
    "wd (deg)": "missing_wd_deg",
}

FEATURE_COLUMNS: list[str] = (
    RAW_NUMERIC_COLUMNS + TIME_FEATURE_COLUMNS + list(INDICATOR_NAME_MAP.values())
)
TARGET_INDEX = FEATURE_COLUMNS.index(TARGET_COLUMN)

# Target is excluded: T (degC) is never forward-filled or median-filled (see impute_missing).
INPUT_IMPUTE_COLUMNS: list[str] = [c for c in RAW_NUMERIC_COLUMNS if c != TARGET_COLUMN]


# ---------------------------------------------------------------------------
# A. Load raw data
# ---------------------------------------------------------------------------


def load_raw_data(csv_path: Path) -> pd.DataFrame:
    """Read the immutable raw CSV and parse Date Time. Never writes to csv_path."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Raw data file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    missing_columns = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_columns:
        raise ValueError(f"Raw CSV is missing expected columns: {missing_columns}")

    df[TIMESTAMP_COLUMN] = pd.to_datetime(df[TIMESTAMP_COLUMN], format=TIMESTAMP_FORMAT)
    logger.info("Loaded %d raw rows, %d columns from %s", len(df), len(df.columns), csv_path)
    return df


# ---------------------------------------------------------------------------
# B. Timestamp cleaning
# ---------------------------------------------------------------------------


def clean_timestamps(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Sort by time and resolve duplicate timestamps with an explicit policy.

    Two duplicate cases are handled differently, never silently:
      - Exact duplicate rows (identical values in every column): the repeat
        carries no new information, so the first occurrence is kept and the
        rest dropped.
      - Same timestamp, different feature values: dropping either row would
        discard a real observation, so the conflicting rows for that
        timestamp are aggregated with a column-wise mean instead. This is a
        defensive fallback - for this raw file every duplicate-timestamp
        group is value-identical (see artifacts/data_quality_report.json), so
        this branch is not exercised on this dataset, but is kept so the
        pipeline degrades safely if a future export is not value-identical.
    """
    df = df.sort_values(TIMESTAMP_COLUMN, kind="mergesort").reset_index(drop=True)
    input_rows = len(df)

    is_exact_dup = df.duplicated(keep=False)
    has_timestamp_dup = df.duplicated(subset=[TIMESTAMP_COLUMN], keep=False)
    conflicting_mask = has_timestamp_dup & ~is_exact_dup

    n_duplicate_timestamp_groups = int(df.loc[has_timestamp_dup, TIMESTAMP_COLUMN].nunique())
    n_conflicting_groups = int(df.loc[conflicting_mask, TIMESTAMP_COLUMN].nunique())

    before_exact_drop = len(df)
    df = df.drop_duplicates(keep="first").reset_index(drop=True)
    n_exact_duplicates_dropped = before_exact_drop - len(df)

    if n_conflicting_groups > 0:
        feature_cols = [c for c in df.columns if c != TIMESTAMP_COLUMN]
        df = df.groupby(TIMESTAMP_COLUMN, as_index=False)[feature_cols].mean()

    df = df.sort_values(TIMESTAMP_COLUMN, kind="mergesort").reset_index(drop=True)

    report = {
        "input_rows": input_rows,
        "output_rows": int(len(df)),
        "duplicate_timestamp_groups": n_duplicate_timestamp_groups,
        "exact_duplicate_rows_dropped": int(n_exact_duplicates_dropped),
        "conflicting_duplicate_timestamp_groups": n_conflicting_groups,
        "conflicting_resolution": (
            "column_wise_mean"
            if n_conflicting_groups
            else "not_needed_all_duplicates_were_identical"
        ),
        "monotonic_after_sort": bool(df[TIMESTAMP_COLUMN].is_monotonic_increasing),
    }
    logger.info(
        "Timestamp cleaning: %d -> %d rows "
        "(dropped %d exact duplicates, aggregated %d conflicting groups)",
        input_rows,
        len(df),
        n_exact_duplicates_dropped,
        n_conflicting_groups,
    )
    return df, report


# ---------------------------------------------------------------------------
# C. Hourly resampling
# ---------------------------------------------------------------------------


def _circular_mean_deg(angles_deg: pd.Series) -> float:
    """Mean of angles in degrees, respecting wrap-around at 360."""
    angles_deg = angles_deg.dropna()
    if angles_deg.empty:
        return np.nan
    radians = np.deg2rad(angles_deg.to_numpy())
    mean_rad = np.arctan2(np.sin(radians).mean(), np.cos(radians).mean())
    return float(np.rad2deg(mean_rad) % 360)


def resample_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Resample from 10-minute to 1-hour bins using column-appropriate aggregation.

    Wind direction (`wd (deg)`) is NOT arithmetic-averaged: it is a circular
    quantity, so 359 deg and 1 deg are one degree apart in reality but an
    arithmetic mean gives 180 deg (the opposite direction). Instead the unit
    vectors (sin, cos) of the angles are averaged and converted back with
    atan2, which respects the wrap-around at 360 deg. Hours with no raw
    observations become NaN rows rather than being reconstructed - long gaps
    are marked, not fabricated (see artifacts/preprocessing/tv2_data_cleaning_policy.json).
    """
    indexed = df.set_index(TIMESTAMP_COLUMN)

    mean_part = indexed[MEAN_FEATURES].resample("1h").mean()
    max_part = indexed[MAX_FEATURES].resample("1h").max()
    circular_part = pd.DataFrame(
        {col: indexed[col].resample("1h").apply(_circular_mean_deg) for col in CIRCULAR_FEATURES}
    )

    hourly = pd.concat([mean_part, max_part, circular_part], axis=1)[RAW_NUMERIC_COLUMNS]
    hourly = hourly.reset_index().rename(columns={hourly.index.name or "index": TIMESTAMP_COLUMN})
    logger.info(
        "Resampled to hourly: %d rows spanning %s to %s",
        len(hourly),
        hourly[TIMESTAMP_COLUMN].min(),
        hourly[TIMESTAMP_COLUMN].max(),
    )
    return hourly


# ---------------------------------------------------------------------------
# D. Time feature engineering
# ---------------------------------------------------------------------------


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical (sin/cos) encodings of hour-of-day and day-of-year.

    Raw integer hour/day values imply a false discontinuity - hour 23 and
    hour 0 look maximally far apart to a model even though they are
    adjacent in time. Sine/cosine encoding maps each value onto a circle so
    adjacent times stay close together in feature space.
    """
    df = df.copy()
    hour = df[TIMESTAMP_COLUMN].dt.hour
    day_of_year = df[TIMESTAMP_COLUMN].dt.dayofyear
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["dayofyear_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    df["dayofyear_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)
    return df


def add_missing_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Flag hourly bins that had no raw observation, before any filling.

    Computed on the hourly data (post-resample), so an indicator is 1 only
    when an entire hour had no raw readings - not when raw data was merely
    sparse within that hour. These flags are the only record of which
    values were later fabricated by imputation (or, for the target, left
    missing on purpose); the future sliding-window step relies on
    `missing_T_degC` to drop forecast windows whose target was originally
    missing instead of training against an imputed "ground truth".
    """
    df = df.copy()
    for raw_col, indicator_col in INDICATOR_NAME_MAP.items():
        df[indicator_col] = df[raw_col].isna().astype("float32")
    return df


# ---------------------------------------------------------------------------
# E. Temporal split
# ---------------------------------------------------------------------------


def temporal_split(
    df: pd.DataFrame, train_fraction: float = 0.70, val_fraction: float = 0.15
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Chronological, non-shuffled split: train is the oldest slice, test the newest.

    No randomness, no shuffling: a forecasting model must be evaluated on
    data strictly after what it trained on, matching how it is actually
    used (predicting the future from the past). The three slices are
    contiguous row ranges, so they cannot overlap by construction.
    """
    if not 0 < train_fraction < 1 or not 0 < val_fraction < 1:
        raise ValueError("train_fraction and val_fraction must be in (0, 1)")
    if train_fraction + val_fraction >= 1:
        raise ValueError("train_fraction + val_fraction must leave a positive test share")

    n = len(df)
    train_end = int(round(n * train_fraction))
    val_end = train_end + int(round(n * val_fraction))

    train_df = df.iloc[:train_end].reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].reset_index(drop=True)
    test_df = df.iloc[val_end:].reset_index(drop=True)

    def _bounds(part: pd.DataFrame) -> dict[str, Any]:
        return {
            "start": part[TIMESTAMP_COLUMN].iloc[0].isoformat(),
            "end": part[TIMESTAMP_COLUMN].iloc[-1].isoformat(),
            "n_samples": int(len(part)),
        }

    metadata = {
        "method": "chronological_no_shuffle",
        "train_fraction": train_fraction,
        "validation_fraction": val_fraction,
        "test_fraction": round(1 - train_fraction - val_fraction, 10),
        "total_samples": int(n),
        "train": _bounds(train_df),
        "validation": _bounds(val_df),
        "test": _bounds(test_df),
    }
    logger.info(
        "Temporal split: train=%d val=%d test=%d rows (train ends %s, val ends %s)",
        len(train_df),
        len(val_df),
        len(test_df),
        metadata["train"]["end"],
        metadata["validation"]["end"],
    )
    return train_df, val_df, test_df, metadata


# ---------------------------------------------------------------------------
# F. Missing data handling
# ---------------------------------------------------------------------------


def impute_missing(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    forward_fill_limit_hours: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Fill missing INPUT features without leaking future or cross-split information.

    Order is fixed by policy (artifacts/preprocessing/tv2_data_cleaning_policy.json):
      1. Split first (done by the caller before this runs).
      2. Forward-fill up to `forward_fill_limit_hours`, independently within
         each split - never across the train/val/test boundary, and never
         backward (backfilling would leak a future value into the past).
      3. Anything still missing is filled with the TRAIN split's median,
         computed once on train and reused for validation/test, so no
         statistic is ever derived from data the model should not see yet.

    The target column T (degC) is excluded entirely: it is never
    forward-filled or median-filled. Its gaps are preserved so a later step
    can drop forecast windows with a missing target instead of training
    against a fabricated value (see missing_T_degC in add_missing_indicators).
    """
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    for split_df in (train_df, val_df, test_df):
        split_df[INPUT_IMPUTE_COLUMNS] = split_df[INPUT_IMPUTE_COLUMNS].ffill(
            limit=forward_fill_limit_hours
        )

    train_medians = train_df[INPUT_IMPUTE_COLUMNS].median()

    for split_df in (train_df, val_df, test_df):
        split_df[INPUT_IMPUTE_COLUMNS] = split_df[INPUT_IMPUTE_COLUMNS].fillna(train_medians)

    remaining_input_missing = {
        "train": int(train_df[INPUT_IMPUTE_COLUMNS].isna().sum().sum()),
        "validation": int(val_df[INPUT_IMPUTE_COLUMNS].isna().sum().sum()),
        "test": int(test_df[INPUT_IMPUTE_COLUMNS].isna().sum().sum()),
    }
    remaining_target_missing = {
        "train": int(train_df[TARGET_COLUMN].isna().sum()),
        "validation": int(val_df[TARGET_COLUMN].isna().sum()),
        "test": int(test_df[TARGET_COLUMN].isna().sum()),
    }
    report = {
        "forward_fill_limit_hours": forward_fill_limit_hours,
        "median_fit_scope": "train_only",
        "backward_fill": False,
        "target_column_imputed": False,
        "train_median": {k: float(v) for k, v in train_medians.items()},
        "remaining_missing_after_fill": remaining_input_missing,
        "remaining_missing_target": remaining_target_missing,
    }
    logger.info(
        "Missing-value handling done. Remaining input NaNs: train=%d val=%d test=%d. "
        "Target left unimputed with NaNs: train=%d val=%d test=%d",
        remaining_input_missing["train"],
        remaining_input_missing["validation"],
        remaining_input_missing["test"],
        remaining_target_missing["train"],
        remaining_target_missing["validation"],
        remaining_target_missing["test"],
    )
    return train_df, val_df, test_df, report


# ---------------------------------------------------------------------------
# G. Feature scaling
# ---------------------------------------------------------------------------


def scale_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Fit StandardScaler on TRAIN ONLY, then transform all three splits.

    Fitting on train alone (never on val/test, never on train+val+test
    combined) keeps validation and test statistically unseen - fitting on
    everything would leak future information into the mean/std used to
    normalize the past. scikit-learn's StandardScaler computes nanmean/
    nanstd and passes NaN through transform() unchanged, so the
    intentionally-unimputed target keeps its missing values here too.
    """
    scaler = StandardScaler()
    scaler.fit(train_df[feature_columns])

    def _transform(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out[feature_columns] = scaler.transform(out[feature_columns]).astype("float32")
        return out

    logger.info(
        "Fitted StandardScaler on %d train rows, %d features", len(train_df), len(feature_columns)
    )
    return _transform(train_df), _transform(val_df), _transform(test_df), scaler


# ---------------------------------------------------------------------------
# Contract check + I/O helpers
# ---------------------------------------------------------------------------


def _assert_matches_locked_schema(schema_path: Path) -> dict[str, Any]:
    """Fail fast if this pipeline's feature layout drifts from the locked schema."""
    if not schema_path.exists():
        raise FileNotFoundError(
            f"Locked feature schema not found at {schema_path}; it must be approved "
            "and committed before preprocessing can run."
        )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    if schema.get("ordered_features") != FEATURE_COLUMNS:
        raise ValueError(
            "Pipeline FEATURE_COLUMNS no longer matches ordered_features in "
            f"{schema_path}. This is a breaking contract change for TV3-TV5 model "
            "code and must be updated deliberately in both places."
        )
    if schema.get("target", {}).get("name") != TARGET_COLUMN:
        raise ValueError(f"Locked schema target does not match TARGET_COLUMN={TARGET_COLUMN}")
    if schema.get("target", {}).get("index") != TARGET_INDEX:
        raise ValueError(
            f"Locked schema target index {schema.get('target', {}).get('index')} "
            f"does not match computed TARGET_INDEX={TARGET_INDEX}"
        )
    return schema


def _save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Wrote %d rows to %s", len(df), path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Wrote %s", path)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def preprocess(raw_path: Path, output_dir: Path, config: dict[str, Any] | None = None) -> Path:
    """Run the full leakage-safe preprocessing pipeline and write processed artifacts.

    Raw -> clean timestamps -> hourly resample -> time features -> missing
    indicators -> chronological split -> input imputation -> scaling ->
    CSV + scaler + metadata written to disk. Never mutates raw_path. Safe to
    call repeatedly; every output is overwritten deterministically.

    CSV splits (train/val/test_processed.csv) go to `output_dir`. The
    scaler, split metadata, and feature schema are written both to
    `output_dir` (as the deliverable requested for this step) and to
    `config["artifacts_dir"]` (default artifacts/preprocessing/, the path
    already referenced by the locked feature_schema.json as the single
    source of truth other pipeline stages read from).
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    raw_path = Path(raw_path)
    output_dir = Path(output_dir)
    artifacts_dir = Path(cfg["artifacts_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    schema = _assert_matches_locked_schema(LOCKED_SCHEMA_PATH)

    raw_df = load_raw_data(raw_path)
    clean_df, cleaning_report = clean_timestamps(raw_df)
    hourly_df = resample_hourly(clean_df)
    hourly_df = add_time_features(hourly_df)
    hourly_df = add_missing_indicators(hourly_df)

    train_df, val_df, test_df, split_metadata = temporal_split(
        hourly_df, cfg["train_fraction"], cfg["val_fraction"]
    )
    train_df, val_df, test_df, impute_report = impute_missing(
        train_df, val_df, test_df, cfg["forward_fill_limit_hours"]
    )
    train_scaled, val_scaled, test_scaled, scaler = scale_features(
        train_df, val_df, test_df, FEATURE_COLUMNS
    )

    output_columns = [TIMESTAMP_COLUMN] + FEATURE_COLUMNS
    _save_csv(train_scaled[output_columns], output_dir / "train_processed.csv")
    _save_csv(val_scaled[output_columns], output_dir / "val_processed.csv")
    _save_csv(test_scaled[output_columns], output_dir / "test_processed.csv")

    joblib.dump(scaler, artifacts_dir / "scaler.joblib")
    joblib.dump(scaler, output_dir / "scaler.joblib")
    logger.info(
        "Wrote scaler to %s and %s",
        artifacts_dir / "scaler.joblib",
        output_dir / "scaler.joblib",
    )

    split_metadata = {
        **split_metadata,
        "cleaning": cleaning_report,
        "missing_handling": impute_report,
    }
    _write_json(artifacts_dir / "split_metadata.json", split_metadata)
    _write_json(output_dir / "split_metadata.json", split_metadata)

    _write_json(output_dir / "feature_schema.json", schema)

    logger.info(
        "Preprocessing complete: train=%d val=%d test=%d rows, %d features -> %s",
        len(train_scaled),
        len(val_scaled),
        len(test_scaled),
        len(FEATURE_COLUMNS),
        output_dir,
    )
    return output_dir


def main() -> None:
    """Run preprocessing on the default raw file, logging progress to stdout."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raw_path = PROJECT_ROOT / "data" / "raw" / "jena_climate_2009_2016.csv"
    output_dir = PROJECT_ROOT / "data" / "processed"
    preprocess(raw_path, output_dir)


if __name__ == "__main__":
    main()
