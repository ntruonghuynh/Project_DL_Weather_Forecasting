"""Fetch and verify the raw Jena Climate CSV declared in data/raw/manifest.json.

`data/raw/*.csv` is gitignored (see `data/README.md`, `.gitignore`) - every
clone, including CI, starts without it. This is the single place that:

1. Verifies an already-present raw file against the sha256 `manifest.json`
   records, refusing to let a wrong or corrupted file through silently.
2. Downloads it from the exact Kaggle dataset `manifest.json` declares when
   it is missing - never a different, guessed mirror - then verifies it the
   same way.

Requires the `kaggle` package and Kaggle API credentials (`KAGGLE_USERNAME`/
`KAGGLE_KEY` env vars, or `~/.kaggle/kaggle.json`) only when a download is
actually needed; verifying an already-present file needs neither.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "data" / "raw" / "manifest.json"


def load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(csv_path: Path, expected_sha256: str) -> None:
    actual = sha256_of(csv_path)
    if actual.lower() != expected_sha256.lower():
        raise ValueError(
            f"sha256 mismatch for {csv_path}: expected {expected_sha256}, got {actual}. "
            "The Kaggle source may have changed - update data/raw/manifest.json "
            "deliberately if so; never silently accept a different file."
        )


def download(dataset_slug: str, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", dataset_slug, "-p", str(dest_dir), "--unzip"],
        check=True,
    )


def main() -> None:
    manifest = load_manifest()
    file_entry = manifest["files"][0]
    csv_path = MANIFEST_PATH.parent / file_entry["name"]

    if not csv_path.exists():
        slug = manifest["dataset_slug"]
        print(f"{csv_path} not found - downloading from Kaggle dataset {slug!r}")
        download(slug, csv_path.parent)

    if not csv_path.exists():
        print(f"Error: {csv_path} still missing after download attempt.", file=sys.stderr)
        raise SystemExit(2)

    verify(csv_path, file_entry["sha256"])
    print(f"OK: {csv_path} verified against {MANIFEST_PATH} (sha256 match).")


if __name__ == "__main__":
    main()
