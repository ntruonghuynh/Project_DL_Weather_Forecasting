"""Executable tests for scripts/prepare_data.py's checksum-verification logic.

Only the pure sha256/verify functions are tested here - `download()` shells
out to the `kaggle` CLI and is intentionally not exercised (no network calls
in tests).
"""

import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_data import sha256_of, verify  # noqa: E402


def test_sha256_of_matches_hashlib_reference(tmp_path: Path) -> None:
    """sha256_of reads the file in chunks but must match a plain hashlib digest."""
    path = tmp_path / "data.bin"
    path.write_bytes(b"jena climate raw bytes" * 1000)

    expected = hashlib.sha256(path.read_bytes()).hexdigest()

    assert sha256_of(path) == expected


def test_verify_accepts_matching_checksum_case_insensitively(tmp_path: Path) -> None:
    """A checksum recorded in uppercase (as manifest.json does) must still verify."""
    path = tmp_path / "data.bin"
    path.write_bytes(b"some content")
    expected = hashlib.sha256(path.read_bytes()).hexdigest()

    verify(path, expected.upper())  # must not raise


def test_verify_rejects_mismatched_checksum(tmp_path: Path) -> None:
    """A wrong/corrupted file must fail loudly, never be silently accepted."""
    path = tmp_path / "data.bin"
    path.write_bytes(b"some content")

    with pytest.raises(ValueError, match="sha256 mismatch"):
        verify(path, "0" * 64)
