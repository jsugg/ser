"""Contract tests for archive extraction helper functions."""

from __future__ import annotations

import tarfile
from pathlib import Path

import pytest

from ser.data import archive_extraction


def test_extract_archive_supports_tar_gz(tmp_path: Path) -> None:
    """Generic helper extraction should support tar.gz sources."""
    archive_path = tmp_path / "sample.tar.gz"
    source_dir = tmp_path / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    source_file = source_dir / "nested" / "clip.wav"
    source_file.parent.mkdir(parents=True, exist_ok=True)
    source_file.write_bytes(b"fake")
    with tarfile.open(archive_path, mode="w:gz") as handle:
        handle.add(source_file, arcname="nested/clip.wav")

    extract_root = tmp_path / "extract"
    archive_extraction.extract_archive(
        archive_path=archive_path,
        extract_root=extract_root,
    )

    assert (extract_root / "nested" / "clip.wav").is_file()


def test_extract_archives_from_tree_errors_when_none_found(tmp_path: Path) -> None:
    """Tree extraction should fail fast when no archives exist."""
    (tmp_path / "plain.txt").write_text("content", encoding="utf-8")
    with pytest.raises(RuntimeError, match="No extractable archives found"):
        _ = archive_extraction.extract_archives_from_tree(
            search_root=tmp_path,
            extract_root=tmp_path / "extract",
        )


def test_ensure_extracted_archive_is_idempotent(tmp_path: Path) -> None:
    """Marker-based extraction guard should skip repeated extractions."""
    archive_path = tmp_path / "bundle.tar.gz"
    source_file = tmp_path / "source.wav"
    source_file.write_bytes(b"fake")
    with tarfile.open(archive_path, mode="w:gz") as handle:
        handle.add(source_file, arcname="source.wav")
    extract_root = tmp_path / "extract"

    archive_extraction.ensure_extracted_archive(
        archive_path=archive_path,
        extract_root=extract_root,
    )
    marker_path = extract_root / f".extract-ok-{archive_path.name}.json"
    assert marker_path.is_file()
    marker_before = marker_path.read_text(encoding="utf-8")

    archive_extraction.ensure_extracted_archive(
        archive_path=archive_path,
        extract_root=extract_root,
    )
    marker_after = marker_path.read_text(encoding="utf-8")
    assert marker_after == marker_before
