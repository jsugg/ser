"""Contract tests for OpenSLR dataset preparation helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ser.data import openslr_dataset_preparation


@dataclass(frozen=True)
class _Stats:
    files_seen: int
    labels_written: int
    dropped_files: int
    duplicate_conflicts: int


def test_prepare_openslr_dataset_writes_manifest_and_honors_extensions(
    tmp_path: Path,
) -> None:
    """OpenSLR helper should extract archives, generate labels, and persist manifest."""
    captured_manifest: dict[str, object] = {}
    extracted_archives: list[Path] = []

    def _download_openslr_archives(
        *,
        dataset_root: Path,
        dataset_id: str,
        archive_suffixes: tuple[str, ...],
    ) -> list[Path]:
        assert dataset_root == tmp_path
        assert dataset_id == "115"
        assert archive_suffixes == (".tar.gz", ".tgz")
        first = dataset_root / "downloads" / "a.tar.gz"
        second = dataset_root / "downloads" / "b.tgz"
        first.parent.mkdir(parents=True, exist_ok=True)
        first.write_bytes(b"archive")
        second.write_bytes(b"archive")
        return [first, second]

    def _ensure_extracted_archive(*, archive_path: Path, extract_root: Path) -> None:
        extracted_archives.append(archive_path)
        extract_root.mkdir(parents=True, exist_ok=True)

    def _generate_labels_from_audio_tree(
        *,
        dataset_root: Path,
        search_root: Path,
        labels_csv_path: Path,
        resolver: Callable[[Path], str | None],
        extensions: frozenset[str] | None = None,
    ) -> _Stats:
        del resolver
        assert dataset_root == tmp_path
        assert search_root == tmp_path / "raw" / "emov-db"
        assert labels_csv_path == tmp_path / "labels.csv"
        assert extensions == frozenset({".wav", ".flac"})
        return _Stats(
            files_seen=8,
            labels_written=7,
            dropped_files=1,
            duplicate_conflicts=0,
        )

    def _write_source_manifest(
        *,
        dataset_root: Path,
        source_manifest_path: Path,
        source_payload: dict[str, object],
        labels_csv_path: Path | None,
        labels_stats: openslr_dataset_preparation.GeneratedLabelsStatsLike | None,
    ) -> None:
        del dataset_root
        captured_manifest["source_manifest_path"] = source_manifest_path
        captured_manifest["source_payload"] = source_payload
        captured_manifest["labels_csv_path"] = labels_csv_path
        captured_manifest["labels_stats"] = labels_stats

    artifacts = openslr_dataset_preparation.prepare_openslr_dataset(
        dataset_root=tmp_path,
        dataset_id="115",
        archive_suffixes=(".tar.gz", ".tgz"),
        extract_dir_name="emov-db",
        labels_file_name="labels.csv",
        source_manifest_file_name="source_manifest.json",
        label_resolver=lambda _path: "neutral",
        label_semantics=None,
        extensions=frozenset({".wav", ".flac"}),
        download_openslr_archives=_download_openslr_archives,
        ensure_extracted_archive=_ensure_extracted_archive,
        generate_labels_from_audio_tree=_generate_labels_from_audio_tree,
        write_source_manifest=_write_source_manifest,
    )

    assert [path.name for path in extracted_archives] == ["a.tar.gz", "b.tgz"]
    assert artifacts.dataset_root == tmp_path
    assert artifacts.labels_csv_path == tmp_path / "labels.csv"
    assert artifacts.audio_base_dir == tmp_path
    assert artifacts.files_seen == 8
    assert artifacts.labels_written == 7
    payload = captured_manifest["source_payload"]
    assert isinstance(payload, dict)
    assert payload["provider"] == "openslr"
    assert payload["dataset_id"] == "115"
    assert payload["archive_paths"] == [
        str(tmp_path / "downloads" / "a.tar.gz"),
        str(tmp_path / "downloads" / "b.tgz"),
    ]
    assert "label_semantics" not in payload


def test_prepare_openslr_dataset_emits_label_semantics_when_requested(
    tmp_path: Path,
) -> None:
    """OpenSLR helper should include optional label semantics in manifest payload."""
    captured_manifest: dict[str, object] = {}

    def _download_openslr_archives(
        *,
        dataset_root: Path,
        dataset_id: str,
        archive_suffixes: tuple[str, ...],
    ) -> list[Path]:
        del dataset_id, archive_suffixes
        archive_path = dataset_root / "downloads" / "wav.tgz"
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        archive_path.write_bytes(b"archive")
        return [archive_path]

    def _ensure_extracted_archive(*, archive_path: Path, extract_root: Path) -> None:
        del archive_path
        extract_root.mkdir(parents=True, exist_ok=True)

    def _generate_labels_from_audio_tree(
        *,
        dataset_root: Path,
        search_root: Path,
        labels_csv_path: Path,
        resolver: Callable[[Path], str | None],
        extensions: frozenset[str] | None = None,
    ) -> _Stats:
        del dataset_root, search_root, labels_csv_path, resolver
        assert extensions is None
        return _Stats(
            files_seen=5,
            labels_written=4,
            dropped_files=1,
            duplicate_conflicts=0,
        )

    def _write_source_manifest(
        *,
        dataset_root: Path,
        source_manifest_path: Path,
        source_payload: dict[str, object],
        labels_csv_path: Path | None,
        labels_stats: openslr_dataset_preparation.GeneratedLabelsStatsLike | None,
    ) -> None:
        del dataset_root, source_manifest_path, labels_csv_path, labels_stats
        captured_manifest["source_payload"] = source_payload

    _ = openslr_dataset_preparation.prepare_openslr_dataset(
        dataset_root=tmp_path,
        dataset_id="88",
        archive_suffixes=(".tgz",),
        extract_dir_name="att-hack",
        labels_file_name="labels.csv",
        source_manifest_file_name="source_manifest.json",
        label_resolver=lambda _path: "friendly",
        label_semantics="social_attitudes",
        extensions=None,
        download_openslr_archives=_download_openslr_archives,
        ensure_extracted_archive=_ensure_extracted_archive,
        generate_labels_from_audio_tree=_generate_labels_from_audio_tree,
        write_source_manifest=_write_source_manifest,
    )

    payload = captured_manifest["source_payload"]
    assert isinstance(payload, dict)
    assert payload["label_semantics"] == "social_attitudes"
