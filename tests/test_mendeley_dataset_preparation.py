"""Tests for Mendeley dataset preparation helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ser.data import mendeley_dataset_preparation as preparation


@dataclass(frozen=True, slots=True)
class _Stats:
    files_seen: int
    labels_written: int
    dropped_files: int
    duplicate_conflicts: int


def test_prepare_mesd_from_mendeley_generates_labels_and_manifest(
    tmp_path: Path,
) -> None:
    """MESD helper should orchestrate download, labels, and source manifest content."""
    captured_download_root: Path | None = None
    captured_source_payload: dict[str, object] = {}

    def _download_tree(
        *,
        dataset_id: str,
        version: int,
        destination_root: Path,
    ) -> int:
        nonlocal captured_download_root
        assert dataset_id == "cy34mh68j9"
        assert version == 5
        captured_download_root = destination_root
        destination_root.mkdir(parents=True, exist_ok=True)
        return 12

    def _generate_labels(
        *,
        dataset_root: Path,
        search_root: Path,
        labels_csv_path: Path,
        resolver: Callable[[Path], str | None],
        extensions: frozenset[str] = frozenset({".wav"}),
    ) -> _Stats:
        del resolver, extensions
        assert dataset_root == tmp_path
        assert search_root == tmp_path / "raw" / "mesd"
        assert labels_csv_path == tmp_path / "labels.csv"
        return _Stats(
            files_seen=10,
            labels_written=9,
            dropped_files=1,
            duplicate_conflicts=0,
        )

    def _write_source_manifest(
        *,
        dataset_root: Path,
        source_manifest_path: Path,
        source_payload: dict[str, object],
        labels_csv_path: Path | None,
        labels_stats: preparation.GeneratedLabelsStatsLike | None,
    ) -> None:
        assert dataset_root == tmp_path
        assert source_manifest_path == tmp_path / "source_manifest.json"
        assert labels_csv_path == tmp_path / "labels.csv"
        assert labels_stats is not None
        captured_source_payload.update(source_payload)

    artifacts = preparation.prepare_mesd_from_mendeley(
        dataset_root=tmp_path,
        dataset_id="cy34mh68j9",
        version=5,
        extract_dir_name="mesd",
        labels_file_name="labels.csv",
        source_manifest_file_name="source_manifest.json",
        download_mendeley_dataset_tree=_download_tree,
        generate_labels_from_audio_tree=_generate_labels,
        infer_mesd_label=lambda _path: "angry",
        write_source_manifest=_write_source_manifest,
    )

    assert captured_download_root == tmp_path / "raw" / "mesd"
    assert captured_source_payload == {
        "provider": "mendeley",
        "dataset_id": "cy34mh68j9",
        "version": 5,
        "files_downloaded": 12,
    }
    assert artifacts.dataset_root == tmp_path
    assert artifacts.labels_csv_path == tmp_path / "labels.csv"
    assert artifacts.audio_base_dir == tmp_path
    assert artifacts.source_manifest_path == tmp_path / "source_manifest.json"
    assert artifacts.files_seen == 10
    assert artifacts.labels_written == 9
