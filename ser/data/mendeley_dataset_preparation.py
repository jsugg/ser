"""Mendeley dataset preparation helpers for public SER corpora."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from ser.data.provider_dataset_preparation import (
    AutoDownloadArtifacts,
    GeneratedLabelsStatsLike,
)


class DownloadMendeleyDatasetTree(Protocol):
    """Callable contract for one Mendeley dataset-tree download operation."""

    def __call__(
        self,
        *,
        dataset_id: str,
        version: int,
        destination_root: Path,
    ) -> int: ...


class GenerateLabelsFromAudioTree(Protocol):
    """Callable contract for deterministic labels.csv generation from audio trees."""

    def __call__(
        self,
        *,
        dataset_root: Path,
        search_root: Path,
        labels_csv_path: Path,
        resolver: Callable[[Path], str | None],
        extensions: frozenset[str] = ...,
    ) -> GeneratedLabelsStatsLike: ...


class WriteSourceManifest(Protocol):
    """Callable contract for source manifest persistence."""

    def __call__(
        self,
        *,
        dataset_root: Path,
        source_manifest_path: Path,
        source_payload: dict[str, object],
        labels_csv_path: Path | None,
        labels_stats: GeneratedLabelsStatsLike | None,
    ) -> None: ...


def prepare_mesd_from_mendeley(
    *,
    dataset_root: Path,
    dataset_id: str,
    version: int,
    extract_dir_name: str,
    labels_file_name: str,
    source_manifest_file_name: str,
    download_mendeley_dataset_tree: DownloadMendeleyDatasetTree,
    generate_labels_from_audio_tree: GenerateLabelsFromAudioTree,
    infer_mesd_label: Callable[[Path], str | None],
    write_source_manifest: WriteSourceManifest,
) -> AutoDownloadArtifacts:
    """Downloads one Mendeley dataset and generates deterministic labels."""
    root = dataset_root.expanduser()
    root.mkdir(parents=True, exist_ok=True)
    extract_root = root / "raw" / extract_dir_name
    extract_root.mkdir(parents=True, exist_ok=True)
    files_downloaded = download_mendeley_dataset_tree(
        dataset_id=dataset_id,
        version=version,
        destination_root=extract_root,
    )
    labels_csv_path = root / labels_file_name
    stats = generate_labels_from_audio_tree(
        dataset_root=root,
        search_root=extract_root,
        labels_csv_path=labels_csv_path,
        resolver=infer_mesd_label,
    )
    source_manifest_path = root / source_manifest_file_name
    write_source_manifest(
        dataset_root=root,
        source_manifest_path=source_manifest_path,
        source_payload={
            "provider": "mendeley",
            "dataset_id": dataset_id,
            "version": version,
            "files_downloaded": files_downloaded,
        },
        labels_csv_path=labels_csv_path,
        labels_stats=stats,
    )
    return AutoDownloadArtifacts(
        dataset_root=root,
        labels_csv_path=labels_csv_path,
        audio_base_dir=root,
        source_manifest_path=source_manifest_path,
        files_seen=stats.files_seen,
        labels_written=stats.labels_written,
    )


__all__ = ["AutoDownloadArtifacts", "prepare_mesd_from_mendeley"]
