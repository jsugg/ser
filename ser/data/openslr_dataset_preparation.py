"""OpenSLR dataset preparation helpers for public SER corpora."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol, TypeVar

from ser.data.provider_dataset_preparation import (
    AutoDownloadArtifacts,
    GeneratedLabelsStatsLike,
)

_StatsT = TypeVar("_StatsT", bound=GeneratedLabelsStatsLike)
_StatsCovT = TypeVar("_StatsCovT", bound=GeneratedLabelsStatsLike, covariant=True)
_StatsContraT = TypeVar(
    "_StatsContraT",
    bound=GeneratedLabelsStatsLike,
    contravariant=True,
)


class DownloadOpenSlrArchives(Protocol):
    """Callable contract for OpenSLR archive acquisition."""

    def __call__(
        self,
        *,
        dataset_root: Path,
        dataset_id: str,
        archive_suffixes: tuple[str, ...],
    ) -> list[Path]: ...


class EnsureExtractedArchive(Protocol):
    """Callable contract for idempotent archive extraction."""

    def __call__(
        self,
        *,
        archive_path: Path,
        extract_root: Path,
    ) -> None: ...


class GenerateLabelsFromAudioTree(Protocol[_StatsCovT]):
    """Callable contract for deterministic label generation from extracted audio."""

    def __call__(
        self,
        *,
        dataset_root: Path,
        search_root: Path,
        labels_csv_path: Path,
        resolver: Callable[[Path], str | None],
        extensions: frozenset[str] = ...,
    ) -> _StatsCovT: ...


class WriteSourceManifest(Protocol[_StatsContraT]):
    """Callable contract for source provenance manifest persistence."""

    def __call__(
        self,
        *,
        dataset_root: Path,
        source_manifest_path: Path,
        source_payload: dict[str, object],
        labels_csv_path: Path | None,
        labels_stats: _StatsContraT | None,
    ) -> None: ...


def prepare_openslr_dataset(
    *,
    dataset_root: Path,
    dataset_id: str,
    archive_suffixes: tuple[str, ...],
    extract_dir_name: str,
    labels_file_name: str,
    source_manifest_file_name: str,
    label_resolver: Callable[[Path], str | None],
    label_semantics: str | None,
    extensions: frozenset[str] | None,
    download_openslr_archives: DownloadOpenSlrArchives,
    ensure_extracted_archive: EnsureExtractedArchive,
    generate_labels_from_audio_tree: GenerateLabelsFromAudioTree[_StatsT],
    write_source_manifest: WriteSourceManifest[_StatsT],
) -> AutoDownloadArtifacts:
    """Downloads one OpenSLR dataset and generates deterministic labels/manifest artifacts."""
    root = dataset_root.expanduser()
    root.mkdir(parents=True, exist_ok=True)
    archives = download_openslr_archives(
        dataset_root=root,
        dataset_id=dataset_id,
        archive_suffixes=archive_suffixes,
    )
    extract_root = root / "raw" / extract_dir_name
    for archive_path in archives:
        ensure_extracted_archive(archive_path=archive_path, extract_root=extract_root)
    labels_csv_path = root / labels_file_name
    if extensions is None:
        stats = generate_labels_from_audio_tree(
            dataset_root=root,
            search_root=extract_root,
            labels_csv_path=labels_csv_path,
            resolver=label_resolver,
        )
    else:
        stats = generate_labels_from_audio_tree(
            dataset_root=root,
            search_root=extract_root,
            labels_csv_path=labels_csv_path,
            resolver=label_resolver,
            extensions=extensions,
        )
    source_payload: dict[str, object] = {
        "provider": "openslr",
        "dataset_id": dataset_id,
        "archive_paths": [str(path) for path in archives],
    }
    if label_semantics is not None:
        source_payload["label_semantics"] = label_semantics
    source_manifest_path = root / source_manifest_file_name
    write_source_manifest(
        dataset_root=root,
        source_manifest_path=source_manifest_path,
        source_payload=source_payload,
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


__all__ = ["AutoDownloadArtifacts", "prepare_openslr_dataset"]
