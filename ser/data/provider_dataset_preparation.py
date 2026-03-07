"""Provider-sourced dataset preparation helpers for public SER corpora."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypeVar


@dataclass(frozen=True, slots=True)
class GeneratedLabelsStats:
    """Deterministic label-generation summary."""

    files_seen: int
    labels_written: int
    dropped_files: int
    duplicate_conflicts: int


class GeneratedLabelsStatsLike(Protocol):
    """Minimal label-generation statistics contract."""

    @property
    def files_seen(self) -> int: ...

    @property
    def labels_written(self) -> int: ...

    @property
    def dropped_files(self) -> int: ...

    @property
    def duplicate_conflicts(self) -> int: ...


class GitHubReleaseAssetMetadataLike(Protocol):
    """Minimal GitHub release asset metadata contract."""

    @property
    def name(self) -> str: ...

    @property
    def download_url(self) -> str: ...

    @property
    def size(self) -> int | None: ...


@dataclass(frozen=True, slots=True)
class AutoDownloadArtifacts:
    """Typed artifact paths produced by one dataset acquisition run."""

    dataset_root: Path
    labels_csv_path: Path | None
    audio_base_dir: Path | None
    source_manifest_path: Path
    files_seen: int
    labels_written: int


class ReadGitHubLatestReleaseAssets(Protocol):
    """Callable contract for latest-release GitHub asset lookup."""

    def __call__(
        self,
        *,
        owner: str,
        repo: str,
    ) -> tuple[str, Sequence[GitHubReleaseAssetMetadataLike]]: ...


class DownloadGoogleDriveFolder(Protocol):
    """Callable contract for Google Drive folder download."""

    def __call__(
        self,
        *,
        folder_url: str,
        destination_root: Path,
    ) -> list[Path]: ...


class DownloadFile(Protocol):
    """Callable contract for one file-download operation."""

    def __call__(
        self,
        *,
        url: str,
        destination_path: Path,
        expected_md5: str | None = None,
        expected_size: int | None = None,
        headers: dict[str, str] | None = None,
    ) -> Path: ...


class ExtractArchivesFromTree(Protocol):
    """Callable contract for tree-scoped archive extraction."""

    def __call__(self, *, search_root: Path, extract_root: Path) -> list[Path]: ...


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


_GeneratedLabelsStatsT = TypeVar(
    "_GeneratedLabelsStatsT",
    bound=GeneratedLabelsStatsLike,
)


def _source_manifest_artifacts_payload(
    *,
    dataset_root: Path,
    labels_csv_path: Path | None,
) -> dict[str, object]:
    """Builds normalized artifact payload fields for one source manifest."""
    return {
        "dataset_root": str(dataset_root),
        "labels_csv_path": (
            str(labels_csv_path) if labels_csv_path is not None else None
        ),
    }


def _source_manifest_stats_payload(
    labels_stats: GeneratedLabelsStatsLike | None,
) -> dict[str, object]:
    """Builds normalized stats payload fields for one source manifest."""
    if labels_stats is None:
        return {}
    return {
        "files_seen": labels_stats.files_seen,
        "labels_written": labels_stats.labels_written,
        "dropped_files": labels_stats.dropped_files,
        "duplicate_conflicts": labels_stats.duplicate_conflicts,
    }


def build_source_manifest_payload(
    *,
    dataset_root: Path,
    source_payload: dict[str, object],
    labels_csv_path: Path | None,
    labels_stats: GeneratedLabelsStatsLike | None,
) -> dict[str, object]:
    """Builds deterministic source-manifest payload fields."""
    return {
        "generated_at_unix": time.time(),
        "source": source_payload,
        "artifacts": _source_manifest_artifacts_payload(
            dataset_root=dataset_root,
            labels_csv_path=labels_csv_path,
        ),
        "stats": _source_manifest_stats_payload(labels_stats),
    }


def write_source_manifest(
    *,
    dataset_root: Path,
    source_manifest_path: Path,
    source_payload: dict[str, object],
    labels_csv_path: Path | None,
    labels_stats: GeneratedLabelsStatsLike | None,
) -> None:
    """Persists one deterministic source manifest payload."""
    payload = build_source_manifest_payload(
        dataset_root=dataset_root,
        source_payload=source_payload,
        labels_csv_path=labels_csv_path,
        labels_stats=labels_stats,
    )
    source_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    source_manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def generate_labels_from_audio_tree(
    *,
    dataset_root: Path,
    search_root: Path,
    labels_csv_path: Path,
    resolver: Callable[[Path], str | None],
    collect_audio_files: Callable[..., list[Path]],
    compute_relative_to_dataset_root: Callable[..., str],
    write_labels_csv: Callable[..., None],
    stats_factory: Callable[..., _GeneratedLabelsStatsT],
    extensions: frozenset[str] = frozenset({".wav"}),
) -> _GeneratedLabelsStatsT:
    """Builds deterministic labels.csv rows from one provider audio tree."""
    audio_files = collect_audio_files(
        search_root=search_root,
        extensions=extensions,
    )
    labels_by_file: dict[str, str] = {}
    dropped_files = 0
    duplicate_conflicts = 0
    for audio_path in audio_files:
        try:
            infer_path = audio_path.relative_to(search_root)
        except ValueError:
            infer_path = audio_path
        mapped_label = resolver(infer_path)
        if mapped_label is None:
            dropped_files += 1
            continue
        relative_path = compute_relative_to_dataset_root(
            dataset_root=dataset_root,
            path=audio_path,
        )
        existing_label = labels_by_file.get(relative_path)
        if existing_label is not None:
            if existing_label != mapped_label:
                duplicate_conflicts += 1
            dropped_files += 1
            continue
        labels_by_file[relative_path] = mapped_label
    write_labels_csv(
        labels_csv_path=labels_csv_path,
        labels_by_file=labels_by_file,
    )
    return stats_factory(
        files_seen=len(audio_files),
        labels_written=len(labels_by_file),
        dropped_files=dropped_files,
        duplicate_conflicts=duplicate_conflicts,
    )


def prepare_pavoque_from_github_release(
    *,
    dataset_root: Path,
    owner: str,
    repo: str,
    labels_file_name: str,
    source_manifest_file_name: str,
    read_github_latest_release_assets: ReadGitHubLatestReleaseAssets,
    download_file: DownloadFile,
    generate_labels_from_audio_tree: GenerateLabelsFromAudioTree,
    infer_label_from_path_tokens: Callable[[Path], str | None],
    write_source_manifest: WriteSourceManifest,
) -> AutoDownloadArtifacts:
    """Downloads one GitHub-release dataset and generates deterministic labels."""
    root = dataset_root.expanduser()
    root.mkdir(parents=True, exist_ok=True)
    tag_name, assets = read_github_latest_release_assets(
        owner=owner,
        repo=repo,
    )
    selected_assets = [
        asset
        for asset in assets
        if asset.name.lower().startswith("pavoque-")
        and asset.name.lower().endswith((".flac", ".wav", ".yaml", ".yml"))
    ]
    if not selected_assets:
        raise RuntimeError(
            "PAVOQUE latest release does not expose expected pavoque-* assets."
        )
    extract_root = root / "raw" / "pavoque"
    extract_root.mkdir(parents=True, exist_ok=True)
    downloaded_audio_assets: list[Path] = []
    downloaded_non_audio_assets: list[Path] = []
    for asset in selected_assets:
        destination_path = extract_root / asset.name
        downloaded_path = download_file(
            url=asset.download_url,
            destination_path=destination_path,
            expected_size=asset.size,
        )
        if downloaded_path.suffix.lower() in {".flac", ".wav"}:
            downloaded_audio_assets.append(downloaded_path)
        else:
            downloaded_non_audio_assets.append(downloaded_path)
    labels_csv_path = root / labels_file_name
    stats = generate_labels_from_audio_tree(
        dataset_root=root,
        search_root=extract_root,
        labels_csv_path=labels_csv_path,
        resolver=infer_label_from_path_tokens,
        extensions=frozenset({".wav", ".flac"}),
    )
    source_manifest_path = root / source_manifest_file_name
    write_source_manifest(
        dataset_root=root,
        source_manifest_path=source_manifest_path,
        source_payload={
            "provider": "github-release",
            "owner": owner,
            "repo": repo,
            "tag_name": tag_name,
            "audio_assets": [path.name for path in downloaded_audio_assets],
            "metadata_assets": [path.name for path in downloaded_non_audio_assets],
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


def prepare_coraa_ser_from_google_drive(
    *,
    dataset_root: Path,
    folder_url: str,
    label_semantics: str,
    labels_file_name: str,
    source_manifest_file_name: str,
    download_google_drive_folder: DownloadGoogleDriveFolder,
    extract_archives_from_tree: ExtractArchivesFromTree,
    generate_labels_from_audio_tree: GenerateLabelsFromAudioTree,
    infer_coraa_ser_label: Callable[[Path], str | None],
    write_source_manifest: WriteSourceManifest,
) -> AutoDownloadArtifacts:
    """Downloads one Google Drive dataset and generates deterministic labels."""
    root = dataset_root.expanduser()
    root.mkdir(parents=True, exist_ok=True)
    downloads_root = root / "downloads" / "coraa-ser"
    downloaded_files = download_google_drive_folder(
        folder_url=folder_url,
        destination_root=downloads_root,
    )
    extract_root = root / "raw" / "coraa-ser"
    extracted_archives = extract_archives_from_tree(
        search_root=downloads_root,
        extract_root=extract_root,
    )
    labels_csv_path = root / labels_file_name
    stats = generate_labels_from_audio_tree(
        dataset_root=root,
        search_root=extract_root,
        labels_csv_path=labels_csv_path,
        resolver=infer_coraa_ser_label,
    )
    source_manifest_path = root / source_manifest_file_name
    write_source_manifest(
        dataset_root=root,
        source_manifest_path=source_manifest_path,
        source_payload={
            "provider": "google-drive",
            "folder_url": folder_url,
            "downloaded_files_count": len(downloaded_files),
            "extracted_archives": [str(path) for path in extracted_archives],
            "label_semantics": label_semantics,
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


__all__ = [
    "AutoDownloadArtifacts",
    "GeneratedLabelsStats",
    "GeneratedLabelsStatsLike",
    "build_source_manifest_payload",
    "generate_labels_from_audio_tree",
    "prepare_coraa_ser_from_google_drive",
    "prepare_pavoque_from_github_release",
    "write_source_manifest",
]
