"""Zenodo dataset preparation helpers for public SER corpora."""

from __future__ import annotations

import csv
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ser.data.provider_dataset_preparation import (
    AutoDownloadArtifacts,
    GeneratedLabelsStats,
    GeneratedLabelsStatsLike,
)


@dataclass(frozen=True, slots=True)
class ZenodoFileMetadata:
    """Resolved download metadata for one file in a Zenodo record."""

    key: str
    url: str
    md5: str | None
    size: int | None


class DownloadZenodoArchive(Protocol):
    """Callable contract for downloading one Zenodo record file."""

    def __call__(
        self,
        *,
        dataset_root: Path,
        record_id: str,
        file_key: str,
    ) -> Path: ...


class EnsureExtractedArchive(Protocol):
    """Callable contract for archive extraction idempotency."""

    def __call__(
        self,
        *,
        archive_path: Path,
        extract_root: Path,
    ) -> None: ...


class GenerateLabelsFromAudioTree(Protocol):
    """Callable contract for filename-token label generation."""

    def __call__(
        self,
        *,
        dataset_root: Path,
        search_root: Path,
        labels_csv_path: Path,
        resolver: Callable[[Path], str | None],
    ) -> GeneratedLabelsStatsLike: ...


class ComputeRelativeToDatasetRoot(Protocol):
    """Callable contract for dataset-root relative path projections."""

    def __call__(
        self,
        *,
        dataset_root: Path,
        path: Path,
    ) -> str: ...


class WriteLabelsCsv(Protocol):
    """Callable contract for deterministic labels.csv persistence."""

    def __call__(
        self, *, labels_csv_path: Path, labels_by_file: dict[str, str]
    ) -> None: ...


class GenerateLabelsFromMetadataCsv(Protocol):
    """Callable contract for metadata-driven label generation."""

    def __call__(
        self,
        *,
        dataset_root: Path,
        metadata_csv_path: Path,
        labels_csv_path: Path,
        audio_search_roots: tuple[Path, ...],
        file_name_keys: tuple[str, ...],
        label_keys: tuple[str, ...],
        label_resolver: Callable[[str], str | None],
    ) -> GeneratedLabelsStatsLike: ...


class CopyFile(Protocol):
    """Callable contract for metadata-file stabilization copies."""

    def __call__(self, source_path: Path, destination_path: Path) -> object: ...


class RequestJson(Protocol):
    """Callable contract for JSON metadata fetchers."""

    def __call__(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
    ) -> object: ...


class DownloadFile(Protocol):
    """Callable contract for one file download operation."""

    def __call__(
        self,
        *,
        url: str,
        destination_path: Path,
        expected_md5: str | None = None,
        expected_size: int | None = None,
        headers: dict[str, str] | None = None,
    ) -> Path: ...


def parse_zenodo_md5(checksum: str | None) -> str | None:
    """Parses md5 checksum tokens from Zenodo file checksum fields."""
    if checksum is None:
        return None
    normalized = checksum.strip().lower()
    if normalized.startswith("md5:"):
        digest = normalized.removeprefix("md5:").strip()
        return digest or None
    return None


def read_zenodo_file_metadata(
    *,
    record_id: str,
    file_key: str,
    request_json: RequestJson,
    parse_md5_checksum: Callable[[str | None], str | None] = parse_zenodo_md5,
) -> ZenodoFileMetadata:
    """Reads metadata for one Zenodo record file key."""
    record_url = f"https://zenodo.org/api/records/{record_id}"
    raw = request_json(record_url)
    if not isinstance(raw, dict):
        raise RuntimeError(f"Unexpected Zenodo payload shape for record {record_id}.")
    files = raw.get("files")
    if not isinstance(files, list):
        raise RuntimeError(f"Zenodo record {record_id} does not expose files metadata.")
    for entry in files:
        if not isinstance(entry, dict):
            continue
        if entry.get("key") != file_key:
            continue
        links = entry.get("links")
        if not isinstance(links, dict):
            break
        url = links.get("self")
        if not isinstance(url, str) or not url:
            break
        checksum = entry.get("checksum")
        md5 = parse_md5_checksum(checksum if isinstance(checksum, str) else None)
        size_raw = entry.get("size")
        size = size_raw if isinstance(size_raw, int) and size_raw >= 0 else None
        return ZenodoFileMetadata(key=file_key, url=url, md5=md5, size=size)
    raise RuntimeError(
        f"Zenodo record {record_id} does not contain expected file key {file_key!r}."
    )


def download_zenodo_archive(
    *,
    dataset_root: Path,
    record_id: str,
    file_key: str,
    request_json: RequestJson,
    download_file: DownloadFile,
) -> Path:
    """Downloads one archive key from a Zenodo record."""
    file_metadata = read_zenodo_file_metadata(
        record_id=record_id,
        file_key=file_key,
        request_json=request_json,
    )
    downloads_dir = dataset_root / "downloads"
    archive_path = downloads_dir / file_metadata.key
    return download_file(
        url=file_metadata.url,
        destination_path=archive_path,
        expected_md5=file_metadata.md5,
        expected_size=file_metadata.size,
    )


def generate_labels_from_metadata_csv(
    *,
    dataset_root: Path,
    metadata_csv_path: Path,
    labels_csv_path: Path,
    audio_search_roots: tuple[Path, ...],
    file_name_keys: tuple[str, ...],
    label_keys: tuple[str, ...],
    label_resolver: Callable[[str], str | None],
    compute_relative_to_dataset_root: ComputeRelativeToDatasetRoot,
    write_labels_csv: WriteLabelsCsv,
) -> GeneratedLabelsStats:
    """Generates labels.csv rows from metadata records with local-audio filtering."""
    rows_seen = 0
    dropped_rows = 0
    duplicate_conflicts = 0
    labels_by_file: dict[str, str] = {}
    normalized_audio_roots = tuple(
        root.expanduser().resolve() for root in audio_search_roots
    )

    with metadata_csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows_seen += 1
            file_name = ""
            for key in file_name_keys:
                value = row.get(key)
                if isinstance(value, str) and value.strip():
                    file_name = value.strip()
                    break
            if not file_name:
                dropped_rows += 1
                continue
            raw_label = ""
            for key in label_keys:
                value = row.get(key)
                if isinstance(value, str) and value.strip():
                    raw_label = value.strip()
                    break
            if not raw_label:
                dropped_rows += 1
                continue
            mapped_label = label_resolver(raw_label)
            if mapped_label is None:
                dropped_rows += 1
                continue

            normalized_file_name = file_name.replace("\\", "/").lstrip("./")
            candidate_paths: list[Path] = []
            for audio_root in normalized_audio_roots:
                candidate_paths.append(audio_root / normalized_file_name)
            basename = Path(normalized_file_name).name
            if basename:
                for audio_root in normalized_audio_roots:
                    matches = sorted(audio_root.rglob(basename))
                    if matches:
                        candidate_paths.extend(matches)

            selected_path: Path | None = None
            for candidate in candidate_paths:
                if candidate.is_file():
                    selected_path = candidate
                    break
            if selected_path is None:
                dropped_rows += 1
                continue

            rel = compute_relative_to_dataset_root(
                dataset_root=dataset_root,
                path=selected_path,
            )
            existing = labels_by_file.get(rel)
            if existing is not None:
                if existing != mapped_label:
                    duplicate_conflicts += 1
                dropped_rows += 1
                continue
            labels_by_file[rel] = mapped_label

    write_labels_csv(labels_csv_path=labels_csv_path, labels_by_file=labels_by_file)
    return GeneratedLabelsStats(
        files_seen=rows_seen,
        labels_written=len(labels_by_file),
        dropped_files=dropped_rows,
        duplicate_conflicts=duplicate_conflicts,
    )


def prepare_ravdess_from_zenodo(
    *,
    dataset_root: Path,
    record_id: str,
    file_key: str,
    source_manifest_file_name: str,
    download_zenodo_archive: DownloadZenodoArchive,
    ensure_extracted_archive: EnsureExtractedArchive,
    collect_wav_files: Callable[[Path], list[Path]],
    write_source_manifest: Callable[..., None],
) -> AutoDownloadArtifacts:
    """Downloads and extracts RAVDESS archive payload from Zenodo."""
    root = dataset_root.expanduser()
    root.mkdir(parents=True, exist_ok=True)
    archive_path = download_zenodo_archive(
        dataset_root=root,
        record_id=record_id,
        file_key=file_key,
    )
    ensure_extracted_archive(archive_path=archive_path, extract_root=root)
    source_manifest_path = root / source_manifest_file_name
    write_source_manifest(
        dataset_root=root,
        source_manifest_path=source_manifest_path,
        source_payload={
            "provider": "zenodo",
            "record_id": record_id,
            "file_key": file_key,
            "archive_path": str(archive_path),
        },
        labels_csv_path=None,
        labels_stats=None,
    )
    speech_files = len(collect_wav_files(root))
    return AutoDownloadArtifacts(
        dataset_root=root,
        labels_csv_path=None,
        audio_base_dir=None,
        source_manifest_path=source_manifest_path,
        files_seen=speech_files,
        labels_written=0,
    )


def prepare_escorpus_pe_from_zenodo(
    *,
    dataset_root: Path,
    record_id: str,
    file_key: str,
    labels_file_name: str,
    source_manifest_file_name: str,
    download_zenodo_archive: DownloadZenodoArchive,
    ensure_extracted_archive: EnsureExtractedArchive,
    generate_labels_from_audio_tree: GenerateLabelsFromAudioTree,
    infer_escorpus_pe_label: Callable[[Path], str | None],
    write_source_manifest: Callable[..., None],
) -> AutoDownloadArtifacts:
    """Downloads ESCorpus-PE and generates deterministic weak labels."""
    root = dataset_root.expanduser()
    root.mkdir(parents=True, exist_ok=True)
    archive_path = download_zenodo_archive(
        dataset_root=root,
        record_id=record_id,
        file_key=file_key,
    )
    extract_root = root / "raw" / "escorpus-pe"
    ensure_extracted_archive(archive_path=archive_path, extract_root=extract_root)
    labels_csv_path = root / labels_file_name
    stats = generate_labels_from_audio_tree(
        dataset_root=root,
        search_root=extract_root,
        labels_csv_path=labels_csv_path,
        resolver=infer_escorpus_pe_label,
    )
    source_manifest_path = root / source_manifest_file_name
    write_source_manifest(
        dataset_root=root,
        source_manifest_path=source_manifest_path,
        source_payload={
            "provider": "zenodo",
            "record_id": record_id,
            "file_key": file_key,
            "archive_path": str(archive_path),
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


def prepare_oreau_french_esd_from_zenodo(
    *,
    dataset_root: Path,
    record_id: str,
    rar_keys: tuple[str, ...],
    doc_key: str,
    labels_file_name: str,
    source_manifest_file_name: str,
    download_zenodo_archive: DownloadZenodoArchive,
    ensure_extracted_archive: EnsureExtractedArchive,
    generate_labels_from_audio_tree: GenerateLabelsFromAudioTree,
    infer_label_from_path_tokens: Callable[[Path], str | None],
    write_source_manifest: Callable[..., None],
) -> AutoDownloadArtifacts:
    """Downloads Oreau French ESD archives and generates inferred labels."""
    root = dataset_root.expanduser()
    root.mkdir(parents=True, exist_ok=True)
    archives: list[Path] = []
    for key in (*rar_keys, doc_key):
        archives.append(
            download_zenodo_archive(
                dataset_root=root,
                record_id=record_id,
                file_key=key,
            )
        )
    extract_root = root / "raw" / "oreau-french-esd"
    for archive_path in archives:
        ensure_extracted_archive(archive_path=archive_path, extract_root=extract_root)
    labels_csv_path = root / labels_file_name
    stats = generate_labels_from_audio_tree(
        dataset_root=root,
        search_root=extract_root,
        labels_csv_path=labels_csv_path,
        resolver=infer_label_from_path_tokens,
    )
    source_manifest_path = root / source_manifest_file_name
    write_source_manifest(
        dataset_root=root,
        source_manifest_path=source_manifest_path,
        source_payload={
            "provider": "zenodo",
            "record_id": record_id,
            "file_keys": [*rar_keys, doc_key],
            "archive_paths": [str(path) for path in archives],
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


def prepare_emodb_2_from_zenodo(
    *,
    dataset_root: Path,
    record_id: str,
    file_key: str,
    labels_file_name: str,
    source_manifest_file_name: str,
    emodb_label_map: Mapping[str, str],
    download_zenodo_archive: DownloadZenodoArchive,
    ensure_extracted_archive: EnsureExtractedArchive,
    compute_relative_to_dataset_root: ComputeRelativeToDatasetRoot,
    write_labels_csv: WriteLabelsCsv,
    write_source_manifest: Callable[..., None],
) -> AutoDownloadArtifacts:
    """Downloads EmoDB 2.0 metadata/audio and materializes deterministic labels."""
    root = dataset_root.expanduser()
    root.mkdir(parents=True, exist_ok=True)
    archive_path = download_zenodo_archive(
        dataset_root=root,
        record_id=record_id,
        file_key=file_key,
    )
    extract_root = root / "raw" / "emodb-2.0"
    ensure_extracted_archive(archive_path=archive_path, extract_root=extract_root)

    metadata_candidates = sorted(
        extract_root.rglob("db.emotion.categories.ambiguous.csv")
    )
    if not metadata_candidates:
        raise RuntimeError(
            "EmoDB 2.0 metadata file `db.emotion.categories.ambiguous.csv` not found after extraction."
        )
    metadata_csv_path = metadata_candidates[0]
    wav_root_candidates = [path for path in extract_root.rglob("wav") if path.is_dir()]
    if not wav_root_candidates:
        raise RuntimeError("EmoDB 2.0 WAV directory not found after extraction.")
    wav_root = wav_root_candidates[0]

    labels_by_file: dict[str, str] = {}
    dropped_rows = 0
    rows_seen = 0
    with metadata_csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows_seen += 1
            file_name = (row.get("file") or "").strip()
            emotion_raw = (row.get("emotion") or "").strip().lower()
            if not file_name or not emotion_raw:
                dropped_rows += 1
                continue
            mapped = emodb_label_map.get(emotion_raw)
            if mapped is None:
                dropped_rows += 1
                continue
            normalized_name = (
                file_name if file_name.lower().endswith(".wav") else f"{file_name}.wav"
            )
            candidate_path = wav_root / normalized_name
            if not candidate_path.is_file():
                basename_matches = list(wav_root.rglob(Path(normalized_name).name))
                if not basename_matches:
                    dropped_rows += 1
                    continue
                candidate_path = basename_matches[0]
            rel = compute_relative_to_dataset_root(
                dataset_root=root,
                path=candidate_path,
            )
            labels_by_file[rel] = mapped

    labels_csv_path = root / labels_file_name
    write_labels_csv(labels_csv_path=labels_csv_path, labels_by_file=labels_by_file)
    source_manifest_path = root / source_manifest_file_name
    write_source_manifest(
        dataset_root=root,
        source_manifest_path=source_manifest_path,
        source_payload={
            "provider": "zenodo",
            "record_id": record_id,
            "file_key": file_key,
            "archive_path": str(archive_path),
            "metadata_csv_path": str(metadata_csv_path),
        },
        labels_csv_path=labels_csv_path,
        labels_stats=GeneratedLabelsStats(
            files_seen=rows_seen,
            labels_written=len(labels_by_file),
            dropped_files=dropped_rows,
            duplicate_conflicts=0,
        ),
    )
    return AutoDownloadArtifacts(
        dataset_root=root,
        labels_csv_path=labels_csv_path,
        audio_base_dir=root,
        source_manifest_path=source_manifest_path,
        files_seen=rows_seen,
        labels_written=len(labels_by_file),
    )


def prepare_cafe_from_zenodo(
    *,
    dataset_root: Path,
    record_id: str,
    archive_keys: tuple[str, ...],
    labels_file_name: str,
    source_manifest_file_name: str,
    download_zenodo_archive: DownloadZenodoArchive,
    ensure_extracted_archive: EnsureExtractedArchive,
    generate_labels_from_audio_tree: GenerateLabelsFromAudioTree,
    infer_label_from_path_tokens: Callable[[Path], str | None],
    write_source_manifest: Callable[..., None],
) -> AutoDownloadArtifacts:
    """Downloads CaFE archives and generates inferred labels."""
    root = dataset_root.expanduser()
    root.mkdir(parents=True, exist_ok=True)
    archives = [
        download_zenodo_archive(
            dataset_root=root,
            record_id=record_id,
            file_key=file_key,
        )
        for file_key in archive_keys
    ]
    extract_root = root / "raw" / "cafe"
    for archive_path in archives:
        ensure_extracted_archive(archive_path=archive_path, extract_root=extract_root)
    labels_csv_path = root / labels_file_name
    stats = generate_labels_from_audio_tree(
        dataset_root=root,
        search_root=extract_root,
        labels_csv_path=labels_csv_path,
        resolver=infer_label_from_path_tokens,
    )
    source_manifest_path = root / source_manifest_file_name
    write_source_manifest(
        dataset_root=root,
        source_manifest_path=source_manifest_path,
        source_payload={
            "provider": "zenodo",
            "record_id": record_id,
            "file_keys": [*archive_keys],
            "archive_paths": [str(path) for path in archives],
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


def prepare_spanish_meacorpus_2023_from_zenodo(
    *,
    dataset_root: Path,
    record_id: str,
    metadata_key: str,
    labels_file_name: str,
    source_manifest_file_name: str,
    download_zenodo_archive: DownloadZenodoArchive,
    copy_file: CopyFile,
    generate_labels_from_metadata_csv: GenerateLabelsFromMetadataCsv,
    write_source_manifest: Callable[..., None],
) -> AutoDownloadArtifacts:
    """Downloads Spanish MEACorpus metadata and labels present local clips."""
    root = dataset_root.expanduser()
    root.mkdir(parents=True, exist_ok=True)
    metadata_path = download_zenodo_archive(
        dataset_root=root,
        record_id=record_id,
        file_key=metadata_key,
    )
    metadata_root = root / "metadata"
    metadata_root.mkdir(parents=True, exist_ok=True)
    stable_metadata_path = metadata_root / metadata_path.name
    if metadata_path != stable_metadata_path:
        copy_file(metadata_path, stable_metadata_path)

    meacorpus_label_map: dict[str, str] = {
        "anger": "angry",
        "disgust": "disgust",
        "fear": "fearful",
        "joy": "happy",
        "neutral": "neutral",
        "sadness": "sad",
    }

    def _resolve_meacorpus_label(raw_label: str) -> str | None:
        return meacorpus_label_map.get(raw_label.strip().lower())

    labels_csv_path = root / labels_file_name
    stats = generate_labels_from_metadata_csv(
        dataset_root=root,
        metadata_csv_path=stable_metadata_path,
        labels_csv_path=labels_csv_path,
        audio_search_roots=(root / "raw" / "spanish-meacorpus-2023",),
        file_name_keys=("filename", "file_name", "FileName"),
        label_keys=("label", "emotion", "EmoClass"),
        label_resolver=_resolve_meacorpus_label,
    )
    source_manifest_path = root / source_manifest_file_name
    write_source_manifest(
        dataset_root=root,
        source_manifest_path=source_manifest_path,
        source_payload={
            "provider": "zenodo",
            "record_id": record_id,
            "metadata_key": metadata_key,
            "metadata_csv_path": str(stable_metadata_path),
            "audio_distribution_note": (
                "Audio clips are not redistributed in the Zenodo package; "
                "labels.csv only includes rows for locally available audio files."
            ),
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


def prepare_asvp_esd_from_zenodo(
    *,
    dataset_root: Path,
    record_id: str,
    file_key: str,
    labels_file_name: str,
    source_manifest_file_name: str,
    download_zenodo_archive: DownloadZenodoArchive,
    ensure_extracted_archive: EnsureExtractedArchive,
    generate_labels_from_audio_tree: GenerateLabelsFromAudioTree,
    infer_label_from_path_tokens: Callable[[Path], str | None],
    write_source_manifest: Callable[..., None],
) -> AutoDownloadArtifacts:
    """Downloads ASVP-ESD archive and generates inferred labels."""
    root = dataset_root.expanduser()
    root.mkdir(parents=True, exist_ok=True)
    archive_path = download_zenodo_archive(
        dataset_root=root,
        record_id=record_id,
        file_key=file_key,
    )
    extract_root = root / "raw" / "asvp-esd"
    ensure_extracted_archive(archive_path=archive_path, extract_root=extract_root)
    labels_csv_path = root / labels_file_name
    stats = generate_labels_from_audio_tree(
        dataset_root=root,
        search_root=extract_root,
        labels_csv_path=labels_csv_path,
        resolver=infer_label_from_path_tokens,
    )
    source_manifest_path = root / source_manifest_file_name
    write_source_manifest(
        dataset_root=root,
        source_manifest_path=source_manifest_path,
        source_payload={
            "provider": "zenodo",
            "record_id": record_id,
            "file_key": file_key,
            "archive_path": str(archive_path),
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
    "ZenodoFileMetadata",
    "download_zenodo_archive",
    "generate_labels_from_metadata_csv",
    "parse_zenodo_md5",
    "prepare_emodb_2_from_zenodo",
    "prepare_spanish_meacorpus_2023_from_zenodo",
    "prepare_asvp_esd_from_zenodo",
    "prepare_cafe_from_zenodo",
    "prepare_escorpus_pe_from_zenodo",
    "prepare_oreau_french_esd_from_zenodo",
    "prepare_ravdess_from_zenodo",
    "read_zenodo_file_metadata",
]
