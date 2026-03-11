"""JL-Corpus download helpers for Hugging Face rows fallback orchestration."""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import Protocol
from urllib import parse

from ser.data.provider_dataset_preparation import (
    AutoDownloadArtifacts,
    GeneratedLabelsStats,
    GeneratedLabelsStatsLike,
)

JlCorpusDownloadStats = GeneratedLabelsStats


class _RequestJson(Protocol):
    def __call__(self, url: str, *, headers: dict[str, str] | None = None) -> object: ...


class _DownloadFile(Protocol):
    def __call__(
        self,
        *,
        url: str,
        destination_path: Path,
        expected_md5: str | None = None,
        expected_size: int | None = None,
        headers: dict[str, str] | None = None,
    ) -> Path: ...


class _InferLabelFromPathTokens(Protocol):
    def __call__(self, path: Path) -> str | None: ...


class _ComputeRelativeToDatasetRoot(Protocol):
    def __call__(self, *, dataset_root: Path, path: Path) -> str: ...


class _WriteLabelsCsv(Protocol):
    def __call__(self, *, labels_csv_path: Path, labels_by_file: dict[str, str]) -> None: ...


class _DownloadKaggleArchive(Protocol):
    def __call__(self, *, dataset_ref: str, destination_path: Path) -> Path: ...


class _EnsureExtractedArchive(Protocol):
    def __call__(self, *, archive_path: Path, extract_root: Path) -> None: ...


class _GenerateLabelsFromAudioTree(Protocol):
    def __call__(
        self,
        *,
        dataset_root: Path,
        search_root: Path,
        labels_csv_path: Path,
        resolver: Callable[[Path], str | None],
        extensions: frozenset[str] = ...,
    ) -> GeneratedLabelsStatsLike: ...


class _WriteSourceManifest(Protocol):
    def __call__(
        self,
        *,
        dataset_root: Path,
        source_manifest_path: Path,
        source_payload: dict[str, object],
        labels_csv_path: Path | None,
        labels_stats: GeneratedLabelsStatsLike | None,
    ) -> None: ...


class _PrepareJlCorpusFromHfRowsFallback(Protocol):
    def __call__(
        self,
        *,
        dataset_root: Path,
        fallback_reason: str,
    ) -> AutoDownloadArtifacts: ...


class _LoggerWarning(Protocol):
    def __call__(self, msg: str, *args: object) -> None: ...


def sanitize_jl_corpus_index(index: str) -> str | None:
    """Sanitizes one JL-Corpus index into a safe local file stem."""
    normalized = re.sub(r"[^a-zA-Z0-9._-]+", "_", index.strip())
    sanitized = normalized.strip("._")
    if not sanitized:
        return None
    return sanitized


def extract_jl_corpus_audio_src(value: object) -> str | None:
    """Extracts one JL-Corpus source audio URL from supported rows API shapes."""
    if isinstance(value, dict):
        src_raw = value.get("src")
        if isinstance(src_raw, str):
            src = src_raw.strip()
            return src or None
        return None
    if isinstance(value, list):
        for item in value:
            if not isinstance(item, dict):
                continue
            src_raw = item.get("src")
            if not isinstance(src_raw, str):
                continue
            src = src_raw.strip()
            if src:
                return src
    return None


def download_jl_corpus_via_hf_rows(
    *,
    dataset_root: Path,
    labels_csv_path: Path,
    rows_api_url: str,
    dataset_id: str,
    config: str,
    split: str,
    page_size: int,
    request_json: _RequestJson,
    download_file: _DownloadFile,
    infer_label_from_path_tokens: _InferLabelFromPathTokens,
    compute_relative_to_dataset_root: _ComputeRelativeToDatasetRoot,
    write_labels_csv: _WriteLabelsCsv,
    sanitize_index: Callable[[str], str | None] = sanitize_jl_corpus_index,
    extract_audio_src: Callable[[object], str | None] = extract_jl_corpus_audio_src,
) -> JlCorpusDownloadStats:
    """Downloads JL-Corpus rows fallback audio and writes deterministic labels."""
    if page_size <= 0:
        raise RuntimeError("JL-Corpus rows API page size must be positive.")

    raw_root = dataset_root / "raw" / "jl-corpus"
    raw_root.mkdir(parents=True, exist_ok=True)
    labels_by_file: dict[str, str] = {}
    files_seen = 0
    dropped_files = 0
    duplicate_conflicts = 0
    offset = 0
    num_rows_total: int | None = None

    while True:
        query = parse.urlencode(
            {
                "dataset": dataset_id,
                "config": config,
                "split": split,
                "offset": str(offset),
                "length": str(page_size),
            }
        )
        payload = request_json(f"{rows_api_url}?{query}")
        if not isinstance(payload, dict):
            raise RuntimeError("Unexpected Hugging Face rows API payload for JL-Corpus.")
        if num_rows_total is None:
            total_raw = payload.get("num_rows_total")
            if not isinstance(total_raw, int) or total_raw < 0:
                raise RuntimeError("Hugging Face rows API did not return a valid `num_rows_total`.")
            num_rows_total = total_raw
        rows_raw = payload.get("rows")
        if not isinstance(rows_raw, list):
            raise RuntimeError("Hugging Face rows API did not return a valid `rows` payload.")
        if not rows_raw:
            break

        for item in rows_raw:
            if not isinstance(item, dict):
                dropped_files += 1
                continue
            row = item.get("row")
            if not isinstance(row, dict):
                dropped_files += 1
                continue
            files_seen += 1
            index_raw = row.get("index")
            if not isinstance(index_raw, str) or not index_raw.strip():
                dropped_files += 1
                continue
            safe_stem = sanitize_index(index_raw)
            if safe_stem is None:
                dropped_files += 1
                continue
            label = infer_label_from_path_tokens(Path(index_raw))
            if label is None:
                dropped_files += 1
                continue
            audio_src = extract_audio_src(row.get("audio"))
            if audio_src is None:
                dropped_files += 1
                continue
            audio_path = raw_root / f"{safe_stem}.wav"
            if not audio_path.is_file() or audio_path.stat().st_size <= 0:
                download_file(url=audio_src, destination_path=audio_path)
            rel_path = compute_relative_to_dataset_root(
                dataset_root=dataset_root,
                path=audio_path,
            )
            existing_label = labels_by_file.get(rel_path)
            if existing_label is not None:
                if existing_label != label:
                    duplicate_conflicts += 1
                dropped_files += 1
                continue
            labels_by_file[rel_path] = label

        offset += len(rows_raw)
        if num_rows_total is not None and offset >= num_rows_total:
            break

    if not labels_by_file:
        raise RuntimeError("Hugging Face JL-Corpus fallback produced no labeled audio samples.")
    write_labels_csv(labels_csv_path=labels_csv_path, labels_by_file=labels_by_file)
    return JlCorpusDownloadStats(
        files_seen=files_seen,
        labels_written=len(labels_by_file),
        dropped_files=dropped_files,
        duplicate_conflicts=duplicate_conflicts,
    )


def prepare_jl_corpus_from_kaggle(
    *,
    dataset_root: Path,
    dataset_ref: str,
    labels_file_name: str,
    source_manifest_file_name: str,
    download_kaggle_archive: _DownloadKaggleArchive,
    ensure_extracted_archive: _EnsureExtractedArchive,
    generate_labels_from_audio_tree: _GenerateLabelsFromAudioTree,
    infer_label_from_path_tokens: _InferLabelFromPathTokens,
    write_source_manifest: _WriteSourceManifest,
    prepare_hf_rows_fallback: _PrepareJlCorpusFromHfRowsFallback,
    logger_warning: _LoggerWarning,
) -> AutoDownloadArtifacts:
    """Downloads JL-Corpus via Kaggle and falls back to HF rows when unavailable."""
    root = dataset_root.expanduser()
    root.mkdir(parents=True, exist_ok=True)
    try:
        archive_path = download_kaggle_archive(
            dataset_ref=dataset_ref,
            destination_path=root / "downloads" / "jl-corpus.zip",
        )
    except RuntimeError as err:
        logger_warning(
            "JL-Corpus Kaggle download unavailable; falling back to public Hugging Face rows API. reason=%s",
            err,
        )
        fallback_artifacts = prepare_hf_rows_fallback(
            dataset_root=root,
            fallback_reason=str(err),
        )
        return AutoDownloadArtifacts(
            dataset_root=fallback_artifacts.dataset_root,
            labels_csv_path=fallback_artifacts.labels_csv_path,
            audio_base_dir=fallback_artifacts.audio_base_dir,
            source_manifest_path=fallback_artifacts.source_manifest_path,
            files_seen=fallback_artifacts.files_seen,
            labels_written=fallback_artifacts.labels_written,
        )
    extract_root = root / "raw" / "jl-corpus"
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
            "provider": "kaggle",
            "dataset_ref": dataset_ref,
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


def prepare_jl_corpus_from_hf_rows(
    *,
    dataset_root: Path,
    fallback_reason: str,
    labels_file_name: str,
    source_manifest_file_name: str,
    dataset_id: str,
    source_url: str,
    rows_api_url: str,
    config: str,
    split: str,
    page_size: int,
    request_json: _RequestJson,
    download_file: _DownloadFile,
    infer_label_from_path_tokens: _InferLabelFromPathTokens,
    compute_relative_to_dataset_root: _ComputeRelativeToDatasetRoot,
    write_labels_csv: _WriteLabelsCsv,
    write_source_manifest: _WriteSourceManifest,
    sanitize_index: Callable[[str], str | None] = sanitize_jl_corpus_index,
    extract_audio_src: Callable[[object], str | None] = extract_jl_corpus_audio_src,
) -> AutoDownloadArtifacts:
    """Downloads JL-Corpus via HF rows fallback and writes source-manifest metadata."""
    root = dataset_root.expanduser()
    root.mkdir(parents=True, exist_ok=True)
    labels_csv_path = root / labels_file_name
    stats = download_jl_corpus_via_hf_rows(
        dataset_root=root,
        labels_csv_path=labels_csv_path,
        rows_api_url=rows_api_url,
        dataset_id=dataset_id,
        config=config,
        split=split,
        page_size=page_size,
        request_json=request_json,
        download_file=download_file,
        infer_label_from_path_tokens=infer_label_from_path_tokens,
        compute_relative_to_dataset_root=compute_relative_to_dataset_root,
        write_labels_csv=write_labels_csv,
        sanitize_index=sanitize_index,
        extract_audio_src=extract_audio_src,
    )
    source_manifest_path = root / source_manifest_file_name
    write_source_manifest(
        dataset_root=root,
        source_manifest_path=source_manifest_path,
        source_payload={
            "provider": "huggingface_rows_api",
            "dataset_id": dataset_id,
            "source_url": source_url,
            "fallback_reason": fallback_reason,
            "rows_api_url": rows_api_url,
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
    "JlCorpusDownloadStats",
    "download_jl_corpus_via_hf_rows",
    "extract_jl_corpus_audio_src",
    "prepare_jl_corpus_from_hf_rows",
    "prepare_jl_corpus_from_kaggle",
    "sanitize_jl_corpus_index",
]
