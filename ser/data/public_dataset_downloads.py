"""Programmatic acquisition helpers for public SER datasets.

This module centralizes source-specific download logic (Zenodo, Mendeley Data,
Kaggle), archive extraction, and deterministic labels.csv generation for
datasets that expose machine-downloadable artifacts.
"""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

from ser.data import archive_extraction as archive_extraction_helpers
from ser.data import jl_corpus_downloads as jl_corpus_download_helpers
from ser.data import (
    mendeley_dataset_preparation as mendeley_dataset_preparation_helpers,
)
from ser.data import mendeley_downloads as mendeley_download_helpers
from ser.data import openslr_dataset_preparation as openslr_dataset_preparation_helpers
from ser.data import openslr_downloads as openslr_download_helpers
from ser.data import openslr_resolution as openslr_resolution_helpers
from ser.data import (
    provider_dataset_preparation as provider_dataset_preparation_helpers,
)
from ser.data import provider_downloads as provider_download_helpers
from ser.data import zenodo_downloads as zenodo_download_helpers
from ser.data.provider_dataset_preparation import (
    AutoDownloadArtifacts,
    GeneratedLabelsStats,
)
from ser.data.public_dataset_label_inference import (
    infer_att_hack_label as _infer_att_hack_label,
)
from ser.data.public_dataset_label_inference import (
    infer_coraa_ser_label as _infer_coraa_ser_label,
)
from ser.data.public_dataset_label_inference import (
    infer_escorpus_pe_label as _infer_escorpus_pe_label,
)
from ser.data.public_dataset_label_inference import (
    infer_label_from_path_tokens as _infer_label_from_path_tokens,
)
from ser.data.public_dataset_label_inference import (
    infer_mesd_label as _infer_mesd_label,
)
from ser.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_HTTP_TIMEOUT_SECONDS = 60.0
DEFAULT_HTTP_RETRIES = 3
DEFAULT_HTTP_RETRY_BASE_SECONDS = 1.0
HTTP_CHUNK_SIZE = 1024 * 1024
DEFAULT_LABELS_FILE_NAME = "labels.csv"
DEFAULT_SOURCE_MANIFEST_FILE_NAME = "source_manifest.json"

EMODB_2_ZENODO_RECORD_ID = "17651657"
EMODB_2_ZENODO_ARCHIVE_KEY = "emodb_2.0.zip"

ESCORPUS_PE_ZENODO_RECORD_ID = "5793223"
ESCORPUS_PE_ZENODO_ARCHIVE_KEY = "Corpus_Globalv1.zip"

RAVDESS_ZENODO_RECORD_ID = "1188976"
RAVDESS_ZENODO_ARCHIVE_KEY = "Audio_Speech_Actors_01-24.zip"

OREAU_FRENCH_ESD_ZENODO_RECORD_ID = "4405783"
OREAU_FRENCH_ESD_RAR_KEYS = ("Or\u00e9auFR_01.rar", "Or\u00e9auFR_02.rar")
OREAU_FRENCH_ESD_DOC_KEY = "Doc_FR.rar"

MESD_MENDELEY_DATASET_ID = "cy34mh68j9"
MESD_DEFAULT_VERSION = 5

JL_CORPUS_KAGGLE_DATASET_REF = "tli725/jl-corpus"
JL_CORPUS_HF_DATASET_ID = "CLAPv2/JL-Corpus"
JL_CORPUS_HF_SOURCE_URL = "https://huggingface.co/datasets/CLAPv2/JL-Corpus"
JL_CORPUS_HF_ROWS_API_URL = "https://datasets-server.huggingface.co/rows"
JL_CORPUS_HF_CONFIG = "default"
JL_CORPUS_HF_SPLIT = "train"
JL_CORPUS_HF_PAGE_SIZE = 100

CAFE_ZENODO_RECORD_ID = "1478765"
CAFE_ZENODO_ARCHIVE_KEYS = ("CaFE_192k_1.zip", "CaFE_192k_2.zip")

ASVP_ESD_ZENODO_RECORD_ID = "7132783"
ASVP_ESD_ZENODO_ARCHIVE_KEY = "ASVP-ESD-Update.zip"

EMOV_DB_OPENSLR_DATASET_ID = "115"
EMOV_DB_OPENSLR_ARCHIVE_SUFFIXES = (".tar.gz", ".tgz")

PAVOQUE_GITHUB_OWNER = "marytts"
PAVOQUE_GITHUB_REPO = "pavoque-data"

ATT_HACK_OPENSLR_DATASET_ID = "88"
ATT_HACK_OPENSLR_ARCHIVE_SUFFIXES = (".tgz",)

CORAA_SER_GOOGLE_DRIVE_FOLDER_URL = (
    "https://drive.google.com/drive/folders/12Nuv8J7pBHJuNU3nH2c7F8VwCDEE6GDt"
)

SPANISH_MEACORPUS_2023_ZENODO_RECORD_ID = "18606423"
SPANISH_MEACORPUS_2023_METADATA_KEY = "spanish-meacorpus-2023-dataset.csv"


_GitHubReleaseAssetMetadata = provider_download_helpers.GitHubReleaseAssetMetadata


def _request_json(url: str, *, headers: dict[str, str] | None = None) -> object:
    """Fetches one JSON payload with retries."""
    return provider_download_helpers.request_json_with_retries(
        url=url,
        headers=headers,
        timeout_seconds=DEFAULT_HTTP_TIMEOUT_SECONDS,
        with_retries=_with_retries,
    )


def _with_retries[T](*, description: str, action: Callable[[], T]) -> T:
    """Runs one callable with bounded retries and jittered backoff."""
    return provider_download_helpers.run_with_retries(
        description=description,
        action=action,
        retries=DEFAULT_HTTP_RETRIES,
        retry_base_seconds=DEFAULT_HTTP_RETRY_BASE_SECONDS,
        logger=logger,
    )


def _compute_file_md5(path: Path) -> str:
    return provider_download_helpers.compute_file_md5(
        path=path,
        chunk_size=HTTP_CHUNK_SIZE,
    )


def _compute_relative_to_dataset_root(*, dataset_root: Path, path: Path) -> str:
    return path.expanduser().resolve().relative_to(dataset_root.resolve()).as_posix()


def _download_file(
    *,
    url: str,
    destination_path: Path,
    expected_md5: str | None = None,
    expected_size: int | None = None,
    headers: dict[str, str] | None = None,
) -> Path:
    """Downloads one remote file atomically and verifies size/checksum when available."""
    return provider_download_helpers.download_file_with_retries(
        url=url,
        destination_path=destination_path,
        expected_md5=expected_md5,
        expected_size=expected_size,
        headers=headers,
        with_retries=_with_retries,
        compute_file_md5=_compute_file_md5,
        timeout_seconds=DEFAULT_HTTP_TIMEOUT_SECONDS,
        chunk_size=HTTP_CHUNK_SIZE,
    )


def _is_safe_destination_path(*, extract_root: Path, destination_path: Path) -> bool:
    return archive_extraction_helpers.is_safe_destination_path(
        extract_root=extract_root,
        destination_path=destination_path,
    )


def _extract_zip_archive(*, archive_path: Path, extract_root: Path) -> None:
    archive_extraction_helpers.extract_zip_archive(
        archive_path=archive_path,
        extract_root=extract_root,
        is_safe_destination=_is_safe_destination_path,
    )


def _extract_rar_archive(*, archive_path: Path, extract_root: Path) -> None:
    archive_extraction_helpers.extract_rar_archive(
        archive_path=archive_path,
        extract_root=extract_root,
        os_name=os.name,
        which=shutil.which,
        run=subprocess.run,
        logger_warning=logger.warning,
    )


def _extract_tar_archive(*, archive_path: Path, extract_root: Path) -> None:
    archive_extraction_helpers.extract_tar_archive(
        archive_path=archive_path,
        extract_root=extract_root,
        is_safe_destination=_is_safe_destination_path,
    )


def _extract_archive(*, archive_path: Path, extract_root: Path) -> None:
    archive_extraction_helpers.extract_archive(
        archive_path=archive_path,
        extract_root=extract_root,
        extract_zip=_extract_zip_archive,
        extract_rar=_extract_rar_archive,
        extract_tar=_extract_tar_archive,
    )


def _ensure_extracted_archive(*, archive_path: Path, extract_root: Path) -> None:
    archive_extraction_helpers.ensure_extracted_archive(
        archive_path=archive_path,
        extract_root=extract_root,
        extract_archive_fn=_extract_archive,
        current_time=time.time,
    )


def _write_labels_csv(*, labels_csv_path: Path, labels_by_file: dict[str, str]) -> None:
    labels_csv_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = labels_csv_path.with_suffix(labels_csv_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["FileName", "emotion"])
        writer.writeheader()
        for file_name in sorted(labels_by_file):
            writer.writerow(
                {"FileName": file_name, "emotion": labels_by_file[file_name]}
            )
    os.replace(tmp_path, labels_csv_path)


_EMODB_LABEL_MAP: dict[str, str] = {
    "anger": "angry",
    "boredom": "neutral",
    "disgust": "disgust",
    "fear": "fearful",
    "happiness": "happy",
    "neutral": "neutral",
    "sadness": "sad",
}


def _collect_audio_files(
    *,
    search_root: Path,
    extensions: frozenset[str],
) -> list[Path]:
    normalized_extensions = frozenset(ext.lower() for ext in extensions)
    files: list[Path] = []
    for path in sorted(search_root.rglob("*")):
        if path.is_file() and path.suffix.lower() in normalized_extensions:
            files.append(path)
    return files


def _collect_wav_files(search_root: Path) -> list[Path]:
    return _collect_audio_files(search_root=search_root, extensions=frozenset({".wav"}))


def _generate_labels_from_audio_tree(
    *,
    dataset_root: Path,
    search_root: Path,
    labels_csv_path: Path,
    resolver: Callable[[Path], str | None],
    extensions: frozenset[str] = frozenset({".wav"}),
) -> GeneratedLabelsStats:
    return provider_dataset_preparation_helpers.generate_labels_from_audio_tree(
        dataset_root=dataset_root,
        search_root=search_root,
        labels_csv_path=labels_csv_path,
        resolver=resolver,
        collect_audio_files=_collect_audio_files,
        compute_relative_to_dataset_root=_compute_relative_to_dataset_root,
        write_labels_csv=_write_labels_csv,
        stats_factory=GeneratedLabelsStats,
        extensions=extensions,
    )


def _generate_labels_from_metadata_csv(
    *,
    dataset_root: Path,
    metadata_csv_path: Path,
    labels_csv_path: Path,
    audio_search_roots: tuple[Path, ...],
    file_name_keys: tuple[str, ...],
    label_keys: tuple[str, ...],
    label_resolver: Callable[[str], str | None],
) -> GeneratedLabelsStats:
    return zenodo_download_helpers.generate_labels_from_metadata_csv(
        dataset_root=dataset_root,
        metadata_csv_path=metadata_csv_path,
        labels_csv_path=labels_csv_path,
        audio_search_roots=audio_search_roots,
        file_name_keys=file_name_keys,
        label_keys=label_keys,
        label_resolver=label_resolver,
        compute_relative_to_dataset_root=_compute_relative_to_dataset_root,
        write_labels_csv=_write_labels_csv,
    )


def _write_source_manifest(
    *,
    dataset_root: Path,
    source_manifest_path: Path,
    source_payload: dict[str, object],
    labels_csv_path: Path | None,
    labels_stats: provider_dataset_preparation_helpers.GeneratedLabelsStatsLike | None,
) -> None:
    provider_dataset_preparation_helpers.write_source_manifest(
        dataset_root=dataset_root,
        source_manifest_path=source_manifest_path,
        source_payload=source_payload,
        labels_csv_path=labels_csv_path,
        labels_stats=labels_stats,
    )


def _download_zenodo_archive(
    *,
    dataset_root: Path,
    record_id: str,
    file_key: str,
) -> Path:
    return zenodo_download_helpers.download_zenodo_archive(
        dataset_root=dataset_root,
        record_id=record_id,
        file_key=file_key,
        request_json=_request_json,
        download_file=_download_file,
    )


def _download_openslr_archives(
    *,
    dataset_root: Path,
    dataset_id: str,
    archive_suffixes: tuple[str, ...],
) -> list[Path]:
    return openslr_download_helpers.download_openslr_archives(
        dataset_root=dataset_root,
        dataset_id=dataset_id,
        archive_suffixes=archive_suffixes,
        resolve_pinned_artifacts=openslr_resolution_helpers.resolve_openslr_pinned_artifacts,
        read_archive_urls=lambda dataset_id, archive_suffixes: (
            openslr_download_helpers.read_openslr_archive_urls(
                dataset_id=dataset_id,
                archive_suffixes=archive_suffixes,
                read_archive_urls_from_hf_script=lambda dataset_id, archive_suffixes: (
                    openslr_download_helpers.read_openslr_archive_urls_from_hf_script(
                        dataset_id=dataset_id,
                        archive_suffixes=archive_suffixes,
                        script_url=openslr_resolution_helpers.OPENSLR_HF_SCRIPT_URL,
                        with_retries=_with_retries,
                        timeout_seconds=DEFAULT_HTTP_TIMEOUT_SECONDS,
                        extract_openslr_files_from_hf_script=(
                            openslr_resolution_helpers.extract_openslr_files_from_hf_script
                        ),
                        build_canonical_archive_urls=(
                            openslr_resolution_helpers.build_canonical_archive_urls
                        ),
                    )
                ),
                with_retries=_with_retries,
                timeout_seconds=DEFAULT_HTTP_TIMEOUT_SECONDS,
                extract_archive_urls_from_listing_html=(
                    openslr_resolution_helpers.extract_archive_urls_from_listing_html
                ),
                log_hf_metadata_resolution_failure=lambda *, dataset_id, error: (
                    logger.warning(
                        "Could not resolve OpenSLR SLR%s via Hugging Face metadata; falling back to page parsing: %s",
                        dataset_id,
                        error,
                    )
                ),
            )
        ),
        download_file=_download_file,
        log_mirror_failure=lambda dataset_id, file_name, url: logger.warning(
            "OpenSLR SLR%s artifact %s failed from %s; trying next mirror.",
            dataset_id,
            file_name,
            url,
        ),
    )


def _read_github_latest_release_assets(
    *,
    owner: str,
    repo: str,
) -> tuple[str, list[_GitHubReleaseAssetMetadata]]:
    return provider_download_helpers.read_github_latest_release_assets(
        owner=owner,
        repo=repo,
        request_json=_request_json,
    )


def _download_google_drive_folder(
    *,
    folder_url: str,
    destination_root: Path,
) -> list[Path]:
    return provider_download_helpers.download_google_drive_folder(
        folder_url=folder_url,
        destination_root=destination_root,
        which=shutil.which,
        run=subprocess.run,
    )


def _extract_archives_from_tree(*, search_root: Path, extract_root: Path) -> list[Path]:
    return archive_extraction_helpers.extract_archives_from_tree(
        search_root=search_root,
        extract_root=extract_root,
        ensure_extracted=_ensure_extracted_archive,
    )


def _kaggle_credentials_from_env() -> tuple[str | None, str | None]:
    return provider_download_helpers.kaggle_credentials_from_env()


def _download_kaggle_archive(*, dataset_ref: str, destination_path: Path) -> Path:
    return provider_download_helpers.download_kaggle_archive(
        dataset_ref=dataset_ref,
        destination_path=destination_path,
        download_file=_download_file,
        logger_warning=logger.warning,
        resolve_credentials=_kaggle_credentials_from_env,
        which=shutil.which,
        run=subprocess.run,
        replace_file=os.replace,
    )


def _sanitize_jl_corpus_index(index: str) -> str | None:
    return jl_corpus_download_helpers.sanitize_jl_corpus_index(index)


def _extract_jl_corpus_audio_src(value: object) -> str | None:
    return jl_corpus_download_helpers.extract_jl_corpus_audio_src(value)


def _download_jl_corpus_via_hf_rows(*, dataset_root: Path) -> GeneratedLabelsStats:
    labels_csv_path = dataset_root / DEFAULT_LABELS_FILE_NAME
    return jl_corpus_download_helpers.download_jl_corpus_via_hf_rows(
        dataset_root=dataset_root,
        labels_csv_path=labels_csv_path,
        rows_api_url=JL_CORPUS_HF_ROWS_API_URL,
        dataset_id=JL_CORPUS_HF_DATASET_ID,
        config=JL_CORPUS_HF_CONFIG,
        split=JL_CORPUS_HF_SPLIT,
        page_size=JL_CORPUS_HF_PAGE_SIZE,
        request_json=_request_json,
        download_file=_download_file,
        infer_label_from_path_tokens=_infer_label_from_path_tokens,
        compute_relative_to_dataset_root=_compute_relative_to_dataset_root,
        write_labels_csv=_write_labels_csv,
        sanitize_index=_sanitize_jl_corpus_index,
        extract_audio_src=_extract_jl_corpus_audio_src,
    )


def _copy_file(source_path: Path, destination_path: Path) -> object:
    return shutil.copy2(source_path, destination_path)


def _prepare_jl_corpus_from_hf_rows(
    *,
    dataset_root: Path,
    fallback_reason: str,
) -> AutoDownloadArtifacts:
    return jl_corpus_download_helpers.prepare_jl_corpus_from_hf_rows(
        dataset_root=dataset_root,
        fallback_reason=fallback_reason,
        labels_file_name=DEFAULT_LABELS_FILE_NAME,
        source_manifest_file_name=DEFAULT_SOURCE_MANIFEST_FILE_NAME,
        dataset_id=JL_CORPUS_HF_DATASET_ID,
        source_url=JL_CORPUS_HF_SOURCE_URL,
        rows_api_url=JL_CORPUS_HF_ROWS_API_URL,
        config=JL_CORPUS_HF_CONFIG,
        split=JL_CORPUS_HF_SPLIT,
        page_size=JL_CORPUS_HF_PAGE_SIZE,
        request_json=_request_json,
        download_file=_download_file,
        infer_label_from_path_tokens=_infer_label_from_path_tokens,
        compute_relative_to_dataset_root=_compute_relative_to_dataset_root,
        write_labels_csv=_write_labels_csv,
        write_source_manifest=_write_source_manifest,
        sanitize_index=_sanitize_jl_corpus_index,
        extract_audio_src=_extract_jl_corpus_audio_src,
    )


def prepare_ravdess_from_zenodo(*, dataset_root: Path) -> AutoDownloadArtifacts:
    """Downloads and extracts the canonical RAVDESS speech archive from Zenodo."""
    return zenodo_download_helpers.prepare_ravdess_from_zenodo(
        dataset_root=dataset_root,
        record_id=RAVDESS_ZENODO_RECORD_ID,
        file_key=RAVDESS_ZENODO_ARCHIVE_KEY,
        source_manifest_file_name=DEFAULT_SOURCE_MANIFEST_FILE_NAME,
        download_zenodo_archive=_download_zenodo_archive,
        ensure_extracted_archive=_ensure_extracted_archive,
        collect_wav_files=_collect_wav_files,
        write_source_manifest=_write_source_manifest,
    )


def prepare_emodb_2_from_zenodo(*, dataset_root: Path) -> AutoDownloadArtifacts:
    """Downloads EmoDB 2.0 from Zenodo and generates deterministic labels.csv."""
    return zenodo_download_helpers.prepare_emodb_2_from_zenodo(
        dataset_root=dataset_root,
        record_id=EMODB_2_ZENODO_RECORD_ID,
        file_key=EMODB_2_ZENODO_ARCHIVE_KEY,
        labels_file_name=DEFAULT_LABELS_FILE_NAME,
        source_manifest_file_name=DEFAULT_SOURCE_MANIFEST_FILE_NAME,
        emodb_label_map=_EMODB_LABEL_MAP,
        download_zenodo_archive=_download_zenodo_archive,
        ensure_extracted_archive=_ensure_extracted_archive,
        compute_relative_to_dataset_root=_compute_relative_to_dataset_root,
        write_labels_csv=_write_labels_csv,
        write_source_manifest=_write_source_manifest,
    )


def prepare_escorpus_pe_from_zenodo(*, dataset_root: Path) -> AutoDownloadArtifacts:
    """Downloads ESCorpus-PE and infers weak categorical labels from VAD filename codes."""
    return zenodo_download_helpers.prepare_escorpus_pe_from_zenodo(
        dataset_root=dataset_root,
        record_id=ESCORPUS_PE_ZENODO_RECORD_ID,
        file_key=ESCORPUS_PE_ZENODO_ARCHIVE_KEY,
        labels_file_name=DEFAULT_LABELS_FILE_NAME,
        source_manifest_file_name=DEFAULT_SOURCE_MANIFEST_FILE_NAME,
        download_zenodo_archive=_download_zenodo_archive,
        ensure_extracted_archive=_ensure_extracted_archive,
        generate_labels_from_audio_tree=_generate_labels_from_audio_tree,
        infer_escorpus_pe_label=_infer_escorpus_pe_label,
        write_source_manifest=_write_source_manifest,
    )


def _download_mendeley_dataset_tree(
    *,
    dataset_id: str,
    version: int,
    destination_root: Path,
) -> int:
    return mendeley_download_helpers.download_mendeley_dataset_tree_from_api(
        dataset_id=dataset_id,
        version=version,
        destination_root=destination_root,
        request_json=_request_json,
        download_file=_download_file,
    )


def prepare_mesd_from_mendeley(*, dataset_root: Path) -> AutoDownloadArtifacts:
    """Downloads MESD from Mendeley and generates filename-based labels."""
    return mendeley_dataset_preparation_helpers.prepare_mesd_from_mendeley(
        dataset_root=dataset_root,
        dataset_id=MESD_MENDELEY_DATASET_ID,
        version=MESD_DEFAULT_VERSION,
        extract_dir_name="mesd",
        labels_file_name=DEFAULT_LABELS_FILE_NAME,
        source_manifest_file_name=DEFAULT_SOURCE_MANIFEST_FILE_NAME,
        download_mendeley_dataset_tree=_download_mendeley_dataset_tree,
        generate_labels_from_audio_tree=_generate_labels_from_audio_tree,
        infer_mesd_label=_infer_mesd_label,
        write_source_manifest=_write_source_manifest,
    )


def prepare_oreau_french_esd_from_zenodo(
    *, dataset_root: Path
) -> AutoDownloadArtifacts:
    """Downloads Oreau French ESD from Zenodo and generates inferred labels."""
    return zenodo_download_helpers.prepare_oreau_french_esd_from_zenodo(
        dataset_root=dataset_root,
        record_id=OREAU_FRENCH_ESD_ZENODO_RECORD_ID,
        rar_keys=OREAU_FRENCH_ESD_RAR_KEYS,
        doc_key=OREAU_FRENCH_ESD_DOC_KEY,
        labels_file_name=DEFAULT_LABELS_FILE_NAME,
        source_manifest_file_name=DEFAULT_SOURCE_MANIFEST_FILE_NAME,
        download_zenodo_archive=_download_zenodo_archive,
        ensure_extracted_archive=_ensure_extracted_archive,
        generate_labels_from_audio_tree=_generate_labels_from_audio_tree,
        infer_label_from_path_tokens=_infer_label_from_path_tokens,
        write_source_manifest=_write_source_manifest,
    )


def prepare_jl_corpus_from_kaggle(*, dataset_root: Path) -> AutoDownloadArtifacts:
    """Downloads JL-Corpus via Kaggle API/CLI and generates inferred labels."""
    return jl_corpus_download_helpers.prepare_jl_corpus_from_kaggle(
        dataset_root=dataset_root,
        dataset_ref=JL_CORPUS_KAGGLE_DATASET_REF,
        labels_file_name=DEFAULT_LABELS_FILE_NAME,
        source_manifest_file_name=DEFAULT_SOURCE_MANIFEST_FILE_NAME,
        download_kaggle_archive=_download_kaggle_archive,
        ensure_extracted_archive=_ensure_extracted_archive,
        generate_labels_from_audio_tree=_generate_labels_from_audio_tree,
        infer_label_from_path_tokens=_infer_label_from_path_tokens,
        write_source_manifest=_write_source_manifest,
        prepare_hf_rows_fallback=_prepare_jl_corpus_from_hf_rows,
        logger_warning=logger.warning,
    )


def prepare_cafe_from_zenodo(*, dataset_root: Path) -> AutoDownloadArtifacts:
    """Downloads CaFE from Zenodo and generates inferred labels."""
    return zenodo_download_helpers.prepare_cafe_from_zenodo(
        dataset_root=dataset_root,
        record_id=CAFE_ZENODO_RECORD_ID,
        archive_keys=CAFE_ZENODO_ARCHIVE_KEYS,
        labels_file_name=DEFAULT_LABELS_FILE_NAME,
        source_manifest_file_name=DEFAULT_SOURCE_MANIFEST_FILE_NAME,
        download_zenodo_archive=_download_zenodo_archive,
        ensure_extracted_archive=_ensure_extracted_archive,
        generate_labels_from_audio_tree=_generate_labels_from_audio_tree,
        infer_label_from_path_tokens=_infer_label_from_path_tokens,
        write_source_manifest=_write_source_manifest,
    )


def prepare_asvp_esd_from_zenodo(*, dataset_root: Path) -> AutoDownloadArtifacts:
    """Downloads ASVP-ESD from Zenodo and generates inferred labels."""
    return zenodo_download_helpers.prepare_asvp_esd_from_zenodo(
        dataset_root=dataset_root,
        record_id=ASVP_ESD_ZENODO_RECORD_ID,
        file_key=ASVP_ESD_ZENODO_ARCHIVE_KEY,
        labels_file_name=DEFAULT_LABELS_FILE_NAME,
        source_manifest_file_name=DEFAULT_SOURCE_MANIFEST_FILE_NAME,
        download_zenodo_archive=_download_zenodo_archive,
        ensure_extracted_archive=_ensure_extracted_archive,
        generate_labels_from_audio_tree=_generate_labels_from_audio_tree,
        infer_label_from_path_tokens=_infer_label_from_path_tokens,
        write_source_manifest=_write_source_manifest,
    )


def prepare_emov_db_from_openslr(*, dataset_root: Path) -> AutoDownloadArtifacts:
    """Downloads EmoV-DB from OpenSLR and generates inferred labels."""
    return openslr_dataset_preparation_helpers.prepare_openslr_dataset(
        dataset_root=dataset_root,
        dataset_id=EMOV_DB_OPENSLR_DATASET_ID,
        archive_suffixes=EMOV_DB_OPENSLR_ARCHIVE_SUFFIXES,
        extract_dir_name="emov-db",
        labels_file_name=DEFAULT_LABELS_FILE_NAME,
        source_manifest_file_name=DEFAULT_SOURCE_MANIFEST_FILE_NAME,
        label_resolver=_infer_label_from_path_tokens,
        label_semantics=None,
        extensions=frozenset({".wav", ".flac"}),
        download_openslr_archives=_download_openslr_archives,
        ensure_extracted_archive=_ensure_extracted_archive,
        generate_labels_from_audio_tree=_generate_labels_from_audio_tree,
        write_source_manifest=_write_source_manifest,
    )


def prepare_pavoque_from_github_release(*, dataset_root: Path) -> AutoDownloadArtifacts:
    """Downloads PAVOQUE release assets from GitHub and generates inferred labels."""
    return provider_dataset_preparation_helpers.prepare_pavoque_from_github_release(
        dataset_root=dataset_root,
        owner=PAVOQUE_GITHUB_OWNER,
        repo=PAVOQUE_GITHUB_REPO,
        labels_file_name=DEFAULT_LABELS_FILE_NAME,
        source_manifest_file_name=DEFAULT_SOURCE_MANIFEST_FILE_NAME,
        read_github_latest_release_assets=_read_github_latest_release_assets,
        download_file=_download_file,
        generate_labels_from_audio_tree=_generate_labels_from_audio_tree,
        infer_label_from_path_tokens=_infer_label_from_path_tokens,
        write_source_manifest=_write_source_manifest,
    )


def prepare_att_hack_from_openslr(*, dataset_root: Path) -> AutoDownloadArtifacts:
    """Downloads Att-HACK from OpenSLR and generates social-attitude labels."""
    return openslr_dataset_preparation_helpers.prepare_openslr_dataset(
        dataset_root=dataset_root,
        dataset_id=ATT_HACK_OPENSLR_DATASET_ID,
        archive_suffixes=ATT_HACK_OPENSLR_ARCHIVE_SUFFIXES,
        extract_dir_name="att-hack",
        labels_file_name=DEFAULT_LABELS_FILE_NAME,
        source_manifest_file_name=DEFAULT_SOURCE_MANIFEST_FILE_NAME,
        label_resolver=_infer_att_hack_label,
        label_semantics="social_attitudes",
        extensions=None,
        download_openslr_archives=_download_openslr_archives,
        ensure_extracted_archive=_ensure_extracted_archive,
        generate_labels_from_audio_tree=_generate_labels_from_audio_tree,
        write_source_manifest=_write_source_manifest,
    )


def prepare_coraa_ser_from_google_drive(*, dataset_root: Path) -> AutoDownloadArtifacts:
    """Downloads CORAA SER artifacts from Google Drive and generates labels."""
    return provider_dataset_preparation_helpers.prepare_coraa_ser_from_google_drive(
        dataset_root=dataset_root,
        folder_url=CORAA_SER_GOOGLE_DRIVE_FOLDER_URL,
        label_semantics="neutral_vs_non_neutral_by_gender",
        labels_file_name=DEFAULT_LABELS_FILE_NAME,
        source_manifest_file_name=DEFAULT_SOURCE_MANIFEST_FILE_NAME,
        download_google_drive_folder=_download_google_drive_folder,
        extract_archives_from_tree=_extract_archives_from_tree,
        generate_labels_from_audio_tree=_generate_labels_from_audio_tree,
        infer_coraa_ser_label=_infer_coraa_ser_label,
        write_source_manifest=_write_source_manifest,
    )


def prepare_spanish_meacorpus_2023_from_zenodo(
    *,
    dataset_root: Path,
) -> AutoDownloadArtifacts:
    """Downloads Spanish MEACorpus metadata and generates labels for present audio."""
    return zenodo_download_helpers.prepare_spanish_meacorpus_2023_from_zenodo(
        dataset_root=dataset_root,
        record_id=SPANISH_MEACORPUS_2023_ZENODO_RECORD_ID,
        metadata_key=SPANISH_MEACORPUS_2023_METADATA_KEY,
        labels_file_name=DEFAULT_LABELS_FILE_NAME,
        source_manifest_file_name=DEFAULT_SOURCE_MANIFEST_FILE_NAME,
        download_zenodo_archive=_download_zenodo_archive,
        copy_file=_copy_file,
        generate_labels_from_metadata_csv=_generate_labels_from_metadata_csv,
        write_source_manifest=_write_source_manifest,
    )
