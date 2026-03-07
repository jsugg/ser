"""Contract tests for OpenSLR download execution helpers."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from ser.data.openslr_downloads import (
    download_openslr_archives,
    download_openslr_pinned_artifacts,
    read_openslr_archive_urls,
    read_openslr_archive_urls_from_hf_script,
)
from ser.data.openslr_resolution import OpenSlrPinnedArtifact


def test_download_openslr_pinned_artifacts_retries_mirrors(
    tmp_path: Path,
) -> None:
    """Pinned downloads should fail over to mirror URLs for one artifact."""
    artifact = OpenSlrPinnedArtifact(
        file_name="wav.tgz",
        urls=(
            "https://openslr.org/resources/88/wav.tgz",
            "https://openslr.trmal.net/resources/88/wav.tgz",
        ),
    )
    attempted_urls: list[str] = []
    warnings: list[tuple[str, str, str]] = []

    def _download_file(url: str, destination_path: Path) -> Path:
        attempted_urls.append(url)
        if "trmal.net" not in url:
            raise RuntimeError("primary mirror unavailable")
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        destination_path.write_bytes(b"ok")
        return destination_path

    paths = download_openslr_pinned_artifacts(
        dataset_root=tmp_path,
        dataset_id="88",
        artifacts=(artifact,),
        download_file=_download_file,
        log_mirror_failure=lambda dataset_id, file_name, url: warnings.append(
            (dataset_id, file_name, url)
        ),
    )

    assert [path.name for path in paths] == ["wav.tgz"]
    assert attempted_urls == [
        "https://openslr.org/resources/88/wav.tgz",
        "https://openslr.trmal.net/resources/88/wav.tgz",
    ]
    assert warnings == [("88", "wav.tgz", "https://openslr.org/resources/88/wav.tgz")]


def test_download_openslr_archives_uses_discovered_urls_when_not_pinned(
    tmp_path: Path,
) -> None:
    """Archive downloader should use discovered URLs when no pinned registry exists."""
    downloaded_urls: list[str] = []

    def _download_file(url: str, destination_path: Path) -> Path:
        downloaded_urls.append(url)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        destination_path.write_bytes(b"ok")
        return destination_path

    paths = download_openslr_archives(
        dataset_root=tmp_path,
        dataset_id="999",
        archive_suffixes=(".tgz",),
        resolve_pinned_artifacts=lambda dataset_id, archive_suffixes: (),
        read_archive_urls=lambda dataset_id, archive_suffixes: [
            "https://www.openslr.org/resources/999/a.tgz",
            "https://www.openslr.org/resources/999/b.tgz",
        ],
        download_file=_download_file,
        log_mirror_failure=lambda dataset_id, file_name, url: None,
    )

    assert [path.name for path in paths] == ["a.tgz", "b.tgz"]
    assert downloaded_urls == [
        "https://www.openslr.org/resources/999/a.tgz",
        "https://www.openslr.org/resources/999/b.tgz",
    ]


def test_download_openslr_archives_raises_when_no_downloadable_files(
    tmp_path: Path,
) -> None:
    """Archive downloader should fail fast when discovery yields no archive files."""
    with pytest.raises(RuntimeError, match="did not resolve any archive files"):
        _ = download_openslr_archives(
            dataset_root=tmp_path,
            dataset_id="999",
            archive_suffixes=(".tgz",),
            resolve_pinned_artifacts=lambda dataset_id, archive_suffixes: (),
            read_archive_urls=lambda dataset_id, archive_suffixes: [],
            download_file=lambda url, destination_path: destination_path,
            log_mirror_failure=lambda dataset_id, file_name, url: None,
        )


def test_read_openslr_archive_urls_prefers_hf_metadata() -> None:
    """Archive URL resolver should return HF metadata URLs without listing fallback."""
    retry_calls: list[str] = []

    def _read_archive_urls_from_hf_script(
        *,
        dataset_id: str,
        archive_suffixes: tuple[str, ...],
    ) -> list[str]:
        del archive_suffixes
        return [f"https://openslr.org/resources/{dataset_id}/wav.tgz"]

    def _with_retries(*, description: str, action: Callable[[], str]) -> str:
        del action
        retry_calls.append(description)
        return ""

    def _extract_archive_urls_from_listing_html(
        *,
        listing_url: str,
        html_text: str,
        archive_suffixes: tuple[str, ...],
    ) -> list[str]:
        del listing_url, html_text, archive_suffixes
        return []

    urls = read_openslr_archive_urls(
        dataset_id="88",
        archive_suffixes=(".tgz",),
        read_archive_urls_from_hf_script=_read_archive_urls_from_hf_script,
        with_retries=_with_retries,
        timeout_seconds=60.0,
        extract_archive_urls_from_listing_html=_extract_archive_urls_from_listing_html,
        log_hf_metadata_resolution_failure=lambda dataset_id, error: None,
    )

    assert urls == ["https://openslr.org/resources/88/wav.tgz"]
    assert retry_calls == []


def test_read_openslr_archive_urls_from_hf_script_reads_and_builds_urls() -> None:
    """HF-script helper should parse filenames and map them to canonical URLs."""

    def _with_retries(*, description: str, action: Callable[[], str]) -> str:
        del description, action
        return '"SLR88": {"Files": ["wav.tgz", "notes.txt"]}'

    urls = read_openslr_archive_urls_from_hf_script(
        dataset_id="88",
        archive_suffixes=(".tgz",),
        script_url="https://huggingface.co/datasets/openslr/openslr/raw/main/openslr.py",
        with_retries=_with_retries,
        timeout_seconds=60.0,
        extract_openslr_files_from_hf_script=lambda *, script_text, dataset_id: (
            ["wav.tgz", "notes.txt"]
            if "SLR88" in script_text and dataset_id == "88"
            else []
        ),
        build_canonical_archive_urls=lambda *, dataset_id, file_names, archive_suffixes: [
            f"https://openslr.org/resources/{dataset_id}/{name}"
            for name in file_names
            if name.endswith(archive_suffixes)
        ],
    )

    assert urls == ["https://openslr.org/resources/88/wav.tgz"]


def test_read_openslr_archive_urls_falls_back_to_listing_on_hf_error() -> None:
    """Archive URL resolver should use listing-page extraction after HF errors."""
    warning_calls: list[tuple[str, str]] = []

    def _read_archive_urls_from_hf_script(
        *,
        dataset_id: str,
        archive_suffixes: tuple[str, ...],
    ) -> list[str]:
        del dataset_id, archive_suffixes
        raise RuntimeError("hf-unavailable")

    def _with_retries(*, description: str, action: Callable[[], str]) -> str:
        del description, action
        return "<html>listing</html>"

    def _extract_archive_urls_from_listing_html(
        *,
        listing_url: str,
        html_text: str,
        archive_suffixes: tuple[str, ...],
    ) -> list[str]:
        del listing_url, html_text, archive_suffixes
        return ["https://www.openslr.org/resources/115/a.tar.gz"]

    def _log_hf_metadata_resolution_failure(
        *, dataset_id: str, error: Exception
    ) -> None:
        warning_calls.append((dataset_id, str(error)))

    urls = read_openslr_archive_urls(
        dataset_id="115",
        archive_suffixes=(".tar.gz", ".tgz"),
        read_archive_urls_from_hf_script=_read_archive_urls_from_hf_script,
        with_retries=_with_retries,
        timeout_seconds=60.0,
        extract_archive_urls_from_listing_html=_extract_archive_urls_from_listing_html,
        log_hf_metadata_resolution_failure=_log_hf_metadata_resolution_failure,
    )

    assert urls == ["https://www.openslr.org/resources/115/a.tar.gz"]
    assert warning_calls == [("115", "hf-unavailable")]
