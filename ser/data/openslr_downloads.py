"""OpenSLR download execution helpers for dataset acquisition workflows."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol
from urllib import parse, request

from ser.data.openslr_resolution import OpenSlrPinnedArtifact


class _DownloadFile(Protocol):
    def __call__(self, *, url: str, destination_path: Path) -> Path: ...


class _ResolvePinnedArtifacts(Protocol):
    def __call__(
        self,
        *,
        dataset_id: str,
        archive_suffixes: tuple[str, ...],
    ) -> tuple[OpenSlrPinnedArtifact, ...]: ...


class _ReadArchiveUrls(Protocol):
    def __call__(
        self,
        *,
        dataset_id: str,
        archive_suffixes: tuple[str, ...],
    ) -> list[str]: ...


class _LogMirrorFailure(Protocol):
    def __call__(self, *, dataset_id: str, file_name: str, url: str) -> None: ...


class _ReadArchiveUrlsFromHfScript(Protocol):
    def __call__(
        self,
        *,
        dataset_id: str,
        archive_suffixes: tuple[str, ...],
    ) -> list[str]: ...


class _RunWithRetries(Protocol):
    def __call__(
        self,
        *,
        description: str,
        action: Callable[[], str],
    ) -> str: ...


class _ExtractArchiveUrlsFromListingHtml(Protocol):
    def __call__(
        self,
        *,
        listing_url: str,
        html_text: str,
        archive_suffixes: tuple[str, ...],
    ) -> list[str]: ...


class _LogHfMetadataResolutionFailure(Protocol):
    def __call__(self, *, dataset_id: str, error: Exception) -> None: ...


class _ExtractOpenSlrFilesFromHfScript(Protocol):
    def __call__(self, *, script_text: str, dataset_id: str) -> list[str]: ...


class _BuildCanonicalArchiveUrls(Protocol):
    def __call__(
        self,
        *,
        dataset_id: str,
        file_names: list[str],
        archive_suffixes: tuple[str, ...],
    ) -> list[str]: ...


def read_openslr_archive_urls_from_hf_script(
    *,
    dataset_id: str,
    archive_suffixes: tuple[str, ...],
    script_url: str,
    with_retries: _RunWithRetries,
    timeout_seconds: float,
    extract_openslr_files_from_hf_script: _ExtractOpenSlrFilesFromHfScript,
    build_canonical_archive_urls: _BuildCanonicalArchiveUrls,
) -> list[str]:
    """Reads OpenSLR archive URLs from Hugging Face dataset script metadata."""

    def _read_script() -> str:
        req = request.Request(
            script_url,
            headers={
                "Accept": "text/plain",
                "User-Agent": "ser-data-downloader/1.0",
            },
            method="GET",
        )
        with request.urlopen(req, timeout=timeout_seconds) as response:
            payload = response.read()
        if not isinstance(payload, bytes):
            raise RuntimeError("Unexpected Hugging Face OpenSLR script payload type.")
        return payload.decode("utf-8", errors="replace")

    script_text = with_retries(description=f"GET {script_url}", action=_read_script)
    file_names = extract_openslr_files_from_hf_script(
        script_text=script_text,
        dataset_id=dataset_id,
    )
    return build_canonical_archive_urls(
        dataset_id=dataset_id,
        file_names=file_names,
        archive_suffixes=archive_suffixes,
    )


def read_openslr_archive_urls(
    *,
    dataset_id: str,
    archive_suffixes: tuple[str, ...],
    read_archive_urls_from_hf_script: _ReadArchiveUrlsFromHfScript,
    with_retries: _RunWithRetries,
    timeout_seconds: float,
    extract_archive_urls_from_listing_html: _ExtractArchiveUrlsFromListingHtml,
    log_hf_metadata_resolution_failure: _LogHfMetadataResolutionFailure,
) -> list[str]:
    """Reads OpenSLR archive URLs from HF metadata with listing-page fallback."""
    try:
        urls_from_hf = read_archive_urls_from_hf_script(
            dataset_id=dataset_id,
            archive_suffixes=archive_suffixes,
        )
    except Exception as err:
        log_hf_metadata_resolution_failure(dataset_id=dataset_id, error=err)
    else:
        if urls_from_hf:
            return urls_from_hf

    listing_url = f"https://www.openslr.org/{dataset_id}/"

    def _read_html() -> str:
        req = request.Request(
            listing_url,
            headers={
                "Accept": "text/html",
                "User-Agent": "ser-data-downloader/1.0",
            },
            method="GET",
        )
        with request.urlopen(req, timeout=timeout_seconds) as response:
            payload = response.read()
        if not isinstance(payload, bytes):
            raise RuntimeError("Unexpected OpenSLR response payload type.")
        return payload.decode("utf-8", errors="replace")

    html_text = with_retries(
        description=f"GET {listing_url}",
        action=_read_html,
    )
    urls = extract_archive_urls_from_listing_html(
        listing_url=listing_url,
        html_text=html_text,
        archive_suffixes=archive_suffixes,
    )
    if not urls:
        raise RuntimeError(
            f"No downloadable archives were discovered on OpenSLR page {listing_url}."
        )
    return urls


def download_openslr_pinned_artifacts(
    *,
    dataset_root: Path,
    dataset_id: str,
    artifacts: tuple[OpenSlrPinnedArtifact, ...],
    download_file: _DownloadFile,
    log_mirror_failure: _LogMirrorFailure,
) -> list[Path]:
    """Downloads pinned OpenSLR artifacts with per-artifact mirror fallback."""
    downloads_dir = dataset_root / "downloads"
    downloaded: list[Path] = []
    for artifact in artifacts:
        destination_path = downloads_dir / artifact.file_name
        last_error: RuntimeError | None = None
        for url in artifact.urls:
            try:
                path = download_file(url=url, destination_path=destination_path)
            except RuntimeError as err:
                last_error = err
                log_mirror_failure(
                    dataset_id=dataset_id,
                    file_name=artifact.file_name,
                    url=url,
                )
                continue
            downloaded.append(path)
            break
        else:
            raise RuntimeError(
                "OpenSLR pinned artifact download failed for "
                f"SLR{dataset_id} file {artifact.file_name} after trying all mirrors."
            ) from last_error
    return downloaded


def download_openslr_archives(
    *,
    dataset_root: Path,
    dataset_id: str,
    archive_suffixes: tuple[str, ...],
    resolve_pinned_artifacts: _ResolvePinnedArtifacts,
    read_archive_urls: _ReadArchiveUrls,
    download_file: _DownloadFile,
    log_mirror_failure: _LogMirrorFailure,
) -> list[Path]:
    """Downloads OpenSLR archives from pinned registry or discovered archive URLs."""
    pinned_artifacts = resolve_pinned_artifacts(
        dataset_id=dataset_id,
        archive_suffixes=archive_suffixes,
    )
    if pinned_artifacts:
        return download_openslr_pinned_artifacts(
            dataset_root=dataset_root,
            dataset_id=dataset_id,
            artifacts=pinned_artifacts,
            download_file=download_file,
            log_mirror_failure=log_mirror_failure,
        )

    archive_urls = read_archive_urls(
        dataset_id=dataset_id,
        archive_suffixes=archive_suffixes,
    )
    downloads_dir = dataset_root / "downloads"
    downloaded: list[Path] = []
    for archive_url in archive_urls:
        parsed_url = parse.urlparse(archive_url)
        archive_name = Path(parsed_url.path).name
        if not archive_name:
            continue
        destination_path = downloads_dir / archive_name
        downloaded.append(download_file(url=archive_url, destination_path=destination_path))
    if not downloaded:
        raise RuntimeError(f"OpenSLR {dataset_id} did not resolve any archive files to download.")
    return downloaded


__all__ = [
    "download_openslr_archives",
    "download_openslr_pinned_artifacts",
    "read_openslr_archive_urls",
    "read_openslr_archive_urls_from_hf_script",
]
