"""Provider-specific download helpers for public dataset acquisition."""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import random
import shutil
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from urllib import error, request


@dataclass(frozen=True, slots=True)
class GitHubReleaseAssetMetadata:
    """One downloadable asset from a GitHub release payload."""

    name: str
    download_url: str
    size: int | None


class RequestJson(Protocol):
    """Callable contract for JSON fetchers used by provider helpers."""

    def __call__(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
    ) -> object: ...


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


class RunWithRetries(Protocol):
    """Callable contract for retry wrappers used by download helpers."""

    def __call__(
        self,
        *,
        description: str,
        action: Callable[[], None],
    ) -> None: ...


def is_retryable_http_status(status_code: int) -> bool:
    """Returns whether one HTTP status is safe to retry."""
    return status_code == 429 or 500 <= status_code <= 599


def run_with_retries[T](
    *,
    description: str,
    action: Callable[[], T],
    retries: int,
    retry_base_seconds: float,
    logger: logging.Logger,
) -> T:
    """Runs one callable with bounded retries and jittered backoff."""
    if retries < 1:
        raise RuntimeError("Retry count must be at least 1.")
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return action()
        except error.HTTPError as err:
            if not is_retryable_http_status(err.code):
                raise RuntimeError(
                    f"{description} failed with HTTP {err.code}"
                ) from err
            last_error = err
        except (error.URLError, TimeoutError, OSError) as err:
            last_error = err
        if attempt < retries:
            sleep_seconds = retry_base_seconds * float(attempt) + random.uniform(
                0.0,
                retry_base_seconds,
            )
            logger.warning(
                "%s failed (attempt=%s/%s); retrying in %.2fs",
                description,
                attempt,
                retries,
                sleep_seconds,
            )
            time.sleep(sleep_seconds)
    raise RuntimeError(
        f"{description} failed after {retries} attempts."
    ) from last_error


def request_json_with_retries(
    *,
    url: str,
    headers: dict[str, str] | None,
    timeout_seconds: float,
    with_retries: Callable[..., object],
) -> object:
    """Fetches one JSON payload with retry orchestration delegated by caller."""

    def _action() -> object:
        req = request.Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": "ser-data-downloader/1.0",
                **(headers or {}),
            },
            method="GET",
        )
        with request.urlopen(req, timeout=timeout_seconds) as response:
            payload = response.read()
        return json.loads(payload.decode("utf-8"))

    return with_retries(description=f"GET {url}", action=_action)


def compute_file_md5(*, path: Path, chunk_size: int) -> str:
    """Computes deterministic MD5 for one file path using streamed reads."""
    digest = hashlib.md5()  # nosec: B324 - upstream datasets expose md5 checksums.
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def download_file_with_retries(
    *,
    url: str,
    destination_path: Path,
    expected_md5: str | None = None,
    expected_size: int | None = None,
    headers: dict[str, str] | None = None,
    with_retries: RunWithRetries,
    compute_file_md5: Callable[[Path], str],
    timeout_seconds: float,
    chunk_size: int,
) -> Path:
    """Downloads one file atomically with retries and optional integrity checks."""
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if destination_path.is_file():
        existing_size = destination_path.stat().st_size
        if expected_size is not None and existing_size != expected_size:
            destination_path.unlink()
        elif expected_md5 is not None:
            if compute_file_md5(destination_path) == expected_md5:
                return destination_path
            destination_path.unlink()
        elif existing_size > 0:
            return destination_path
    tmp_path = destination_path.with_suffix(destination_path.suffix + ".partial")

    def _action() -> None:
        req = request.Request(
            url,
            headers={
                "User-Agent": "ser-data-downloader/1.0",
                **(headers or {}),
            },
            method="GET",
        )
        with request.urlopen(req, timeout=timeout_seconds) as response:
            with tmp_path.open("wb") as output_handle:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    output_handle.write(chunk)

    try:
        with_retries(description=f"download {url}", action=_action)
        os.replace(tmp_path, destination_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)

    if expected_size is not None:
        actual_size = destination_path.stat().st_size
        if actual_size != expected_size:
            raise RuntimeError(
                f"Downloaded file size mismatch for {destination_path.name}: "
                f"expected {expected_size}, got {actual_size}."
            )
    if expected_md5 is not None:
        actual_md5 = compute_file_md5(destination_path)
        if actual_md5 != expected_md5:
            raise RuntimeError(
                f"Downloaded file checksum mismatch for {destination_path.name}: "
                f"expected md5 {expected_md5}, got {actual_md5}."
            )
    return destination_path


def read_github_latest_release_assets(
    *,
    owner: str,
    repo: str,
    request_json: RequestJson,
) -> tuple[str, list[GitHubReleaseAssetMetadata]]:
    """Reads GitHub latest-release metadata and returns downloadable assets."""
    api_url = f"https://api.github.com/repos/{owner}/{repo}/releases/latest"
    raw = request_json(
        api_url,
        headers={
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    if not isinstance(raw, dict):
        raise RuntimeError(f"Unexpected GitHub release payload for {owner}/{repo}.")

    tag_name_raw = raw.get("tag_name")
    tag_name = str(tag_name_raw).strip() if tag_name_raw is not None else ""
    if not tag_name:
        raise RuntimeError(f"GitHub latest release for {owner}/{repo} has no tag_name.")

    assets_raw = raw.get("assets")
    if not isinstance(assets_raw, list):
        raise RuntimeError(f"GitHub latest release for {owner}/{repo} has no assets.")

    assets: list[GitHubReleaseAssetMetadata] = []
    for item in assets_raw:
        if not isinstance(item, dict):
            continue
        name_raw = item.get("name")
        url_raw = item.get("browser_download_url")
        if not isinstance(name_raw, str) or not name_raw.strip():
            continue
        if not isinstance(url_raw, str) or not url_raw.strip():
            continue
        size_raw = item.get("size")
        size = size_raw if isinstance(size_raw, int) and size_raw >= 0 else None
        assets.append(
            GitHubReleaseAssetMetadata(
                name=name_raw.strip(),
                download_url=url_raw.strip(),
                size=size,
            )
        )

    if not assets:
        raise RuntimeError(f"GitHub latest release for {owner}/{repo} has no assets.")
    return tag_name, assets


def download_google_drive_folder(
    *,
    folder_url: str,
    destination_root: Path,
    which: Callable[[str], str | None] = shutil.which,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> list[Path]:
    """Downloads one Google Drive folder via gdown and returns local files."""
    destination_root.mkdir(parents=True, exist_ok=True)
    gdown_bin = which("gdown")
    if gdown_bin is None:
        raise RuntimeError(
            "Google Drive folder download requires `gdown` in PATH. "
            "Install with `pip install gdown` and retry."
        )
    completed = run(
        [
            gdown_bin,
            "--folder",
            "--fuzzy",
            "--continue",
            "-O",
            str(destination_root),
            folder_url,
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"gdown folder download failed. stderr: {completed.stderr.strip()}"
        )

    files = [path for path in sorted(destination_root.rglob("*")) if path.is_file()]
    if not files:
        raise RuntimeError(
            f"gdown completed but no files were downloaded to {destination_root}."
        )
    return files


def kaggle_credentials_from_env() -> tuple[str | None, str | None]:
    """Reads Kaggle credentials from environment variables."""
    username = os.getenv("KAGGLE_USERNAME", "").strip()
    key = os.getenv("KAGGLE_KEY", "").strip()
    if username and key:
        return username, key
    return None, None


def download_kaggle_archive(
    *,
    dataset_ref: str,
    destination_path: Path,
    download_file: DownloadFile,
    logger_warning: Callable[[str, object], None],
    resolve_credentials: Callable[[], tuple[str | None, str | None]] = (
        kaggle_credentials_from_env
    ),
    which: Callable[[str], str | None] = shutil.which,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    replace_file: Callable[[Path, Path], None] = os.replace,
) -> Path:
    """Downloads one Kaggle dataset archive via API or CLI fallback."""
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if destination_path.is_file() and destination_path.stat().st_size > 0:
        return destination_path

    username, key = resolve_credentials()
    download_url = f"https://www.kaggle.com/api/v1/datasets/download/{dataset_ref}"
    if username is not None and key is not None:
        token = base64.b64encode(f"{username}:{key}".encode()).decode("ascii")
        try:
            return download_file(
                url=download_url,
                destination_path=destination_path,
                headers={"Authorization": f"Basic {token}"},
            )
        except RuntimeError as err:
            logger_warning(
                "Kaggle direct API download failed; falling back to CLI if available: %s",
                err,
            )

    kaggle_bin = which("kaggle")
    if kaggle_bin is None:
        raise RuntimeError(
            "JL-Corpus download requires Kaggle credentials. "
            "Set KAGGLE_USERNAME/KAGGLE_KEY or install/configure Kaggle CLI."
        )
    expected_cli_archive = (
        destination_path.parent / f"{dataset_ref.replace('/', '-')}.zip"
    )
    completed = run(
        [
            kaggle_bin,
            "datasets",
            "download",
            "-d",
            dataset_ref,
            "-p",
            str(destination_path.parent),
            "--force",
            "--quiet",
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Kaggle CLI download failed. Configure credentials via ~/.kaggle/kaggle.json "
            "or KAGGLE_USERNAME/KAGGLE_KEY. "
            f"stderr: {completed.stderr.strip()}"
        )
    if expected_cli_archive.is_file():
        replace_file(expected_cli_archive, destination_path)
    if not destination_path.is_file() or destination_path.stat().st_size <= 0:
        raise RuntimeError(
            f"Kaggle download completed but archive not found at {destination_path}."
        )
    return destination_path


__all__ = [
    "GitHubReleaseAssetMetadata",
    "compute_file_md5",
    "download_file_with_retries",
    "download_google_drive_folder",
    "download_kaggle_archive",
    "is_retryable_http_status",
    "kaggle_credentials_from_env",
    "request_json_with_retries",
    "run_with_retries",
    "read_github_latest_release_assets",
]
