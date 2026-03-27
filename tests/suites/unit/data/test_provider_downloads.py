"""Contract tests for provider-specific public dataset download helpers."""

from __future__ import annotations

from collections.abc import Callable
from email.message import Message
from pathlib import Path
from typing import Any
from urllib import error

import pytest

from ser.data import provider_downloads


def test_download_file_with_retries_reuses_existing_file_without_hash(
    tmp_path: Path,
) -> None:
    """Existing non-empty files should be reused when no integrity hints are provided."""
    destination_path = tmp_path / "archive.zip"
    destination_path.write_bytes(b"cached")
    retried = False

    def _with_retries(*, description: str, action: Callable[[], None]) -> None:
        nonlocal retried
        del description, action
        retried = True

    resolved = provider_downloads.download_file_with_retries(
        url="https://example.invalid/archive.zip",
        destination_path=destination_path,
        with_retries=_with_retries,
        compute_file_md5=lambda _path: "unused",
        timeout_seconds=1.0,
        chunk_size=1024,
    )

    assert resolved == destination_path
    assert retried is False
    assert destination_path.read_bytes() == b"cached"


def test_download_file_with_retries_streams_and_validates_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Downloader should stream to partial file then atomically publish validated output."""
    destination_path = tmp_path / "archive.zip"
    seen: dict[str, object] = {}

    class _FakeResponse:
        def __init__(self, chunks: list[bytes]) -> None:
            self._chunks = chunks

        def __enter__(self) -> _FakeResponse:
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            del exc_type, exc, tb

        def read(self, _size: int) -> bytes:
            return self._chunks.pop(0) if self._chunks else b""

    def _urlopen(req: object, timeout: float) -> _FakeResponse:
        if not isinstance(req, provider_downloads.request.Request):
            raise AssertionError("request object expected")
        seen["url"] = req.full_url
        seen["headers"] = {key.lower(): value for key, value in req.header_items()}
        seen["timeout"] = timeout
        return _FakeResponse([b"abc", b"def"])

    monkeypatch.setattr(provider_downloads.request, "urlopen", _urlopen)

    def _with_retries(*, description: str, action: Callable[[], None]) -> None:
        seen["description"] = description
        action()

    resolved = provider_downloads.download_file_with_retries(
        url="https://example.invalid/archive.zip",
        destination_path=destination_path,
        expected_md5="md5-ok",
        expected_size=6,
        headers={"X-Test": "1"},
        with_retries=_with_retries,
        compute_file_md5=lambda _path: "md5-ok",
        timeout_seconds=9.5,
        chunk_size=3,
    )

    assert resolved == destination_path
    assert destination_path.read_bytes() == b"abcdef"
    assert seen["description"] == "download https://example.invalid/archive.zip"
    assert seen["url"] == "https://example.invalid/archive.zip"
    assert seen["timeout"] == 9.5
    headers = seen["headers"]
    assert isinstance(headers, dict)
    assert headers["user-agent"] == "ser-data-downloader/1.0"
    assert headers["x-test"] == "1"
    assert not destination_path.with_suffix(".zip.partial").exists()


def test_read_github_latest_release_assets_parses_expected_payload() -> None:
    """GitHub helper should read latest release tag and downloadable assets."""

    payload = {
        "tag_name": "v1.2.3",
        "assets": [
            {
                "name": "dataset.zip",
                "browser_download_url": "https://example.org/dataset.zip",
                "size": 123,
            },
            {
                "name": " ",
                "browser_download_url": "https://example.org/skip.bin",
            },
        ],
    }

    def _request_json(
        url: str,
        *,
        headers: dict[str, str] | None = None,
    ) -> object:
        del url, headers
        return payload

    tag_name, assets = provider_downloads.read_github_latest_release_assets(
        owner="acme",
        repo="repo",
        request_json=_request_json,
    )

    assert tag_name == "v1.2.3"
    assert len(assets) == 1
    assert assets[0].name == "dataset.zip"
    assert assets[0].download_url == "https://example.org/dataset.zip"
    assert assets[0].size == 123


def test_download_google_drive_folder_requires_gdown(tmp_path: Path) -> None:
    """Google Drive helper should fail with actionable guidance without gdown."""

    with pytest.raises(RuntimeError, match="requires `gdown`"):
        _ = provider_downloads.download_google_drive_folder(
            folder_url="https://drive.google.com/drive/folders/example",
            destination_root=tmp_path / "downloads",
            which=lambda _binary: None,
        )


def test_download_kaggle_archive_uses_direct_api_when_credentials_present(
    tmp_path: Path,
) -> None:
    """Kaggle helper should use direct API path when env credentials are present."""
    destination_path = tmp_path / "archive.zip"
    captured: dict[str, Any] = {}

    def _download_file(
        *,
        url: str,
        destination_path: Path,
        expected_md5: str | None = None,
        expected_size: int | None = None,
        headers: dict[str, str] | None = None,
    ) -> Path:
        del expected_md5, expected_size
        captured["url"] = url
        captured["headers"] = headers
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        destination_path.write_bytes(b"zip")
        return destination_path

    archive_path = provider_downloads.download_kaggle_archive(
        dataset_ref="owner/dataset",
        destination_path=destination_path,
        download_file=_download_file,
        logger_warning=lambda _msg, _err: None,
        resolve_credentials=lambda: ("user", "secret"),
    )

    assert archive_path == destination_path
    assert captured["url"] == "https://www.kaggle.com/api/v1/datasets/download/owner/dataset"
    headers = captured["headers"]
    assert isinstance(headers, dict)
    authorization = headers.get("Authorization")
    assert isinstance(authorization, str)
    assert authorization.startswith("Basic ")


def test_run_with_retries_retries_retryable_http_errors_before_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retry helper should retry retryable HTTP failures and then return success."""
    attempts = 0
    monkeypatch.setattr(provider_downloads.random, "uniform", lambda _low, _high: 0.0)
    monkeypatch.setattr(provider_downloads.time, "sleep", lambda _seconds: None)

    def _action() -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise error.HTTPError(
                url="https://example.invalid",
                code=503,
                msg="service unavailable",
                hdrs=Message(),
                fp=None,
            )
        return "ok"

    resolved = provider_downloads.run_with_retries(
        description="GET https://example.invalid",
        action=_action,
        retries=3,
        retry_base_seconds=0.01,
        logger=provider_downloads.logging.getLogger("tests.provider_downloads"),
    )

    assert resolved == "ok"
    assert attempts == 2


def test_request_json_with_retries_reads_json_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """JSON request helper should decode payload and route through retry callable."""
    seen: dict[str, object] = {}

    class _FakeResponse:
        def __enter__(self) -> _FakeResponse:
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            del exc_type, exc, tb

        def read(self) -> bytes:
            return b'{"ok": true}'

    def _fake_urlopen(req: object, timeout: float) -> _FakeResponse:
        if not isinstance(req, provider_downloads.request.Request):
            raise AssertionError("Request object expected")
        seen["url"] = req.full_url
        seen["timeout"] = timeout
        return _FakeResponse()

    monkeypatch.setattr(provider_downloads.request, "urlopen", _fake_urlopen)

    def _with_retries(*, description: str, action: Callable[[], object]) -> object:
        seen["description"] = description
        return action()

    payload = provider_downloads.request_json_with_retries(
        url="https://example.invalid/data.json",
        headers={"X-Test": "1"},
        timeout_seconds=2.5,
        with_retries=_with_retries,
    )

    assert payload == {"ok": True}
    assert seen["description"] == "GET https://example.invalid/data.json"
    assert seen["url"] == "https://example.invalid/data.json"
    assert seen["timeout"] == 2.5


def test_compute_file_md5_streams_deterministic_digest(tmp_path: Path) -> None:
    """MD5 helper should compute deterministic digest for streamed file content."""
    path = tmp_path / "payload.bin"
    path.write_bytes(b"abcdef")

    md5 = provider_downloads.compute_file_md5(path=path, chunk_size=2)

    assert md5 == "e80b5017098950fc58aad83c8c14978e"
