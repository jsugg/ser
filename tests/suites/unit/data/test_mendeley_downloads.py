"""Contract tests for Mendeley download execution helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from ser.data.mendeley_downloads import (
    build_mendeley_folder_paths,
    download_mendeley_dataset_tree,
    download_mendeley_dataset_tree_from_api,
    read_mendeley_files,
    read_mendeley_folders,
)


def test_read_mendeley_folders_filters_non_dict_entries() -> None:
    """Folder reader should keep only dict entries from API payload lists."""

    def _request_json(url: str, *, headers: dict[str, str] | None = None) -> object:
        del url, headers
        return [{"id": "folder-1", "name": "audio"}, "unexpected", {"id": "folder-2"}]

    folders = read_mendeley_folders(
        dataset_id="cy34mh68j9",
        version=5,
        request_json=_request_json,
    )

    assert folders == [{"id": "folder-1", "name": "audio"}, {"id": "folder-2"}]


def test_build_mendeley_folder_paths_builds_nested_safe_paths() -> None:
    """Folder-path builder should resolve parent relationships deterministically."""
    paths = build_mendeley_folder_paths(
        [
            {"id": "root-child", "name": "Root Child"},
            {"id": "nested", "name": "Nested Folder", "parent_id": "root-child"},
        ]
    )

    assert paths["root-child"] == Path("Root_Child")
    assert paths["nested"] == Path("Root_Child") / "Nested_Folder"


def test_build_mendeley_folder_paths_rejects_cycles() -> None:
    """Folder-path builder should fail fast on cyclic folder relationships."""
    with pytest.raises(RuntimeError, match="Cycle detected"):
        _ = build_mendeley_folder_paths(
            [
                {"id": "a", "name": "A", "parent_id": "b"},
                {"id": "b", "name": "B", "parent_id": "a"},
            ]
        )


def test_read_mendeley_files_filters_non_dict_entries() -> None:
    """File reader should keep only dict entries from API payload lists."""

    def _request_json(url: str, *, headers: dict[str, str] | None = None) -> object:
        del url, headers
        return [
            {"filename": "a.wav", "content_details": {"download_url": "https://x/a"}},
            123,
            {"filename": "b.wav", "content_details": {"download_url": "https://x/b"}},
        ]

    files = read_mendeley_files(
        dataset_id="cy34mh68j9",
        folder_id="root",
        version=5,
        request_json=_request_json,
    )

    assert len(files) == 2
    assert files[0]["filename"] == "a.wav"
    assert files[1]["filename"] == "b.wav"


def test_download_mendeley_dataset_tree_downloads_root_and_nested_files(
    tmp_path: Path,
) -> None:
    """Dataset-tree downloader should iterate root and resolved folder ids."""
    downloaded_paths: list[Path] = []

    def _read_folders(*, dataset_id: str, version: int) -> list[dict[str, object]]:
        del dataset_id, version
        return [{"id": "folder-a", "name": "Folder A"}]

    def _build_folder_paths(folders: list[dict[str, object]]) -> dict[str, Path]:
        return build_mendeley_folder_paths(folders)

    def _read_files(
        *,
        dataset_id: str,
        folder_id: str,
        version: int,
    ) -> list[dict[str, object]]:
        del dataset_id, version
        if folder_id == "root":
            return [
                {
                    "filename": "root.wav",
                    "content_details": {"download_url": "https://example/root.wav"},
                    "size": 11,
                }
            ]
        return [
            {
                "filename": "child.wav",
                "content_details": {"download_url": "https://example/child.wav"},
                "size": 22,
            }
        ]

    def _download_file(
        *,
        url: str,
        destination_path: Path,
        expected_md5: str | None = None,
        expected_size: int | None = None,
        headers: dict[str, str] | None = None,
    ) -> Path:
        del url, expected_md5, expected_size, headers
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        destination_path.write_bytes(b"ok")
        downloaded_paths.append(destination_path)
        return destination_path

    downloaded_count = download_mendeley_dataset_tree(
        dataset_id="cy34mh68j9",
        version=5,
        destination_root=tmp_path,
        read_folders=_read_folders,
        build_folder_paths=_build_folder_paths,
        read_files=_read_files,
        download_file=_download_file,
    )

    assert downloaded_count == 2
    assert downloaded_paths == [
        tmp_path / "root.wav",
        tmp_path / "Folder_A" / "child.wav",
    ]


def test_download_mendeley_dataset_tree_from_api_wires_api_helpers(
    tmp_path: Path,
) -> None:
    """API-bound helper should reuse folder/file readers and tree downloader."""
    request_urls: list[str] = []
    downloaded_paths: list[Path] = []

    def _request_json(url: str, *, headers: dict[str, str] | None = None) -> object:
        del headers
        request_urls.append(url)
        if "/folders/" in url:
            return [{"id": "folder-a", "name": "Folder A"}]
        if "folder_id=root" in url:
            return [
                {
                    "filename": "root.wav",
                    "content_details": {"download_url": "https://example/root.wav"},
                    "size": 11,
                }
            ]
        if "folder_id=folder-a" in url:
            return [
                {
                    "filename": "child.wav",
                    "content_details": {"download_url": "https://example/child.wav"},
                    "size": 22,
                }
            ]
        raise AssertionError(f"Unexpected URL: {url}")

    def _download_file(
        *,
        url: str,
        destination_path: Path,
        expected_md5: str | None = None,
        expected_size: int | None = None,
        headers: dict[str, str] | None = None,
    ) -> Path:
        del url, expected_md5, expected_size, headers
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        destination_path.write_bytes(b"ok")
        downloaded_paths.append(destination_path)
        return destination_path

    downloaded_count = download_mendeley_dataset_tree_from_api(
        dataset_id="cy34mh68j9",
        version=5,
        destination_root=tmp_path,
        request_json=_request_json,
        download_file=_download_file,
    )

    assert downloaded_count == 2
    assert len(request_urls) == 3
    assert downloaded_paths == [
        tmp_path / "root.wav",
        tmp_path / "Folder_A" / "child.wav",
    ]
