"""Mendeley download execution helpers for dataset acquisition workflows."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Protocol


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


class _ReadMendeleyFolders(Protocol):
    def __call__(self, *, dataset_id: str, version: int) -> list[dict[str, object]]: ...


class _BuildMendeleyFolderPaths(Protocol):
    def __call__(self, folders: list[dict[str, object]]) -> dict[str, Path]: ...


class _ReadMendeleyFiles(Protocol):
    def __call__(
        self,
        *,
        dataset_id: str,
        folder_id: str,
        version: int,
    ) -> list[dict[str, object]]: ...


def read_mendeley_folders(
    *,
    dataset_id: str,
    version: int,
    request_json: _RequestJson,
) -> list[dict[str, object]]:
    """Reads one Mendeley folders listing with strict payload-shape checks."""
    payload = request_json(
        f"https://data.mendeley.com/public-api/datasets/{dataset_id}/folders/{version}"
    )
    if not isinstance(payload, list):
        raise RuntimeError("Unexpected Mendeley folder payload shape.")
    parsed: list[dict[str, object]] = []
    for item in payload:
        if isinstance(item, dict):
            parsed.append(item)
    return parsed


def build_mendeley_folder_paths(
    folders: list[dict[str, object]],
) -> dict[str, Path]:
    """Builds stable safe local paths for one Mendeley folder tree."""
    by_id: dict[str, dict[str, object]] = {}
    for item in folders:
        folder_id = item.get("id")
        if isinstance(folder_id, str) and folder_id:
            by_id[folder_id] = item

    cache: dict[str, Path] = {}

    def _resolve(folder_id: str, seen: set[str]) -> Path:
        cached = cache.get(folder_id)
        if cached is not None:
            return cached
        if folder_id in seen:
            raise RuntimeError("Cycle detected while resolving Mendeley folder paths.")
        seen.add(folder_id)
        item = by_id.get(folder_id)
        if item is None:
            raise RuntimeError(f"Unknown Mendeley folder id: {folder_id}")
        raw_name = item.get("name")
        folder_name = str(raw_name).strip() if raw_name is not None else folder_id
        safe_name = re.sub(r"[^0-9a-zA-Z._-]+", "_", folder_name).strip("._-") or folder_id
        parent_id_raw = item.get("parent_id")
        if isinstance(parent_id_raw, str) and parent_id_raw and parent_id_raw in by_id:
            parent_path = _resolve(parent_id_raw, seen)
            resolved = parent_path / safe_name
        else:
            resolved = Path(safe_name)
        cache[folder_id] = resolved
        seen.remove(folder_id)
        return resolved

    for folder_id in by_id:
        _resolve(folder_id, set())
    return cache


def read_mendeley_files(
    *,
    dataset_id: str,
    folder_id: str,
    version: int,
    request_json: _RequestJson,
) -> list[dict[str, object]]:
    """Reads one Mendeley files listing for one folder identifier."""
    payload = request_json(
        f"https://data.mendeley.com/public-api/datasets/{dataset_id}/files"
        f"?folder_id={folder_id}&version={version}",
        headers={"Accept": "application/vnd.mendeley-public-dataset.1+json"},
    )
    if not isinstance(payload, list):
        raise RuntimeError("Unexpected Mendeley files payload shape.")
    parsed: list[dict[str, object]] = []
    for item in payload:
        if isinstance(item, dict):
            parsed.append(item)
    return parsed


def download_mendeley_dataset_tree(
    *,
    dataset_id: str,
    version: int,
    destination_root: Path,
    read_folders: _ReadMendeleyFolders,
    build_folder_paths: _BuildMendeleyFolderPaths,
    read_files: _ReadMendeleyFiles,
    download_file: _DownloadFile,
) -> int:
    """Downloads one Mendeley dataset tree to a local destination root."""
    folders = read_folders(dataset_id=dataset_id, version=version)
    folder_paths = build_folder_paths(folders)
    files_downloaded = 0

    all_folder_ids = ["root", *sorted(folder_paths)]
    for folder_id in all_folder_ids:
        files = read_files(
            dataset_id=dataset_id,
            folder_id=folder_id,
            version=version,
        )
        relative_folder = Path(".") if folder_id == "root" else folder_paths[folder_id]
        for entry in files:
            filename_raw = entry.get("filename")
            if not isinstance(filename_raw, str) or not filename_raw.strip():
                continue
            filename = filename_raw.strip()
            content_details = entry.get("content_details")
            if not isinstance(content_details, dict):
                continue
            download_url = content_details.get("download_url")
            if not isinstance(download_url, str) or not download_url:
                continue
            size_raw = entry.get("size")
            expected_size = size_raw if isinstance(size_raw, int) and size_raw >= 0 else None
            destination_path = destination_root / relative_folder / filename
            download_file(
                url=download_url,
                destination_path=destination_path,
                expected_size=expected_size,
            )
            files_downloaded += 1
    return files_downloaded


def download_mendeley_dataset_tree_from_api(
    *,
    dataset_id: str,
    version: int,
    destination_root: Path,
    request_json: _RequestJson,
    download_file: _DownloadFile,
) -> int:
    """Downloads one Mendeley dataset tree using API-bound helpers."""

    def _read_folders(*, dataset_id: str, version: int) -> list[dict[str, object]]:
        return read_mendeley_folders(
            dataset_id=dataset_id,
            version=version,
            request_json=request_json,
        )

    def _read_files(
        *,
        dataset_id: str,
        folder_id: str,
        version: int,
    ) -> list[dict[str, object]]:
        return read_mendeley_files(
            dataset_id=dataset_id,
            folder_id=folder_id,
            version=version,
            request_json=request_json,
        )

    return download_mendeley_dataset_tree(
        dataset_id=dataset_id,
        version=version,
        destination_root=destination_root,
        read_folders=_read_folders,
        build_folder_paths=build_mendeley_folder_paths,
        read_files=_read_files,
        download_file=download_file,
    )


__all__ = [
    "build_mendeley_folder_paths",
    "download_mendeley_dataset_tree",
    "download_mendeley_dataset_tree_from_api",
    "read_mendeley_files",
    "read_mendeley_folders",
]
