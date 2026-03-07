"""Archive extraction helpers for public dataset acquisition workflows."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tarfile
import time
import zipfile
from collections.abc import Callable
from pathlib import Path

SUPPORTED_ARCHIVE_SUFFIXES: tuple[str, ...] = (
    ".zip",
    ".rar",
    ".tar",
    ".tar.gz",
    ".tgz",
    ".tar.bz2",
    ".tbz2",
    ".tar.xz",
    ".txz",
)


def is_safe_destination_path(*, extract_root: Path, destination_path: Path) -> bool:
    """Returns whether extraction destination is inside extract root."""
    root_resolved = extract_root.resolve()
    target_resolved = destination_path.resolve()
    return target_resolved == root_resolved or root_resolved in target_resolved.parents


def extract_zip_archive(
    *,
    archive_path: Path,
    extract_root: Path,
    is_safe_destination: Callable[..., bool] = is_safe_destination_path,
) -> None:
    """Extracts one zip archive with path-traversal protections."""
    extract_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, mode="r") as archive:
        for member in archive.infolist():
            member_path = Path(member.filename)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise RuntimeError(
                    f"Unsafe archive member path in {archive_path.name}: {member.filename!r}"
                )
            target_path = extract_root / member_path
            if not is_safe_destination(
                extract_root=extract_root,
                destination_path=target_path,
            ):
                raise RuntimeError(
                    f"Unsafe archive extraction target in {archive_path.name}: {member.filename!r}"
                )
            if member.is_dir():
                target_path.mkdir(parents=True, exist_ok=True)
                continue
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member, mode="r") as source_handle:
                with target_path.open("wb") as target_handle:
                    shutil.copyfileobj(source_handle, target_handle)


def extract_rar_archive(
    *,
    archive_path: Path,
    extract_root: Path,
    os_name: str = os.name,
    which: Callable[[str], str | None] = shutil.which,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    logger_warning: Callable[..., object] | None = None,
) -> None:
    """Extracts one rar archive using first available backend in PATH."""
    extract_root.mkdir(parents=True, exist_ok=True)
    backend_commands: list[list[str]]
    if os_name == "nt":
        backend_commands = [
            ["7z", "x", "-y", f"-o{extract_root}", str(archive_path)],
            ["unrar", "x", "-o+", str(archive_path), f"{extract_root}/"],
            [
                "unar",
                "-quiet",
                "-force-overwrite",
                "-o",
                str(extract_root),
                str(archive_path),
            ],
            ["bsdtar", "-xf", str(archive_path), "-C", str(extract_root)],
        ]
    else:
        backend_commands = [
            [
                "unar",
                "-quiet",
                "-force-overwrite",
                "-o",
                str(extract_root),
                str(archive_path),
            ],
            ["7z", "x", "-y", f"-o{extract_root}", str(archive_path)],
            ["unrar", "x", "-o+", str(archive_path), f"{extract_root}/"],
            ["bsdtar", "-xf", str(archive_path), "-C", str(extract_root)],
        ]

    attempted_backends: list[str] = []
    for command in backend_commands:
        binary_name = command[0]
        binary_path = which(binary_name)
        if binary_path is None:
            continue
        attempted_backends.append(binary_name)
        command_to_run = [binary_path, *command[1:]]
        completed = run(
            command_to_run,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode == 0:
            return
        if logger_warning is not None:
            logger_warning(
                "RAR extraction backend `%s` failed for %s (exit=%s). stderr=%s",
                binary_name,
                archive_path.name,
                completed.returncode,
                completed.stderr.strip(),
            )

    raise RuntimeError(
        "RAR extraction requires one supported backend in PATH "
        "(tried: "
        + ", ".join(attempted_backends or ["none detected"])
        + "). Install one of: `unar`, `7z`, `unrar`, or `bsdtar`."
    )


def extract_tar_archive(
    *,
    archive_path: Path,
    extract_root: Path,
    is_safe_destination: Callable[..., bool] = is_safe_destination_path,
) -> None:
    """Extracts one tar-family archive with member-path validation."""
    extract_root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, mode="r:*") as archive:
        for member in archive.getmembers():
            member_path = Path(member.name)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise RuntimeError(
                    f"Unsafe archive member path in {archive_path.name}: {member.name!r}"
                )
            target_path = extract_root / member_path
            if not is_safe_destination(
                extract_root=extract_root,
                destination_path=target_path,
            ):
                raise RuntimeError(
                    f"Unsafe archive extraction target in {archive_path.name}: {member.name!r}"
                )
            if member.isdir():
                target_path.mkdir(parents=True, exist_ok=True)
                continue
            if member.isreg():
                target_path.parent.mkdir(parents=True, exist_ok=True)
                source_handle = archive.extractfile(member)
                if source_handle is None:
                    raise RuntimeError(
                        f"Could not read archive member in {archive_path.name}: {member.name!r}"
                    )
                with source_handle:
                    with target_path.open("wb") as target_handle:
                        shutil.copyfileobj(source_handle, target_handle)
                continue
            raise RuntimeError(
                f"Unsupported tar member type in {archive_path.name}: {member.name!r}"
            )


def extract_archive(
    *,
    archive_path: Path,
    extract_root: Path,
    extract_zip: Callable[..., None] = extract_zip_archive,
    extract_rar: Callable[..., None] = extract_rar_archive,
    extract_tar: Callable[..., None] = extract_tar_archive,
) -> None:
    """Extracts one supported archive based on file extension."""
    lower_name = archive_path.name.lower()
    if lower_name.endswith(".zip"):
        extract_zip(archive_path=archive_path, extract_root=extract_root)
        return
    if lower_name.endswith(".rar"):
        extract_rar(archive_path=archive_path, extract_root=extract_root)
        return
    if (
        lower_name.endswith(".tar")
        or lower_name.endswith(".tar.gz")
        or lower_name.endswith(".tgz")
        or lower_name.endswith(".tar.bz2")
        or lower_name.endswith(".tbz2")
        or lower_name.endswith(".tar.xz")
        or lower_name.endswith(".txz")
    ):
        extract_tar(archive_path=archive_path, extract_root=extract_root)
        return
    raise RuntimeError(f"Unsupported archive format for {archive_path.name}.")


def ensure_extracted_archive(
    *,
    archive_path: Path,
    extract_root: Path,
    extract_archive_fn: Callable[..., None] = extract_archive,
    current_time: Callable[[], float] = time.time,
) -> None:
    """Extracts one archive once and writes an extraction-marker file."""
    marker_path = extract_root / f".extract-ok-{archive_path.name}.json"
    if marker_path.is_file():
        return
    extract_archive_fn(archive_path=archive_path, extract_root=extract_root)
    payload = {
        "archive": str(archive_path),
        "size_bytes": archive_path.stat().st_size,
        "extracted_at_unix": current_time(),
    }
    marker_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def extract_archives_from_tree(
    *,
    search_root: Path,
    extract_root: Path,
    ensure_extracted: Callable[..., None] = ensure_extracted_archive,
) -> list[Path]:
    """Extracts all supported archives under one tree and returns paths."""
    extracted_archives: list[Path] = []
    for archive_path in sorted(search_root.rglob("*")):
        if not archive_path.is_file():
            continue
        lower_name = archive_path.name.lower()
        if not any(
            lower_name.endswith(suffix) for suffix in SUPPORTED_ARCHIVE_SUFFIXES
        ):
            continue
        ensure_extracted(archive_path=archive_path, extract_root=extract_root)
        extracted_archives.append(archive_path)
    if not extracted_archives:
        raise RuntimeError(f"No extractable archives found under {search_root}.")
    return extracted_archives


__all__ = [
    "SUPPORTED_ARCHIVE_SUFFIXES",
    "ensure_extracted_archive",
    "extract_archive",
    "extract_archives_from_tree",
    "extract_rar_archive",
    "extract_tar_archive",
    "extract_zip_archive",
    "is_safe_destination_path",
]
