"""Dataset uninstall workflow orchestration."""

from __future__ import annotations

import shutil
from pathlib import Path

from ser._internal.data.application.models import DatasetUninstallWorkflowResult
from ser.config import AppConfig
from ser.data.dataset_prepare import (
    default_dataset_root,
    default_manifest_path,
    resolve_dataset_descriptor,
)
from ser.data.dataset_registry import remove_dataset_registry_entry


def run_dataset_uninstall_workflow(
    *,
    settings: AppConfig,
    dataset_id: str,
    remove_files: bool = True,
) -> DatasetUninstallWorkflowResult:
    """Runs one dataset uninstall workflow against local registry/artifacts."""

    descriptor = resolve_dataset_descriptor(dataset_id)
    existing = remove_dataset_registry_entry(
        settings=settings,
        dataset_id=descriptor.dataset_id,
    )
    if existing is None:
        return DatasetUninstallWorkflowResult(
            descriptor=descriptor,
            removed_from_registry=False,
            removed_manifest_paths=(),
            removed_dataset_roots=(),
        )

    removed_manifest_paths: list[Path] = []
    removed_dataset_roots: list[Path] = []
    if remove_files:
        manifest_paths_to_remove = {
            existing.manifest_path.expanduser(),
            default_manifest_path(settings, descriptor.dataset_id).expanduser(),
        }
        for manifest_path in sorted(manifest_paths_to_remove):
            if not manifest_path.is_file():
                continue
            manifest_path.unlink()
            removed_manifest_paths.append(manifest_path)

        dataset_roots_to_remove = {
            existing.dataset_root.expanduser(),
            default_dataset_root(settings, descriptor.dataset_id).expanduser(),
        }
        for dataset_root in sorted(dataset_roots_to_remove):
            if not dataset_root.is_dir():
                continue
            shutil.rmtree(dataset_root)
            removed_dataset_roots.append(dataset_root)

    return DatasetUninstallWorkflowResult(
        descriptor=descriptor,
        removed_from_registry=True,
        removed_manifest_paths=tuple(removed_manifest_paths),
        removed_dataset_roots=tuple(removed_dataset_roots),
    )
