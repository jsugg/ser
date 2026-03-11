"""Dataset prepare workflow orchestration."""

from __future__ import annotations

from pathlib import Path

from ser._internal.data.application.models import DatasetPrepareWorkflowResult
from ser.config import AppConfig
from ser.data.dataset_prepare import (
    default_dataset_root,
    default_manifest_path,
    download_dataset,
    prepare_dataset_manifest,
    resolve_dataset_descriptor,
)
from ser.data.dataset_registry import (
    load_dataset_registry,
    parse_dataset_registry_options,
)
from ser.data.label_ontology import resolve_label_ontology


def run_dataset_prepare_workflow(
    *,
    settings: AppConfig,
    dataset_id: str,
    dataset_root: Path | None = None,
    manifest_path: Path | None = None,
    labels_csv_path: Path | None = None,
    audio_base_dir: Path | None = None,
    source_repo_id: str | None = None,
    source_revision: str | None = None,
    default_language: str | None = None,
    skip_download: bool = False,
) -> DatasetPrepareWorkflowResult:
    """Runs one dataset acquisition + manifest-preparation workflow."""

    descriptor = resolve_dataset_descriptor(dataset_id)
    resolved_dataset_root = (
        dataset_root.expanduser()
        if dataset_root is not None
        else default_dataset_root(settings, descriptor.dataset_id)
    )
    resolved_manifest_path = (
        manifest_path.expanduser()
        if manifest_path is not None
        else default_manifest_path(settings, descriptor.dataset_id)
    )
    resolved_labels_csv_path = labels_csv_path.expanduser() if labels_csv_path is not None else None
    resolved_audio_base_dir = audio_base_dir.expanduser() if audio_base_dir is not None else None
    normalized_source_repo_id = source_repo_id.strip() if source_repo_id is not None else None
    normalized_source_revision = source_revision.strip() if source_revision is not None else None
    if normalized_source_repo_id == "":
        normalized_source_repo_id = None
    if normalized_source_revision == "":
        normalized_source_revision = None
    if skip_download and (
        normalized_source_repo_id is not None or normalized_source_revision is not None
    ):
        raise ValueError("Download source overrides cannot be used when skip_download=True.")

    downloaded = False
    resolved_source_repo_id: str | None = None
    resolved_source_revision: str | None = None
    resolved_source_commit_sha: str | None = None
    if not skip_download:
        resolved_source_repo_id, resolved_source_revision = download_dataset(
            settings=settings,
            dataset_id=descriptor.dataset_id,
            dataset_root=resolved_dataset_root,
            source_repo_id=normalized_source_repo_id,
            source_revision=normalized_source_revision,
        )
        downloaded = True

    ontology = resolve_label_ontology(settings)
    built_paths = prepare_dataset_manifest(
        settings=settings,
        dataset_id=descriptor.dataset_id,
        dataset_root=resolved_dataset_root,
        ontology=ontology,
        manifest_path=resolved_manifest_path,
        labels_csv_path=resolved_labels_csv_path,
        audio_base_dir=resolved_audio_base_dir,
        source_repo_id=resolved_source_repo_id,
        source_revision=resolved_source_revision,
        default_language=default_language,
    )
    if descriptor.dataset_id == "msp-podcast":
        registry = load_dataset_registry(settings=settings)
        persisted_entry = registry.get(descriptor.dataset_id)
        if persisted_entry is not None:
            parsed_options = parse_dataset_registry_options(persisted_entry.options)
            resolved_source_commit_sha = parsed_options.source_commit_sha

    return DatasetPrepareWorkflowResult(
        descriptor=descriptor,
        dataset_root=resolved_dataset_root,
        manifest_path=resolved_manifest_path,
        manifest_paths=tuple(built_paths),
        downloaded=downloaded,
        source_repo_id=resolved_source_repo_id,
        source_revision=resolved_source_revision,
        source_commit_sha=resolved_source_commit_sha,
    )
