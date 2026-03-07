"""Application services for dataset download/prepare workflows."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from ser.config import AppConfig
from ser.data.dataset_capabilities import resolve_dataset_capability_profile
from ser.data.dataset_consents import (
    is_policy_restricted,
    load_persisted_dataset_consents,
    persist_dataset_consents,
)
from ser.data.dataset_prepare import (
    SUPPORTED_DATASETS,
    DatasetDescriptor,
    collect_dataset_registry_health_issues,
    default_dataset_root,
    default_manifest_path,
    download_dataset,
    prepare_dataset_manifest,
    resolve_dataset_descriptor,
)
from ser.data.dataset_registry import (
    load_dataset_registry,
    parse_dataset_registry_options,
    remove_dataset_registry_entry,
)
from ser.data.label_ontology import resolve_label_ontology


@dataclass(frozen=True, slots=True)
class DatasetDescriptorConsentStatus:
    """Missing consent status for one dataset descriptor."""

    descriptor: DatasetDescriptor
    missing_policy_consents: tuple[str, ...]
    missing_license_consents: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DatasetPrepareWorkflowResult:
    """Result of one dataset acquire+prepare workflow execution."""

    descriptor: DatasetDescriptor
    dataset_root: Path
    manifest_path: Path
    manifest_paths: tuple[Path, ...]
    downloaded: bool
    source_repo_id: str | None
    source_revision: str | None
    source_commit_sha: str | None


@dataclass(frozen=True, slots=True)
class DatasetUninstallWorkflowResult:
    """Result of one dataset uninstall workflow execution."""

    descriptor: DatasetDescriptor
    removed_from_registry: bool
    removed_manifest_paths: tuple[Path, ...]
    removed_dataset_roots: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class DatasetRegistrySnapshotEntry:
    """One typed dataset registry entry for UI/API consumers."""

    dataset_id: str
    dataset_root: Path
    manifest_path: Path
    options: dict[str, str]
    source_repo_id: str | None
    source_revision: str | None
    source_commit_sha: str | None = None


@dataclass(frozen=True, slots=True)
class DatasetRegistrySnapshotIssue:
    """One typed dataset registry health issue."""

    dataset_id: str
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class DatasetRegistrySnapshot:
    """Typed snapshot payload for registry listing and health checks."""

    entries: tuple[DatasetRegistrySnapshotEntry, ...]
    issues: tuple[DatasetRegistrySnapshotIssue, ...]


@dataclass(frozen=True, slots=True)
class DatasetCapabilitySnapshotEntry:
    """Typed capability + readiness snapshot row for one dataset."""

    dataset_id: str
    display_name: str
    registered: bool
    installed: bool
    manifest_exists: bool
    dataset_root: Path | None
    manifest_path: Path | None
    referenced_audio_files: int
    present_audio_files: int
    nonempty_audio_files: int
    dataset_size_bytes: int
    source_url: str
    policy_id: str
    license_id: str
    modalities: tuple[str, ...]
    label_schema: str
    has_label_mapping: bool
    supervised_ser_candidate: bool
    ssl_candidate: bool
    multimodal_candidate: bool
    mergeable_with_emotion_ontology: bool
    recommended_uses: tuple[str, ...]
    notes: tuple[str, ...]


def compute_dataset_descriptor_missing_consents(
    *,
    settings: AppConfig,
    dataset_id: str,
) -> DatasetDescriptorConsentStatus:
    """Computes missing restricted consents for one dataset id."""

    descriptor = resolve_dataset_descriptor(dataset_id)
    persisted = load_persisted_dataset_consents(settings=settings)
    normalized_policy = descriptor.policy_id.strip().lower()
    normalized_license = descriptor.license_id.strip().lower()

    missing_policies: tuple[str, ...] = ()
    missing_licenses: tuple[str, ...] = ()
    if (
        is_policy_restricted(normalized_policy)
        and normalized_policy not in persisted.policy_consents
    ):
        missing_policies = (normalized_policy,)
    if (
        is_policy_restricted(normalized_policy)
        and normalized_license
        and normalized_license not in persisted.license_consents
    ):
        missing_licenses = (normalized_license,)

    return DatasetDescriptorConsentStatus(
        descriptor=descriptor,
        missing_policy_consents=missing_policies,
        missing_license_consents=missing_licenses,
    )


def persist_missing_dataset_descriptor_consents(
    *,
    settings: AppConfig,
    missing_policy_consents: tuple[str, ...],
    missing_license_consents: tuple[str, ...],
    source: str,
) -> None:
    """Persists one set of missing descriptor consents, if any."""

    if not missing_policy_consents and not missing_license_consents:
        return
    persist_dataset_consents(
        settings=settings,
        accept_policy_ids=list(missing_policy_consents),
        accept_license_ids=list(missing_license_consents),
        source=source,
    )


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
    resolved_labels_csv_path = (
        labels_csv_path.expanduser() if labels_csv_path is not None else None
    )
    resolved_audio_base_dir = (
        audio_base_dir.expanduser() if audio_base_dir is not None else None
    )
    normalized_source_repo_id = (
        source_repo_id.strip() if source_repo_id is not None else None
    )
    normalized_source_revision = (
        source_revision.strip() if source_revision is not None else None
    )
    if normalized_source_repo_id == "":
        normalized_source_repo_id = None
    if normalized_source_revision == "":
        normalized_source_revision = None
    if skip_download and (
        normalized_source_repo_id is not None or normalized_source_revision is not None
    ):
        raise ValueError(
            "Download source overrides cannot be used when skip_download=True."
        )

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


def collect_dataset_registry_snapshot(
    *,
    settings: AppConfig,
) -> DatasetRegistrySnapshot:
    """Collects typed registry entries and deterministic health issues."""

    registry = load_dataset_registry(settings=settings)
    entries: list[DatasetRegistrySnapshotEntry] = []
    for entry in sorted(registry.values(), key=lambda item: item.dataset_id):
        parsed_options = parse_dataset_registry_options(entry.options)
        entries.append(
            DatasetRegistrySnapshotEntry(
                dataset_id=entry.dataset_id,
                dataset_root=entry.dataset_root,
                manifest_path=entry.manifest_path,
                options=parsed_options.as_dict(),
                source_repo_id=parsed_options.source_repo_id,
                source_revision=parsed_options.source_revision,
                source_commit_sha=parsed_options.source_commit_sha,
            )
        )

    issues = tuple(
        DatasetRegistrySnapshotIssue(
            dataset_id=issue.dataset_id,
            code=issue.code,
            message=issue.message,
        )
        for issue in collect_dataset_registry_health_issues(settings=settings)
    )
    return DatasetRegistrySnapshot(entries=tuple(entries), issues=issues)


def build_dataset_registry_snapshot_json_payload(
    snapshot: DatasetRegistrySnapshot,
) -> dict[str, object]:
    """Builds a JSON-serializable payload for registry inspection output."""

    payload_entries: list[dict[str, object]] = []
    for entry in snapshot.entries:
        payload_entries.append(
            {
                "dataset_id": entry.dataset_id,
                "dataset_root": str(entry.dataset_root),
                "manifest_path": str(entry.manifest_path),
                "source_repo_id": entry.source_repo_id,
                "source_revision": entry.source_revision,
                "source_commit_sha": entry.source_commit_sha,
                "options": entry.options,
            }
        )
    payload_issues: list[dict[str, str]] = []
    for issue in snapshot.issues:
        payload_issues.append(
            {
                "dataset_id": issue.dataset_id,
                "code": issue.code,
                "message": issue.message,
            }
        )
    return {
        "entries": payload_entries,
        "issues": payload_issues,
    }


def _resolve_manifest_audio_path(
    *,
    raw_audio_path: str,
    dataset_root: Path | None,
) -> Path:
    candidate = Path(raw_audio_path).expanduser()
    if candidate.is_absolute():
        return candidate
    if dataset_root is not None:
        return (dataset_root / candidate).expanduser()
    return candidate


def _collect_manifest_audio_stats(
    *,
    manifest_path: Path | None,
    dataset_root: Path | None,
) -> tuple[int, int, int, int]:
    if manifest_path is None or not manifest_path.is_file():
        return (0, 0, 0, 0)

    referenced = 0
    present = 0
    nonempty = 0
    total_bytes = 0
    seen_paths: set[Path] = set()

    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            raw_audio_path = payload.get("audio_path")
            if not isinstance(raw_audio_path, str) or not raw_audio_path.strip():
                continue
            resolved_path = _resolve_manifest_audio_path(
                raw_audio_path=raw_audio_path,
                dataset_root=dataset_root,
            )
            if resolved_path in seen_paths:
                continue
            seen_paths.add(resolved_path)
            referenced += 1
            if not resolved_path.is_file():
                continue
            present += 1
            size_bytes = resolved_path.stat().st_size
            total_bytes += size_bytes
            if size_bytes > 0:
                nonempty += 1
    return (referenced, present, nonempty, total_bytes)


def collect_dataset_capability_snapshot(
    *,
    settings: AppConfig,
    include_uninstalled: bool = False,
) -> tuple[DatasetCapabilitySnapshotEntry, ...]:
    """Collects capability records for installed datasets (or all supported)."""

    registry_snapshot = collect_dataset_registry_snapshot(settings=settings)
    registry_by_id = {entry.dataset_id: entry for entry in registry_snapshot.entries}
    rows: list[DatasetCapabilitySnapshotEntry] = []
    for dataset_id in sorted(SUPPORTED_DATASETS):
        descriptor = SUPPORTED_DATASETS[dataset_id]
        registry_entry = registry_by_id.get(dataset_id)
        registered = registry_entry is not None
        manifest_exists = (
            registry_entry.manifest_path.is_file()
            if registry_entry is not None
            else False
        )
        (
            referenced_audio_files,
            present_audio_files,
            nonempty_audio_files,
            dataset_size_bytes,
        ) = _collect_manifest_audio_stats(
            manifest_path=registry_entry.manifest_path if registry_entry else None,
            dataset_root=registry_entry.dataset_root if registry_entry else None,
        )
        installed = registered and manifest_exists and nonempty_audio_files > 0
        if not include_uninstalled and not installed:
            continue
        profile = resolve_dataset_capability_profile(dataset_id)
        rows.append(
            DatasetCapabilitySnapshotEntry(
                dataset_id=dataset_id,
                display_name=descriptor.display_name,
                registered=registered,
                installed=installed,
                manifest_exists=manifest_exists,
                dataset_root=registry_entry.dataset_root if registry_entry else None,
                manifest_path=registry_entry.manifest_path if registry_entry else None,
                referenced_audio_files=referenced_audio_files,
                present_audio_files=present_audio_files,
                nonempty_audio_files=nonempty_audio_files,
                dataset_size_bytes=dataset_size_bytes,
                source_url=descriptor.source_url,
                policy_id=descriptor.policy_id,
                license_id=descriptor.license_id,
                modalities=profile.modalities,
                label_schema=profile.label_schema,
                has_label_mapping=profile.has_label_mapping,
                supervised_ser_candidate=profile.supervised_ser_candidate,
                ssl_candidate=profile.ssl_candidate,
                multimodal_candidate=profile.multimodal_candidate,
                mergeable_with_emotion_ontology=profile.mergeable_with_emotion_ontology,
                recommended_uses=profile.recommended_uses,
                notes=profile.notes,
            )
        )
    return tuple(rows)


def build_dataset_capability_snapshot_json_payload(
    rows: tuple[DatasetCapabilitySnapshotEntry, ...],
) -> dict[str, object]:
    """Builds JSON-serializable payload for dataset capability snapshot output."""

    entries: list[dict[str, object]] = []
    for row in rows:
        entries.append(
            {
                "dataset_id": row.dataset_id,
                "display_name": row.display_name,
                "registered": row.registered,
                "installed": row.installed,
                "manifest_exists": row.manifest_exists,
                "dataset_root": (
                    str(row.dataset_root) if row.dataset_root is not None else None
                ),
                "manifest_path": (
                    str(row.manifest_path) if row.manifest_path is not None else None
                ),
                "referenced_audio_files": row.referenced_audio_files,
                "present_audio_files": row.present_audio_files,
                "nonempty_audio_files": row.nonempty_audio_files,
                "dataset_size_bytes": row.dataset_size_bytes,
                "source_url": row.source_url,
                "policy_id": row.policy_id,
                "license_id": row.license_id,
                "modalities": list(row.modalities),
                "label_schema": row.label_schema,
                "has_label_mapping": row.has_label_mapping,
                "supervised_ser_candidate": row.supervised_ser_candidate,
                "ssl_candidate": row.ssl_candidate,
                "multimodal_candidate": row.multimodal_candidate,
                "mergeable_with_emotion_ontology": row.mergeable_with_emotion_ontology,
                "recommended_uses": list(row.recommended_uses),
                "notes": list(row.notes),
            }
        )
    return {"entries": entries}
