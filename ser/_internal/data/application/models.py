"""Shared result models for dataset application services."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ser.data.dataset_prepare import DatasetDescriptor


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
