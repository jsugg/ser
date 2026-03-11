"""Dataset registry snapshot services."""

from __future__ import annotations

from ser._internal.data.application.models import (
    DatasetRegistrySnapshot,
    DatasetRegistrySnapshotEntry,
    DatasetRegistrySnapshotIssue,
)
from ser.config import AppConfig
from ser.data.dataset_prepare import collect_dataset_registry_health_issues
from ser.data.dataset_registry import (
    load_dataset_registry,
    parse_dataset_registry_options,
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
