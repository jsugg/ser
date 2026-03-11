"""Public dataset application facade."""

from __future__ import annotations

from ser._internal.data.application.capability_snapshot import (
    build_dataset_capability_snapshot_json_payload,
    collect_dataset_capability_snapshot,
)
from ser._internal.data.application.models import (
    DatasetCapabilitySnapshotEntry,
    DatasetDescriptorConsentStatus,
    DatasetPrepareWorkflowResult,
    DatasetRegistrySnapshot,
    DatasetRegistrySnapshotEntry,
    DatasetRegistrySnapshotIssue,
    DatasetUninstallWorkflowResult,
)
from ser._internal.data.application.prepare import run_dataset_prepare_workflow
from ser._internal.data.application.registry_snapshot import (
    build_dataset_registry_snapshot_json_payload,
    collect_dataset_registry_snapshot,
)
from ser._internal.data.application.uninstall import run_dataset_uninstall_workflow
from ser.config import AppConfig
from ser.data.dataset_consents import (
    is_policy_restricted,
    load_persisted_dataset_consents,
    persist_dataset_consents,
)
from ser.data.dataset_prepare import resolve_dataset_descriptor

__all__ = [
    "DatasetCapabilitySnapshotEntry",
    "DatasetDescriptorConsentStatus",
    "DatasetPrepareWorkflowResult",
    "DatasetRegistrySnapshot",
    "DatasetRegistrySnapshotEntry",
    "DatasetRegistrySnapshotIssue",
    "DatasetUninstallWorkflowResult",
    "build_dataset_capability_snapshot_json_payload",
    "build_dataset_registry_snapshot_json_payload",
    "collect_dataset_capability_snapshot",
    "collect_dataset_registry_snapshot",
    "compute_dataset_descriptor_missing_consents",
    "persist_missing_dataset_descriptor_consents",
    "run_dataset_prepare_workflow",
    "run_dataset_uninstall_workflow",
]


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
