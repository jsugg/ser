"""Dataset-focused public API helpers for library and CLI orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from ser.config import AppConfig
from ser.utils.logger import get_logger

logger = get_logger(__name__)

type ComplianceMode = Literal["advisory", "strict"]

if TYPE_CHECKING:
    from ser.data.msp_podcast_mirror import MspPodcastMirrorArtifacts


@dataclass(frozen=True, slots=True)
class DatasetPrepareResult:
    """Result payload for one programmatic dataset preparation request."""

    dataset_id: str
    dataset_root: Path
    manifest_paths: tuple[Path, ...]
    downloaded: bool
    missing_policy_consents: tuple[str, ...]
    missing_license_consents: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DatasetRegistryRecord:
    """Read model for one dataset registry entry."""

    dataset_id: str
    dataset_root: Path
    manifest_path: Path
    options: dict[str, str]
    source_repo_id: str | None
    source_revision: str | None
    source_commit_sha: str | None = None


@dataclass(frozen=True, slots=True)
class DatasetRegistryHealthIssueRecord:
    """Read model for one dataset registry health issue."""

    dataset_id: str
    code: str
    message: str


def run_configure_command(argv: list[str], *, settings: AppConfig) -> int:
    """Runs `ser configure ...` command argv via the public API boundary."""
    from ser.data.cli import run_configure_command as _run_configure_command

    return _run_configure_command(argv, settings=settings)


def run_data_command(argv: list[str], *, settings: AppConfig) -> int:
    """Runs `ser data ...` command argv via the public API boundary."""
    from ser.data.cli import run_data_command as _run_data_command

    return _run_data_command(argv, settings=settings)


def list_datasets() -> tuple[str, ...]:
    """Returns sorted supported dataset identifiers."""
    from ser.data.dataset_prepare import SUPPORTED_DATASETS

    return tuple(sorted(SUPPORTED_DATASETS))


def list_registered_datasets(
    *,
    settings: AppConfig,
) -> tuple[DatasetRegistryRecord, ...]:
    """Returns persisted dataset registry records ordered by dataset id."""
    from ser.data.application import collect_dataset_registry_snapshot

    snapshot = collect_dataset_registry_snapshot(settings=settings)
    records: list[DatasetRegistryRecord] = []
    for entry in snapshot.entries:
        records.append(
            DatasetRegistryRecord(
                dataset_id=entry.dataset_id,
                dataset_root=entry.dataset_root,
                manifest_path=entry.manifest_path,
                options=entry.options,
                source_repo_id=entry.source_repo_id,
                source_revision=entry.source_revision,
                source_commit_sha=entry.source_commit_sha,
            )
        )
    return tuple(records)


def list_dataset_registry_health_issues(
    *,
    settings: AppConfig,
) -> tuple[DatasetRegistryHealthIssueRecord, ...]:
    """Returns deterministic dataset registry health issues."""
    from ser.data.application import collect_dataset_registry_snapshot

    snapshot = collect_dataset_registry_snapshot(settings=settings)
    return tuple(
        DatasetRegistryHealthIssueRecord(
            dataset_id=issue.dataset_id,
            code=issue.code,
            message=issue.message,
        )
        for issue in snapshot.issues
    )


def show_dataset_consents(
    *,
    settings: AppConfig,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Returns persisted dataset consent IDs as `(policy_ids, license_ids)`."""
    from ser.data.dataset_consents import load_persisted_dataset_consents

    consents = load_persisted_dataset_consents(settings=settings)
    return (
        tuple(sorted(consents.policy_consents)),
        tuple(sorted(consents.license_consents)),
    )


def configure_dataset_consents(
    *,
    accept_policy_ids: tuple[str, ...] = (),
    accept_license_ids: tuple[str, ...] = (),
    settings: AppConfig,
    source: str = "ser.api.configure_dataset_consents",
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Persists dataset consents and returns updated `(policy_ids, license_ids)`."""
    from ser.data.dataset_consents import (
        load_persisted_dataset_consents,
        persist_dataset_consents,
    )

    persist_dataset_consents(
        settings=settings,
        accept_policy_ids=list(accept_policy_ids),
        accept_license_ids=list(accept_license_ids),
        source=source,
    )
    updated = load_persisted_dataset_consents(settings=settings)
    return (
        tuple(sorted(updated.policy_consents)),
        tuple(sorted(updated.license_consents)),
    )


def prepare_dataset(
    *,
    dataset_id: str,
    dataset_root: Path | None = None,
    manifest_path: Path | None = None,
    labels_csv_path: Path | None = None,
    audio_base_dir: Path | None = None,
    source_repo_id: str | None = None,
    source_revision: str | None = None,
    default_language: str | None = None,
    skip_download: bool = False,
    accept_license: bool = False,
    compliance_mode: ComplianceMode = "advisory",
    settings: AppConfig,
) -> DatasetPrepareResult:
    """Programmatic dataset acquisition + manifest preparation."""
    from ser.data.application import (
        compute_dataset_descriptor_missing_consents,
        persist_missing_dataset_descriptor_consents,
        run_dataset_prepare_workflow,
    )
    from ser.data.dataset_consents import DatasetConsentError

    if compliance_mode not in {"advisory", "strict"}:
        raise ValueError(
            "Unsupported compliance_mode "
            f"{compliance_mode!r}; expected 'advisory' or 'strict'."
        )

    consent_status = compute_dataset_descriptor_missing_consents(
        settings=settings,
        dataset_id=dataset_id,
    )
    descriptor = consent_status.descriptor

    missing_policies = consent_status.missing_policy_consents
    missing_licenses = consent_status.missing_license_consents

    if missing_policies or missing_licenses:
        if accept_license:
            persist_missing_dataset_descriptor_consents(
                settings=settings,
                missing_policy_consents=missing_policies,
                missing_license_consents=missing_licenses,
                source=f"ser.api.prepare_dataset:{descriptor.dataset_id}",
            )
            missing_policies = ()
            missing_licenses = ()
        elif compliance_mode == "strict":
            policies_text = (
                ", ".join(missing_policies) if missing_policies else "(none)"
            )
            licenses_text = (
                ", ".join(missing_licenses) if missing_licenses else "(none)"
            )
            raise DatasetConsentError(
                "Missing dataset acknowledgements for strict library execution.\n"
                f"Missing policy consent(s): {policies_text}\n"
                f"Missing license consent(s): {licenses_text}\n"
                "Persist consents via `ser configure ... --persist` or "
                "`ser.api.configure_dataset_consents(...)`."
            )
        else:
            logger.warning(
                "Proceeding in advisory mode with missing dataset acknowledgements "
                "for '%s' (policy=%s licenses=%s).",
                descriptor.dataset_id,
                ",".join(missing_policies) if missing_policies else "(none)",
                ",".join(missing_licenses) if missing_licenses else "(none)",
            )

    workflow_result = run_dataset_prepare_workflow(
        settings=settings,
        dataset_id=descriptor.dataset_id,
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        labels_csv_path=labels_csv_path,
        audio_base_dir=audio_base_dir,
        source_repo_id=source_repo_id,
        source_revision=source_revision,
        default_language=default_language,
        skip_download=skip_download,
    )
    return DatasetPrepareResult(
        dataset_id=descriptor.dataset_id,
        dataset_root=workflow_result.dataset_root,
        manifest_paths=workflow_result.manifest_paths,
        downloaded=workflow_result.downloaded,
        missing_policy_consents=missing_policies,
        missing_license_consents=missing_licenses,
    )


def prepare_msp_podcast_mirror(
    *,
    dataset_root: Path,
    repo_id: str = "AbstractTTS/PODCAST",
    revision: str = "main",
    max_workers: int = 8,
    batch_size: int = 64,
    token: str | None = None,
) -> MspPodcastMirrorArtifacts:
    """Prepares MSP-Podcast artifacts from the configured Hugging Face mirror."""
    from ser.data.msp_podcast_mirror import prepare_msp_podcast_from_hf_mirror

    return prepare_msp_podcast_from_hf_mirror(
        dataset_root=dataset_root,
        repo_id=repo_id,
        revision=revision,
        max_workers=max_workers,
        batch_size=batch_size,
        token=token,
    )


__all__ = [
    "ComplianceMode",
    "DatasetPrepareResult",
    "DatasetRegistryHealthIssueRecord",
    "DatasetRegistryRecord",
    "configure_dataset_consents",
    "list_dataset_registry_health_issues",
    "list_datasets",
    "list_registered_datasets",
    "prepare_dataset",
    "prepare_msp_podcast_mirror",
    "run_configure_command",
    "run_data_command",
    "show_dataset_consents",
]
