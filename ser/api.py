"""Public API facade for runtime, dataset, and diagnostics workflows."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import ser._internal.api.data as _data_api
import ser._internal.api.diagnostics as _diagnostics_api
import ser._internal.api.runtime as _runtime_api
from ser.config import AppConfig, reload_settings
from ser.profiles import ProfileName

if TYPE_CHECKING:
    from ser.diagnostics.domain import DiagnosticReport
    from ser.runtime.contracts import InferenceExecution, InferenceRequest, SubtitleFormat

ComplianceMode = _data_api.ComplianceMode
DatasetPrepareResult = _data_api.DatasetPrepareResult
DatasetRegistryHealthIssueRecord = _data_api.DatasetRegistryHealthIssueRecord
DatasetRegistryRecord = _data_api.DatasetRegistryRecord


class _PublicRuntimePipeline(Protocol):
    """Minimal runtime pipeline contract exposed at the public API facade."""

    def run_training(self) -> None:
        """Runs training for the active profile."""
        ...

    def run_inference(self, request: InferenceRequest) -> InferenceExecution:
        """Runs inference for one audio request."""
        ...


type RuntimePipelineBuilder = Callable[[AppConfig], _PublicRuntimePipeline]


def _resolve_boundary_settings(settings: AppConfig | None) -> AppConfig:
    """Returns explicit settings or reloads a boundary-local settings snapshot."""
    return settings if settings is not None else reload_settings()


def list_datasets() -> tuple[str, ...]:
    """Returns all supported dataset identifiers in deterministic order."""
    return _data_api.list_datasets()


def list_registered_datasets(
    *,
    settings: AppConfig | None = None,
) -> tuple[DatasetRegistryRecord, ...]:
    """Returns registered dataset records using the active settings snapshot."""
    return _data_api.list_registered_datasets(settings=_resolve_boundary_settings(settings))


def list_dataset_registry_health_issues(
    *,
    settings: AppConfig | None = None,
) -> tuple[DatasetRegistryHealthIssueRecord, ...]:
    """Returns dataset registry health issues using the active settings snapshot."""
    return _data_api.list_dataset_registry_health_issues(
        settings=_resolve_boundary_settings(settings)
    )


def show_dataset_consents(
    *,
    settings: AppConfig | None = None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Returns persisted dataset consents using the active settings snapshot."""
    return _data_api.show_dataset_consents(settings=_resolve_boundary_settings(settings))


def configure_dataset_consents(
    *,
    accept_policy_ids: tuple[str, ...] = (),
    accept_license_ids: tuple[str, ...] = (),
    settings: AppConfig | None = None,
    source: str = "ser.api.configure_dataset_consents",
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Persists dataset consents using the active settings snapshot."""
    return _data_api.configure_dataset_consents(
        accept_policy_ids=accept_policy_ids,
        accept_license_ids=accept_license_ids,
        settings=_resolve_boundary_settings(settings),
        source=source,
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
    settings: AppConfig | None = None,
) -> DatasetPrepareResult:
    """Runs programmatic dataset preparation using the active settings snapshot."""
    return _data_api.prepare_dataset(
        dataset_id=dataset_id,
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        labels_csv_path=labels_csv_path,
        audio_base_dir=audio_base_dir,
        source_repo_id=source_repo_id,
        source_revision=source_revision,
        default_language=default_language,
        skip_download=skip_download,
        accept_license=accept_license,
        compliance_mode=compliance_mode,
        settings=_resolve_boundary_settings(settings),
    )


def list_profiles() -> tuple[ProfileName, ...]:
    """Returns all registered runtime profile names."""
    return _runtime_api.list_profiles()


def load_profile(
    profile: ProfileName,
    *,
    settings: AppConfig | None = None,
) -> None:
    """Validates one runtime profile using the active settings snapshot."""
    return _runtime_api.load_profile(
        profile,
        settings=_resolve_boundary_settings(settings),
    )


def train(
    *,
    profile: ProfileName | None = None,
    settings: AppConfig | None = None,
    use_profile_pipeline: bool = True,
    pipeline_builder: RuntimePipelineBuilder | None = None,
) -> None:
    """Runs training using the active settings snapshot via the runtime pipeline."""
    return _runtime_api.train(
        profile=profile,
        settings=_resolve_boundary_settings(settings),
        use_profile_pipeline=use_profile_pipeline,
        pipeline_builder=pipeline_builder,
    )


def infer(
    file_path: str | Path,
    *,
    profile: ProfileName | None = None,
    language: str | None = None,
    save_transcript: bool = False,
    include_transcript: bool = True,
    subtitle_output_path: str | None = None,
    subtitle_format: SubtitleFormat | None = None,
    settings: AppConfig | None = None,
    pipeline_builder: RuntimePipelineBuilder | None = None,
) -> InferenceExecution:
    """Runs inference using the active settings snapshot."""
    return _runtime_api.infer(
        file_path,
        profile=profile,
        language=language,
        save_transcript=save_transcript,
        include_transcript=include_transcript,
        subtitle_output_path=subtitle_output_path,
        subtitle_format=subtitle_format,
        settings=_resolve_boundary_settings(settings),
        pipeline_builder=pipeline_builder,
    )


def run_startup_preflight(
    *,
    include_transcription_checks: bool,
    settings: AppConfig | None = None,
) -> DiagnosticReport:
    """Runs structured startup diagnostics using the active settings snapshot."""
    return _diagnostics_api.run_startup_preflight(
        settings=_resolve_boundary_settings(settings),
        include_transcription_checks=include_transcription_checks,
    )


__all__ = [
    "ComplianceMode",
    "DatasetPrepareResult",
    "DatasetRegistryHealthIssueRecord",
    "DatasetRegistryRecord",
    "configure_dataset_consents",
    "infer",
    "list_dataset_registry_health_issues",
    "list_datasets",
    "list_profiles",
    "list_registered_datasets",
    "load_profile",
    "prepare_dataset",
    "run_startup_preflight",
    "show_dataset_consents",
    "train",
]
