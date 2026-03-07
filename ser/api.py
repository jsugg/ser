"""Public API facade for runtime, dataset, and diagnostics workflows."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from ser._internal.api.data import (
    ComplianceMode,
    DatasetPrepareResult,
    DatasetRegistryHealthIssueRecord,
    DatasetRegistryRecord,
)
from ser._internal.api.data import (
    configure_dataset_consents as _configure_dataset_consents,
)
from ser._internal.api.data import (
    list_dataset_registry_health_issues as _list_dataset_registry_health_issues,
)
from ser._internal.api.data import (
    list_datasets,
)
from ser._internal.api.data import list_registered_datasets as _list_registered_datasets
from ser._internal.api.data import prepare_dataset as _prepare_dataset
from ser._internal.api.data import (
    prepare_msp_podcast_mirror,
)
from ser._internal.api.data import run_configure_command as _run_configure_command
from ser._internal.api.data import run_data_command as _run_data_command
from ser._internal.api.data import show_dataset_consents as _show_dataset_consents
from ser._internal.api.diagnostics import (
    format_startup_preflight_one_liner,
    parse_preflight_mode,
    preflight_command_requested,
    preflight_includes_transcription_checks,
    resolve_doctor_command,
    run_doctor_command,
    run_startup_preflight,
    run_startup_preflight_cli_gate,
    should_fail_preflight,
    suppress_preflight_transcription_operational_relogs,
)
from ser._internal.api.runtime import (
    apply_cli_profile_override,
    apply_cli_timeout_override,
    build_runtime_pipeline,
)
from ser._internal.api.runtime import infer as _infer
from ser._internal.api.runtime import (
    list_profiles,
)
from ser._internal.api.runtime import load_profile as _load_profile
from ser._internal.api.runtime import (
    profile_pipeline_enabled,
    profile_resolution_requested,
    resolve_cli_workflow_profile,
    run_inference_command,
    run_inference_workflow,
    run_restricted_backend_cli_gate,
    run_training_command,
    run_training_workflow,
    run_transcription_runtime_calibration_cli,
    run_transcription_runtime_calibration_command,
)
from ser._internal.api.runtime import train as _train
from ser._internal.runtime.commands import (
    WorkflowErrorDisposition,
    classify_inference_exception,
    classify_training_exception,
    run_transcription_runtime_calibration_workflow,
)
from ser._internal.runtime.restricted_backends import (
    RestrictedBackendOptInState,
    RestrictedBackendPrompt,
    collect_missing_restricted_backend_consents,
    enforce_restricted_backends_for_cli,
    ensure_restricted_backends_ready_for_command,
    persist_all_restricted_backend_consents,
    persist_required_restricted_backends,
    prepare_restricted_backend_opt_in_state,
    required_restricted_backends_for_current_profile,
)
from ser.config import AppConfig, get_settings
from ser.profiles import ProfileName

if TYPE_CHECKING:
    from ser.runtime.contracts import InferenceExecution, InferenceRequest


class _PublicRuntimePipeline(Protocol):
    """Minimal runtime pipeline contract exposed at the public API facade."""

    def run_training(self) -> None:
        """Runs training for the active profile."""
        ...

    def run_inference(self, request: InferenceRequest) -> InferenceExecution:
        """Runs inference for one audio request."""
        ...


type RuntimePipelineBuilder = Callable[[AppConfig], _PublicRuntimePipeline]


def run_configure_command(argv: list[str], *, settings: AppConfig | None = None) -> int:
    """Runs `ser configure ...` with an explicit or default settings snapshot."""
    active_settings = settings if settings is not None else get_settings()
    return _run_configure_command(argv, settings=active_settings)


def run_data_command(argv: list[str], *, settings: AppConfig | None = None) -> int:
    """Runs `ser data ...` with an explicit or default settings snapshot."""
    active_settings = settings if settings is not None else get_settings()
    return _run_data_command(argv, settings=active_settings)


def list_registered_datasets(
    *,
    settings: AppConfig | None = None,
) -> tuple[DatasetRegistryRecord, ...]:
    """Returns registered dataset records using the active settings snapshot."""
    active_settings = settings if settings is not None else get_settings()
    return _list_registered_datasets(settings=active_settings)


def list_dataset_registry_health_issues(
    *,
    settings: AppConfig | None = None,
) -> tuple[DatasetRegistryHealthIssueRecord, ...]:
    """Returns dataset registry health issues using the active settings snapshot."""
    active_settings = settings if settings is not None else get_settings()
    return _list_dataset_registry_health_issues(settings=active_settings)


def show_dataset_consents(
    *,
    settings: AppConfig | None = None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Returns persisted dataset consents using the active settings snapshot."""
    active_settings = settings if settings is not None else get_settings()
    return _show_dataset_consents(settings=active_settings)


def configure_dataset_consents(
    *,
    accept_policy_ids: tuple[str, ...] = (),
    accept_license_ids: tuple[str, ...] = (),
    settings: AppConfig | None = None,
    source: str = "ser.api.configure_dataset_consents",
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Persists dataset consents using the active settings snapshot."""
    active_settings = settings if settings is not None else get_settings()
    return _configure_dataset_consents(
        accept_policy_ids=accept_policy_ids,
        accept_license_ids=accept_license_ids,
        settings=active_settings,
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
    active_settings = settings if settings is not None else get_settings()
    return _prepare_dataset(
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
        settings=active_settings,
    )


def load_profile(
    profile: ProfileName,
    *,
    settings: AppConfig | None = None,
) -> None:
    """Validates one runtime profile using the active settings snapshot."""
    active_settings = settings if settings is not None else get_settings()
    return _load_profile(profile, settings=active_settings)


def train(
    *,
    profile: ProfileName | None = None,
    settings: AppConfig | None = None,
    use_profile_pipeline: bool = True,
    pipeline_builder: RuntimePipelineBuilder | None = None,
) -> None:
    """Runs training using the active settings snapshot."""
    active_settings = settings if settings is not None else get_settings()
    return _train(
        profile=profile,
        settings=active_settings,
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
    settings: AppConfig | None = None,
    pipeline_builder: RuntimePipelineBuilder | None = None,
) -> InferenceExecution:
    """Runs inference using the active settings snapshot."""
    active_settings = settings if settings is not None else get_settings()
    return _infer(
        file_path,
        profile=profile,
        language=language,
        save_transcript=save_transcript,
        include_transcript=include_transcript,
        settings=active_settings,
        pipeline_builder=pipeline_builder,
    )


__all__ = [
    "ComplianceMode",
    "DatasetPrepareResult",
    "DatasetRegistryHealthIssueRecord",
    "DatasetRegistryRecord",
    "RestrictedBackendOptInState",
    "RestrictedBackendPrompt",
    "WorkflowErrorDisposition",
    "apply_cli_profile_override",
    "apply_cli_timeout_override",
    "build_runtime_pipeline",
    "classify_inference_exception",
    "classify_training_exception",
    "collect_missing_restricted_backend_consents",
    "configure_dataset_consents",
    "enforce_restricted_backends_for_cli",
    "ensure_restricted_backends_ready_for_command",
    "format_startup_preflight_one_liner",
    "infer",
    "list_dataset_registry_health_issues",
    "list_datasets",
    "list_profiles",
    "list_registered_datasets",
    "load_profile",
    "parse_preflight_mode",
    "preflight_command_requested",
    "preflight_includes_transcription_checks",
    "persist_all_restricted_backend_consents",
    "persist_required_restricted_backends",
    "prepare_restricted_backend_opt_in_state",
    "profile_pipeline_enabled",
    "profile_resolution_requested",
    "prepare_dataset",
    "prepare_msp_podcast_mirror",
    "resolve_cli_workflow_profile",
    "resolve_doctor_command",
    "required_restricted_backends_for_current_profile",
    "run_configure_command",
    "run_data_command",
    "run_doctor_command",
    "run_inference_command",
    "run_inference_workflow",
    "run_restricted_backend_cli_gate",
    "run_startup_preflight_cli_gate",
    "run_training_command",
    "run_transcription_runtime_calibration_command",
    "run_transcription_runtime_calibration_cli",
    "run_startup_preflight",
    "run_training_workflow",
    "run_transcription_runtime_calibration_workflow",
    "suppress_preflight_transcription_operational_relogs",
    "should_fail_preflight",
    "show_dataset_consents",
    "train",
]
