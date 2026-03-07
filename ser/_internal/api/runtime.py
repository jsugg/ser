"""Runtime-focused public API helpers for library and CLI orchestration."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from ser._internal.runtime.commands import (
    WorkflowErrorDisposition,
)
from ser._internal.runtime.commands import (
    classify_inference_exception as classify_inference_exception,
)
from ser._internal.runtime.commands import (
    classify_training_exception as classify_training_exception,
)
from ser._internal.runtime.commands import (
    run_inference_command as _run_inference_command,
)
from ser._internal.runtime.commands import run_training_command as _run_training_command
from ser._internal.runtime.commands import (
    run_transcription_runtime_calibration_cli as _run_transcription_runtime_calibration_cli,
)
from ser._internal.runtime.commands import (
    run_transcription_runtime_calibration_command as _run_transcription_runtime_calibration_command,
)
from ser._internal.runtime.commands import (
    run_transcription_runtime_calibration_workflow as _run_transcription_runtime_calibration_workflow,
)
from ser._internal.runtime.restricted_backends import (
    RuntimeCliLogRecord,
)
from ser._internal.runtime.restricted_backends import (
    enforce_restricted_backends_for_cli as _enforce_restricted_backends_for_cli,
)
from ser._internal.runtime.restricted_backends import (
    prepare_restricted_backend_opt_in_state as _prepare_restricted_backend_opt_in_state,
)
from ser._internal.runtime.restricted_backends import (
    run_restricted_backend_cli_gate as _run_restricted_backend_cli_gate,
)
from ser.config import (
    AppConfig,
    profile_artifact_file_names,
    resolve_profile_transcription_config,
    settings_override,
)
from ser.profiles import ProfileName, get_profile_catalog, resolve_profile_name
from ser.utils.logger import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    from ser.runtime.contracts import InferenceExecution, InferenceRequest
    from ser.transcript.profiling import RuntimeCalibrationResult


class _RuntimePipeline(Protocol):
    """Minimal runtime pipeline contract for train/inference orchestration."""

    def run_training(self) -> None:
        """Runs training for the active runtime profile."""
        ...

    def run_inference(self, request: InferenceRequest) -> InferenceExecution:
        """Runs inference for one audio file request."""
        ...


type _RuntimePipelineBuilder = Callable[[AppConfig], _RuntimePipeline]


def _resolved_artifact_name(
    *,
    env_var: str,
    profile_default: str,
    current: str,
) -> str:
    """Returns profile default unless explicitly overridden by environment."""
    return current if os.getenv(env_var) is not None else profile_default


def apply_cli_profile_override(
    settings: AppConfig,
    cli_profile: ProfileName | None,
) -> AppConfig:
    """Returns settings with runtime flags overridden by explicit CLI profile."""
    if cli_profile is None:
        return settings

    profile_overrides = {
        "profile_pipeline": True,
        "medium_profile": cli_profile == "medium",
        "accurate_profile": cli_profile == "accurate",
        "accurate_research_profile": cli_profile == "accurate-research",
        "restricted_backends": bool(settings.runtime_flags.restricted_backends),
    }
    resolved_settings = replace(
        settings,
        runtime_flags=replace(settings.runtime_flags, **profile_overrides),
    )
    (
        profile_model_file_name,
        profile_secure_model_file_name,
        profile_training_report_file_name,
    ) = profile_artifact_file_names(
        profile=cli_profile,
        medium_model_id=resolved_settings.models.medium_model_id,
        accurate_model_id=resolved_settings.models.accurate_model_id,
        accurate_research_model_id=(
            resolved_settings.models.accurate_research_model_id
        ),
    )
    (
        profile_transcription_backend_id,
        profile_whisper_model_name,
        profile_use_demucs,
        profile_use_vad,
    ) = resolve_profile_transcription_config(cli_profile)
    has_explicit_artifact_override = any(
        os.getenv(env_var) is not None
        for env_var in (
            "SER_MODEL_FILE_NAME",
            "SER_SECURE_MODEL_FILE_NAME",
            "SER_TRAINING_REPORT_FILE_NAME",
        )
    )
    if cli_profile != "fast" and has_explicit_artifact_override:
        logger.warning(
            "Explicit artifact filename overrides are active for profile '%s'. "
            "Tuple-scoped default naming by backend_model_id is bypassed.",
            cli_profile,
        )
    return replace(
        resolved_settings,
        models=replace(
            resolved_settings.models,
            whisper_model=replace(
                resolved_settings.models.whisper_model,
                name=profile_whisper_model_name,
            ),
            model_file_name=_resolved_artifact_name(
                env_var="SER_MODEL_FILE_NAME",
                profile_default=profile_model_file_name,
                current=resolved_settings.models.model_file_name,
            ),
            secure_model_file_name=_resolved_artifact_name(
                env_var="SER_SECURE_MODEL_FILE_NAME",
                profile_default=profile_secure_model_file_name,
                current=resolved_settings.models.secure_model_file_name,
            ),
            training_report_file_name=_resolved_artifact_name(
                env_var="SER_TRAINING_REPORT_FILE_NAME",
                profile_default=profile_training_report_file_name,
                current=resolved_settings.models.training_report_file_name,
            ),
        ),
        transcription=replace(
            resolved_settings.transcription,
            backend_id=profile_transcription_backend_id,
            use_demucs=profile_use_demucs,
            use_vad=profile_use_vad,
        ),
    )


def apply_cli_timeout_override(
    settings: AppConfig,
    *,
    disable_timeouts: bool,
) -> AppConfig:
    """Returns settings with profile timeout budgets disabled when requested."""
    if not disable_timeouts:
        return settings

    return replace(
        settings,
        fast_runtime=replace(settings.fast_runtime, timeout_seconds=0.0),
        medium_runtime=replace(settings.medium_runtime, timeout_seconds=0.0),
        accurate_runtime=replace(settings.accurate_runtime, timeout_seconds=0.0),
        accurate_research_runtime=replace(
            settings.accurate_research_runtime,
            timeout_seconds=0.0,
        ),
    )


def profile_pipeline_enabled(settings: AppConfig) -> bool:
    """Returns whether runtime pipeline routing is enabled in settings."""
    return bool(settings.runtime_flags.profile_pipeline)


def profile_resolution_requested(
    *,
    use_profile_pipeline: bool,
    file_path: str | None,
) -> bool:
    """Returns whether profile resolution should run for this invocation."""
    return bool(use_profile_pipeline or file_path)


def resolve_cli_workflow_profile(settings: AppConfig) -> ProfileName:
    """Resolves workflow profile label for CLI logs and telemetry."""
    return resolve_profile_name(settings)


def _build_runtime_pipeline(settings: AppConfig) -> _RuntimePipeline:
    """Builds one runtime pipeline for API-managed train/inference workflows."""
    from ser.runtime.pipeline import create_runtime_pipeline

    return create_runtime_pipeline(settings)


def build_runtime_pipeline(settings: AppConfig) -> _RuntimePipeline:
    """Builds one runtime pipeline instance for CLI or library execution."""
    return _build_runtime_pipeline(settings)


def _settings_for_profile(
    settings: AppConfig,
    *,
    profile: ProfileName | None,
) -> AppConfig:
    """Returns settings with runtime flags scoped to the requested profile."""
    if profile is None:
        return settings
    runtime_flags = replace(
        settings.runtime_flags,
        profile_pipeline=True,
        medium_profile=profile == "medium",
        accurate_profile=profile == "accurate",
        accurate_research_profile=profile == "accurate-research",
        restricted_backends=bool(settings.runtime_flags.restricted_backends),
    )
    return replace(settings, runtime_flags=runtime_flags)


def list_profiles() -> tuple[ProfileName, ...]:
    """Returns all registered profile names in deterministic order."""
    names = tuple(get_profile_catalog().keys())
    return tuple(name for name in names if isinstance(name, str))


def load_profile(
    profile: ProfileName,
    *,
    settings: AppConfig,
) -> None:
    """Validates profile availability and dependency capability for execution."""
    from ser.runtime.backend_hooks import build_backend_hooks
    from ser.runtime.registry import (
        ensure_profile_supported,
        resolve_runtime_capability,
    )

    scoped_settings = _settings_for_profile(settings, profile=profile)
    backend_hooks = build_backend_hooks(scoped_settings)
    implemented_backends: frozenset[str] = frozenset({"handcrafted", *backend_hooks})
    capability = resolve_runtime_capability(
        scoped_settings,
        available_backend_hooks=implemented_backends,
    )
    ensure_profile_supported(capability)


def run_training_workflow(
    *,
    settings: AppConfig,
    use_profile_pipeline: bool,
    pipeline_builder: _RuntimePipelineBuilder | None = None,
) -> None:
    """Runs CLI-equivalent training workflow through one API boundary."""
    if use_profile_pipeline:
        builder = (
            pipeline_builder
            if pipeline_builder is not None
            else _build_runtime_pipeline
        )
        builder(settings).run_training()
        return
    from ser.models.emotion_model import train_model

    with settings_override(settings):
        train_model()


def train(
    *,
    profile: ProfileName | None = None,
    settings: AppConfig,
    use_profile_pipeline: bool = True,
    pipeline_builder: _RuntimePipelineBuilder | None = None,
) -> None:
    """Runs training for the selected profile with optional explicit settings."""
    scoped_settings = _settings_for_profile(settings, profile=profile)
    run_training_workflow(
        settings=scoped_settings,
        use_profile_pipeline=use_profile_pipeline,
        pipeline_builder=pipeline_builder,
    )


def run_inference_workflow(
    *,
    settings: AppConfig,
    file_path: str | Path,
    language: str,
    save_transcript: bool,
    include_transcript: bool,
    pipeline_builder: _RuntimePipelineBuilder | None = None,
) -> InferenceExecution:
    """Runs CLI-equivalent inference workflow through one API boundary."""
    from ser.runtime import InferenceRequest

    builder = (
        pipeline_builder if pipeline_builder is not None else _build_runtime_pipeline
    )
    request = InferenceRequest(
        file_path=str(file_path),
        language=language,
        save_transcript=save_transcript,
        include_transcript=include_transcript,
    )
    return builder(settings).run_inference(request)


def infer(
    file_path: str | Path,
    *,
    profile: ProfileName | None = None,
    language: str | None = None,
    save_transcript: bool = False,
    include_transcript: bool = True,
    settings: AppConfig,
    pipeline_builder: _RuntimePipelineBuilder | None = None,
) -> InferenceExecution:
    """Runs inference for one file using selected profile and runtime settings."""
    scoped_settings = _settings_for_profile(settings, profile=profile)
    resolved_language = (
        language
        if isinstance(language, str) and language.strip()
        else scoped_settings.default_language
    )
    return run_inference_workflow(
        settings=scoped_settings,
        file_path=file_path,
        language=resolved_language,
        save_transcript=save_transcript,
        include_transcript=include_transcript,
        pipeline_builder=pipeline_builder,
    )


def run_restricted_backend_cli_gate(
    *,
    settings: AppConfig,
    use_profile_pipeline: bool,
    train_requested: bool,
    file_path: str | None,
    accept_restricted_backends: bool,
    accept_all_restricted_backends: bool,
    is_interactive: bool,
) -> tuple[tuple[RuntimeCliLogRecord, ...], int | None]:
    """Evaluates restricted-backend CLI gate and returns logs plus optional exit code."""
    return _run_restricted_backend_cli_gate(
        settings=settings,
        use_profile_pipeline=use_profile_pipeline,
        train_requested=train_requested,
        file_path=file_path,
        accept_restricted_backends=accept_restricted_backends,
        accept_all_restricted_backends=accept_all_restricted_backends,
        is_interactive=is_interactive,
        prepare_opt_in_state=_prepare_restricted_backend_opt_in_state,
        enforce_for_cli=_enforce_restricted_backends_for_cli,
    )


def run_training_command(
    *,
    settings: AppConfig,
    use_profile_pipeline: bool,
    pipeline_builder: _RuntimePipelineBuilder | None = None,
) -> WorkflowErrorDisposition | None:
    """Runs training command and returns one exit disposition on failure."""
    return _run_training_command(
        settings=settings,
        use_profile_pipeline=use_profile_pipeline,
        pipeline_builder=pipeline_builder,
        run_training_workflow=run_training_workflow,
        classify_training_error=classify_training_exception,
    )


def run_inference_command(
    *,
    settings: AppConfig,
    file_path: str | None,
    language: str,
    save_transcript: bool,
    include_transcript: bool,
    pipeline_builder: _RuntimePipelineBuilder | None = None,
) -> tuple[InferenceExecution | None, WorkflowErrorDisposition | None]:
    """Runs inference command and returns execution plus optional failure disposition."""
    return _run_inference_command(
        settings=settings,
        file_path=file_path,
        language=language,
        save_transcript=save_transcript,
        include_transcript=include_transcript,
        pipeline_builder=pipeline_builder,
        run_inference_workflow=run_inference_workflow,
        classify_inference_error=classify_inference_exception,
    )


def run_transcription_runtime_calibration_cli(
    *,
    file_path: str | None,
    language: str,
    calibration_iterations: int,
    calibration_profiles: str,
) -> RuntimeCalibrationResult:
    """Runs CLI calibration with argument validation and workflow delegation."""
    return _run_transcription_runtime_calibration_cli(
        file_path=file_path,
        language=language,
        calibration_iterations=calibration_iterations,
        calibration_profiles=calibration_profiles,
        run_workflow=_run_transcription_runtime_calibration_workflow,
    )


def run_transcription_runtime_calibration_command(
    *,
    file_path: str | None,
    language: str,
    calibration_iterations: int,
    calibration_profiles: str,
) -> tuple[RuntimeCalibrationResult | None, WorkflowErrorDisposition | None]:
    """Runs CLI calibration and maps failures to workflow exit dispositions."""
    return _run_transcription_runtime_calibration_command(
        file_path=file_path,
        language=language,
        calibration_iterations=calibration_iterations,
        calibration_profiles=calibration_profiles,
        run_calibration_cli=run_transcription_runtime_calibration_cli,
    )


__all__ = [
    "apply_cli_profile_override",
    "apply_cli_timeout_override",
    "build_runtime_pipeline",
    "infer",
    "list_profiles",
    "load_profile",
    "profile_pipeline_enabled",
    "profile_resolution_requested",
    "resolve_cli_workflow_profile",
    "run_inference_command",
    "run_inference_workflow",
    "run_restricted_backend_cli_gate",
    "run_training_command",
    "run_transcription_runtime_calibration_command",
    "run_transcription_runtime_calibration_cli",
    "run_training_workflow",
    "train",
]
