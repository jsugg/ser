"""Command-line entry point for the speech emotion recognition tool."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import replace
from typing import TYPE_CHECKING, cast

from dotenv import load_dotenv

from ser.config import (
    AppConfig,
    RuntimeFlags,
    apply_settings,
    profile_artifact_file_names,
    reload_settings,
    resolve_profile_transcription_config,
)
from ser.license_check import (
    BackendLicensePolicy,
    BackendLicensePolicyError,
    LicenseDecision,
    evaluate_backend_access,
    get_backend_policy,
    load_persisted_backend_consents,
    parse_allowed_restricted_backends_env,
    persist_all_restricted_backend_consents,
    persist_backend_consent,
)
from ser.profiles import ProfileName, get_profile_catalog, resolve_profile_name
from ser.runtime.contracts import InferenceExecution
from ser.runtime.phase_timing import format_duration
from ser.utils.logger import configure_logging, get_logger

if TYPE_CHECKING:
    from ser.domain import EmotionSegment, TimelineEntry, TranscriptWord
    from ser.runtime.pipeline import RuntimePipeline

logger: logging.Logger = get_logger("ser")
type CliProfileName = ProfileName
_PROFILE_CHOICES: tuple[CliProfileName, ...] = (
    "fast",
    "medium",
    "accurate",
    "accurate-research",
)


def _profile_pipeline_enabled(settings: object) -> bool:
    """Returns whether runtime pipeline routing is enabled in settings."""
    runtime_flags: object | None = getattr(settings, "runtime_flags", None)
    return bool(getattr(runtime_flags, "profile_pipeline", False))


def _resolved_artifact_name(
    *,
    env_var: str,
    profile_default: str,
    current: str,
) -> str:
    """Returns profile default unless explicitly overridden by environment."""
    return current if os.getenv(env_var) is not None else profile_default


def _build_runtime_pipeline(settings: AppConfig) -> RuntimePipeline:
    """Builds the runtime pipeline used by flag-gated orchestration."""
    from ser.runtime.pipeline import create_runtime_pipeline

    return create_runtime_pipeline(settings)


def _apply_cli_profile_override(
    settings: object,
    cli_profile: CliProfileName | None,
) -> object:
    """Returns settings with runtime flags overridden by explicit CLI profile."""
    if cli_profile is None:
        return settings

    runtime_flags: object | None = getattr(settings, "runtime_flags", None)
    if runtime_flags is None:
        raise RuntimeError("CLI profile override requires runtime_flags in settings.")

    profile_overrides = {
        "profile_pipeline": True,
        "medium_profile": cli_profile == "medium",
        "accurate_profile": cli_profile == "accurate",
        "accurate_research_profile": cli_profile == "accurate-research",
        "restricted_backends": bool(
            getattr(runtime_flags, "restricted_backends", False)
        ),
    }

    resolved_runtime_flags: object
    if isinstance(runtime_flags, RuntimeFlags):
        resolved_runtime_flags = replace(runtime_flags, **profile_overrides)
    else:
        for key, value in profile_overrides.items():
            runtime_flags.__dict__[key] = value
        resolved_runtime_flags = runtime_flags

    if isinstance(settings, AppConfig):
        resolved_settings = replace(
            settings,
            runtime_flags=cast(RuntimeFlags, resolved_runtime_flags),
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
        resolved_settings = replace(
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
                use_demucs=profile_use_demucs,
                use_vad=profile_use_vad,
            ),
        )
        return resolved_settings
    settings.__dict__["runtime_flags"] = resolved_runtime_flags
    return settings


def _apply_cli_timeout_override(
    settings: object,
    *,
    disable_timeouts: bool,
) -> object:
    """Returns settings with profile timeout budgets disabled when requested."""
    if not disable_timeouts:
        return settings

    if isinstance(settings, AppConfig):
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

    for runtime_field in (
        "fast_runtime",
        "medium_runtime",
        "accurate_runtime",
        "accurate_research_runtime",
    ):
        runtime_config = getattr(settings, runtime_field, None)
        if runtime_config is None:
            continue
        if hasattr(runtime_config, "__dict__"):
            runtime_config.__dict__["timeout_seconds"] = 0.0
            continue
        try:
            runtime_config.timeout_seconds = 0.0
        except Exception:
            continue
    return settings


def _required_restricted_backends_for_current_profile(
    settings: AppConfig,
    *,
    use_profile_pipeline: bool,
) -> tuple[str, ...]:
    """Returns restricted backend ids required by the active runtime profile."""
    if not use_profile_pipeline:
        return ()
    profile_name = resolve_profile_name(settings)
    backend_id = get_profile_catalog()[profile_name].backend_id
    policy = get_backend_policy(backend_id)
    if policy is None or not policy.restricted:
        return ()
    return (backend_id,)


def _prompt_restricted_backend_opt_in(policy: BackendLicensePolicy) -> bool:
    """Prompts for one restricted-backend acknowledgement in interactive shells."""
    print(
        "Restricted backend acknowledgement required:",
        file=sys.stderr,
    )
    print(f"  backend: {policy.backend_id}", file=sys.stderr)
    print(f"  license: {policy.license_id}", file=sys.stderr)
    print(f"  source: {policy.source_url}", file=sys.stderr)
    answer = input("Persist opt-in for this backend now? [y/N]: ").strip().lower()
    return answer in {"y", "yes"}


def _persist_required_restricted_backends(
    settings: AppConfig,
    *,
    use_profile_pipeline: bool,
    consent_source: str,
) -> tuple[str, ...]:
    """Persists consent for restricted backends required by the active profile."""
    required_backends = _required_restricted_backends_for_current_profile(
        settings,
        use_profile_pipeline=use_profile_pipeline,
    )
    persisted: list[str] = []
    for backend_id in required_backends:
        persist_backend_consent(
            settings=settings,
            backend_id=backend_id,
            consent_source=consent_source,
        )
        persisted.append(backend_id)
    return tuple(persisted)


def _collect_missing_restricted_backend_consents(
    settings: AppConfig,
    *,
    required_backend_ids: tuple[str, ...],
) -> tuple[LicenseDecision, ...]:
    """Returns restricted backend decisions that still require explicit consent."""
    if not required_backend_ids:
        return ()
    persisted_consents = load_persisted_backend_consents(settings=settings)
    allowed_restricted_backends = parse_allowed_restricted_backends_env()
    missing: list[LicenseDecision] = []
    for backend_id in required_backend_ids:
        decision = evaluate_backend_access(
            backend_id=backend_id,
            restricted_backends_enabled=settings.runtime_flags.restricted_backends,
            allowed_restricted_backends=allowed_restricted_backends,
            persisted_consents=persisted_consents,
        )
        if decision.allowed:
            continue
        missing.append(decision)
    return tuple(missing)


def _ensure_restricted_backends_ready_for_command(
    settings: AppConfig,
    *,
    required_backend_ids: tuple[str, ...],
) -> None:
    """Ensures required restricted backends have explicit opt-in before execution."""
    while True:
        missing_decisions = _collect_missing_restricted_backend_consents(
            settings,
            required_backend_ids=required_backend_ids,
        )
        if not missing_decisions:
            return
        decision = missing_decisions[0]
        policy = get_backend_policy(decision.policy.backend_id)
        if policy is None:
            raise BackendLicensePolicyError(decision.reason)
        if not (sys.stdin.isatty() and sys.stdout.isatty()):
            raise BackendLicensePolicyError(
                f"{decision.reason} Non-interactive shell cannot prompt for consent. "
                "Use `--accept-restricted-backends`, "
                "`--accept-all-restricted-backends`, "
                "`SER_ALLOWED_RESTRICTED_BACKENDS`, or "
                "`SER_ENABLE_RESTRICTED_BACKENDS=true`."
            )
        if not _prompt_restricted_backend_opt_in(policy):
            raise BackendLicensePolicyError(
                f"Restricted backend {policy.backend_id!r} requires explicit "
                "acknowledgement before execution."
            )
        persist_backend_consent(
            settings=settings,
            backend_id=policy.backend_id,
            consent_source="interactive_prompt",
        )


def main() -> None:
    """Parses CLI arguments and runs training or inference workflows."""
    load_dotenv()
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--log-level", type=str, default=None)
    pre_args, _ = pre_parser.parse_known_args()
    configure_logging(pre_args.log_level)
    settings: AppConfig = reload_settings()

    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Speech Emotion Recognition Tool",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        help=(
            "Set log verbosity (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL). "
            "Overrides LOG_LEVEL env var for this invocation."
        ),
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train the emotion classification model",
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Path to the audio file for emotion prediction",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=settings.default_language,
        help="Language of the audio file",
    )
    parser.add_argument(
        "--save_transcript",
        action="store_true",
        help="Save the transcript to a CSV file",
    )
    parser.add_argument(
        "--profile",
        choices=_PROFILE_CHOICES,
        default=None,
        help="Runtime profile to execute (fast, medium, accurate, accurate-research).",
    )
    parser.add_argument(
        "--disable-timeouts",
        "--no-timeout",
        action="store_true",
        dest="disable_timeouts",
        help="Disable inference timeout budgets for this CLI invocation.",
    )
    parser.add_argument(
        "--accept-restricted-backends",
        action="store_true",
        help=(
            "Persist consent for restricted backends required by the active profile "
            "before command execution."
        ),
    )
    parser.add_argument(
        "--accept-all-restricted-backends",
        action="store_true",
        help=(
            "Persist consent for all currently known restricted backends "
            "(can be used as a standalone management command)."
        ),
    )
    args: argparse.Namespace = parser.parse_args()
    configure_logging(args.log_level)
    active_settings = _apply_cli_profile_override(
        settings,
        cast(CliProfileName | None, args.profile),
    )
    active_settings = _apply_cli_timeout_override(
        active_settings,
        disable_timeouts=bool(args.disable_timeouts),
    )
    if isinstance(active_settings, AppConfig):
        apply_settings(active_settings)
    use_profile_pipeline: bool = _profile_pipeline_enabled(active_settings)
    required_restricted_backends: tuple[str, ...] = ()
    if isinstance(active_settings, AppConfig):
        required_restricted_backends = (
            _required_restricted_backends_for_current_profile(
                active_settings,
                use_profile_pipeline=use_profile_pipeline,
            )
        )
        if args.accept_all_restricted_backends:
            records = persist_all_restricted_backend_consents(
                settings=active_settings,
                consent_source="cli_flag_accept_all",
            )
            logger.info(
                "Persisted restricted-backend consent for %s backend(s).",
                len(records),
            )
        if args.accept_restricted_backends:
            persisted = _persist_required_restricted_backends(
                active_settings,
                use_profile_pipeline=use_profile_pipeline,
                consent_source="cli_flag_accept_restricted",
            )
            if persisted:
                logger.info(
                    "Persisted restricted-backend consent for active profile backend(s): %s",
                    ", ".join(persisted),
                )
        if (
            args.accept_restricted_backends or args.accept_all_restricted_backends
        ) and (not args.train and not args.file):
            sys.exit(0)
        if (args.train or args.file) and required_restricted_backends:
            try:
                _ensure_restricted_backends_ready_for_command(
                    active_settings,
                    required_backend_ids=required_restricted_backends,
                )
            except BackendLicensePolicyError as err:
                logger.error("%s", err)
                sys.exit(2)
    elif args.accept_restricted_backends or args.accept_all_restricted_backends:
        logger.error(
            "Restricted backend opt-in flags require concrete AppConfig settings."
        )
        sys.exit(2)

    if args.train:
        logger.info("Starting model training...")
        start_time: float = time.perf_counter()
        try:
            if use_profile_pipeline:
                _build_runtime_pipeline(cast(AppConfig, active_settings)).run_training()
            else:
                from ser.models.emotion_model import train_model

                train_model()
        except RuntimeError as err:
            logger.error("%s", err)
            sys.exit(2)
        except Exception as err:
            logger.error("Training workflow failed: %s", err, exc_info=True)
            sys.exit(1)
        logger.info(
            "Training completed in %s.",
            format_duration(time.perf_counter() - start_time),
        )
        sys.exit(0)

    if not args.file:
        logger.error(msg="No audio file provided for prediction.")
        sys.exit(1)

    from ser.runtime import UnsupportedProfileError
    from ser.runtime.accurate_inference import (
        AccurateInferenceExecutionError,
        AccurateInferenceTimeoutError,
        AccurateModelLoadError,
        AccurateModelUnavailableError,
        AccurateRuntimeDependencyError,
    )
    from ser.runtime.fast_inference import (
        FastInferenceExecutionError,
        FastInferenceTimeoutError,
        FastModelLoadError,
        FastModelUnavailableError,
    )
    from ser.runtime.medium_inference import (
        MediumInferenceExecutionError,
        MediumInferenceTimeoutError,
        MediumModelLoadError,
        MediumModelUnavailableError,
        MediumRuntimeDependencyError,
    )
    from ser.transcript import TranscriptionError

    workflow_profile = (
        resolve_profile_name(active_settings)
        if isinstance(active_settings, AppConfig)
        else "legacy"
    )
    logger.info("SER workflow started (profile=%s).", workflow_profile)
    start_time = time.perf_counter()
    try:
        if use_profile_pipeline:
            from ser.runtime import InferenceRequest

            execution: InferenceExecution = _build_runtime_pipeline(
                cast(AppConfig, active_settings)
            ).run_inference(
                InferenceRequest(
                    file_path=args.file,
                    language=args.language,
                    save_transcript=args.save_transcript,
                )
            )
            if execution.timeline_csv_path is not None:
                logger.info(msg=f"Timeline saved to {execution.timeline_csv_path}")
        else:
            _run_legacy_inference_workflow(args)
    except UnsupportedProfileError as err:
        logger.error("%s", err)
        sys.exit(2)
    except (
        BackendLicensePolicyError,
        AccurateRuntimeDependencyError,
        AccurateModelLoadError,
        AccurateModelUnavailableError,
        AccurateInferenceTimeoutError,
        FastModelLoadError,
        FastModelUnavailableError,
        FastInferenceTimeoutError,
        MediumRuntimeDependencyError,
        MediumModelLoadError,
        MediumModelUnavailableError,
        MediumInferenceTimeoutError,
    ) as err:
        logger.error("%s", err)
        sys.exit(2)
    except AccurateInferenceExecutionError as err:
        logger.error("Accurate inference failed: %s", err)
        sys.exit(1)
    except FastInferenceExecutionError as err:
        logger.error("Fast inference failed: %s", err)
        sys.exit(1)
    except MediumInferenceExecutionError as err:
        logger.error("Medium inference failed: %s", err)
        sys.exit(1)
    except TranscriptionError as err:
        logger.error("Transcription failed: %s", err, exc_info=True)
        sys.exit(3)
    except FileNotFoundError as err:
        logger.error("%s", err)
        sys.exit(2)
    except Exception as err:
        logger.error("Prediction workflow failed: %s", err, exc_info=True)
        sys.exit(1)

    logger.info(
        "SER workflow completed in %s.",
        format_duration(time.perf_counter() - start_time),
    )


def _run_legacy_inference_workflow(args: argparse.Namespace) -> None:
    from ser.models.emotion_model import predict_emotions
    from ser.transcript import extract_transcript
    from ser.utils.timeline_utils import (
        build_timeline,
        print_timeline,
        save_timeline_to_csv,
    )

    emotions: list[EmotionSegment] = predict_emotions(args.file)
    transcript: list[TranscriptWord] = extract_transcript(args.file, args.language)
    timeline: list[TimelineEntry] = build_timeline(transcript, emotions)
    print_timeline(timeline)

    if args.save_transcript:
        csv_file_name: str = save_timeline_to_csv(timeline, args.file)
        logger.info(msg=f"Timeline saved to {csv_file_name}")


if __name__ == "__main__":
    main()
