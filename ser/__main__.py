"""Command-line entry point for the speech emotion recognition tool."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Sequence
from contextlib import AbstractContextManager, nullcontext
from typing import cast

from dotenv import load_dotenv

from ser._internal.cli.data import run_configure_command, run_data_command
from ser._internal.cli.diagnostics import (
    parse_preflight_mode,
    run_doctor_command,
    run_startup_preflight_cli_gate,
)
from ser._internal.cli.runtime import (
    apply_cli_profile_override,
    apply_cli_timeout_override,
    build_runtime_pipeline,
    profile_pipeline_enabled,
    resolve_cli_workflow_profile,
    run_inference_command,
    run_restricted_backend_cli_gate,
    run_training_command,
    run_transcription_runtime_calibration_command,
)
from ser.config import AppConfig, reload_settings, settings_override
from ser.diagnostics.domain import PreflightMode
from ser.profiles import ProfileName
from ser.runtime.contracts import SubtitleFormat
from ser.runtime.phase_timing import format_duration
from ser.utils.logger import configure_logging, get_logger
from ser.utils.subtitles import SUPPORTED_SUBTITLE_FORMATS

logger: logging.Logger = get_logger("ser")
type CliProfileName = ProfileName
type CliSubtitleFormat = SubtitleFormat
_PROFILE_CHOICES: tuple[CliProfileName, ...] = (
    "fast",
    "medium",
    "accurate",
    "accurate-research",
)
_DATASET_COMMAND_HELP = (
    "Dataset workflow commands:\n"
    "  ser configure --show\n"
    "  ser configure --accept-dataset-policy <policy ...> "
    "--accept-dataset-license <license ...> --persist\n"
    "  ser data registry [--show] [--format text|json] [--strict]\n"
    "  ser data download --dataset <ravdess|crema-d|msp-podcast|biic-podcast> "
    "[--dataset-root PATH] [--manifest-path PATH]\n"
    "                     [--labels-csv-path PATH] [--audio-base-dir PATH] "
    "[--skip-download]\n"
    "                     [--source HF_DATASET_ID] [--source-revision REV] "
    "[--accept-license]\n"
    "\n"
    "Diagnostics commands:\n"
    "  ser doctor [--profile fast|medium|accurate|accurate-research] "
    "[--format text|json] [--strict]\n"
    "\n"
    "Run `ser configure --help`, `ser data download --help`, and "
    "`ser doctor --help` for details."
)


def _build_pre_parser() -> argparse.ArgumentParser:
    """Builds the minimal parser used for early logging configuration."""
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--log-level", type=str, default=None)
    return pre_parser


def _command_file_path(value: object) -> str | None:
    """Returns the CLI file-path argument when present."""
    return value if isinstance(value, str) else None


def _log_cli_records(records: Sequence[tuple[str, str]]) -> None:
    """Emits structured CLI gate logs through the top-level logger."""
    for level, message in records:
        if level == "error":
            logger.error("%s", message)
            continue
        logger.info("%s", message)


def _dispatch_subcommand(command: str, command_argv: Sequence[str], *, settings: AppConfig) -> None:
    """Runs one CLI subcommand and exits with its return code."""
    try:
        if command == "configure":
            raise SystemExit(run_configure_command(list(command_argv), settings=settings))
        if command == "data":
            raise SystemExit(run_data_command(list(command_argv), settings=settings))
        raise SystemExit(run_doctor_command(list(command_argv), settings=settings))
    except SystemExit:
        raise
    except Exception as err:
        logger.error("Command '%s' failed: %s", command, err)
        raise SystemExit(1) from err


def _maybe_dispatch_subcommand(pre_remaining: Sequence[str], *, settings: AppConfig) -> bool:
    """Runs subcommand flow when one CLI subcommand is requested."""
    if not pre_remaining or pre_remaining[0] not in {"configure", "data", "doctor"}:
        return False
    _dispatch_subcommand(pre_remaining[0], pre_remaining[1:], settings=settings)
    return True


def _build_main_parser(settings: AppConfig) -> argparse.ArgumentParser:
    """Builds the main CLI parser for inference, training, and maintenance flows."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Speech Emotion Recognition Tool",
        epilog=_DATASET_COMMAND_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
        "--subtitle-output",
        type=str,
        default=None,
        help=(
            "Write timeline subtitles to the given path. "
            "If --subtitle-format is omitted, the format is inferred from the file suffix."
        ),
    )
    parser.add_argument(
        "--subtitle-format",
        choices=SUPPORTED_SUBTITLE_FORMATS,
        default=None,
        help=(
            "Subtitle export format. When provided without --subtitle-output, the file is "
            "written under the configured timeline folder using the source audio stem."
        ),
    )
    parser.add_argument(
        "--no-transcript",
        action="store_true",
        help=(
            "Skip transcription and build timeline with emotion timestamps only "
            "(empty speech column)."
        ),
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
    parser.add_argument(
        "--calibrate-transcription-runtime",
        action="store_true",
        help=(
            "Run runtime calibration for transcription model/profile recommendations "
            "with confidence scoring."
        ),
    )
    parser.add_argument(
        "--preflight",
        choices=("off", "warn", "strict"),
        default="warn",
        help=(
            "Startup diagnostics gate: off=skip checks, warn=log findings and fail only "
            "on blocking errors, strict=fail on warnings/errors."
        ),
    )
    parser.add_argument(
        "--calibration-iterations",
        type=int,
        default=2,
        help="Number of calibration runs per profile/model candidate.",
    )
    parser.add_argument(
        "--calibration-profiles",
        type=str,
        default="accurate,medium,accurate-research,fast",
        help=(
            "Comma-separated profile list for calibration "
            "(fast,medium,accurate,accurate-research)."
        ),
    )
    return parser


def _settings_scope(active_settings: object) -> AbstractContextManager[object]:
    """Returns the scoped settings context when one concrete config is active."""
    if isinstance(active_settings, AppConfig):
        return settings_override(active_settings)
    return nullcontext()


def _run_restricted_backend_gate(
    args: argparse.Namespace, *, active_settings: object
) -> int | None:
    """Runs restricted-backend CLI gating and returns an optional exit code."""
    restricted_logs, restricted_exit_code = run_restricted_backend_cli_gate(
        settings=cast(AppConfig, active_settings),
        use_profile_pipeline=profile_pipeline_enabled(cast(AppConfig, active_settings)),
        train_requested=bool(args.train),
        file_path=_command_file_path(args.file),
        accept_restricted_backends=bool(args.accept_restricted_backends),
        accept_all_restricted_backends=bool(args.accept_all_restricted_backends),
        is_interactive=bool(sys.stdin.isatty() and sys.stdout.isatty()),
    )
    _log_cli_records(restricted_logs)
    return restricted_exit_code


def _run_preflight_gate(
    args: argparse.Namespace,
    *,
    active_settings: object,
    preflight_mode: PreflightMode,
) -> int | None:
    """Runs startup preflight gate when a concrete config is available."""
    if not isinstance(active_settings, AppConfig):
        return None
    preflight_logs, preflight_exit_code = run_startup_preflight_cli_gate(
        settings=active_settings,
        mode=preflight_mode,
        profile=args.profile if isinstance(args.profile, str) else None,
        train_requested=bool(args.train),
        file_path=_command_file_path(args.file),
        no_transcript=bool(args.no_transcript),
        calibrate_transcription_runtime=bool(args.calibrate_transcription_runtime),
    )
    _log_cli_records(preflight_logs)
    return preflight_exit_code


def _run_calibration_or_exit(args: argparse.Namespace) -> None:
    """Runs calibration workflow and exits with success or failure disposition."""
    calibration_result, calibration_error = run_transcription_runtime_calibration_command(
        file_path=_command_file_path(args.file),
        language=str(args.language),
        calibration_iterations=int(args.calibration_iterations),
        calibration_profiles=str(args.calibration_profiles),
    )
    if calibration_error is not None:
        logger.error(
            "%s",
            calibration_error.message,
            exc_info=calibration_error.include_traceback,
        )
        sys.exit(calibration_error.exit_code)
    if calibration_result is None:
        logger.error("Calibration command returned no result.")
        sys.exit(1)

    logger.info(
        "Transcription runtime calibration completed. Report: %s",
        calibration_result.report_path,
    )
    for recommendation in calibration_result.recommendations:
        logger.info(
            "Calibration recommendation (%s/%s): %s (confidence=%s, reason=%s).",
            recommendation.profile.source_profile,
            recommendation.profile.model_name,
            recommendation.recommendation,
            recommendation.confidence,
            recommendation.reason,
        )
    sys.exit(0)


def _run_training_or_exit(*, active_settings: object, use_profile_pipeline: bool) -> None:
    """Runs training flow and exits with the appropriate status code."""
    logger.info("Starting model training...")
    start_time = time.perf_counter()
    disposition = run_training_command(
        settings=cast(AppConfig, active_settings),
        use_profile_pipeline=use_profile_pipeline,
        pipeline_builder=build_runtime_pipeline,
    )
    if disposition is not None:
        logger.error(
            "%s",
            disposition.message,
            exc_info=disposition.include_traceback,
        )
        sys.exit(disposition.exit_code)
    logger.info(
        "Training completed in %s.",
        format_duration(time.perf_counter() - start_time),
    )
    sys.exit(0)


def _run_inference_or_exit(args: argparse.Namespace, *, active_settings: object) -> None:
    """Runs inference flow and exits on any workflow disposition."""
    workflow_profile = resolve_cli_workflow_profile(cast(AppConfig, active_settings))
    logger.info("SER workflow started (profile=%s).", workflow_profile)
    start_time = time.perf_counter()
    execution, disposition = run_inference_command(
        settings=cast(AppConfig, active_settings),
        file_path=_command_file_path(args.file),
        language=str(args.language),
        save_transcript=bool(args.save_transcript),
        include_transcript=not bool(args.no_transcript),
        subtitle_output_path=(
            args.subtitle_output if isinstance(args.subtitle_output, str) else None
        ),
        subtitle_format=cast(
            CliSubtitleFormat | None,
            args.subtitle_format if isinstance(args.subtitle_format, str) else None,
        ),
        pipeline_builder=build_runtime_pipeline,
    )
    if disposition is not None:
        logger.error(
            "%s",
            disposition.message,
            exc_info=disposition.include_traceback,
        )
        sys.exit(disposition.exit_code)
    timeline_csv_path = getattr(execution, "timeline_csv_path", None) if execution else None
    subtitle_path = getattr(execution, "subtitle_path", None) if execution else None
    if timeline_csv_path is not None:
        logger.info(msg=f"Timeline saved to {timeline_csv_path}")
    if subtitle_path is not None:
        logger.info("Timeline subtitles saved to %s", subtitle_path)
    logger.info(
        "SER workflow completed in %s.",
        format_duration(time.perf_counter() - start_time),
    )


def main() -> None:
    """Parses CLI arguments and runs training or inference workflows."""
    load_dotenv()
    pre_parser = _build_pre_parser()
    pre_args, pre_remaining = pre_parser.parse_known_args()
    configure_logging(pre_args.log_level)
    settings: AppConfig = reload_settings()
    if _maybe_dispatch_subcommand(pre_remaining, settings=settings):
        return

    parser = _build_main_parser(settings)
    args: argparse.Namespace = parser.parse_args()
    configure_logging(args.log_level)
    active_settings = apply_cli_profile_override(
        settings,
        cast(CliProfileName | None, args.profile),
    )
    active_settings = apply_cli_timeout_override(
        active_settings,
        disable_timeouts=bool(args.disable_timeouts),
    )
    preflight_mode = parse_preflight_mode(str(args.preflight))
    with _settings_scope(active_settings):
        restricted_exit_code = _run_restricted_backend_gate(
            args,
            active_settings=active_settings,
        )
        if restricted_exit_code is not None:
            sys.exit(restricted_exit_code)

        preflight_exit_code = _run_preflight_gate(
            args,
            active_settings=active_settings,
            preflight_mode=preflight_mode,
        )
        if preflight_exit_code is not None:
            sys.exit(preflight_exit_code)

        if args.calibrate_transcription_runtime:
            _run_calibration_or_exit(args)

        use_profile_pipeline = profile_pipeline_enabled(active_settings)
        if args.train:
            _run_training_or_exit(
                active_settings=active_settings,
                use_profile_pipeline=use_profile_pipeline,
            )

        _run_inference_or_exit(args, active_settings=active_settings)


if __name__ == "__main__":
    main()
