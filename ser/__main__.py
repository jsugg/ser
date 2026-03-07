"""Command-line entry point for the speech emotion recognition tool."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from contextlib import nullcontext
from typing import cast

from dotenv import load_dotenv

from ser.api import (
    apply_cli_profile_override,
    apply_cli_timeout_override,
    build_runtime_pipeline,
    parse_preflight_mode,
    profile_pipeline_enabled,
    resolve_cli_workflow_profile,
    run_configure_command,
    run_data_command,
    run_doctor_command,
    run_inference_command,
    run_restricted_backend_cli_gate,
    run_startup_preflight_cli_gate,
    run_training_command,
    run_transcription_runtime_calibration_command,
)
from ser.config import AppConfig, reload_settings, settings_override
from ser.profiles import ProfileName
from ser.runtime.phase_timing import format_duration
from ser.utils.logger import configure_logging, get_logger

logger: logging.Logger = get_logger("ser")
type CliProfileName = ProfileName
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


def main() -> None:
    """Parses CLI arguments and runs training or inference workflows."""
    load_dotenv()
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--log-level", type=str, default=None)
    pre_args, pre_remaining = pre_parser.parse_known_args()
    configure_logging(pre_args.log_level)
    settings: AppConfig = reload_settings()

    # Subcommand dispatch for dataset and diagnostics workflows.
    if pre_remaining and pre_remaining[0] in {"configure", "data", "doctor"}:
        command = pre_remaining[0]
        command_argv = pre_remaining[1:]
        try:
            if command == "configure":
                raise SystemExit(run_configure_command(command_argv, settings=settings))
            if command == "data":
                raise SystemExit(run_data_command(command_argv, settings=settings))
            raise SystemExit(run_doctor_command(command_argv, settings=settings))
        except SystemExit:
            raise
        except Exception as err:
            logger.error("Command '%s' failed: %s", command, err)
            raise SystemExit(1) from err

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
    settings_scope = (
        settings_override(active_settings)
        if isinstance(active_settings, AppConfig)
        else nullcontext()
    )
    with settings_scope:
        use_profile_pipeline: bool = profile_pipeline_enabled(active_settings)
        restricted_logs, restricted_exit_code = run_restricted_backend_cli_gate(
            settings=active_settings,
            use_profile_pipeline=use_profile_pipeline,
            train_requested=bool(args.train),
            file_path=args.file if isinstance(args.file, str) else None,
            accept_restricted_backends=bool(args.accept_restricted_backends),
            accept_all_restricted_backends=bool(args.accept_all_restricted_backends),
            is_interactive=bool(sys.stdin.isatty() and sys.stdout.isatty()),
        )
        for level, message in restricted_logs:
            if level == "error":
                logger.error("%s", message)
            else:
                logger.info("%s", message)
        if restricted_exit_code is not None:
            sys.exit(restricted_exit_code)

        if isinstance(active_settings, AppConfig):
            preflight_logs, preflight_exit_code = run_startup_preflight_cli_gate(
                settings=active_settings,
                mode=preflight_mode,
                profile=args.profile if isinstance(args.profile, str) else None,
                train_requested=bool(args.train),
                file_path=args.file if isinstance(args.file, str) else None,
                no_transcript=bool(args.no_transcript),
                calibrate_transcription_runtime=bool(
                    args.calibrate_transcription_runtime
                ),
            )
            for level, message in preflight_logs:
                if level == "error":
                    logger.error("%s", message)
                else:
                    logger.info("%s", message)
            if preflight_exit_code is not None:
                sys.exit(preflight_exit_code)

        if args.calibrate_transcription_runtime:
            calibration_result, calibration_error = (
                run_transcription_runtime_calibration_command(
                    file_path=args.file if isinstance(args.file, str) else None,
                    language=args.language,
                    calibration_iterations=args.calibration_iterations,
                    calibration_profiles=args.calibration_profiles,
                )
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

        if args.train:
            logger.info("Starting model training...")
            start_time: float = time.perf_counter()
            disposition = run_training_command(
                settings=active_settings,
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

        workflow_profile = resolve_cli_workflow_profile(active_settings)
        logger.info("SER workflow started (profile=%s).", workflow_profile)
        start_time = time.perf_counter()
        execution, disposition = run_inference_command(
            settings=active_settings,
            file_path=args.file if isinstance(args.file, str) else None,
            language=str(args.language),
            save_transcript=bool(args.save_transcript),
            include_transcript=not bool(args.no_transcript),
            pipeline_builder=build_runtime_pipeline,
        )
        if disposition is not None:
            logger.error(
                "%s",
                disposition.message,
                exc_info=disposition.include_traceback,
            )
            sys.exit(disposition.exit_code)
        if execution is not None and execution.timeline_csv_path is not None:
            logger.info(msg=f"Timeline saved to {execution.timeline_csv_path}")
        logger.info(
            "SER workflow completed in %s.",
            format_duration(time.perf_counter() - start_time),
        )


if __name__ == "__main__":
    main()
