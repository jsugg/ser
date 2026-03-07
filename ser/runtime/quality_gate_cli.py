"""CLI helpers for profile quality-gate execution."""

from __future__ import annotations

import argparse
import logging
import warnings
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class QualityGateCliDefaults:
    """Default CLI values derived from active runtime settings."""

    dataset_glob: str
    test_size: float
    random_state: int
    fast_model_file_name: str
    fast_secure_model_file_name: str
    fast_training_report_file_name: str
    medium_secure_model_file_name: str
    medium_training_report_file_name: str
    min_uar_delta: float
    min_macro_f1_delta: float
    max_medium_segments_per_minute: float | None
    min_medium_median_segment_duration_seconds: float | None


def configure_cli_noise_controls() -> None:
    """Suppresses non-actionable warning/log noise for long gate executions."""
    warnings.filterwarnings(
        "ignore",
        message=r"n_fft=\d+ is too large for input signal of length=.*",
        category=UserWarning,
        module=r"librosa\.core\.spectrum",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"Trying to estimate tuning from empty frequency set\.",
        category=UserWarning,
        module=r"librosa\.core\.pitch",
    )
    logging.getLogger("ser.models.emotion_model").setLevel(logging.WARNING)
    logging.getLogger("ser.features.feature_extractor").setLevel(logging.ERROR)
    logging.getLogger("ser.runtime.medium_inference").setLevel(logging.WARNING)


def build_arg_parser(defaults: QualityGateCliDefaults) -> argparse.ArgumentParser:
    """Builds CLI parser for fast-vs-medium quality gate evaluation."""
    parser = argparse.ArgumentParser(
        description="SER fast-vs-medium profile quality gate harness"
    )
    parser.add_argument(
        "--dataset-glob",
        default=defaults.dataset_glob,
        help="Dataset glob pattern used for shared evaluation set.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional maximum number of files from dataset glob.",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Grouped CV fold count before fallback to grouped holdout.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=defaults.test_size,
        help="Grouped holdout test-size fallback when CV split is infeasible.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=defaults.random_state,
        help="Deterministic seed for split and fallback behavior.",
    )
    parser.add_argument(
        "--fast-model-file-name",
        default=defaults.fast_model_file_name,
        help="Model artifact filename used for fast-profile inference.",
    )
    parser.add_argument(
        "--fast-secure-model-file-name",
        default=defaults.fast_secure_model_file_name,
        help="Secure model artifact filename used for fast-profile inference.",
    )
    parser.add_argument(
        "--fast-training-report-file-name",
        default=defaults.fast_training_report_file_name,
        help="Training report filename used for fast-profile feature-size hints.",
    )
    parser.add_argument(
        "--medium-model-file-name",
        required=True,
        help="Model artifact filename used for medium-profile inference.",
    )
    parser.add_argument(
        "--medium-secure-model-file-name",
        default=defaults.medium_secure_model_file_name,
        help="Secure model artifact filename used for medium-profile inference.",
    )
    parser.add_argument(
        "--medium-training-report-file-name",
        default=defaults.medium_training_report_file_name,
        help="Training report filename used for medium-profile feature-size hints.",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Optional language passed through to medium inference request.",
    )
    parser.add_argument(
        "--min-uar-delta",
        type=float,
        default=defaults.min_uar_delta,
        help="Minimum required (medium - fast) UAR delta.",
    )
    parser.add_argument(
        "--min-macro-f1-delta",
        type=float,
        default=defaults.min_macro_f1_delta,
        help="Minimum required (medium - fast) macro-F1 delta.",
    )
    parser.add_argument(
        "--max-medium-segments-per-minute",
        type=float,
        default=defaults.max_medium_segments_per_minute,
        help="Optional upper bound for medium segment count per minute.",
    )
    parser.add_argument(
        "--min-medium-median-segment-duration",
        type=float,
        default=defaults.min_medium_median_segment_duration_seconds,
        help="Optional lower bound for medium median segment duration in seconds.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional output path for JSON quality-gate report.",
    )
    parser.add_argument(
        "--require-pass",
        action="store_true",
        help="Exit with code 1 when medium profile does not pass thresholds.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=0,
        help="Emit progress every N evaluated clips per profile (0 disables).",
    )
    return parser


def parse_args(
    *,
    defaults: QualityGateCliDefaults,
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    """Parses CLI arguments for quality-gate execution."""
    parser = build_arg_parser(defaults)
    return parser.parse_args(argv)


def normalize_progress_every(progress_every: int) -> int | None:
    """Converts non-positive progress cadence to disabled sentinel."""
    return progress_every if progress_every > 0 else None
