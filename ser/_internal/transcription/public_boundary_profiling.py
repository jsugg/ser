"""Public-boundary orchestration helpers for transcription profiling wrappers."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Protocol, TypeVar

from ser._internal.config.bootstrap import resolve_profile_transcription_config
from ser.config import AppConfig, ArtifactProfileName
from ser.profiles import TranscriptionBackendId

from . import default_benchmark as default_benchmark_helpers
from . import profile_candidates as profile_candidate_helpers
from . import profiling_reporting as profiling_reporting_helpers
from . import runtime_calibration as runtime_calibration_helpers
from . import runtime_calibration_workflow as runtime_calibration_workflow_helpers

CandidateT = TypeVar("CandidateT")
SummaryT = TypeVar("SummaryT", covariant=True)
GateT = TypeVar("GateT", covariant=True)
RecommendationT = TypeVar("RecommendationT", covariant=True)
ProfilingResultT = TypeVar("ProfilingResultT")
MetricsT = TypeVar("MetricsT")
RuntimeCalibrationResultT = TypeVar("RuntimeCalibrationResultT")
RecommendationValueT = TypeVar("RecommendationValueT", bound=str)
ConfidenceT = TypeVar("ConfidenceT", bound=str)
CliResultT = TypeVar("CliResultT")


class _DefaultBenchmarkExecutionLike(Protocol[SummaryT, GateT, RecommendationT]):
    """Execution result contract returned by the default benchmark owner."""

    @property
    def reference_file_count(self) -> int:
        """Returns the number of reference files used in the benchmark."""
        ...

    @property
    def gate(self) -> GateT:
        """Returns the derived accuracy gate."""
        ...

    @property
    def summaries(self) -> tuple[SummaryT, ...]:
        """Returns public benchmark summaries."""
        ...

    @property
    def recommendation(self) -> RecommendationT:
        """Returns the default-profile recommendation."""
        ...

    @property
    def report_path(self) -> Path:
        """Returns the persisted report path."""
        ...


class _RuntimeCalibrationExecutionLike(Protocol[RecommendationT]):
    """Execution result contract returned by the calibration workflow owner."""

    @property
    def recommendations(self) -> tuple[RecommendationT, ...]:
        """Returns all runtime recommendations."""
        ...

    @property
    def report_path(self) -> Path:
        """Returns the persisted report path."""
        ...


type _CandidateFactory[CandidateT] = Callable[
    [str, ArtifactProfileName, TranscriptionBackendId, str, bool, bool],
    CandidateT,
]
type _DefaultProfileCandidatesFactory[CandidateT] = Callable[[], tuple[CandidateT, ...]]
type _RuntimeCalibrationCandidatesFactory[CandidateT] = Callable[
    [tuple[ArtifactProfileName, ...]], tuple[CandidateT, ...]
]
type _ProfileCandidateRunner[CandidateT, SummaryT] = Callable[
    [CandidateT, list[Path], str], SummaryT
]
type _DeriveAccuracyGate[SummaryT, GateT] = Callable[[SummaryT, float, float], GateT]
type _RecommendDefaultProfile[SummaryT, GateT, RecommendationT] = Callable[
    [tuple[SummaryT, ...], GateT, int], RecommendationT
]
type _ProfilingResultFactory[SummaryT, GateT, RecommendationT, ProfilingResultT] = (
    Callable[[int, GateT, tuple[SummaryT, ...], RecommendationT, Path], ProfilingResultT]
)
type _CalibrateCandidate[CandidateT, MetricsT] = Callable[[CandidateT, Path, str, int], MetricsT]
type _DeriveRuntimeRecommendation[MetricsT, RecommendationValueT, ConfidenceT] = (
    Callable[[MetricsT], tuple[RecommendationValueT, ConfidenceT, str]]
)
type _RuntimeRecommendationFactory[
    CandidateT, MetricsT, RecommendationValueT, ConfidenceT, RecommendationT
] = Callable[
    [CandidateT, RecommendationValueT, ConfidenceT, str, MetricsT],
    RecommendationT,
]
type _RuntimeCalibrationResultFactory[RecommendationT, RuntimeCalibrationResultT] = (
    Callable[[tuple[RecommendationT, ...], Path], RuntimeCalibrationResultT]
)
type _SettingsOverride = Callable[[AppConfig], AbstractContextManager[object]]
type _SummaryLineBuilder[CliResultT] = Callable[[CliResultT], tuple[str, ...]]
type _PrintFn = Callable[[str], None]


def build_profile_candidates_from_public_boundary(
    *,
    profiles: tuple[ArtifactProfileName, ...],
    candidate_factory: _CandidateFactory[CandidateT],
) -> tuple[CandidateT, ...]:
    """Builds public profile candidates from the resolved profile catalog."""
    return profile_candidate_helpers.build_profile_candidates(
        profiles=profiles,
        resolve_profile_config=resolve_profile_transcription_config,
        candidate_factory=candidate_factory,
    )


def run_default_profile_benchmark_from_public_boundary(
    *,
    language: str,
    sample_limit: int | None,
    absolute_accuracy_floor: float,
    maximum_accuracy_drop: float,
    minimum_required_samples_for_recommendation: int,
    sampling_strategy: str,
    random_seed: int,
    report_path: Path | None,
    active_settings: AppConfig,
    reference_glob: str,
    collect_reference_files: Callable[[int | None, str, int], list[Path]],
    default_profile_candidates: _DefaultProfileCandidatesFactory[CandidateT],
    profile_candidate: _ProfileCandidateRunner[CandidateT, SummaryT],
    derive_accuracy_gate: _DeriveAccuracyGate[SummaryT, GateT],
    recommend_default_profile: _RecommendDefaultProfile[SummaryT, GateT, RecommendationT],
    summarize_subset_coverage: Callable[[list[Path]], dict[str, int]],
    serialize_gate: Callable[[GateT], object],
    serialize_summary: Callable[[SummaryT], object],
    serialize_recommendation: Callable[[RecommendationT], object],
    result_factory: _ProfilingResultFactory[SummaryT, GateT, RecommendationT, ProfilingResultT],
) -> ProfilingResultT:
    """Runs the public default-benchmark wrapper using explicit boundary wiring."""
    execution: _DefaultBenchmarkExecutionLike[SummaryT, GateT, RecommendationT] = (
        default_benchmark_helpers.execute_default_profile_benchmark(
            language=language,
            sample_limit=sample_limit,
            absolute_accuracy_floor=absolute_accuracy_floor,
            maximum_accuracy_drop=maximum_accuracy_drop,
            minimum_required_samples_for_recommendation=(
                minimum_required_samples_for_recommendation
            ),
            sampling_strategy=sampling_strategy,
            random_seed=random_seed,
            report_path=report_path,
            default_report_folder=active_settings.models.folder,
            reference_glob=reference_glob,
            collect_reference_files=collect_reference_files,
            default_profile_candidates=default_profile_candidates,
            profile_candidate=profile_candidate,
            derive_accuracy_gate=derive_accuracy_gate,
            recommend_default_profile=recommend_default_profile,
            summarize_subset_coverage=summarize_subset_coverage,
            persist_profile_report=profiling_reporting_helpers.persist_profile_report,
            serialize_gate=serialize_gate,
            serialize_summary=serialize_summary,
            serialize_recommendation=serialize_recommendation,
        )
    )
    return result_factory(
        execution.reference_file_count,
        execution.gate,
        execution.summaries,
        execution.recommendation,
        execution.report_path,
    )


def run_runtime_calibration_from_public_boundary(
    *,
    calibration_file: Path,
    language: str,
    iterations_per_profile: int,
    profile_names: tuple[ArtifactProfileName, ...],
    report_path: Path | None,
    active_settings: AppConfig,
    settings_override: _SettingsOverride,
    runtime_calibration_candidates: _RuntimeCalibrationCandidatesFactory[CandidateT],
    calibrate_candidate: _CalibrateCandidate[CandidateT, MetricsT],
    derive_runtime_recommendation: _DeriveRuntimeRecommendation[
        MetricsT, RecommendationValueT, ConfidenceT
    ],
    recommendation_factory: _RuntimeRecommendationFactory[
        CandidateT, MetricsT, RecommendationValueT, ConfidenceT, RecommendationT
    ],
    serialize_recommendation: Callable[[RecommendationT], dict[str, object]],
    result_factory: _RuntimeCalibrationResultFactory[RecommendationT, RuntimeCalibrationResultT],
) -> RuntimeCalibrationResultT:
    """Runs the public runtime-calibration wrapper using explicit boundary wiring."""
    execution: _RuntimeCalibrationExecutionLike[RecommendationT] = (
        runtime_calibration_workflow_helpers.execute_runtime_calibration(
            active_settings=active_settings,
            calibration_file=calibration_file,
            language=language,
            iterations_per_profile=iterations_per_profile,
            profile_names=profile_names,
            report_path=report_path,
            build_runtime_calibration_settings=(
                runtime_calibration_helpers.build_runtime_calibration_settings
            ),
            settings_override=settings_override,
            runtime_calibration_candidates=runtime_calibration_candidates,
            calibrate_candidate=calibrate_candidate,
            derive_runtime_recommendation=derive_runtime_recommendation,
            recommendation_factory=recommendation_factory,
            runtime_calibration_report_path=(
                runtime_calibration_helpers.runtime_calibration_report_path
            ),
            persist_profile_report=profiling_reporting_helpers.persist_profile_report,
            serialize_recommendation=serialize_recommendation,
        )
    )
    return result_factory(execution.recommendations, execution.report_path)


def run_cli_from_public_boundary(
    *,
    run_default_profile_benchmark: Callable[..., ProfilingResultT],
    run_runtime_calibration: Callable[..., RuntimeCalibrationResultT],
    parse_calibration_profiles: Callable[[str], tuple[ArtifactProfileName, ...]],
    profiling_summary_lines: _SummaryLineBuilder[ProfilingResultT],
    runtime_calibration_summary_lines: _SummaryLineBuilder[RuntimeCalibrationResultT],
    argv: Sequence[str] | None = None,
    print_fn: _PrintFn = print,
) -> None:
    """Runs profiling CLI argument parsing and dispatch for the public wrapper."""
    parser = argparse.ArgumentParser(description="Internal transcription default profiling utility")
    parser.add_argument(
        "--mode",
        type=str,
        default="benchmark",
        choices=("benchmark", "runtime-calibration"),
    )
    parser.add_argument("--language", type=str, default="en")
    parser.add_argument("--sample-limit", type=int, default=None)
    parser.add_argument("--accuracy-floor", type=float, default=0.90)
    parser.add_argument("--max-accuracy-drop", type=float, default=0.02)
    parser.add_argument("--min-samples-for-recommendation", type=int, default=100)
    parser.add_argument(
        "--sampling-strategy",
        type=str,
        default="stratified",
        choices=["stratified", "head"],
    )
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--report-path", type=str, default=None)
    parser.add_argument("--calibration-file", type=str, default=None)
    parser.add_argument("--calibration-iterations", type=int, default=2)
    parser.add_argument(
        "--calibration-profiles",
        type=str,
        default="accurate,medium,accurate-research,fast",
        help=(
            "Comma-separated profile list for runtime calibration "
            "(fast,medium,accurate,accurate-research)."
        ),
    )
    parsed_args = parser.parse_args(list(argv) if argv is not None else None)

    report_path = None if parsed_args.report_path is None else Path(parsed_args.report_path)
    if parsed_args.mode == "runtime-calibration":
        if parsed_args.calibration_file is None:
            raise ValueError("--calibration-file is required for runtime-calibration mode.")
        calibration_result = run_runtime_calibration(
            calibration_file=Path(parsed_args.calibration_file),
            language=parsed_args.language,
            iterations_per_profile=parsed_args.calibration_iterations,
            profile_names=parse_calibration_profiles(parsed_args.calibration_profiles),
            report_path=report_path,
        )
        for line in runtime_calibration_summary_lines(calibration_result):
            print_fn(line)
        return

    result = run_default_profile_benchmark(
        language=parsed_args.language,
        sample_limit=parsed_args.sample_limit,
        absolute_accuracy_floor=parsed_args.accuracy_floor,
        maximum_accuracy_drop=parsed_args.max_accuracy_drop,
        minimum_required_samples_for_recommendation=(parsed_args.min_samples_for_recommendation),
        sampling_strategy=parsed_args.sampling_strategy,
        random_seed=parsed_args.random_seed,
        report_path=report_path,
    )
    for line in profiling_summary_lines(result):
        print_fn(line)


__all__ = [
    "build_profile_candidates_from_public_boundary",
    "run_cli_from_public_boundary",
    "run_default_profile_benchmark_from_public_boundary",
    "run_runtime_calibration_from_public_boundary",
]
