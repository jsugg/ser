"""Default benchmark orchestration helpers for transcription profiling."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Generic, TypeVar

CandidateT = TypeVar("CandidateT")
SummaryT = TypeVar("SummaryT")
GateT = TypeVar("GateT")
RecommendationT = TypeVar("RecommendationT")


@dataclass(frozen=True)
class DefaultBenchmarkExecution(Generic[SummaryT, GateT, RecommendationT]):
    """Computed benchmark outputs ready for public result mapping."""

    reference_file_count: int
    gate: GateT
    summaries: tuple[SummaryT, ...]
    recommendation: RecommendationT
    report_path: Path


def _utc_now() -> datetime:
    """Return the current UTC timestamp."""

    return datetime.now(tz=UTC)


def execute_default_profile_benchmark(
    *,
    language: str,
    sample_limit: int | None,
    absolute_accuracy_floor: float,
    maximum_accuracy_drop: float,
    minimum_required_samples_for_recommendation: int,
    sampling_strategy: str,
    random_seed: int,
    report_path: Path | None,
    default_report_folder: Path,
    reference_glob: str,
    collect_reference_files: Callable[[int | None, str, int], list[Path]],
    default_profile_candidates: Callable[[], tuple[CandidateT, ...]],
    profile_candidate: Callable[[CandidateT, list[Path], str], SummaryT],
    derive_accuracy_gate: Callable[[SummaryT, float, float], GateT],
    recommend_default_profile: Callable[
        [tuple[SummaryT, ...], GateT, int],
        RecommendationT,
    ],
    summarize_subset_coverage: Callable[[list[Path]], dict[str, int]],
    persist_profile_report: Callable[[Path, dict[str, object]], Path],
    serialize_gate: Callable[[GateT], object],
    serialize_summary: Callable[[SummaryT], object],
    serialize_recommendation: Callable[[RecommendationT], object],
    now_utc: Callable[[], datetime] = _utc_now,
) -> DefaultBenchmarkExecution[SummaryT, GateT, RecommendationT]:
    """Run the default benchmark flow and persist its report payload."""

    if not 0.0 <= absolute_accuracy_floor <= 1.0:
        raise ValueError("absolute_accuracy_floor must be between 0 and 1.")
    if not 0.0 <= maximum_accuracy_drop <= 1.0:
        raise ValueError("maximum_accuracy_drop must be between 0 and 1.")
    if minimum_required_samples_for_recommendation <= 0:
        raise ValueError("minimum_required_samples_for_recommendation must be greater than zero.")

    reference_files = collect_reference_files(
        sample_limit,
        sampling_strategy,
        random_seed,
    )
    if not reference_files:
        raise RuntimeError(f"No RAVDESS reference files found under {reference_glob}.")

    candidates = default_profile_candidates()
    summaries = tuple(
        profile_candidate(candidate, reference_files, language) for candidate in candidates
    )
    baseline_summary = summaries[0]
    gate = derive_accuracy_gate(
        baseline_summary,
        absolute_accuracy_floor,
        maximum_accuracy_drop,
    )
    recommendation = recommend_default_profile(
        summaries,
        gate,
        minimum_required_samples_for_recommendation,
    )

    output_path = (
        default_report_folder / "transcription_profile_report.json"
        if report_path is None
        else report_path
    )
    payload: dict[str, object] = {
        "created_at_utc": now_utc().isoformat(),
        "reference_glob": reference_glob,
        "reference_files": len(reference_files),
        "sampling_strategy": sampling_strategy,
        "random_seed": random_seed,
        "subset_coverage": summarize_subset_coverage(reference_files),
        "accuracy_gate": serialize_gate(gate),
        "minimum_required_samples_for_recommendation": (
            minimum_required_samples_for_recommendation
        ),
        "profiles": [serialize_summary(summary) for summary in summaries],
        "recommendation": serialize_recommendation(recommendation),
    }
    persisted_path = persist_profile_report(output_path, payload)

    return DefaultBenchmarkExecution(
        reference_file_count=len(reference_files),
        gate=gate,
        summaries=summaries,
        recommendation=recommendation,
        report_path=persisted_path,
    )


__all__ = [
    "DefaultBenchmarkExecution",
    "execute_default_profile_benchmark",
]
