"""Unit tests for internal default-benchmark orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest

from ser._internal.transcription import default_benchmark as helpers


@dataclass(frozen=True)
class _Candidate:
    name: str


@dataclass(frozen=True)
class _Summary:
    name: str
    mean_accuracy: float


@dataclass(frozen=True)
class _Gate:
    minimum_mean_accuracy: float


@dataclass(frozen=True)
class _Recommendation:
    selected_profile: str


def test_execute_default_profile_benchmark_profiles_candidates_and_persists_report(
    tmp_path: Path,
) -> None:
    """Benchmark helper should orchestrate execution and persist one report payload."""

    captured: dict[str, object] = {}
    candidates = (_Candidate(name="accurate"), _Candidate(name="fast"))
    summaries = {
        "accurate": _Summary(name="accurate", mean_accuracy=0.96),
        "fast": _Summary(name="fast", mean_accuracy=0.94),
    }
    references = [tmp_path / "a.wav", tmp_path / "b.wav"]

    def _persist_profile_report(output_path: Path, payload: dict[str, object]) -> Path:
        captured["persist"] = (output_path, payload)
        return output_path

    execution = helpers.execute_default_profile_benchmark(
        language="en",
        sample_limit=2,
        absolute_accuracy_floor=0.90,
        maximum_accuracy_drop=0.02,
        minimum_required_samples_for_recommendation=5,
        sampling_strategy="stratified",
        random_seed=11,
        report_path=None,
        default_report_folder=tmp_path,
        reference_glob="ser/dataset/ravdess/**/*.wav",
        collect_reference_files=lambda limit, sampling_strategy, random_seed: (
            references if (limit, sampling_strategy, random_seed) == (2, "stratified", 11) else []
        ),
        default_profile_candidates=lambda: candidates,
        profile_candidate=lambda candidate, files, language: (
            summaries[candidate.name]
            if files == references and language == "en"
            else _Summary(name="unexpected", mean_accuracy=0.0)
        ),
        derive_accuracy_gate=lambda _baseline_summary, _floor, _drop: _Gate(
            minimum_mean_accuracy=0.94
        ),
        recommend_default_profile=lambda result_summaries, gate, minimum_required_samples: _Recommendation(
            selected_profile=(
                result_summaries[1].name
                if gate.minimum_mean_accuracy == 0.94 and minimum_required_samples == 5
                else "unexpected"
            )
        ),
        summarize_subset_coverage=lambda files: {
            "actors": len(files),
            "emotions": 1,
            "statements": 1,
        },
        persist_profile_report=_persist_profile_report,
        serialize_gate=lambda gate: {"minimum_mean_accuracy": gate.minimum_mean_accuracy},
        serialize_summary=lambda summary: {
            "name": summary.name,
            "mean_accuracy": summary.mean_accuracy,
        },
        serialize_recommendation=lambda recommendation: {
            "selected_profile": recommendation.selected_profile
        },
        now_utc=lambda: datetime(2026, 3, 9, tzinfo=UTC),
    )

    assert execution.reference_file_count == 2
    assert execution.summaries == (summaries["accurate"], summaries["fast"])
    assert execution.gate == _Gate(minimum_mean_accuracy=0.94)
    assert execution.recommendation == _Recommendation(selected_profile="fast")
    assert execution.report_path == tmp_path / "transcription_profile_report.json"
    assert captured["persist"] == (
        tmp_path / "transcription_profile_report.json",
        {
            "created_at_utc": "2026-03-09T00:00:00+00:00",
            "reference_glob": "ser/dataset/ravdess/**/*.wav",
            "reference_files": 2,
            "sampling_strategy": "stratified",
            "random_seed": 11,
            "subset_coverage": {"actors": 2, "emotions": 1, "statements": 1},
            "accuracy_gate": {"minimum_mean_accuracy": 0.94},
            "minimum_required_samples_for_recommendation": 5,
            "profiles": [
                {"name": "accurate", "mean_accuracy": 0.96},
                {"name": "fast", "mean_accuracy": 0.94},
            ],
            "recommendation": {"selected_profile": "fast"},
        },
    )


def test_execute_default_profile_benchmark_raises_when_no_references(
    tmp_path: Path,
) -> None:
    """Benchmark helper should fail fast when no reference files are available."""

    with pytest.raises(RuntimeError, match="No RAVDESS reference files found"):
        helpers.execute_default_profile_benchmark(
            language="en",
            sample_limit=1,
            absolute_accuracy_floor=0.90,
            maximum_accuracy_drop=0.02,
            minimum_required_samples_for_recommendation=5,
            sampling_strategy="head",
            random_seed=42,
            report_path=None,
            default_report_folder=tmp_path,
            reference_glob="ser/dataset/ravdess/**/*.wav",
            collect_reference_files=lambda _limit, _sampling_strategy, _random_seed: [],
            default_profile_candidates=lambda: (_Candidate(name="accurate"),),
            profile_candidate=lambda _candidate, _files, _language: _Summary(
                name="accurate",
                mean_accuracy=1.0,
            ),
            derive_accuracy_gate=lambda _baseline_summary, _floor, _drop: _Gate(
                minimum_mean_accuracy=1.0
            ),
            recommend_default_profile=lambda _summaries, _gate, _minimum_required_samples: _Recommendation(
                selected_profile="accurate"
            ),
            summarize_subset_coverage=lambda _files: {},
            persist_profile_report=lambda output_path, _payload: output_path,
            serialize_gate=lambda gate: {"minimum_mean_accuracy": gate.minimum_mean_accuracy},
            serialize_summary=lambda summary: {"name": summary.name},
            serialize_recommendation=lambda recommendation: {
                "selected_profile": recommendation.selected_profile
            },
        )


def test_execute_default_profile_benchmark_validates_threshold_arguments(
    tmp_path: Path,
) -> None:
    """Benchmark helper should reject invalid threshold parameters before work."""

    with pytest.raises(ValueError, match="maximum_accuracy_drop"):
        helpers.execute_default_profile_benchmark(
            language="en",
            sample_limit=1,
            absolute_accuracy_floor=0.90,
            maximum_accuracy_drop=1.5,
            minimum_required_samples_for_recommendation=5,
            sampling_strategy="head",
            random_seed=42,
            report_path=None,
            default_report_folder=tmp_path,
            reference_glob="ser/dataset/ravdess/**/*.wav",
            collect_reference_files=lambda _limit, _sampling_strategy, _random_seed: [],
            default_profile_candidates=lambda: (_Candidate(name="accurate"),),
            profile_candidate=lambda _candidate, _files, _language: _Summary(
                name="accurate",
                mean_accuracy=1.0,
            ),
            derive_accuracy_gate=lambda _baseline_summary, _floor, _drop: _Gate(
                minimum_mean_accuracy=1.0
            ),
            recommend_default_profile=lambda _summaries, _gate, _minimum_required_samples: _Recommendation(
                selected_profile="accurate"
            ),
            summarize_subset_coverage=lambda _files: {},
            persist_profile_report=lambda output_path, _payload: output_path,
            serialize_gate=lambda gate: {"minimum_mean_accuracy": gate.minimum_mean_accuracy},
            serialize_summary=lambda summary: {"name": summary.name},
            serialize_recommendation=lambda recommendation: {
                "selected_profile": recommendation.selected_profile
            },
        )
