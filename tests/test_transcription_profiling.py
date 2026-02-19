"""Unit tests for transcription profiling and threshold gating."""

from pathlib import Path

import pytest

from ser.transcript import profiling as tp


def _summary(
    *,
    name: str,
    mean_accuracy: float,
    avg_latency: float,
    evaluated_samples: int = 10,
    error_message: str | None = None,
) -> tp.ProfileBenchmarkSummary:
    """Builds synthetic profile summaries for recommendation tests."""
    return tp.ProfileBenchmarkSummary(
        profile=tp.TranscriptionProfileCandidate(
            name=name,
            model_name="large-v2",
            use_demucs=True,
            use_vad=True,
        ),
        evaluated_samples=evaluated_samples,
        failed_samples=0,
        exact_match_rate=0.9,
        mean_word_error_rate=1.0 - mean_accuracy,
        median_word_error_rate=1.0 - mean_accuracy,
        p90_word_error_rate=1.0 - mean_accuracy,
        mean_accuracy=mean_accuracy,
        average_latency_seconds=avg_latency,
        total_runtime_seconds=10.0,
        error_message=error_message,
    )


def test_ravdess_reference_text_maps_statement_codes() -> None:
    """RAVDESS statement code should map to canonical transcript text."""
    kids_path = Path("03-01-02-01-01-01-01.wav")
    dogs_path = Path("03-01-02-01-02-01-01.wav")
    invalid_path = Path("invalid.wav")

    assert tp.ravdess_reference_text(kids_path) == "kids are talking by the door"
    assert tp.ravdess_reference_text(dogs_path) == "dogs are sitting by the door"
    assert tp.ravdess_reference_text(invalid_path) is None


def test_default_profile_candidates_match_internal_fast_balanced_accurate_set() -> None:
    """Internal candidates should stay aligned with the approved three profiles."""
    candidates = tp.default_profile_candidates()
    names = [candidate.name for candidate in candidates]

    assert names == [
        "accurate_large-v2_demucs_vad",
        "balanced_base_no_demucs_vad",
        "fast_tiny_no_demucs_no_vad",
    ]


def test_word_error_rate_handles_normalization_and_exact_match() -> None:
    """Normalization should ignore punctuation/casing differences."""
    reference = "Kids are talking by the door"
    hypothesis = "kids, are talking by the door."

    wer = tp.word_error_rate(reference, hypothesis)

    assert wer == pytest.approx(0.0)


def test_collect_ravdess_reference_files_filters_non_reference_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only files with known statement codes should be returned."""
    monkeypatch.setattr(
        tp.glob,
        "glob",
        lambda _pattern, recursive: [
            "ser/dataset/ravdess/a/03-01-02-01-01-01-01.wav",
            "ser/dataset/ravdess/a/03-01-02-01-02-01-01.wav",
            "ser/dataset/ravdess/a/invalid.wav",
        ],
    )

    files = tp.collect_ravdess_reference_files()

    assert len(files) == 2


def test_collect_ravdess_reference_files_stratified_limit_is_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stratified selection should be deterministic for a given random seed."""
    mock_files = [
        f"ser/dataset/ravdess/Actor_{actor:02d}/03-01-02-01-{statement}-01-{actor:02d}.wav"
        for actor in (1, 2)
        for statement in ("01", "02")
        for _ in range(3)
    ]
    monkeypatch.setattr(tp.glob, "glob", lambda _pattern, recursive: mock_files)

    first = tp.collect_ravdess_reference_files(
        limit=4,
        sampling_strategy="stratified",
        random_seed=11,
    )
    second = tp.collect_ravdess_reference_files(
        limit=4,
        sampling_strategy="stratified",
        random_seed=11,
    )

    assert first == second
    assert len(first) == 4


def test_derive_accuracy_gate_uses_baseline_drop_and_floor() -> None:
    """Accuracy gate should respect both absolute floor and allowed drop."""
    baseline_summary = _summary(name="baseline", mean_accuracy=0.94, avg_latency=3.0)

    gate = tp.derive_accuracy_gate(
        baseline_summary,
        absolute_accuracy_floor=0.90,
        maximum_accuracy_drop=0.02,
    )

    assert gate.minimum_mean_accuracy == pytest.approx(0.92)


def test_recommend_default_profile_selects_fast_candidate_when_gate_passes() -> None:
    """A faster profile should be selected only when it passes the accuracy gate."""
    baseline = _summary(name="baseline", mean_accuracy=0.96, avg_latency=4.0)
    fast_candidate = _summary(
        name="candidate-fast", mean_accuracy=0.95, avg_latency=2.8
    )
    slow_candidate = _summary(
        name="candidate-slow", mean_accuracy=0.97, avg_latency=4.2
    )
    gate = tp.derive_accuracy_gate(
        baseline,
        absolute_accuracy_floor=0.90,
        maximum_accuracy_drop=0.02,
    )

    recommendation = tp.recommend_default_profile(
        (baseline, fast_candidate, slow_candidate),
        gate,
        minimum_speedup_ratio=1.10,
        minimum_required_samples=1,
    )

    assert recommendation.should_change_defaults is True
    assert recommendation.selected_profile == "candidate-fast"


def test_recommend_default_profile_keeps_baseline_when_candidates_fail_gate() -> None:
    """Profiles below minimum accuracy should never replace the baseline."""
    baseline = _summary(name="baseline", mean_accuracy=0.96, avg_latency=4.0)
    low_accuracy_candidate = _summary(
        name="candidate-low-accuracy",
        mean_accuracy=0.80,
        avg_latency=1.5,
    )
    gate = tp.derive_accuracy_gate(
        baseline,
        absolute_accuracy_floor=0.90,
        maximum_accuracy_drop=0.02,
    )

    recommendation = tp.recommend_default_profile(
        (baseline, low_accuracy_candidate),
        gate,
        minimum_required_samples=1,
    )

    assert recommendation.should_change_defaults is False
    assert recommendation.selected_profile == "baseline"


def test_recommend_default_profile_requires_minimum_sample_size() -> None:
    """Low sample-size profiling runs should never flip defaults."""
    baseline = _summary(
        name="baseline",
        mean_accuracy=0.96,
        avg_latency=4.0,
        evaluated_samples=12,
    )
    candidate = _summary(
        name="candidate-fast",
        mean_accuracy=0.95,
        avg_latency=2.0,
        evaluated_samples=12,
    )
    gate = tp.derive_accuracy_gate(
        baseline,
        absolute_accuracy_floor=0.90,
        maximum_accuracy_drop=0.02,
    )

    recommendation = tp.recommend_default_profile(
        (baseline, candidate),
        gate,
        minimum_required_samples=100,
    )

    assert recommendation.should_change_defaults is False
    assert recommendation.selected_profile == "baseline"


def test_run_default_profile_benchmark_validates_threshold_arguments() -> None:
    """Invalid threshold arguments should fail fast before any profiling work."""
    with pytest.raises(ValueError, match="absolute_accuracy_floor"):
        tp.run_default_profile_benchmark(
            language="en",
            sample_limit=1,
            absolute_accuracy_floor=1.5,
            maximum_accuracy_drop=0.02,
        )

    with pytest.raises(ValueError, match="minimum_required_samples_for_recommendation"):
        tp.run_default_profile_benchmark(
            language="en",
            sample_limit=1,
            absolute_accuracy_floor=0.9,
            maximum_accuracy_drop=0.02,
            minimum_required_samples_for_recommendation=0,
        )

    with pytest.raises(ValueError, match="sampling_strategy"):
        tp.collect_ravdess_reference_files(
            limit=1,
            sampling_strategy="unknown",
        )
