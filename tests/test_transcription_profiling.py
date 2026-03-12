"""Unit tests for transcription profiling and threshold gating."""

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from ser._internal.transcription import public_boundary_profiling as public_boundary_helpers
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
            source_profile="accurate",
            backend_id="stable_whisper",
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


def test_default_profile_candidates_match_profile_catalog_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Benchmark candidates should stay aligned with profile transcription defaults."""
    monkeypatch.delenv("WHISPER_BACKEND", raising=False)
    monkeypatch.delenv("WHISPER_MODEL", raising=False)
    candidates = tp.default_profile_candidates()
    assert [candidate.source_profile for candidate in candidates] == [
        "accurate",
        "medium",
        "fast",
    ]
    assert [candidate.backend_id for candidate in candidates] == [
        "stable_whisper",
        "stable_whisper",
        "faster_whisper",
    ]
    assert [candidate.model_name for candidate in candidates] == [
        "large",
        "turbo",
        "distil-large-v3",
    ]


def test_default_profile_candidates_delegate_to_boundary_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Candidate wrapper should delegate catalog assembly to the boundary owner."""
    captured: dict[str, object] = {}
    expected = (
        tp.TranscriptionProfileCandidate(
            name="candidate-fast",
            source_profile="fast",
            backend_id="faster_whisper",
            model_name="distil-large-v3",
            use_demucs=False,
            use_vad=True,
        ),
    )

    def _fake_boundary_impl(
        **kwargs: object,
    ) -> tuple[tp.TranscriptionProfileCandidate, ...]:
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(tp, "_build_profile_candidates_boundary_impl", _fake_boundary_impl)

    candidates = tp.default_profile_candidates()

    assert candidates == expected
    assert captured["profiles"] == tp.DEFAULT_BENCHMARK_PROFILES
    assert captured["candidate_factory"] is tp.TranscriptionProfileCandidate


def test_default_profile_candidates_follow_runtime_backend_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Benchmark candidates should reflect global backend/model env overrides."""
    monkeypatch.setenv("WHISPER_BACKEND", "stable_whisper")
    monkeypatch.setenv("WHISPER_MODEL", "base")

    candidates = tp.default_profile_candidates()

    assert all(candidate.backend_id == "stable_whisper" for candidate in candidates)
    assert all(candidate.model_name == "base" for candidate in candidates)


def test_runtime_calibration_candidates_follow_selected_profiles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Calibration candidates should reflect the requested profile ordering."""

    monkeypatch.delenv("WHISPER_BACKEND", raising=False)
    monkeypatch.delenv("WHISPER_MODEL", raising=False)

    candidates = tp.runtime_calibration_candidates(("fast", "accurate"))

    assert [candidate.source_profile for candidate in candidates] == [
        "fast",
        "accurate",
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


def test_parse_ravdess_metadata_extracts_expected_fields() -> None:
    """Public metadata wrapper should preserve the existing dataclass contract."""

    metadata = tp._parse_ravdess_metadata(Path("03-01-06-01-02-01-24.wav"))

    assert metadata == tp.RavdessMetadata(
        emotion_code="06",
        statement_code="02",
        actor_id="24",
    )


def test_summarize_subset_coverage_counts_unique_groups() -> None:
    """Public coverage wrapper should preserve actor/emotion/statement counting."""

    files = [
        Path("03-01-06-01-02-01-24.wav"),
        Path("03-01-05-01-01-01-24.wav"),
        Path("03-01-05-01-02-01-05.wav"),
        Path("invalid.wav"),
    ]

    assert tp._summarize_subset_coverage(files) == {
        "actors": 2,
        "emotions": 2,
        "statements": 2,
    }


def test_profile_transcription_candidate_delegates_to_internal_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Candidate profiling wrapper should delegate execution to internal helper."""
    captured: dict[str, object] = {}
    stats = tp.default_profiling_helpers.CandidateProfileBenchmarkStats(
        evaluated_samples=2,
        failed_samples=1,
        exact_match_rate=0.5,
        mean_word_error_rate=0.25,
        median_word_error_rate=0.25,
        p90_word_error_rate=0.3,
        mean_accuracy=0.75,
        average_latency_seconds=0.6,
        total_runtime_seconds=1.3,
        error_message=None,
    )

    def _profile_candidate_transcriptions(**kwargs: object) -> object:
        captured.update(kwargs)
        return stats

    monkeypatch.setattr(
        "ser._internal.transcription.default_profiling." "profile_candidate_transcriptions",
        _profile_candidate_transcriptions,
    )
    candidate = tp.TranscriptionProfileCandidate(
        name="candidate-medium",
        source_profile="medium",
        backend_id="stable_whisper",
        model_name="turbo",
        use_demucs=False,
        use_vad=True,
    )
    result = tp.profile_transcription_candidate(
        candidate=candidate,
        files=[Path("sample.wav")],
        language="en",
    )

    assert result.profile is candidate
    assert result.evaluated_samples == stats.evaluated_samples
    assert result.failed_samples == stats.failed_samples
    assert result.exact_match_rate == stats.exact_match_rate
    assert result.mean_word_error_rate == stats.mean_word_error_rate
    assert result.median_word_error_rate == stats.median_word_error_rate
    assert result.p90_word_error_rate == stats.p90_word_error_rate
    assert result.mean_accuracy == stats.mean_accuracy
    assert result.average_latency_seconds == stats.average_latency_seconds
    assert result.total_runtime_seconds == stats.total_runtime_seconds
    assert result.error_message is None
    assert captured["candidate_name"] == "candidate-medium"
    assert captured["profile"] == tp.TranscriptionProfile(
        backend_id="stable_whisper",
        model_name="turbo",
        use_demucs=False,
        use_vad=True,
    )
    assert captured["files"] == [Path("sample.wav")]
    assert captured["language"] == "en"
    assert captured["load_model"] is tp.load_whisper_model
    assert captured["transcribe"] is tp.transcribe_with_model
    assert captured["resolve_reference_text"] is tp.ravdess_reference_text
    assert captured["words_to_text"] is tp.transcript_words_to_text
    assert captured["compute_word_error_rate"] is tp.word_error_rate
    assert captured["percentile"] is tp._percentile
    assert captured["logger"] is tp.logger


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
    fast_candidate = _summary(name="candidate-fast", mean_accuracy=0.95, avg_latency=2.8)
    slow_candidate = _summary(name="candidate-slow", mean_accuracy=0.97, avg_latency=4.2)
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


def test_run_default_profile_benchmark_delegates_to_internal_helper(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Public benchmark wrapper should delegate orchestration to the internal helper."""

    gate = tp.AccuracyGate(
        baseline_mean_accuracy=0.96,
        minimum_mean_accuracy=0.94,
        maximum_accuracy_drop=0.02,
        absolute_accuracy_floor=0.90,
    )
    summary = _summary(name="baseline", mean_accuracy=0.96, avg_latency=4.0)
    recommendation = tp.DefaultRecommendation(
        baseline_profile="baseline",
        selected_profile="baseline",
        should_change_defaults=False,
        reason="keep current defaults",
        selected_mean_accuracy=0.96,
        selected_average_latency_seconds=4.0,
        selected_speedup_vs_baseline=1.0,
        minimum_required_samples=10,
    )
    captured: dict[str, object] = {}
    execution = tp.default_benchmark_helpers.DefaultBenchmarkExecution(
        reference_file_count=3,
        gate=gate,
        summaries=(summary,),
        recommendation=recommendation,
        report_path=tmp_path / "report.json",
    )

    def _execute_default_profile_benchmark(**kwargs: object) -> object:
        captured.update(kwargs)
        return execution

    monkeypatch.setattr(
        "ser._internal.transcription.default_benchmark." "execute_default_profile_benchmark",
        _execute_default_profile_benchmark,
    )
    monkeypatch.setattr(
        tp,
        "reload_settings",
        lambda: SimpleNamespace(models=SimpleNamespace(folder=tmp_path)),
    )

    result = tp.run_default_profile_benchmark(
        language="en",
        sample_limit=3,
        absolute_accuracy_floor=0.90,
        maximum_accuracy_drop=0.02,
        minimum_required_samples_for_recommendation=10,
        sampling_strategy="head",
        random_seed=7,
        report_path=None,
        settings=None,
    )

    assert result == tp.ProfilingResult(
        reference_files=3,
        gate=gate,
        summaries=(summary,),
        recommendation=recommendation,
        report_path=tmp_path / "report.json",
    )
    assert captured["language"] == "en"
    assert captured["sample_limit"] == 3
    assert captured["absolute_accuracy_floor"] == 0.90
    assert captured["maximum_accuracy_drop"] == 0.02
    assert captured["minimum_required_samples_for_recommendation"] == 10
    assert captured["sampling_strategy"] == "head"
    assert captured["random_seed"] == 7
    assert captured["report_path"] is None
    assert captured["default_report_folder"] == tmp_path
    assert captured["reference_glob"] == tp.RAVDESS_REFERENCE_GLOB


def test_run_default_profile_benchmark_delegates_to_boundary_owner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Public benchmark wrapper should delegate boundary assembly to the internal owner."""
    captured: dict[str, object] = {}
    expected = tp.ProfilingResult(
        reference_files=2,
        gate=tp.AccuracyGate(
            baseline_mean_accuracy=0.96,
            minimum_mean_accuracy=0.94,
            maximum_accuracy_drop=0.02,
            absolute_accuracy_floor=0.90,
        ),
        summaries=(_summary(name="baseline", mean_accuracy=0.96, avg_latency=4.0),),
        recommendation=tp.DefaultRecommendation(
            baseline_profile="baseline",
            selected_profile="baseline",
            should_change_defaults=False,
            reason="keep current defaults",
            selected_mean_accuracy=0.96,
            selected_average_latency_seconds=4.0,
            selected_speedup_vs_baseline=1.0,
            minimum_required_samples=10,
        ),
        report_path=tmp_path / "report.json",
    )
    settings = SimpleNamespace(models=SimpleNamespace(folder=tmp_path))

    def _fake_boundary_impl(**kwargs: object) -> tp.ProfilingResult:
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(tp, "_run_default_profile_benchmark_boundary_impl", _fake_boundary_impl)

    result = tp.run_default_profile_benchmark(
        language="en",
        sample_limit=3,
        absolute_accuracy_floor=0.90,
        maximum_accuracy_drop=0.02,
        minimum_required_samples_for_recommendation=10,
        sampling_strategy="head",
        random_seed=7,
        report_path=None,
        settings=cast(tp.AppConfig, settings),
    )

    assert result == expected
    assert captured["language"] == "en"
    assert captured["sample_limit"] == 3
    assert captured["absolute_accuracy_floor"] == 0.90
    assert captured["maximum_accuracy_drop"] == 0.02
    assert captured["minimum_required_samples_for_recommendation"] == 10
    assert captured["sampling_strategy"] == "head"
    assert captured["random_seed"] == 7
    assert captured["report_path"] is None
    assert captured["active_settings"] is settings
    assert captured["reference_glob"] == tp.RAVDESS_REFERENCE_GLOB
    assert captured["default_profile_candidates"] is tp.default_profile_candidates
    assert captured["profile_candidate"] is tp.profile_transcription_candidate
    assert captured["summarize_subset_coverage"] is tp._summarize_subset_coverage


def _calibration_metrics(
    *,
    backend_id: tp.TranscriptionBackendId,
    model_name: str,
    iterations: int,
    mps_loaded_runs: int,
    mps_completed_runs: int,
    mps_to_cpu_failover_runs: int,
    failed_runs: int,
    hard_mps_oom_runs: int,
) -> tp.RuntimeCalibrationMetrics:
    """Builds synthetic runtime-calibration metrics for recommendation tests."""
    return tp.RuntimeCalibrationMetrics(
        profile=tp.TranscriptionProfileCandidate(
            name=f"candidate-{model_name}",
            source_profile="accurate",
            backend_id=backend_id,
            model_name=model_name,
            use_demucs=True,
            use_vad=True,
        ),
        iterations=iterations,
        successful_runs=max(iterations - failed_runs, 0),
        failed_runs=failed_runs,
        mps_loaded_runs=mps_loaded_runs,
        mps_completed_runs=mps_completed_runs,
        mps_to_cpu_failover_runs=mps_to_cpu_failover_runs,
        hard_mps_oom_runs=hard_mps_oom_runs,
        mean_latency_seconds=1.0,
        error_messages=(),
    )


def test_derive_runtime_recommendation_prefers_cpu_on_hard_mps_oom() -> None:
    """Hard MPS OOM should produce CPU recommendation with high confidence."""
    metrics = _calibration_metrics(
        backend_id="stable_whisper",
        model_name="large",
        iterations=3,
        mps_loaded_runs=3,
        mps_completed_runs=0,
        mps_to_cpu_failover_runs=2,
        failed_runs=2,
        hard_mps_oom_runs=2,
    )

    recommendation, confidence, reason = tp.derive_runtime_recommendation(metrics)

    assert recommendation == "prefer_cpu"
    assert confidence == "high"
    assert "oom" in reason.lower()


def test_derive_runtime_recommendation_prefers_mps_when_stable() -> None:
    """Stable MPS calibrations should recommend MPS with high confidence."""
    metrics = _calibration_metrics(
        backend_id="stable_whisper",
        model_name="turbo",
        iterations=3,
        mps_loaded_runs=3,
        mps_completed_runs=3,
        mps_to_cpu_failover_runs=0,
        failed_runs=0,
        hard_mps_oom_runs=0,
    )

    recommendation, confidence, _reason = tp.derive_runtime_recommendation(metrics)

    assert recommendation == "prefer_mps"
    assert confidence == "high"


def test_parse_calibration_profiles_validates_values() -> None:
    """Calibration profile parser should reject unsupported profile names."""
    parsed = tp.parse_calibration_profiles("accurate,medium,fast")
    assert parsed == ("accurate", "medium", "fast")

    with pytest.raises(ValueError, match="Unsupported profile"):
        tp.parse_calibration_profiles("accurate,unknown")


def test_calibrate_runtime_candidate_delegates_probe_execution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Candidate calibration wrapper should map internal probe stats to metrics."""
    captured: dict[str, object] = {}
    stats = tp.runtime_calibration_helpers.RuntimeCalibrationProbeStats(
        successful_runs=2,
        failed_runs=1,
        mps_loaded_runs=2,
        mps_completed_runs=1,
        mps_to_cpu_failover_runs=1,
        hard_mps_oom_runs=1,
        mean_latency_seconds=0.42,
        error_messages=("mps oom",),
    )

    def _run_runtime_calibration_probes(**kwargs: object) -> object:
        captured.update(kwargs)
        return stats

    monkeypatch.setattr(
        "ser._internal.transcription.runtime_calibration." "run_runtime_calibration_probes",
        _run_runtime_calibration_probes,
    )
    candidate = tp.TranscriptionProfileCandidate(
        name="candidate",
        source_profile="accurate",
        backend_id="stable_whisper",
        model_name="large",
        use_demucs=True,
        use_vad=True,
    )

    metrics = tp._calibrate_runtime_candidate(
        candidate=candidate,
        calibration_file=tmp_path / "sample.wav",
        language="en",
        iterations=3,
    )

    assert metrics.profile is candidate
    assert metrics.iterations == 3
    assert metrics.successful_runs == stats.successful_runs
    assert metrics.failed_runs == stats.failed_runs
    assert metrics.mps_loaded_runs == stats.mps_loaded_runs
    assert metrics.mps_completed_runs == stats.mps_completed_runs
    assert metrics.mps_to_cpu_failover_runs == stats.mps_to_cpu_failover_runs
    assert metrics.hard_mps_oom_runs == stats.hard_mps_oom_runs
    assert metrics.mean_latency_seconds == stats.mean_latency_seconds
    assert metrics.error_messages == stats.error_messages
    assert captured["backend_id"] == "stable_whisper"
    assert captured["active_profile"] == tp.TranscriptionProfile(
        backend_id="stable_whisper",
        model_name="large",
        use_demucs=True,
        use_vad=True,
    )
    assert captured["calibration_file"] == tmp_path / "sample.wav"
    assert captured["language"] == "en"
    assert captured["iterations"] == 3
    assert captured["load_model"] is tp.load_whisper_model
    assert captured["transcribe"] is tp.transcribe_with_model


def test_run_transcription_runtime_calibration_delegates_to_internal_helper(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Public calibration wrapper should delegate orchestration to the internal helper."""

    calibration_file = tmp_path / "sample.wav"
    calibration_file.write_bytes(b"audio")
    metrics = _calibration_metrics(
        backend_id="stable_whisper",
        model_name="large",
        iterations=2,
        mps_loaded_runs=2,
        mps_completed_runs=2,
        mps_to_cpu_failover_runs=0,
        failed_runs=0,
        hard_mps_oom_runs=0,
    )
    recommendation = tp.RuntimeCalibrationRecommendation(
        profile=metrics.profile,
        recommendation="prefer_mps",
        confidence="high",
        reason="stable",
        metrics=metrics,
    )
    execution = tp.runtime_calibration_workflow_helpers.RuntimeCalibrationExecution(
        recommendations=(recommendation,),
        report_path=tmp_path / "runtime.json",
    )
    captured: dict[str, object] = {}

    def _execute_runtime_calibration(**kwargs: object) -> object:
        captured.update(kwargs)
        return execution

    monkeypatch.setattr(
        "ser._internal.transcription.runtime_calibration_workflow." "execute_runtime_calibration",
        _execute_runtime_calibration,
    )

    settings = SimpleNamespace(models=SimpleNamespace(folder=tmp_path))
    monkeypatch.setattr(tp, "reload_settings", lambda: settings)

    result = tp.run_transcription_runtime_calibration(
        calibration_file=calibration_file,
        language="en",
        iterations_per_profile=2,
        profile_names=("accurate", "fast"),
        report_path=None,
        settings=None,
    )

    assert result == tp.RuntimeCalibrationResult(
        recommendations=(recommendation,),
        report_path=tmp_path / "runtime.json",
    )
    assert captured["active_settings"] is settings
    assert captured["calibration_file"] == calibration_file
    assert captured["language"] == "en"
    assert captured["iterations_per_profile"] == 2
    assert captured["profile_names"] == ("accurate", "fast")
    assert captured["report_path"] is None


def test_run_transcription_runtime_calibration_delegates_to_boundary_owner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Public runtime-calibration wrapper should delegate boundary assembly to the internal owner."""
    captured: dict[str, object] = {}
    settings = cast(tp.AppConfig, SimpleNamespace(models=SimpleNamespace(folder=tmp_path)))
    calibration_file = tmp_path / "sample.wav"
    expected = tp.RuntimeCalibrationResult(
        recommendations=(),
        report_path=tmp_path / "runtime.json",
    )

    def _fake_boundary_impl(**kwargs: object) -> tp.RuntimeCalibrationResult:
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(tp, "_run_runtime_calibration_boundary_impl", _fake_boundary_impl)

    result = tp.run_transcription_runtime_calibration(
        calibration_file=calibration_file,
        language="en",
        iterations_per_profile=2,
        profile_names=("accurate", "fast"),
        report_path=None,
        settings=settings,
    )

    assert result == expected
    assert captured["calibration_file"] == calibration_file
    assert captured["language"] == "en"
    assert captured["iterations_per_profile"] == 2
    assert captured["profile_names"] == ("accurate", "fast")
    assert captured["report_path"] is None
    assert captured["active_settings"] is settings
    assert captured["settings_override"] is tp.settings_override
    assert captured["runtime_calibration_candidates"] is tp.runtime_calibration_candidates
    assert captured["derive_runtime_recommendation"] is tp.derive_runtime_recommendation


def test_run_cli_from_public_boundary_dispatches_benchmark_mode() -> None:
    """CLI boundary owner should parse benchmark arguments and print summary lines."""
    captured: dict[str, object] = {}
    printed_lines: list[str] = []

    def _run_default_profile_benchmark(**kwargs: object) -> str:
        captured["benchmark_kwargs"] = kwargs
        return "benchmark-result"

    def _run_runtime_calibration(**_kwargs: object) -> object:
        raise AssertionError("runtime calibration path should not run")

    public_boundary_helpers.run_cli_from_public_boundary(
        argv=(
            "--language",
            "es",
            "--sample-limit",
            "3",
            "--accuracy-floor",
            "0.91",
            "--max-accuracy-drop",
            "0.03",
            "--min-samples-for-recommendation",
            "11",
            "--sampling-strategy",
            "head",
            "--random-seed",
            "9",
            "--report-path",
            "profile.json",
        ),
        run_default_profile_benchmark=_run_default_profile_benchmark,
        run_runtime_calibration=_run_runtime_calibration,
        parse_calibration_profiles=lambda raw_profiles: ("fast",),
        profiling_summary_lines=lambda result: (f"summary:{result}",),
        runtime_calibration_summary_lines=lambda _result: ("unexpected",),
        print_fn=printed_lines.append,
    )

    benchmark_kwargs = cast(dict[str, object], captured["benchmark_kwargs"])
    assert benchmark_kwargs == {
        "language": "es",
        "sample_limit": 3,
        "absolute_accuracy_floor": 0.91,
        "maximum_accuracy_drop": 0.03,
        "minimum_required_samples_for_recommendation": 11,
        "sampling_strategy": "head",
        "random_seed": 9,
        "report_path": Path("profile.json"),
    }
    assert printed_lines == ["summary:benchmark-result"]


def test_run_cli_from_public_boundary_dispatches_runtime_calibration_mode() -> None:
    """CLI boundary owner should parse calibration mode and print runtime summary."""
    captured: dict[str, object] = {}
    printed_lines: list[str] = []

    def _run_default_profile_benchmark(**_kwargs: object) -> object:
        raise AssertionError("benchmark path should not run")

    def _run_runtime_calibration(**kwargs: object) -> str:
        captured["runtime_kwargs"] = kwargs
        return "calibration-result"

    raw_profiles: list[str] = []

    def _parse_calibration_profiles(
        raw_profiles_csv: str,
    ) -> tuple[tp.ArtifactProfileName, ...]:
        raw_profiles.append(raw_profiles_csv)
        return cast(tuple[tp.ArtifactProfileName, ...], ("accurate", "fast"))

    public_boundary_helpers.run_cli_from_public_boundary(
        argv=(
            "--mode",
            "runtime-calibration",
            "--language",
            "pt",
            "--calibration-file",
            "sample.wav",
            "--calibration-iterations",
            "4",
            "--calibration-profiles",
            "accurate,fast",
            "--report-path",
            "runtime.json",
        ),
        run_default_profile_benchmark=_run_default_profile_benchmark,
        run_runtime_calibration=_run_runtime_calibration,
        parse_calibration_profiles=_parse_calibration_profiles,
        profiling_summary_lines=lambda _result: ("unexpected",),
        runtime_calibration_summary_lines=lambda result: (f"runtime:{result}",),
        print_fn=printed_lines.append,
    )

    runtime_kwargs = cast(dict[str, object], captured["runtime_kwargs"])
    assert raw_profiles == ["accurate,fast"]
    assert runtime_kwargs == {
        "calibration_file": Path("sample.wav"),
        "language": "pt",
        "iterations_per_profile": 4,
        "profile_names": ("accurate", "fast"),
        "report_path": Path("runtime.json"),
    }
    assert printed_lines == ["runtime:calibration-result"]


def test_run_cli_from_public_boundary_requires_calibration_file() -> None:
    """Calibration mode should fail fast when no calibration file is provided."""
    with pytest.raises(
        ValueError, match="--calibration-file is required for runtime-calibration mode"
    ):
        public_boundary_helpers.run_cli_from_public_boundary(
            argv=("--mode", "runtime-calibration"),
            run_default_profile_benchmark=lambda **_kwargs: "unused",
            run_runtime_calibration=lambda **_kwargs: "unused",
            parse_calibration_profiles=lambda _raw_profiles: cast(
                tuple[tp.ArtifactProfileName, ...], ("fast",)
            ),
            profiling_summary_lines=lambda result: (str(result),),
            runtime_calibration_summary_lines=lambda result: (str(result),),
        )


def test_main_delegates_cli_dispatch_to_boundary_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Public profiling main should delegate CLI wiring to the internal owner."""
    captured: dict[str, object] = {}

    def _fake_cli_boundary_impl(**kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(tp, "_run_cli_from_public_boundary_impl", _fake_cli_boundary_impl)

    tp.main()

    assert captured["run_default_profile_benchmark"] is tp.run_default_profile_benchmark
    assert captured["run_runtime_calibration"] is tp.run_transcription_runtime_calibration
    assert captured["parse_calibration_profiles"] is tp.parse_calibration_profiles
    profiling_summary_lines = cast(
        Callable[[object], tuple[str, ...]], captured["profiling_summary_lines"]
    )
    runtime_summary_lines = cast(
        Callable[[object], tuple[str, ...]],
        captured["runtime_calibration_summary_lines"],
    )
    assert profiling_summary_lines is tp.profiling_reporting_helpers.profiling_summary_lines
    assert runtime_summary_lines is tp.profiling_reporting_helpers.runtime_calibration_summary_lines
