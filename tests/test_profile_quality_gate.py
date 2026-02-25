"""Tests for fast-versus-medium profile quality gate harness."""

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace

import pytest

from ser.domain import EmotionSegment
from ser.runtime.contracts import InferenceRequest
from ser.runtime.profile_quality_gate import (
    QualityGateThresholds,
    _build_fast_predictor,
    _build_medium_predictor,
    _parse_args,
    build_report_payload,
    enforce_quality_gate,
    evaluate_profile_quality_gate,
)

type LabeledAudioSample = tuple[str, str]


def _sample_path(
    *, actor_id: int, emotion_code: str, statement_code: str = "01"
) -> str:
    """Builds deterministic RAVDESS-like file path for grouped split tests."""
    return (
        "ser/dataset/ravdess/"
        f"Actor_{actor_id:02d}/03-01-{emotion_code}-01-{statement_code}-01-{actor_id:02d}.wav"
    )


def _segments(
    label: str, *, start: float = 0.0, end: float = 1.0
) -> list[EmotionSegment]:
    """Builds one deterministic segment with chosen label."""
    return [EmotionSegment(label, start, end)]


def _build_samples() -> list[LabeledAudioSample]:
    """Builds balanced two-class, four-speaker labeled samples."""
    return [
        (_sample_path(actor_id=1, emotion_code="03"), "happy"),
        (_sample_path(actor_id=1, emotion_code="04"), "sad"),
        (_sample_path(actor_id=2, emotion_code="03"), "happy"),
        (_sample_path(actor_id=2, emotion_code="04"), "sad"),
        (_sample_path(actor_id=3, emotion_code="03"), "happy"),
        (_sample_path(actor_id=3, emotion_code="04"), "sad"),
        (_sample_path(actor_id=4, emotion_code="03"), "happy"),
        (_sample_path(actor_id=4, emotion_code="04"), "sad"),
    ]


def test_profile_quality_gate_schema_and_pass_when_medium_beats_fast() -> None:
    """Report payload should include deterministic schema and a passing decision."""
    samples = _build_samples()

    def fast_predictor(_audio_path: str) -> Sequence[EmotionSegment]:
        return _segments("happy")

    def medium_predictor(audio_path: str) -> Sequence[EmotionSegment]:
        return _segments("sad" if "-04-" in audio_path else "happy")

    report = evaluate_profile_quality_gate(
        samples=samples,
        fast_predictor=fast_predictor,
        medium_predictor=medium_predictor,
        dataset_glob_pattern="ser/dataset/ravdess/**/*.wav",
        thresholds=QualityGateThresholds(
            minimum_uar_delta=0.20,
            minimum_macro_f1_delta=0.20,
            maximum_medium_segments_per_minute=120.0,
            minimum_medium_median_segment_duration_seconds=0.25,
        ),
        n_splits=2,
        random_state=7,
    )
    payload = build_report_payload(report)
    grouped_evaluation = payload["grouped_evaluation"]
    assert isinstance(grouped_evaluation, dict)

    assert payload["dataset_glob_pattern"] == "ser/dataset/ravdess/**/*.wav"
    assert payload["folds_evaluated"] == 2
    assert payload["fold_strategy"] == "stratified_group_kfold"
    assert grouped_evaluation["unique_speakers"] == 4
    assert grouped_evaluation["fold_speaker_overlap_counts"] == (0, 0)
    assert report.comparison.passes_quality_gate is True
    assert report.fast.profile == "fast"
    assert report.medium.profile == "medium"
    assert "uar" in report.fast.metrics
    assert "macro_f1" in report.medium.metrics


def test_profile_quality_gate_fails_when_delta_threshold_is_not_met() -> None:
    """Comparison should fail when medium does not exceed requested quality delta."""
    samples = _build_samples()

    def shared_predictor(audio_path: str) -> Sequence[EmotionSegment]:
        return _segments("sad" if "-04-" in audio_path else "happy")

    report = evaluate_profile_quality_gate(
        samples=samples,
        fast_predictor=shared_predictor,
        medium_predictor=shared_predictor,
        dataset_glob_pattern="ser/dataset/ravdess/**/*.wav",
        thresholds=QualityGateThresholds(
            minimum_uar_delta=0.05,
            minimum_macro_f1_delta=0.05,
        ),
        n_splits=2,
        random_state=11,
    )

    assert report.comparison.passes_quality_gate is False
    assert any("uar" in reason for reason in report.comparison.failure_reasons)
    assert any("macro_f1" in reason for reason in report.comparison.failure_reasons)
    with pytest.raises(SystemExit, match="Quality gate failed"):
        enforce_quality_gate(report, require_pass=True)


def test_profile_quality_gate_raises_when_speaker_metadata_is_missing() -> None:
    """Grouped evaluation should reject samples without speaker-identifying names."""
    samples: list[LabeledAudioSample] = [
        ("ser/dataset/ravdess/invalid_filename.wav", "happy"),
        (_sample_path(actor_id=2, emotion_code="04"), "sad"),
    ]

    with pytest.raises(ValueError, match="Speaker ID extraction failed"):
        evaluate_profile_quality_gate(
            samples=samples,
            fast_predictor=lambda _audio_path: _segments("happy"),
            medium_predictor=lambda _audio_path: _segments("happy"),
            dataset_glob_pattern="ser/dataset/ravdess/**/*.wav",
            thresholds=QualityGateThresholds(),
            n_splits=2,
        )


def test_profile_quality_gate_falls_back_to_grouped_holdout() -> None:
    """Infeasible fold count should fall back to deterministic grouped holdout."""
    samples = _build_samples()

    report = evaluate_profile_quality_gate(
        samples=samples,
        fast_predictor=lambda _audio_path: _segments("happy"),
        medium_predictor=lambda _audio_path: _segments("happy"),
        dataset_glob_pattern="ser/dataset/ravdess/**/*.wav",
        thresholds=QualityGateThresholds(),
        n_splits=8,
        random_state=42,
    )

    assert report.fold_strategy == "group_shuffle_holdout"
    assert report.folds_evaluated == 1
    assert report.grouped_evaluation.fold_speaker_overlap_counts == (0,)


def test_enforce_quality_gate_noop_when_not_required() -> None:
    """Gate enforcement helper should not raise when enforcement is disabled."""
    samples = _build_samples()

    report = evaluate_profile_quality_gate(
        samples=samples,
        fast_predictor=lambda _audio_path: _segments("happy"),
        medium_predictor=lambda _audio_path: _segments("happy"),
        dataset_glob_pattern="ser/dataset/ravdess/**/*.wav",
        thresholds=QualityGateThresholds(minimum_uar_delta=0.2),
        n_splits=2,
        random_state=11,
    )

    enforce_quality_gate(report, require_pass=False)


def test_profile_quality_gate_canonicalizes_overlap_for_temporal_metrics() -> None:
    """Temporal metrics should be computed from canonicalized non-overlapping segments."""
    samples = _build_samples()

    def overlapping_predictor(_audio_path: str) -> Sequence[EmotionSegment]:
        return [
            EmotionSegment("happy", 0.0, 1.0),
            EmotionSegment("happy", 0.5, 1.5),
        ]

    report = evaluate_profile_quality_gate(
        samples=samples,
        fast_predictor=overlapping_predictor,
        medium_predictor=overlapping_predictor,
        dataset_glob_pattern="ser/dataset/ravdess/**/*.wav",
        thresholds=QualityGateThresholds(),
        n_splits=2,
        random_state=17,
    )

    assert report.fast.temporal_stability.segment_count_per_minute == pytest.approx(
        40.0
    )
    assert (
        report.fast.temporal_stability.median_segment_duration_seconds
        == pytest.approx(1.5)
    )
    assert report.medium.temporal_stability.segment_count_per_minute == pytest.approx(
        40.0
    )


def test_profile_quality_gate_rejects_negative_thresholds() -> None:
    """Negative quality deltas should be rejected as invalid threshold policy."""
    samples = _build_samples()

    with pytest.raises(ValueError, match="minimum_uar_delta must be >= 0"):
        evaluate_profile_quality_gate(
            samples=samples,
            fast_predictor=lambda _audio_path: _segments("happy"),
            medium_predictor=lambda _audio_path: _segments("happy"),
            dataset_glob_pattern="ser/dataset/ravdess/**/*.wav",
            thresholds=QualityGateThresholds(minimum_uar_delta=-0.01),
            n_splits=2,
            random_state=5,
        )


def test_parse_args_uses_quality_gate_settings_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CLI defaults should come from strict quality-gate settings policy."""
    monkeypatch.setattr(
        "ser.runtime.profile_quality_gate.get_settings",
        lambda: SimpleNamespace(
            dataset=SimpleNamespace(glob_pattern="ser/dataset/ravdess/**/*.wav"),
            training=SimpleNamespace(test_size=0.25, random_state=42),
            models=SimpleNamespace(
                model_file_name="ser_model.pkl",
                secure_model_file_name="ser_model.skops",
                training_report_file_name="training_report.json",
            ),
            quality_gate=SimpleNamespace(
                min_uar_delta=0.015,
                min_macro_f1_delta=0.02,
                max_medium_segments_per_minute=22.5,
                min_medium_median_segment_duration_seconds=2.2,
            ),
        ),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "profile_quality_gate.py",
            "--medium-model-file-name",
            "medium.pkl",
        ],
    )

    args = _parse_args()

    assert args.min_uar_delta == pytest.approx(0.015)
    assert args.min_macro_f1_delta == pytest.approx(0.02)
    assert args.max_medium_segments_per_minute == pytest.approx(22.5)
    assert args.min_medium_median_segment_duration == pytest.approx(2.2)


def test_build_fast_predictor_loads_model_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fast gate predictor should preload model once and reuse it for all clips."""
    loaded_model = object()
    loaded_calls: list[object] = []
    predict_calls: list[tuple[str, object | None]] = []

    def _fake_load_model() -> object:
        loaded_calls.append(loaded_model)
        return loaded_model

    def _fake_predict_emotions(
        audio_path: str,
        *,
        loaded_model: object | None = None,
    ) -> list[EmotionSegment]:
        predict_calls.append((audio_path, loaded_model))
        return _segments("happy")

    monkeypatch.setattr("ser.runtime.profile_quality_gate.load_model", _fake_load_model)
    monkeypatch.setattr(
        "ser.runtime.profile_quality_gate.predict_emotions",
        _fake_predict_emotions,
    )

    predictor = _build_fast_predictor(
        model_file_name="fast.pkl",
        secure_model_file_name="fast.skops",
        training_report_file_name="fast_report.json",
    )

    assert len(loaded_calls) == 1
    predictor("a.wav")
    predictor("b.wav")
    assert predict_calls == [
        ("a.wav", loaded_model),
        ("b.wav", loaded_model),
    ]


def test_build_medium_predictor_reuses_loaded_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Medium gate predictor should reuse preloaded model/backend across clips."""
    loaded_model = object()
    backend = object()
    load_calls: list[object] = []
    backend_calls: list[object] = []
    backend_kwargs: list[dict[str, object]] = []
    run_calls: list[tuple[str, str | None, object, object, bool, bool]] = []

    def _fake_load_model() -> object:
        load_calls.append(loaded_model)
        return loaded_model

    def _fake_backend_factory(**kwargs: object) -> object:
        backend_kwargs.append(dict(kwargs))
        backend_calls.append(backend)
        return backend

    def _fake_run_medium_inference(
        request: InferenceRequest,
        settings: object,
        *,
        loaded_model: object | None = None,
        backend: object | None = None,
        enforce_timeout: bool = True,
        allow_retries: bool = True,
    ) -> object:
        del settings
        run_calls.append(
            (
                request.file_path,
                request.language,
                loaded_model if loaded_model is not None else object(),
                backend if backend is not None else object(),
                enforce_timeout,
                allow_retries,
            )
        )
        return SimpleNamespace(segments=_segments("happy"))

    monkeypatch.setattr("ser.runtime.profile_quality_gate.load_model", _fake_load_model)
    monkeypatch.setattr(
        "ser.runtime.profile_quality_gate.XLSRBackend",
        _fake_backend_factory,
    )
    monkeypatch.setattr(
        "ser.runtime.profile_quality_gate.run_medium_inference",
        _fake_run_medium_inference,
    )

    predictor = _build_medium_predictor(
        model_file_name="medium.pkl",
        secure_model_file_name="medium.skops",
        training_report_file_name="medium_report.json",
        language="pt",
    )

    assert len(load_calls) == 1
    assert len(backend_calls) == 1
    assert backend_kwargs and "cache_dir" in backend_kwargs[0]
    predictor("a.wav")
    predictor("b.wav")
    assert run_calls == [
        ("a.wav", "pt", loaded_model, backend, False, False),
        ("b.wav", "pt", loaded_model, backend, False, False),
    ]
