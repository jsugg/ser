"""Unit tests for internal default-profiling helper orchestration."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from ser._internal.transcription import default_profiling as default_helpers
from ser.domain import TranscriptWord
from ser.transcript.transcript_extractor import TranscriptionProfile


def _profile() -> TranscriptionProfile:
    """Builds one test transcription profile."""
    return TranscriptionProfile(
        backend_id="stable_whisper",
        model_name="turbo",
        use_demucs=False,
        use_vad=True,
    )


def test_profile_candidate_transcriptions_handles_empty_reference_set() -> None:
    """No references should return a deterministic empty benchmark summary."""
    stats = default_helpers.profile_candidate_transcriptions(
        candidate_name="candidate",
        profile=_profile(),
        files=[],
        language="en",
        load_model=lambda _profile: object(),
        transcribe=lambda _model, _file, _language, _profile: [],
        resolve_reference_text=lambda _path: "unused",
        words_to_text=lambda _words: "unused",
        compute_word_error_rate=lambda _ref, _hyp: 0.0,
        percentile=lambda _values, _quantile: 0.0,
        logger=logging.getLogger("tests.default_profiling"),
    )

    assert stats.evaluated_samples == 0
    assert stats.failed_samples == 0
    assert stats.mean_accuracy == 0.0
    assert stats.total_runtime_seconds == 0.0
    assert stats.error_message == "No reference files provided."


def test_profile_candidate_transcriptions_maps_load_failures_to_stats() -> None:
    """Model-load failures should map to failed sample counts and error payload."""

    def _raise_load_error(_profile: TranscriptionProfile | None) -> object:
        raise RuntimeError("load failed")

    stats = default_helpers.profile_candidate_transcriptions(
        candidate_name="candidate",
        profile=_profile(),
        files=[Path("a.wav"), Path("b.wav")],
        language="en",
        load_model=_raise_load_error,
        transcribe=lambda _model, _file, _language, _profile: [],
        resolve_reference_text=lambda _path: "unused",
        words_to_text=lambda _words: "unused",
        compute_word_error_rate=lambda _ref, _hyp: 0.0,
        percentile=lambda _values, _quantile: 0.0,
        logger=logging.getLogger("tests.default_profiling"),
    )

    assert stats.evaluated_samples == 0
    assert stats.failed_samples == 2
    assert stats.error_message == "load failed"
    assert stats.total_runtime_seconds >= 0.0


def test_profile_candidate_transcriptions_aggregates_success_and_failures() -> None:
    """Per-file failures should be counted while successful runs aggregate metrics."""
    load_calls = 0

    def _load_model(_profile: TranscriptionProfile | None) -> object:
        nonlocal load_calls
        load_calls += 1
        return object()

    files = [Path("a.wav"), Path("b.wav"), Path("c.wav")]

    def _resolve_reference_text(file_path: Path) -> str | None:
        if file_path.name == "b.wav":
            return None
        return "kids are talking by the door"

    def _transcribe(
        _model: object,
        file_path: str,
        _language: str,
        _profile: TranscriptionProfile | None,
    ) -> list[TranscriptWord]:
        if file_path.endswith("c.wav"):
            raise RuntimeError("transcribe failed")
        return [TranscriptWord("kids", 0.0, 0.5)]

    stats = default_helpers.profile_candidate_transcriptions(
        candidate_name="candidate",
        profile=_profile(),
        files=files,
        language="en",
        load_model=_load_model,
        transcribe=_transcribe,
        resolve_reference_text=_resolve_reference_text,
        words_to_text=lambda _words: "kids are talking by the door",
        compute_word_error_rate=lambda _ref, _hyp: 0.0,
        percentile=lambda values, _quantile: max(values),
        logger=logging.getLogger("tests.default_profiling"),
    )

    assert load_calls == 1
    assert stats.evaluated_samples == 1
    assert stats.failed_samples == 2
    assert stats.exact_match_rate == pytest.approx(1.0)
    assert stats.mean_word_error_rate == pytest.approx(0.0)
    assert stats.mean_accuracy == pytest.approx(1.0)
    assert stats.average_latency_seconds >= 0.0
    assert stats.error_message is None
