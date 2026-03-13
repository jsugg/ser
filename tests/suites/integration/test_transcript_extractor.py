"""Public transcript extractor behavior tests."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any, cast

import pytest

from ser.domain import TranscriptWord
from ser.transcript import transcript_extractor as te

pytestmark = pytest.mark.integration


class FakeResult:
    """Whisper-like result object with configurable word payload."""

    def __init__(self, words: list[SimpleNamespace]) -> None:
        self._words = words

    def all_words(self) -> list[SimpleNamespace]:
        return self._words


def test_resolve_transcription_profile_delegates_to_boundary_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Profile resolution should delegate to the internal owner with explicit settings."""
    settings = cast(te.AppConfig, SimpleNamespace(default_language="en"))
    captured: dict[str, object] = {}
    expected = te.TranscriptionProfile(backend_id="faster_whisper", model_name="small")

    def _fake_boundary_impl(
        profile: te.TranscriptionProfile | None,
        *,
        settings: te.AppConfig,
        profile_factory: object,
        error_factory: object,
    ) -> te.TranscriptionProfile:
        captured["profile"] = profile
        captured["settings"] = settings
        captured["profile_factory"] = profile_factory
        captured["error_factory"] = error_factory
        return expected

    monkeypatch.setattr(
        te._boundary_support,
        "resolve_transcription_profile_for_settings",
        _fake_boundary_impl,
    )

    resolved = te.resolve_transcription_profile(None, settings=settings)

    assert resolved == expected
    assert captured["settings"] is settings
    assert captured["profile_factory"] is te.TranscriptionProfile
    assert captured["error_factory"] is te.TranscriptionError


def test_extract_transcript_uses_default_language_from_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Public extractor should pass the resolved default language to the owner."""
    settings = cast(te.AppConfig, SimpleNamespace(default_language="pt"))
    captured: dict[str, object] = {}
    expected = [TranscriptWord("ola", 0.0, 0.5)]

    def _fake_boundary_impl(
        file_path: str,
        language: str,
        profile: te.TranscriptionProfile | None,
        *,
        settings: te.AppConfig,
        profile_factory: object,
        logger: logging.Logger,
        error_factory: object,
        release_memory_fn: object,
        phase_started_fn: object,
        phase_completed_fn: object,
        phase_failed_fn: object,
    ) -> list[TranscriptWord]:
        captured["file_path"] = file_path
        captured["language"] = language
        captured["profile"] = profile
        captured["settings"] = settings
        captured["profile_factory"] = profile_factory
        captured["logger"] = logger
        captured["error_factory"] = error_factory
        captured["release_memory_fn"] = release_memory_fn
        captured["phase_started_fn"] = phase_started_fn
        captured["phase_completed_fn"] = phase_completed_fn
        captured["phase_failed_fn"] = phase_failed_fn
        return expected

    monkeypatch.setattr(te._boundary_support, "extract_transcript", _fake_boundary_impl)

    resolved = te.extract_transcript("sample.wav", profile=None, settings=settings)

    assert resolved == expected
    assert captured["file_path"] == "sample.wav"
    assert captured["language"] == "pt"
    assert captured["settings"] is settings
    assert captured["profile_factory"] is te.TranscriptionProfile
    assert captured["logger"] is te.logger
    assert captured["error_factory"] is te.TranscriptionError


def test_extract_transcript_propagates_transcription_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Operational failures should propagate as TranscriptionError."""
    settings = cast(te.AppConfig, SimpleNamespace(default_language="en"))
    monkeypatch.setattr(
        te._boundary_support,
        "extract_transcript",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            te.TranscriptionError("Failed to transcribe audio.")
        ),
    )

    with pytest.raises(te.TranscriptionError, match="Failed to transcribe audio"):
        te.extract_transcript("sample.wav", settings=settings)


def test_load_whisper_model_uses_explicit_settings_without_ambient_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit settings should be passed directly to the owner."""
    settings = cast(te.AppConfig, SimpleNamespace(default_language="en"))
    expected_model = object()
    captured: dict[str, object] = {}

    def _fake_load_model(
        profile: te.TranscriptionProfile | None = None,
        *,
        settings: te.AppConfig,
        profile_factory: object,
        logger: logging.Logger,
        error_factory: object,
    ) -> object:
        captured["profile"] = profile
        captured["settings"] = settings
        captured["profile_factory"] = profile_factory
        captured["logger"] = logger
        captured["error_factory"] = error_factory
        return expected_model

    monkeypatch.setattr(te._boundary_support, "load_whisper_model_for_settings", _fake_load_model)

    resolved = te.load_whisper_model(settings=settings)

    assert resolved is expected_model
    assert captured["settings"] is settings
    assert captured["profile_factory"] is te.TranscriptionProfile
    assert captured["logger"] is te.logger
    assert captured["error_factory"] is te.TranscriptionError


def test_transcribe_with_model_delegates_to_boundary_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pre-loaded transcription calls should delegate to the boundary owner."""
    settings = cast(te.AppConfig, SimpleNamespace(default_language="en"))
    profile = te.TranscriptionProfile(backend_id="stable_whisper", model_name="large-v2")
    expected = [TranscriptWord("hello", 0.1, 0.3)]
    captured: dict[str, object] = {}

    def _fake_boundary_impl(
        model: object,
        language: str,
        file_path: str,
        active_profile: te.TranscriptionProfile | None,
        *,
        settings: te.AppConfig,
        profile_factory: object,
        logger: logging.Logger,
        error_factory: object,
        passthrough_error_cls: object,
    ) -> list[TranscriptWord]:
        captured["model"] = model
        captured["language"] = language
        captured["file_path"] = file_path
        captured["profile"] = active_profile
        captured["settings"] = settings
        captured["profile_factory"] = profile_factory
        captured["logger"] = logger
        captured["error_factory"] = error_factory
        captured["passthrough_error_cls"] = passthrough_error_cls
        return expected

    monkeypatch.setattr(te._boundary_support, "transcribe_with_profile", _fake_boundary_impl)

    model = object()
    resolved = te.transcribe_with_model(
        model,
        "sample.wav",
        "en",
        profile,
        settings=settings,
    )

    assert resolved == expected
    assert captured["model"] is model
    assert captured["profile"] == profile
    assert captured["settings"] is settings
    assert captured["profile_factory"] is te.TranscriptionProfile
    assert captured["logger"] is te.logger
    assert captured["error_factory"] is te.TranscriptionError
    assert captured["passthrough_error_cls"] is te.TranscriptionError


def test_format_transcript_formats_word_timestamps() -> None:
    """Word-level timestamps should be preserved in formatted output."""
    result = FakeResult([SimpleNamespace(word="hello", start=0.1, end=0.3)])

    assert te.format_transcript(cast(Any, result)) == [TranscriptWord("hello", 0.1, 0.3)]


def test_format_transcript_raises_for_invalid_result() -> None:
    """Invalid Whisper results should raise TranscriptionError."""
    with pytest.raises(te.TranscriptionError, match="Invalid Whisper result object"):
        te.format_transcript(cast(Any, object()))


def test_mark_compatibility_issues_as_emitted_suppresses_duplicate_operational_logs(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Pre-emitted compatibility issues should not be logged again."""

    class _Adapter:
        def check_compatibility(self, **_kwargs: object) -> Any:
            issue = SimpleNamespace(
                code="pytest-operational",
                message="sample remediation guidance",
                impact="informational",
            )
            return SimpleNamespace(
                noise_issues=[],
                operational_issues=[issue],
                functional_issues=[],
                has_blocking_issues=False,
            )

    monkeypatch.setattr(
        te._boundary_support,
        "resolve_transcription_backend_adapter",
        lambda _backend_id: _Adapter(),
    )

    te.mark_compatibility_issues_as_emitted(
        backend_id="stable_whisper",
        issue_kind="operational",
        issue_codes=("pytest-operational",),
    )

    with caplog.at_level(logging.INFO):
        _ = te._boundary_support.check_adapter_compatibility(
            active_profile=te.TranscriptionProfile(
                backend_id="stable_whisper",
                model_name="large-v2",
            ),
            settings=cast(te.AppConfig, SimpleNamespace()),
            runtime_request=cast(Any, SimpleNamespace()),
            logger=te.logger,
            error_factory=te.TranscriptionError,
        )

    assert "pytest-operational" not in caplog.text
