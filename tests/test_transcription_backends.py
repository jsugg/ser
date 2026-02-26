"""Unit tests for transcription backend adapter compatibility behavior."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest

from ser.config import AppConfig
from ser.transcript.backends.base import BackendRuntimeRequest
from ser.transcript.backends.faster_whisper import FasterWhisperAdapter
from ser.transcript.backends.stable_whisper import StableWhisperAdapter


def test_stable_whisper_compatibility_report_is_noise_aware(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stable adapter should expose noise policy metadata without blocking."""
    adapter = StableWhisperAdapter()
    monkeypatch.setattr(adapter, "_is_module_available", lambda _name: True)
    report = adapter.check_compatibility(
        runtime_request=BackendRuntimeRequest(
            model_name="large-v2",
            use_demucs=True,
            use_vad=True,
        ),
        settings=cast(AppConfig, SimpleNamespace()),
    )

    assert report.has_blocking_issues is False
    assert report.policy_ids == (
        "stable_whisper.invalid_escape_sequence",
        "stable_whisper.fp16_cpu_fallback_warning",
        "stable_whisper.demucs_deprecated_warning",
    )
    assert report.noise_issues


def test_stable_whisper_compatibility_blocks_on_missing_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stable adapter should block when stable_whisper dependency is unavailable."""
    adapter = StableWhisperAdapter()
    monkeypatch.setattr(adapter, "_is_module_available", lambda _name: False)
    report = adapter.check_compatibility(
        runtime_request=BackendRuntimeRequest(
            model_name="large-v2",
            use_demucs=True,
            use_vad=True,
        ),
        settings=cast(AppConfig, SimpleNamespace()),
    )

    assert report.has_blocking_issues is True
    assert any(
        issue.code == "missing_dependency_stable_whisper"
        for issue in report.functional_issues
    )


def test_faster_whisper_demucs_issue_is_operational_not_blocking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Faster adapter should report demucs limitation without blocking execution."""
    adapter = FasterWhisperAdapter()
    monkeypatch.setattr(adapter, "_is_module_available", lambda _name: True)
    report = adapter.check_compatibility(
        runtime_request=BackendRuntimeRequest(
            model_name="distil-large-v3",
            use_demucs=True,
            use_vad=True,
        ),
        settings=cast(AppConfig, SimpleNamespace()),
    )

    assert report.has_blocking_issues is False
    assert any(
        issue.code == "faster_whisper_demucs_unsupported"
        for issue in report.operational_issues
    )
    assert report.policy_ids == ("faster_whisper.info_demotion",)


def test_stable_whisper_transcribe_prefers_denoiser_and_fp16_control(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stable adapter should pass modern denoiser API and disable fp16 on CPU."""
    adapter = StableWhisperAdapter()
    monkeypatch.setattr(adapter, "_normalize_result", lambda raw: raw)
    monkeypatch.setattr(adapter, "_format_transcript", lambda _result: [])
    captured: dict[str, object] = {}

    class _FakeModel:
        def transcribe(
            self,
            *,
            audio: str,
            language: str,
            verbose: bool,
            word_timestamps: bool,
            no_speech_threshold: object,
            vad: bool,
            fp16: bool,
            denoiser: str,
        ) -> dict[str, object]:
            captured["audio"] = audio
            captured["language"] = language
            captured["verbose"] = verbose
            captured["word_timestamps"] = word_timestamps
            captured["no_speech_threshold"] = no_speech_threshold
            captured["vad"] = vad
            captured["fp16"] = fp16
            captured["denoiser"] = denoiser
            return {"segments": []}

    transcript = adapter.transcribe(
        model=_FakeModel(),
        runtime_request=BackendRuntimeRequest(
            model_name="large-v2",
            use_demucs=True,
            use_vad=True,
        ),
        file_path="sample.wav",
        language="en",
        settings=cast(AppConfig, SimpleNamespace()),
    )

    assert transcript == []
    assert captured["audio"] == "sample.wav"
    assert captured["language"] == "en"
    assert captured["verbose"] is False
    assert captured["word_timestamps"] is True
    assert captured["no_speech_threshold"] is None
    assert captured["vad"] is True
    assert captured["fp16"] is False
    assert captured["denoiser"] == "demucs"


def test_stable_whisper_transcribe_uses_legacy_demucs_when_denoiser_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stable adapter should fall back to legacy demucs argument for old signatures."""
    adapter = StableWhisperAdapter()
    monkeypatch.setattr(adapter, "_normalize_result", lambda raw: raw)
    monkeypatch.setattr(adapter, "_format_transcript", lambda _result: [])
    captured: dict[str, object] = {}

    class _LegacyModel:
        def transcribe(
            self,
            *,
            audio: str,
            language: str,
            verbose: bool,
            word_timestamps: bool,
            no_speech_threshold: object,
            vad: bool,
            demucs: bool,
        ) -> dict[str, object]:
            captured["audio"] = audio
            captured["language"] = language
            captured["verbose"] = verbose
            captured["word_timestamps"] = word_timestamps
            captured["no_speech_threshold"] = no_speech_threshold
            captured["vad"] = vad
            captured["demucs"] = demucs
            return {"segments": []}

    transcript = adapter.transcribe(
        model=_LegacyModel(),
        runtime_request=BackendRuntimeRequest(
            model_name="large-v2",
            use_demucs=True,
            use_vad=False,
        ),
        file_path="legacy.wav",
        language="en",
        settings=cast(AppConfig, SimpleNamespace()),
    )

    assert transcript == []
    assert captured["audio"] == "legacy.wav"
    assert captured["language"] == "en"
    assert captured["verbose"] is False
    assert captured["word_timestamps"] is True
    assert captured["no_speech_threshold"] is None
    assert captured["vad"] is False
    assert captured["demucs"] is True
