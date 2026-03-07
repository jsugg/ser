"""Contracts for stable-whisper transcribe-runtime helper extraction."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest

import ser.transcript.backends.stable_whisper as stable_whisper
import ser.transcript.backends.stable_whisper_transcribe_runtime as runtime_helpers
from ser.config import AppConfig
from ser.transcript.backends.base import BackendRuntimeRequest
from ser.transcript.runtime_failures import (
    FailureDisposition,
    TranscriptionFailureClassification,
)


def test_classify_transcription_failure_for_runtime_downgrades_when_shortcut_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hard-OOM failover should downgrade to precision retry when shortcut is disabled."""
    monkeypatch.setattr(
        runtime_helpers,
        "classify_stable_whisper_transcription_failure",
        lambda **_kwargs: TranscriptionFailureClassification(
            disposition=FailureDisposition.FAILOVER_CPU_NOW,
            reason_code="mps_hard_oom",
            is_retryable=True,
        ),
    )

    classification = runtime_helpers.classify_transcription_failure_for_runtime(
        err=RuntimeError("oom"),
        runtime_device_type="mps",
        precision="float16",
        settings=cast(AppConfig, SimpleNamespace()),
        hard_oom_shortcut_enabled=False,
    )

    assert classification.disposition == FailureDisposition.RETRY_NEXT_PRECISION
    assert classification.reason_code == "mps_hard_oom_disabled"
    assert classification.is_retryable is True


def test_classify_transcription_failure_for_runtime_preserves_failover_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hard-OOM failover should remain immediate CPU failover when shortcut is enabled."""
    expected = TranscriptionFailureClassification(
        disposition=FailureDisposition.FAILOVER_CPU_NOW,
        reason_code="mps_hard_oom",
        is_retryable=True,
    )
    monkeypatch.setattr(
        runtime_helpers,
        "classify_stable_whisper_transcription_failure",
        lambda **_kwargs: expected,
    )

    classification = runtime_helpers.classify_transcription_failure_for_runtime(
        err=RuntimeError("oom"),
        runtime_device_type="mps",
        precision="float16",
        settings=cast(AppConfig, SimpleNamespace()),
        hard_oom_shortcut_enabled=True,
    )

    assert classification is expected


def test_effective_precision_candidates_prefers_cpu_float32() -> None:
    """CPU runtime should force deterministic float32 precision selection."""
    runtime_request = BackendRuntimeRequest(
        model_name="large-v3",
        use_demucs=False,
        use_vad=True,
        device_spec="cpu",
        device_type="cpu",
        precision_candidates=("float16", "float32"),
        memory_tier="low",
    )

    cpu_candidates = runtime_helpers.effective_precision_candidates(
        runtime_request=runtime_request,
        runtime_device_type="cpu",
    )
    mps_candidates = runtime_helpers.effective_precision_candidates(
        runtime_request=runtime_request,
        runtime_device_type="mps",
    )

    assert cpu_candidates == ("float32",)
    assert mps_candidates == ("float16", "float32")


def test_stable_whisper_adapter_classify_transcription_failure_delegates_to_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Adapter wrapper should delegate classification to extracted runtime helper."""
    captured: dict[str, object] = {}
    expected = TranscriptionFailureClassification(
        disposition=FailureDisposition.RETRY_NEXT_PRECISION,
        reason_code="fallback",
        is_retryable=True,
    )
    monkeypatch.setattr(
        stable_whisper,
        "classify_transcription_failure_for_runtime",
        lambda **kwargs: (captured.update(kwargs), expected)[1],
    )
    monkeypatch.setattr(
        stable_whisper.StableWhisperAdapter,
        "_mps_hard_oom_shortcut_enabled",
        staticmethod(lambda _settings: True),
    )

    classification = (
        stable_whisper.StableWhisperAdapter._classify_transcription_failure(
            err=RuntimeError("boom"),
            runtime_device_type="mps",
            precision="float16",
            settings=cast(AppConfig, SimpleNamespace()),
        )
    )

    assert classification is expected
    assert captured["runtime_device_type"] == "mps"
    assert captured["precision"] == "float16"
    assert captured["hard_oom_shortcut_enabled"] is True


def test_stable_whisper_runtime_error_summary_helper_is_single_line() -> None:
    """Extracted runtime-summary helper should normalize multiline messages."""
    summary = runtime_helpers.summarize_runtime_error(
        RuntimeError("line1\nline2 " + ("x" * 400))
    )

    assert "\n" not in summary
    assert len(summary) <= 180
