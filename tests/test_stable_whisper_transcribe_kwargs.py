"""Contract tests for stable-whisper transcribe kwargs helper/delegation."""

from __future__ import annotations

from collections.abc import Callable

import pytest

import ser.transcript.backends.stable_whisper as stable_whisper
from ser.transcript.backends.base import BackendRuntimeRequest
from ser.transcript.backends.stable_whisper_transcribe_kwargs import (
    build_stable_whisper_transcribe_kwargs,
)


def _runtime_request(*, use_demucs: bool) -> BackendRuntimeRequest:
    return BackendRuntimeRequest(
        model_name="openai/whisper-large-v3",
        use_demucs=use_demucs,
        use_vad=True,
        device_spec="mps",
        device_type="mps",
        precision_candidates=("float16", "float32"),
        memory_tier="normal",
    )


def test_build_stable_whisper_transcribe_kwargs_prefers_denoiser_when_available() -> (
    None
):
    """Helper should set fp16 and denoiser fields when both are supported."""
    kwargs = build_stable_whisper_transcribe_kwargs(
        transcribe_callable=lambda **_kwargs: object(),
        runtime_request=_runtime_request(use_demucs=True),
        file_path="sample.wav",
        language="en",
        precision="float16",
        supports_keyword_argument=lambda _callable, name: name in {"fp16", "denoiser"},
    )

    assert kwargs == {
        "audio": "sample.wav",
        "language": "en",
        "verbose": False,
        "word_timestamps": True,
        "no_speech_threshold": None,
        "vad": True,
        "fp16": True,
        "denoiser": "demucs",
    }


def test_build_stable_whisper_transcribe_kwargs_uses_demucs_flag_when_supported() -> (
    None
):
    """Helper should use demucs=True/False when denoiser parameter is unavailable."""
    supports = lambda _callable, name: name in {"fp16", "demucs"}  # noqa: E731
    use_demucs_kwargs = build_stable_whisper_transcribe_kwargs(
        transcribe_callable=lambda **_kwargs: object(),
        runtime_request=_runtime_request(use_demucs=True),
        file_path="sample.wav",
        language="en",
        precision="float32",
        supports_keyword_argument=supports,
    )
    no_demucs_kwargs = build_stable_whisper_transcribe_kwargs(
        transcribe_callable=lambda **_kwargs: object(),
        runtime_request=_runtime_request(use_demucs=False),
        file_path="sample.wav",
        language="en",
        precision="float32",
        supports_keyword_argument=supports,
    )

    assert use_demucs_kwargs["fp16"] is False
    assert use_demucs_kwargs["demucs"] is True
    assert no_demucs_kwargs["fp16"] is False
    assert no_demucs_kwargs["demucs"] is False


def test_stable_whisper_adapter_build_transcribe_kwargs_delegates_to_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Adapter wrapper should delegate kwargs construction to extracted helper."""
    captured: dict[str, object] = {}
    sentinel_kwargs = {"audio": "delegated.wav", "language": "en", "vad": True}

    def _fake_build_kwargs(
        *,
        transcribe_callable: Callable[..., object],
        runtime_request: BackendRuntimeRequest,
        file_path: str,
        language: str,
        precision: str,
        supports_keyword_argument: Callable[[Callable[..., object], str], bool],
    ) -> dict[str, object]:
        captured["transcribe_callable"] = transcribe_callable
        captured["runtime_request"] = runtime_request
        captured["file_path"] = file_path
        captured["language"] = language
        captured["precision"] = precision
        captured["supports_fp16"] = supports_keyword_argument(
            transcribe_callable, "fp16"
        )
        return sentinel_kwargs

    monkeypatch.setattr(
        stable_whisper,
        "build_stable_whisper_transcribe_kwargs",
        _fake_build_kwargs,
    )

    def _transcribe(audio: str, language: str, verbose: bool, fp16: bool) -> object:
        del audio, language, verbose, fp16
        return object()

    kwargs = stable_whisper.StableWhisperAdapter._build_transcribe_kwargs(
        transcribe_callable=_transcribe,
        runtime_request=_runtime_request(use_demucs=False),
        file_path="clip.wav",
        language="en",
        precision="float16",
    )

    assert kwargs is sentinel_kwargs
    assert captured["file_path"] == "clip.wav"
    assert captured["language"] == "en"
    assert captured["precision"] == "float16"
    assert captured["supports_fp16"] is True
