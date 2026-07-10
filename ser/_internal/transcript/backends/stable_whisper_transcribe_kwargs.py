"""Transcribe kwargs composition helpers for stable-whisper."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ser.transcript.backends.base import BackendRuntimeRequest


def build_stable_whisper_transcribe_kwargs(
    *,
    transcribe_callable: Callable[..., object],
    runtime_request: BackendRuntimeRequest,
    file_path: str,
    language: str,
    precision: str,
    supports_keyword_argument: Callable[[Callable[..., object], str], bool],
) -> dict[str, object]:
    """Builds stable-whisper transcribe kwargs for cross-version compatibility."""
    kwargs: dict[str, object] = {
        "audio": file_path,
        "language": language,
        "verbose": False,
        "word_timestamps": True,
        "no_speech_threshold": None,
        "vad": runtime_request.use_vad,
    }
    if supports_keyword_argument(transcribe_callable, "fp16"):
        kwargs["fp16"] = precision == "float16"
    if runtime_request.use_demucs:
        if supports_keyword_argument(transcribe_callable, "denoiser"):
            kwargs["denoiser"] = "demucs"
        elif supports_keyword_argument(transcribe_callable, "demucs"):
            kwargs["demucs"] = True
    elif supports_keyword_argument(transcribe_callable, "demucs"):
        kwargs["demucs"] = False
    return kwargs


__all__ = ["build_stable_whisper_transcribe_kwargs"]
