"""Dependency checks that do not import a transcription backend."""

import importlib.util
import platform
import sys

from ser.profiles import TranscriptionBackendId


def transcription_extra_message(backend_id: TranscriptionBackendId) -> str:
    """Names the extra and any platform restriction for this backend."""
    message = (
        f"Transcription backend '{backend_id}' requires `ser[transcript]` "
        "(`uv sync --extra transcript` in a checkout)."
    )
    if (
        backend_id == "stable_whisper"
        and sys.platform == "darwin"
        and platform.machine() == "x86_64"
        and sys.version_info >= (3, 13)
    ):
        message += " On Intel macOS with Python 3.13, use faster_whisper or Python 3.12."
    return message


def missing_transcription_dependencies(
    backend_id: TranscriptionBackendId, *, use_demucs: bool
) -> str | None:
    """Returns an install error for the selected backend's missing packages."""
    required: tuple[str, ...]
    if backend_id == "stable_whisper":
        required = ("stable_whisper", "whisper", "torch", "torchaudio")
        if use_demucs:
            required += ("demucs",)
    else:
        required = ("faster_whisper", "ctranslate2", "onnxruntime", "huggingface_hub")
    missing = tuple(
        name
        for name in required
        if sys.modules.get(name) is None and importlib.util.find_spec(name) is None
    )
    if not missing:
        return None
    return f"{transcription_extra_message(backend_id)} Missing: {', '.join(missing)}."
