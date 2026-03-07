"""Torio/FFmpeg operational probe helpers for stable-whisper compatibility checks."""

from __future__ import annotations

import ctypes
import importlib
import importlib.util
import sys
from functools import lru_cache
from pathlib import Path
from typing import Protocol

from ser.transcript.backends.base import CompatibilityIssue
from ser.utils.transcription_compat import format_torio_ffmpeg_remediation


class ModuleAvailabilityProbe(Protocol):
    """Callable contract for module-availability checks."""

    def __call__(self, module_name: str, /) -> bool: ...


class ImportModuleCallable(Protocol):
    """Callable contract for dynamic module import."""

    def __call__(self, module_name: str, /) -> object: ...


class LoadDynamicLibraryCallable(Protocol):
    """Callable contract for dynamic-library loading probes."""

    def __call__(self, library_path: str, /) -> object: ...


class FormatRemediationCallable(Protocol):
    """Callable contract for lane-aware torio remediation formatting."""

    def __call__(self, *, missing_library: str | None) -> str: ...


def is_module_available(module_name: str) -> bool:
    """Returns whether one Python module is available or already loaded."""
    if module_name in sys.modules:
        return True
    try:
        return importlib.util.find_spec(module_name) is not None
    except ValueError:
        return module_name in sys.modules


def detect_torio_ffmpeg_operational_issue(
    *,
    is_module_available: ModuleAvailabilityProbe,
    import_module: ImportModuleCallable,
    load_dynamic_library: LoadDynamicLibraryCallable,
    format_remediation: FormatRemediationCallable,
) -> CompatibilityIssue | None:
    """Detects one non-blocking torio FFmpeg runtime issue."""
    if not is_module_available("torchaudio"):
        return None
    try:
        torchaudio_module = import_module("torchaudio")
    except Exception:
        return None
    list_backends = getattr(torchaudio_module, "list_audio_backends", None)
    if not callable(list_backends):
        return None
    try:
        available_backends = list_backends()
    except Exception:
        return None
    if not isinstance(available_backends, list | tuple):
        return None
    normalized_backends = frozenset(
        str(entry).strip().lower()
        for entry in available_backends
        if isinstance(entry, str) and entry.strip()
    )
    if "ffmpeg" in normalized_backends:
        return None
    loader_error = probe_torio_ffmpeg_loader_error(
        is_module_available=is_module_available,
        import_module=import_module,
        load_dynamic_library=load_dynamic_library,
    )
    if loader_error is None:
        remediation = format_remediation(missing_library=None)
        return CompatibilityIssue(
            code="torio_ffmpeg_extension_unavailable",
            message=(
                "torchaudio FFmpeg extension is unavailable in the active runtime; "
                "this is a non-blocking advisory and stable-whisper continues "
                f"with soundfile/sox backends. {remediation}"
            ),
            impact="informational",
        )
    missing_library = extract_missing_dynamic_library(loader_error)
    remediation = format_remediation(missing_library=missing_library)
    if missing_library is None:
        return CompatibilityIssue(
            code="torio_ffmpeg_extension_unavailable",
            message=(
                "torchaudio FFmpeg extension could not be loaded; "
                "this is a non-blocking advisory and stable-whisper continues "
                f"with soundfile/sox backends (loader_error={loader_error}). "
                f"{remediation}"
            ),
            impact="informational",
        )
    if "libavutil" not in missing_library:
        return CompatibilityIssue(
            code="torio_ffmpeg_extension_unavailable",
            message=(
                "torchaudio FFmpeg extension could not be loaded due to missing "
                f"{missing_library}; this is a non-blocking advisory and "
                f"stable-whisper continues with soundfile/sox backends. {remediation}"
            ),
            impact="informational",
        )
    return CompatibilityIssue(
        code="torio_ffmpeg_abi_mismatch",
        message=(
            "torchaudio FFmpeg extension could not be loaded because "
            f"{missing_library} is unavailable (FFmpeg ABI mismatch). "
            "This is a non-blocking advisory and stable-whisper continues "
            f"with soundfile/sox backends. {remediation}"
        ),
        impact="informational",
    )


def probe_torio_ffmpeg_loader_error(
    *,
    is_module_available: ModuleAvailabilityProbe,
    import_module: ImportModuleCallable,
    load_dynamic_library: LoadDynamicLibraryCallable,
) -> str | None:
    """Returns one deterministic torio FFmpeg loader error summary when available."""
    if not is_module_available("torio._extension.utils"):
        return None
    try:
        torio_utils = import_module("torio._extension.utils")
    except Exception:
        return None
    get_lib_path = getattr(torio_utils, "_get_lib_path", None)
    ffmpeg_versions = getattr(torio_utils, "_FFMPEG_VERS", ())
    if not callable(get_lib_path):
        return None
    if not isinstance(ffmpeg_versions, list | tuple):
        return None
    for version in ffmpeg_versions:
        if not isinstance(version, str):
            continue
        library_name = f"libtorio_ffmpeg{version}"
        try:
            library_path_value = get_lib_path(library_name)
        except Exception:
            continue
        library_path = Path(str(library_path_value))
        if not library_path.exists():
            continue
        try:
            load_dynamic_library(str(library_path))
        except OSError as err:
            error_text = str(err).strip()
            if error_text:
                return error_text
        except Exception:
            continue
    return None


def extract_missing_dynamic_library(error_text: str) -> str | None:
    """Extracts one missing dynamic library token from a loader error message."""
    marker = "Library not loaded: "
    marker_index = error_text.find(marker)
    if marker_index < 0:
        return None
    missing = error_text[marker_index + len(marker) :].splitlines()[0].strip()
    return missing or None


@lru_cache(maxsize=1)
def detect_default_torio_ffmpeg_operational_issue() -> CompatibilityIssue | None:
    """Detects one cached torio FFmpeg operational issue using default probes."""
    return detect_torio_ffmpeg_operational_issue(
        is_module_available=is_module_available,
        import_module=importlib.import_module,
        load_dynamic_library=ctypes.CDLL,
        format_remediation=format_torio_ffmpeg_remediation,
    )
