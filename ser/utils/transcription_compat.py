"""Compatibility helpers for transcription backend runtime selection."""

from __future__ import annotations

import importlib.util
import platform
import sys
from functools import lru_cache
from pathlib import Path

FASTER_WHISPER_OPENMP_CONFLICT_ISSUE_CODE = "faster_whisper_openmp_runtime_conflict"
STABLE_WHISPER_SPARSE_MPS_INCOMPATIBLE_ISSUE_CODE = (
    "stable_whisper_sparse_mps_incompatible"
)

_STABLE_WHISPER_SUPPORTED_MODELS = frozenset(
    {
        "tiny.en",
        "tiny",
        "base.en",
        "base",
        "small.en",
        "small",
        "medium.en",
        "medium",
        "large-v1",
        "large-v2",
        "large-v3",
        "large",
        "large-v3-turbo",
        "turbo",
    }
)

_STABLE_WHISPER_MODEL_ALIASES: dict[str, str] = {
    "openai/whisper-large-v1": "large-v1",
    "openai/whisper-large-v2": "large-v2",
    "openai/whisper-large-v3": "large-v3",
    "openai/whisper-large-v3-turbo": "turbo",
}


def has_known_faster_whisper_openmp_runtime_conflict() -> bool:
    """Returns whether known faster-whisper OpenMP collision risk is present."""
    if sys.platform != "darwin":
        return False
    machine = platform.machine().lower()
    if machine not in {"x86_64", "amd64"}:
        return False
    if (
        "faster_whisper" in sys.modules
        and getattr(sys.modules["faster_whisper"], "__file__", None) is None
    ):
        return False
    if importlib.util.find_spec("faster_whisper") is None:
        return False
    ctranslate2_root = _module_root("ctranslate2")
    torch_root = _module_root("torch")
    if ctranslate2_root is None or torch_root is None:
        return False
    ctranslate2_openmp = ctranslate2_root / ".dylibs" / "libiomp5.dylib"
    torch_openmp_candidates = (
        torch_root.parent / "functorch" / ".dylibs" / "libiomp5.dylib",
        torch_root / ".dylibs" / "libiomp5.dylib",
    )
    return ctranslate2_openmp.is_file() and any(
        candidate.is_file() for candidate in torch_openmp_candidates
    )


@lru_cache(maxsize=1)
def has_known_stable_whisper_sparse_mps_incompatibility() -> bool:
    """Returns whether current torch MPS runtime cannot handle sparse allocations."""
    try:
        import torch
    except ModuleNotFoundError:
        return False
    backends = getattr(torch, "backends", None)
    mps_backend = getattr(backends, "mps", None)
    is_available = getattr(mps_backend, "is_available", None)
    is_built = getattr(mps_backend, "is_built", None)
    if not callable(is_available) or not callable(is_built):
        return False
    if not is_available() or not is_built():
        return False
    try:
        sample_sparse = torch.ones((1, 1), dtype=torch.float32).to_sparse()
        _ = sample_sparse.to(device="mps")
    except NotImplementedError as err:
        message = str(err).lower()
        return (
            "sparsemps" in message
            or "aten::empty.memory_format" in message
            or "sparse mps" in message
        )
    except Exception:
        return False
    return False


def resolve_stable_whisper_fallback_model_name(model_name: str) -> str:
    """Maps one requested model name to a stable-whisper compatible fallback."""
    normalized = model_name.strip()
    if not normalized:
        return "turbo"
    if normalized in _STABLE_WHISPER_SUPPORTED_MODELS:
        return normalized
    if normalized in _STABLE_WHISPER_MODEL_ALIASES:
        return _STABLE_WHISPER_MODEL_ALIASES[normalized]
    if normalized.startswith("distil-"):
        return "turbo"
    return "turbo"


def _module_root(module_name: str) -> Path | None:
    """Returns package root for one module spec if available."""
    try:
        spec = importlib.util.find_spec(module_name)
    except ValueError:
        return None
    if spec is None or spec.origin is None:
        return None
    return Path(spec.origin).resolve().parent
