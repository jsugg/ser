"""Runtime policy resolution for transcription backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ser.profiles import TranscriptionBackendId
from ser.utils.torch_inference import maybe_resolve_torch_runtime

_SUPPORTED_DTYPE_SELECTORS: frozenset[str] = frozenset(
    {"auto", "float16", "float32", "bfloat16"}
)
DEFAULT_MPS_LOW_MEMORY_THRESHOLD_GB: float = 16.0
_BACKEND_SUPPORTED_PRECISIONS: dict[TranscriptionBackendId, frozenset[str]] = {
    "stable_whisper": frozenset({"float16", "float32"}),
    "faster_whisper": frozenset({"float16", "float32"}),
}
_BACKEND_SUPPORTED_DEVICE_TYPES: dict[TranscriptionBackendId, frozenset[str]] = {
    "stable_whisper": frozenset({"cpu", "cuda", "mps"}),
    "faster_whisper": frozenset({"cpu", "cuda"}),
}

MemoryTier = Literal["low", "high", "unknown", "not_applicable"]


@dataclass(frozen=True, slots=True)
class TranscriptionRuntimePolicy:
    """Resolved runtime selectors and precision fallback order for transcription."""

    device_spec: str
    device_type: str
    precision_candidates: tuple[str, ...]
    memory_tier: MemoryTier
    reason: str


def _normalize_dtype_selector(dtype: str) -> str:
    """Returns validated dtype selector or auto fallback."""
    normalized = dtype.strip().lower()
    return normalized if normalized in _SUPPORTED_DTYPE_SELECTORS else "auto"


def _normalize_mps_low_memory_threshold_gb(value: float) -> float:
    """Normalizes one configured MPS low-memory threshold value."""
    return value if value > 0.0 else DEFAULT_MPS_LOW_MEMORY_THRESHOLD_GB


def _resolve_mps_memory_tier(*, low_memory_threshold_gb: float) -> MemoryTier:
    """Classifies host memory tier for MPS precision ordering decisions."""
    try:
        import psutil
    except ModuleNotFoundError:
        return "unknown"
    try:
        total_memory_bytes = float(psutil.virtual_memory().total)
    except (AttributeError, OSError, ValueError):
        return "unknown"
    if total_memory_bytes <= 0.0:
        return "unknown"
    threshold_bytes = low_memory_threshold_gb * (1024.0**3)
    return "low" if total_memory_bytes <= threshold_bytes else "high"


def _dedupe_candidates(candidates: tuple[str, ...]) -> tuple[str, ...]:
    """Returns candidates with stable ordering and no duplicates."""
    deduped = tuple(dict.fromkeys(candidates))
    return deduped if deduped else ("float32",)


def _base_precision_candidates(
    *,
    device_type: str,
    requested_dtype: str,
    memory_tier: MemoryTier,
) -> tuple[str, ...]:
    """Returns backend-agnostic base precision fallback order."""
    if device_type == "cpu":
        return ("float32",)
    if requested_dtype == "float16":
        return ("float16",)
    if requested_dtype == "float32":
        return ("float32",)
    if requested_dtype == "bfloat16":
        return ("bfloat16", "float16", "float32")
    if device_type == "cuda":
        return ("float16", "float32")
    if device_type == "mps":
        del memory_tier
        return ("float16", "float32")
    return ("float32",)


def resolve_transcription_runtime_policy(
    *,
    backend_id: TranscriptionBackendId,
    requested_device: str,
    requested_dtype: str,
    mps_low_memory_threshold_gb: float = DEFAULT_MPS_LOW_MEMORY_THRESHOLD_GB,
) -> TranscriptionRuntimePolicy:
    """Resolves runtime device and precision fallback order for one backend."""
    normalized_dtype = _normalize_dtype_selector(requested_dtype)
    normalized_mps_threshold_gb = _normalize_mps_low_memory_threshold_gb(
        mps_low_memory_threshold_gb
    )
    reason_tokens: list[str] = []
    memory_tier: MemoryTier = "not_applicable"
    device_spec = "cpu"
    device_type = "cpu"
    try:
        runtime = maybe_resolve_torch_runtime(device=requested_device, dtype="float32")
    except RuntimeError as err:
        reason_tokens.append(f"device_resolution_fallback_cpu({err})")
        runtime = None
    if runtime is None:
        reason_tokens.append("torch_runtime_unavailable")
    else:
        device_spec = runtime.device_spec
        device_type = runtime.device_type

    supported_device_types = _BACKEND_SUPPORTED_DEVICE_TYPES[backend_id]
    if device_type not in supported_device_types:
        reason_tokens.append(
            f"backend_{backend_id}_does_not_support_device_{device_type}"
        )
        device_spec = "cpu"
        device_type = "cpu"

    if device_type == "mps":
        memory_tier = _resolve_mps_memory_tier(
            low_memory_threshold_gb=normalized_mps_threshold_gb,
        )
        reason_tokens.append(f"mps_memory_tier_{memory_tier}_informational_only")

    base_candidates = _base_precision_candidates(
        device_type=device_type,
        requested_dtype=normalized_dtype,
        memory_tier=memory_tier,
    )
    supported_precisions = _BACKEND_SUPPORTED_PRECISIONS[backend_id]
    filtered_candidates = tuple(
        precision for precision in base_candidates if precision in supported_precisions
    )
    if not filtered_candidates:
        filtered_candidates = ("float32",)
        reason_tokens.append("precision_fallback_float32")
    elif len(filtered_candidates) != len(base_candidates):
        reason_tokens.append("precision_candidates_filtered_by_backend_capabilities")

    if not reason_tokens:
        if normalized_dtype == "auto":
            reason_tokens.append("auto_precision_policy")
        else:
            reason_tokens.append(f"explicit_precision_{normalized_dtype}")

    return TranscriptionRuntimePolicy(
        device_spec=device_spec,
        device_type=device_type,
        precision_candidates=_dedupe_candidates(filtered_candidates),
        memory_tier=memory_tier,
        reason=";".join(reason_tokens),
    )
