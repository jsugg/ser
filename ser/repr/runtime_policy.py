"""Backend-aware runtime policy resolution for emotion feature extractors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ser.utils.torch_inference import TorchRuntime, maybe_resolve_torch_runtime

type FeatureBackendId = Literal["handcrafted", "hf_xlsr", "hf_whisper", "emotion2vec"]

_KNOWN_BACKEND_IDS: frozenset[str] = frozenset(
    {"handcrafted", "hf_xlsr", "hf_whisper", "emotion2vec"}
)
_KNOWN_DTYPE_SELECTORS: frozenset[str] = frozenset(
    {"auto", "float16", "float32", "bfloat16"}
)


@dataclass(frozen=True)
class FeatureRuntimePolicy:
    """Resolved runtime selectors for one feature-backend invocation."""

    device: str
    dtype: str
    reason: str


def resolve_feature_runtime_policy(
    *,
    backend_id: str,
    requested_device: str,
    requested_dtype: str,
    backend_override_device: str | None = None,
    backend_override_dtype: str | None = None,
) -> FeatureRuntimePolicy:
    """Resolves backend-safe runtime selectors for feature extraction.

    The resolver applies backend capabilities first, then caller intent.
    """
    normalized_backend = backend_id.strip().lower()
    normalized_device = _normalize_optional_device_selector(
        backend_override_device
    ) or _normalize_device_selector(requested_device)
    normalized_dtype = _normalize_optional_dtype_selector(
        backend_override_dtype
    ) or _normalize_dtype_selector(requested_dtype)

    if normalized_backend not in _KNOWN_BACKEND_IDS:
        return FeatureRuntimePolicy(
            device=normalized_device,
            dtype=normalized_dtype,
            reason="unknown_backend_passthrough",
        )

    if normalized_backend == "handcrafted":
        return FeatureRuntimePolicy(
            device="cpu",
            dtype="float32",
            reason="handcrafted_cpu_only",
        )

    if normalized_backend == "emotion2vec":
        return _resolve_emotion2vec_policy(
            requested_device=normalized_device,
            requested_dtype=normalized_dtype,
        )

    if normalized_backend == "hf_xlsr":
        return _resolve_xlsr_policy(
            requested_device=normalized_device,
            requested_dtype=normalized_dtype,
        )

    return FeatureRuntimePolicy(
        device=normalized_device,
        dtype=normalized_dtype,
        reason="torch_backend_requested_selectors",
    )


def _resolve_xlsr_policy(
    *,
    requested_device: str,
    requested_dtype: str,
) -> FeatureRuntimePolicy:
    """Resolves safe runtime selectors for XLS-R feature extraction."""
    runtime = _probe_runtime(requested_device)
    if runtime is None:
        return FeatureRuntimePolicy(
            device="cpu",
            dtype="float32",
            reason="torch_runtime_unavailable",
        )
    if runtime.device_type == "mps":
        return FeatureRuntimePolicy(
            device="cpu",
            dtype="float32",
            reason="hf_xlsr_mps_stability_guard",
        )
    return FeatureRuntimePolicy(
        device=requested_device,
        dtype=requested_dtype,
        reason="hf_xlsr_requested_selectors",
    )


def _resolve_emotion2vec_policy(
    *,
    requested_device: str,
    requested_dtype: str,
) -> FeatureRuntimePolicy:
    """Resolves runtime selectors for Emotion2Vec/FunASR backend."""
    runtime = _probe_runtime(requested_device)
    if runtime is None:
        return FeatureRuntimePolicy(
            device="cpu",
            dtype="float32",
            reason="torch_runtime_unavailable",
        )
    if runtime.device_type == "cuda":
        return FeatureRuntimePolicy(
            device=runtime.device_spec,
            dtype=requested_dtype,
            reason="emotion2vec_cuda_enabled",
        )
    return FeatureRuntimePolicy(
        device="cpu",
        dtype="float32",
        reason="emotion2vec_cpu_default",
    )


def _probe_runtime(requested_device: str) -> TorchRuntime | None:
    """Best-effort runtime probe for one device selector."""
    try:
        return maybe_resolve_torch_runtime(device=requested_device, dtype="float32")
    except RuntimeError:
        return None


def _normalize_device_selector(value: str) -> str:
    """Normalizes one runtime device selector."""
    normalized = value.strip().lower()
    if normalized in {"auto", "cpu", "mps", "cuda"}:
        return normalized
    if normalized.startswith("cuda:"):
        return normalized
    return "auto"


def _normalize_dtype_selector(value: str) -> str:
    """Normalizes one runtime dtype selector."""
    normalized = value.strip().lower()
    if normalized in _KNOWN_DTYPE_SELECTORS:
        return normalized
    return "auto"


def _normalize_optional_device_selector(value: str | None) -> str | None:
    """Normalizes an optional runtime device selector override."""
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = value.strip().lower()
    if normalized in {"auto", "cpu", "mps", "cuda"}:
        return normalized
    if normalized.startswith("cuda:"):
        return normalized
    return None


def _normalize_optional_dtype_selector(value: str | None) -> str | None:
    """Normalizes an optional runtime dtype selector override."""
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = value.strip().lower()
    if normalized in _KNOWN_DTYPE_SELECTORS:
        return normalized
    return None
