"""Feature-runtime selector and cache-encoding helpers for profile training."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from ser.config import AppConfig, ProfileRuntimeConfig
from ser.data import EmbeddingCache
from ser.data.embedding_cache import EmbeddingCacheEntry
from ser.repr import EncodedSequence, FeatureBackend
from ser.repr.runtime_policy import resolve_feature_runtime_policy
from ser.utils.audio_utils import read_audio_file


class ReadAudioFile(Protocol):
    """Callable contract for loading one audio segment from disk."""

    def __call__(
        self,
        file_path: str,
        *,
        start_seconds: float | None = None,
        duration_seconds: float | None = None,
    ) -> tuple[NDArray[np.float32], int]: ...


class RuntimePolicyLike(Protocol):
    """Resolved runtime-policy contract used by selector helpers."""

    @property
    def device(self) -> str: ...

    @property
    def dtype(self) -> str: ...

    @property
    def reason(self) -> str: ...


def resolve_profile_runtime_selectors(
    *,
    backend_id: str,
    settings: AppConfig,
    logger: logging.Logger,
    resolve_policy: Callable[..., RuntimePolicyLike] = resolve_feature_runtime_policy,
) -> tuple[str, str]:
    """Resolves backend-aware runtime selectors for feature extraction."""
    backend_override_device: str | None = None
    backend_override_dtype: str | None = None
    feature_runtime_policy = getattr(settings, "feature_runtime_policy", None)
    resolve_backend_override = (
        getattr(feature_runtime_policy, "for_backend", None)
        if feature_runtime_policy is not None
        else None
    )
    if callable(resolve_backend_override):
        backend_override = resolve_backend_override(backend_id)
        if backend_override is not None:
            backend_override_device = getattr(backend_override, "device", None)
            backend_override_dtype = getattr(backend_override, "dtype", None)
    runtime_policy = resolve_policy(
        backend_id=backend_id,
        requested_device=settings.torch_runtime.device,
        requested_dtype=settings.torch_runtime.dtype,
        backend_override_device=backend_override_device,
        backend_override_dtype=backend_override_dtype,
    )
    if (
        runtime_policy.device != settings.torch_runtime.device
        or runtime_policy.dtype != settings.torch_runtime.dtype
    ):
        logger.info(
            "Feature runtime policy adjusted selectors for backend=%s "
            "(device=%s, dtype=%s, reason=%s).",
            backend_id,
            runtime_policy.device,
            runtime_policy.dtype,
            runtime_policy.reason,
        )
    return runtime_policy.device, runtime_policy.dtype


def resolve_accurate_runtime_config(
    *,
    settings: AppConfig,
    backend_id: str,
    accurate_backend_id: str,
    accurate_research_backend_id: str,
) -> ProfileRuntimeConfig:
    """Resolves runtime windowing config for one accurate-profile backend id."""
    if backend_id == accurate_backend_id:
        return settings.accurate_runtime
    if backend_id == accurate_research_backend_id:
        return settings.accurate_research_runtime
    raise ValueError(f"Unknown accurate backend id: {backend_id!r}.")


def resolve_effective_model_id(
    model_id: str | None,
    *,
    default_model_id: str,
) -> str:
    """Returns explicit model id or falls back to resolved default id."""
    resolved_model_id = (
        model_id.strip() if isinstance(model_id, str) and model_id.strip() else None
    )
    return default_model_id if resolved_model_id is None else resolved_model_id


def encode_sequence_with_cache(
    *,
    audio_path: str,
    start_seconds: float | None = None,
    duration_seconds: float | None = None,
    backend: FeatureBackend,
    cache: EmbeddingCache,
    backend_id: str,
    model_id: str,
    frame_size_seconds: float,
    frame_stride_seconds: float,
    log_prefix: str,
    logger: logging.Logger,
    read_audio: ReadAudioFile = read_audio_file,
) -> EncodedSequence:
    """Encodes one audio sequence and caches embeddings with runtime selectors."""

    def _compute_sequence() -> EncodedSequence:
        audio, sample_rate = read_audio(
            file_path=audio_path,
            start_seconds=start_seconds,
            duration_seconds=duration_seconds,
        )
        audio_array = np.asarray(audio, dtype=np.float32)
        return backend.encode_sequence(audio_array, sample_rate)

    cache_entry: EmbeddingCacheEntry = cache.get_or_compute(
        audio_path=audio_path,
        backend_id=backend_id,
        model_id=model_id,
        frame_size_seconds=frame_size_seconds,
        frame_stride_seconds=frame_stride_seconds,
        start_seconds=start_seconds,
        duration_seconds=duration_seconds,
        compute=_compute_sequence,
    )
    logger.debug(
        "%s embedding cache %s for %s (%s).",
        log_prefix,
        "hit" if cache_entry.cache_hit else "miss",
        audio_path,
        cache_entry.cache_key[:12],
    )
    return cache_entry.encoded


__all__ = [
    "encode_sequence_with_cache",
    "resolve_accurate_runtime_config",
    "resolve_effective_model_id",
    "resolve_profile_runtime_selectors",
]
