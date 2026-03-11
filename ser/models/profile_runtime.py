"""Shared profile runtime selector/model-id helpers for model orchestration."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Final, Literal

from ser._internal.config.schema import default_profile_model_id
from ser.config import AppConfig, get_settings
from ser.models.feature_runtime_encoding import resolve_profile_runtime_selectors
from ser.repr import Emotion2VecBackend, WhisperBackend, XLSRBackend
from ser.repr.runtime_policy import resolve_feature_runtime_policy

MEDIUM_BACKEND_ID: Final[str] = "hf_xlsr"
MEDIUM_PROFILE_ID: Final[Literal["medium"]] = "medium"
MEDIUM_MODEL_ID = default_profile_model_id(MEDIUM_PROFILE_ID)
MEDIUM_FRAME_SIZE_SECONDS: Final[float] = 1.0
MEDIUM_FRAME_STRIDE_SECONDS: Final[float] = 1.0
MEDIUM_POOLING_STRATEGY: Final[str] = "mean_std"

ACCURATE_BACKEND_ID: Final[str] = "hf_whisper"
ACCURATE_PROFILE_ID: Final[Literal["accurate"]] = "accurate"
ACCURATE_MODEL_ID = default_profile_model_id(ACCURATE_PROFILE_ID)
ACCURATE_POOLING_STRATEGY: Final[str] = "mean_std"

ACCURATE_RESEARCH_BACKEND_ID: Final[str] = "emotion2vec"
ACCURATE_RESEARCH_PROFILE_ID: Final[Literal["accurate-research"]] = "accurate-research"
ACCURATE_RESEARCH_MODEL_ID = default_profile_model_id(ACCURATE_RESEARCH_PROFILE_ID)


def resolve_model_id_from_settings(
    *,
    settings: AppConfig | None,
    model_attr: str,
    fallback_model_id: str,
    read_settings: Callable[[], AppConfig] = get_settings,
) -> str:
    """Resolves one configured model id from settings with safe fallback."""
    active_settings: AppConfig = settings if settings is not None else read_settings()
    configured_model_id = getattr(
        getattr(active_settings, "models", None),
        model_attr,
        fallback_model_id,
    )
    if not isinstance(configured_model_id, str) or not configured_model_id.strip():
        return fallback_model_id
    return configured_model_id.strip()


def resolve_runtime_selectors_for_backend_id(
    *,
    settings: AppConfig,
    backend_id: str,
    logger: logging.Logger,
) -> tuple[str, str]:
    """Resolves backend-aware runtime selectors for one backend identifier."""
    return resolve_profile_runtime_selectors(
        backend_id=backend_id,
        settings=settings,
        logger=logger,
        resolve_policy=resolve_feature_runtime_policy,
    )


def resolve_medium_model_id(
    settings: AppConfig | None = None,
) -> str:
    """Resolves the medium XLS-R model id from settings with safe fallback."""
    return resolve_model_id_from_settings(
        settings=settings,
        model_attr="medium_model_id",
        fallback_model_id=default_profile_model_id("medium"),
        read_settings=get_settings,
    )


def resolve_accurate_model_id(
    settings: AppConfig | None = None,
) -> str:
    """Resolves the accurate Whisper model id from settings with safe fallback."""
    return resolve_model_id_from_settings(
        settings=settings,
        model_attr="accurate_model_id",
        fallback_model_id=default_profile_model_id("accurate"),
        read_settings=get_settings,
    )


def resolve_accurate_research_model_id(
    settings: AppConfig | None = None,
) -> str:
    """Resolves the accurate-research model id from settings with safe fallback."""
    return resolve_model_id_from_settings(
        settings=settings,
        model_attr="accurate_research_model_id",
        fallback_model_id=default_profile_model_id("accurate-research"),
        read_settings=get_settings,
    )


def build_medium_backend_for_settings(
    model_id: str,
    runtime_device: str,
    runtime_dtype: str,
    settings: AppConfig,
    backend_factory: Callable[..., XLSRBackend] = XLSRBackend,
) -> XLSRBackend:
    """Builds one medium XLS-R backend from settings/runtime selectors."""
    return backend_factory(
        model_id=model_id,
        cache_dir=settings.models.huggingface_cache_root,
        device=runtime_device,
        dtype=runtime_dtype,
    )


def build_accurate_backend_for_settings(
    model_id: str,
    runtime_device: str,
    runtime_dtype: str,
    settings: AppConfig,
    backend_factory: Callable[..., WhisperBackend] = WhisperBackend,
) -> WhisperBackend:
    """Builds one accurate Whisper backend from settings/runtime selectors."""
    return backend_factory(
        model_id=model_id,
        cache_dir=settings.models.huggingface_cache_root,
        device=runtime_device,
        dtype=runtime_dtype,
    )


def build_accurate_research_backend_for_settings(
    model_id: str,
    runtime_device: str,
    settings: AppConfig,
    backend_factory: Callable[..., Emotion2VecBackend] = Emotion2VecBackend,
) -> Emotion2VecBackend:
    """Builds one accurate-research emotion2vec backend from settings selectors."""
    return backend_factory(
        model_id=model_id,
        device=runtime_device,
        modelscope_cache_root=settings.models.modelscope_cache_root,
        huggingface_cache_root=settings.models.huggingface_cache_root,
    )


__all__ = [
    "ACCURATE_BACKEND_ID",
    "ACCURATE_MODEL_ID",
    "ACCURATE_POOLING_STRATEGY",
    "ACCURATE_PROFILE_ID",
    "ACCURATE_RESEARCH_BACKEND_ID",
    "ACCURATE_RESEARCH_MODEL_ID",
    "ACCURATE_RESEARCH_PROFILE_ID",
    "MEDIUM_BACKEND_ID",
    "MEDIUM_FRAME_SIZE_SECONDS",
    "MEDIUM_FRAME_STRIDE_SECONDS",
    "MEDIUM_MODEL_ID",
    "MEDIUM_POOLING_STRATEGY",
    "MEDIUM_PROFILE_ID",
    "build_accurate_backend_for_settings",
    "build_accurate_research_backend_for_settings",
    "build_medium_backend_for_settings",
    "resolve_accurate_model_id",
    "resolve_accurate_research_model_id",
    "resolve_medium_model_id",
    "resolve_model_id_from_settings",
    "resolve_runtime_selectors_for_backend_id",
]
