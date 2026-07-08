"""Accurate-profile backend/runtime setup helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Protocol

from ser.config import AppConfig, ProfileRuntimeConfig
from ser.repr import FeatureBackend


class RuntimePolicyLike(Protocol):
    """Runtime policy contract used during backend construction."""

    @property
    def device(self) -> str: ...

    @property
    def dtype(self) -> str: ...


def settings_with_torch_device(settings: AppConfig, *, device: str) -> AppConfig:
    """Returns process-safe settings with one normalized torch device selector."""
    return replace(
        settings,
        emotions=dict(settings.emotions),
        torch_runtime=replace(settings.torch_runtime, device=device),
    )


def runtime_config_for_profile(
    *,
    settings: AppConfig,
    expected_profile: str,
    unsupported_profile_error: Callable[[str], Exception],
) -> ProfileRuntimeConfig:
    """Returns runtime config for one accurate-profile variant."""
    if expected_profile == "accurate":
        return settings.accurate_runtime
    if expected_profile == "accurate-research":
        return settings.accurate_research_runtime
    raise unsupported_profile_error(f"Unsupported accurate runtime profile {expected_profile!r}.")


def build_backend_for_profile(
    *,
    expected_backend_id: str,
    expected_backend_model_id: str | None,
    settings: AppConfig,
    resolve_accurate_model_id: Callable[[AppConfig], str],
    resolve_accurate_research_model_id: Callable[[AppConfig], str],
    resolve_runtime_policy: Callable[..., RuntimePolicyLike],
    whisper_backend_factory: Callable[..., FeatureBackend],
    emotion2vec_backend_factory: Callable[..., FeatureBackend],
    unsupported_backend_error: Callable[[str], Exception],
) -> FeatureBackend:
    """Builds a feature backend aligned with profile/backend runtime expectations."""
    backend_override = settings.feature_runtime_policy.for_backend(expected_backend_id)
    runtime_policy = resolve_runtime_policy(
        backend_id=expected_backend_id,
        requested_device=settings.torch_runtime.device,
        requested_dtype=settings.torch_runtime.dtype,
        backend_override_device=(backend_override.device if backend_override is not None else None),
        backend_override_dtype=(backend_override.dtype if backend_override is not None else None),
    )
    if expected_backend_id == "hf_whisper":
        model_id = (
            expected_backend_model_id
            if expected_backend_model_id is not None
            else resolve_accurate_model_id(settings)
        )
        return whisper_backend_factory(
            model_id=model_id,
            cache_dir=settings.models.huggingface_cache_root,
            device=runtime_policy.device,
            dtype=runtime_policy.dtype,
        )
    if expected_backend_id == "emotion2vec":
        model_id = (
            expected_backend_model_id
            if expected_backend_model_id is not None
            else resolve_accurate_research_model_id(settings)
        )
        return emotion2vec_backend_factory(
            model_id=model_id,
            device=runtime_policy.device,
            modelscope_cache_root=settings.models.modelscope_cache_root,
            huggingface_cache_root=settings.models.huggingface_cache_root,
        )
    raise unsupported_backend_error(
        f"Unsupported accurate runtime backend id {expected_backend_id!r}."
    )


def prepare_accurate_backend_runtime(
    *,
    backend: FeatureBackend,
    is_dependency_error: Callable[[RuntimeError], bool],
    dependency_error_factory: Callable[[str], Exception],
    transient_error_factory: Callable[[str], Exception],
) -> None:
    """Warms backend runtime components so setup work is outside timeout budgets."""
    prepare_runtime = getattr(backend, "prepare_runtime", None)
    if not callable(prepare_runtime):
        return
    try:
        prepare_runtime()
    except RuntimeError as err:
        if is_dependency_error(err):
            raise dependency_error_factory(str(err)) from err
        raise transient_error_factory(str(err)) from err
