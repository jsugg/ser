"""Medium-profile backend/runtime setup helpers."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import replace
from typing import Protocol

from ser.config import AppConfig
from ser.repr import XLSRBackend
from ser.repr.runtime_policy import FeatureRuntimePolicy, resolve_feature_runtime_policy


class LoadedModelLike(Protocol):
    """Structural contract for loaded model metadata access."""

    @property
    def artifact_metadata(self) -> dict[str, object] | None: ...


def settings_with_torch_runtime(
    settings: AppConfig,
    *,
    device: str,
    dtype: str,
) -> AppConfig:
    """Returns process-safe settings with one normalized torch runtime selector pair."""
    return replace(
        settings,
        emotions=dict(settings.emotions),
        torch_runtime=replace(
            settings.torch_runtime,
            device=device,
            dtype=dtype,
        ),
    )


def resolve_medium_feature_runtime_policy(
    *,
    settings: AppConfig,
) -> FeatureRuntimePolicy:
    """Resolves backend-aware feature runtime selectors for medium profile."""
    backend_override = settings.feature_runtime_policy.for_backend("hf_xlsr")
    return resolve_feature_runtime_policy(
        backend_id="hf_xlsr",
        requested_device=settings.torch_runtime.device,
        requested_dtype=settings.torch_runtime.dtype,
        backend_override_device=(
            backend_override.device if backend_override is not None else None
        ),
        backend_override_dtype=(
            backend_override.dtype if backend_override is not None else None
        ),
    )


def build_medium_backend(
    *,
    settings: AppConfig,
    expected_backend_model_id: str,
    runtime_device: str,
    runtime_dtype: str,
    backend_factory: Callable[..., XLSRBackend],
) -> XLSRBackend:
    """Builds one XLS-R backend with explicit runtime selectors."""
    return backend_factory(
        model_id=expected_backend_model_id,
        cache_dir=settings.models.huggingface_cache_root,
        device=runtime_device,
        dtype=runtime_dtype,
    )


def prepare_medium_backend_runtime(
    *,
    backend: XLSRBackend,
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


def warn_on_runtime_selector_mismatch(
    *,
    loaded_model: LoadedModelLike,
    profile: str,
    runtime_device: str,
    runtime_dtype: str,
    logger: logging.Logger,
) -> None:
    """Warns when artifact runtime selectors differ from current runtime settings."""
    metadata = loaded_model.artifact_metadata
    if not isinstance(metadata, dict):
        return

    artifact_torch_device = metadata.get("torch_device")
    artifact_torch_dtype = metadata.get("torch_dtype")
    if not isinstance(artifact_torch_device, str) or not isinstance(
        artifact_torch_dtype, str
    ):
        return

    normalized_artifact_device = artifact_torch_device.strip().lower()
    normalized_artifact_dtype = artifact_torch_dtype.strip().lower()
    if not normalized_artifact_device or not normalized_artifact_dtype:
        return

    normalized_runtime_device = runtime_device.strip().lower()
    normalized_runtime_dtype = runtime_dtype.strip().lower()
    mismatch_components: list[str] = []
    if normalized_artifact_device != normalized_runtime_device:
        mismatch_components.append(
            "device artifact="
            f"{normalized_artifact_device!r} runtime={normalized_runtime_device!r}"
        )
    if normalized_artifact_dtype != normalized_runtime_dtype:
        mismatch_components.append(
            "dtype artifact="
            f"{normalized_artifact_dtype!r} runtime={normalized_runtime_dtype!r}"
        )
    if mismatch_components:
        logger.warning(
            "Artifact torch runtime selectors differ from current settings for %s "
            "profile (%s); embedding distribution may shift.",
            profile,
            ", ".join(mismatch_components),
        )
