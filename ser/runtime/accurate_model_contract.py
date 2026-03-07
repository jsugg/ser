"""Accurate runtime artifact-compatibility and selector-guard helpers."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Protocol

from ser.config import AppConfig
from ser.repr.runtime_policy import resolve_feature_runtime_policy


class ArtifactMetadataCarrier(Protocol):
    """Model-like contract exposing optional artifact metadata payload."""

    @property
    def artifact_metadata(self) -> dict[str, object] | None: ...


class RuntimeSelectorPolicy(Protocol):
    """Runtime policy contract required for mismatch diagnostics."""

    @property
    def device(self) -> str: ...

    @property
    def dtype(self) -> str: ...


def ensure_accurate_compatible_model(
    loaded_model: ArtifactMetadataCarrier,
    *,
    expected_backend_id: str,
    expected_profile: str,
    expected_backend_model_id: str | None,
    unavailable_error_factory: Callable[[str], Exception],
) -> None:
    """Validates that loaded artifact metadata matches accurate expectations."""
    metadata = loaded_model.artifact_metadata
    if not isinstance(metadata, dict):
        raise unavailable_error_factory(
            "Accurate profile requires a v2 model artifact metadata envelope. "
            "Train an accurate-profile model before inference."
        )

    backend_id = metadata.get("backend_id")
    if backend_id != expected_backend_id:
        raise unavailable_error_factory(
            "No accurate-profile model artifact is available. "
            f"Found backend_id={backend_id!r}; expected {expected_backend_id!r}."
        )

    profile = metadata.get("profile")
    if profile != expected_profile:
        raise unavailable_error_factory(
            "No accurate-profile model artifact is available. "
            f"Found profile={profile!r}; expected {expected_profile!r}."
        )

    if expected_backend_model_id is None:
        return

    backend_model_id = metadata.get("backend_model_id")
    if (
        not isinstance(backend_model_id, str)
        or backend_model_id.strip() != expected_backend_model_id
    ):
        raise unavailable_error_factory(
            "No accurate-profile model artifact is available. "
            f"Found backend_model_id={backend_model_id!r}; "
            f"expected {expected_backend_model_id!r}."
        )


def warn_on_runtime_selector_mismatch(
    *,
    loaded_model: ArtifactMetadataCarrier,
    backend_id: str,
    requested_device: str,
    requested_dtype: str,
    backend_override_device: str | None,
    backend_override_dtype: str | None,
    profile: str,
    logger: logging.Logger,
    resolve_runtime_policy: Callable[..., RuntimeSelectorPolicy] = (
        resolve_feature_runtime_policy
    ),
) -> None:
    """Warns when artifact torch selectors differ from current runtime policy."""
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

    runtime_policy = resolve_runtime_policy(
        backend_id=backend_id,
        requested_device=requested_device,
        requested_dtype=requested_dtype,
        backend_override_device=backend_override_device,
        backend_override_dtype=backend_override_dtype,
    )
    runtime_device = runtime_policy.device.strip().lower()
    runtime_dtype = runtime_policy.dtype.strip().lower()
    mismatch_components: list[str] = []
    if normalized_artifact_device != runtime_device:
        mismatch_components.append(
            f"device artifact={normalized_artifact_device!r} runtime={runtime_device!r}"
        )
    if normalized_artifact_dtype != runtime_dtype:
        mismatch_components.append(
            f"dtype artifact={normalized_artifact_dtype!r} runtime={runtime_dtype!r}"
        )
    if mismatch_components:
        logger.warning(
            "Artifact torch runtime selectors differ from current settings for %s "
            "profile (%s); embedding distribution may shift.",
            profile,
            ", ".join(mismatch_components),
        )


def validate_accurate_loaded_model_runtime_contract(
    loaded_model: ArtifactMetadataCarrier,
    *,
    settings: AppConfig,
    expected_backend_id: str,
    expected_profile: str,
    expected_backend_model_id: str | None,
    unavailable_error_factory: Callable[[str], Exception],
    logger: logging.Logger,
    resolve_runtime_policy: Callable[..., RuntimeSelectorPolicy] = (
        resolve_feature_runtime_policy
    ),
) -> None:
    """Validates artifact compatibility and runtime selector alignment together."""
    ensure_accurate_compatible_model(
        loaded_model,
        expected_backend_id=expected_backend_id,
        expected_profile=expected_profile,
        expected_backend_model_id=expected_backend_model_id,
        unavailable_error_factory=unavailable_error_factory,
    )
    runtime_override = settings.feature_runtime_policy.for_backend(expected_backend_id)
    warn_on_runtime_selector_mismatch(
        loaded_model=loaded_model,
        backend_id=expected_backend_id,
        requested_device=settings.torch_runtime.device,
        requested_dtype=settings.torch_runtime.dtype,
        backend_override_device=(
            runtime_override.device if runtime_override is not None else None
        ),
        backend_override_dtype=(
            runtime_override.dtype if runtime_override is not None else None
        ),
        profile=expected_profile,
        logger=logger,
        resolve_runtime_policy=resolve_runtime_policy,
    )


__all__ = [
    "ensure_accurate_compatible_model",
    "validate_accurate_loaded_model_runtime_contract",
    "warn_on_runtime_selector_mismatch",
]
