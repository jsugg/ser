"""Typed runtime environment deltas derived from explicit settings."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import cast

from ser.config import AppConfig

_PYTORCH_ENABLE_MPS_FALLBACK_ENV = "PYTORCH_ENABLE_MPS_FALLBACK"


@dataclass(frozen=True)
class ProcessEnvDelta:
    """Explicit process-environment variables required for one runtime seam."""

    values: Mapping[str, str]

    def merged(self, *others: ProcessEnvDelta) -> ProcessEnvDelta:
        """Returns one combined env delta with later values taking precedence."""
        merged_values = dict(self.values)
        for other in others:
            merged_values.update(other.values)
        return ProcessEnvDelta(MappingProxyType(merged_values))


@dataclass(frozen=True)
class RuntimeEnvironmentPlan:
    """Env deltas required by runtime policy and cache-owning adapters."""

    torch_runtime: ProcessEnvDelta
    stable_whisper: ProcessEnvDelta
    emotion2vec_modelscope: ProcessEnvDelta
    emotion2vec_huggingface: ProcessEnvDelta


def _frozen_env(values: dict[str, str]) -> Mapping[str, str]:
    """Freezes env-delta mappings for safe reuse across runtime seams."""
    return MappingProxyType(dict(values))


def _resolve_torch_cache_root(settings: AppConfig) -> Path:
    """Returns the torch cache root from full or test-double settings."""
    return settings.models.torch_cache_root


def _resolve_huggingface_cache_root(settings: AppConfig) -> Path:
    """Returns the Hugging Face cache root with a stable test-double fallback."""
    resolved_root = getattr(settings.models, "huggingface_cache_root", None)
    if resolved_root is not None:
        return cast(Path, resolved_root)
    return _resolve_torch_cache_root(settings) / "huggingface"


def _resolve_modelscope_cache_root(settings: AppConfig) -> Path:
    """Returns the ModelScope cache root with a stable test-double fallback."""
    resolved_root = getattr(settings.models, "modelscope_cache_root", None)
    if resolved_root is not None:
        return cast(Path, resolved_root)
    return _resolve_torch_cache_root(settings) / "modelscope"


def _resolve_enable_mps_fallback(settings: AppConfig) -> bool:
    """Returns the torch fallback flag, tolerating partial test doubles."""
    torch_runtime = getattr(settings, "torch_runtime", None)
    return bool(getattr(torch_runtime, "enable_mps_fallback", False))


def build_runtime_environment_plan(settings: AppConfig) -> RuntimeEnvironmentPlan:
    """Builds explicit process-env deltas from one immutable settings snapshot."""
    huggingface_cache_root = _resolve_huggingface_cache_root(settings)
    huggingface_hub_cache = huggingface_cache_root / "hub"
    return RuntimeEnvironmentPlan(
        torch_runtime=ProcessEnvDelta(
            _frozen_env(
                {
                    _PYTORCH_ENABLE_MPS_FALLBACK_ENV: (
                        "1" if _resolve_enable_mps_fallback(settings) else "0"
                    )
                }
            )
        ),
        stable_whisper=ProcessEnvDelta(
            _frozen_env({"TORCH_HOME": str(_resolve_torch_cache_root(settings))})
        ),
        emotion2vec_modelscope=ProcessEnvDelta(
            _frozen_env({"MODELSCOPE_CACHE": str(_resolve_modelscope_cache_root(settings))})
        ),
        emotion2vec_huggingface=ProcessEnvDelta(
            _frozen_env(
                {
                    "HF_HOME": str(huggingface_cache_root),
                    "HF_HUB_CACHE": str(huggingface_hub_cache),
                    "HUGGINGFACE_HUB_CACHE": str(huggingface_hub_cache),
                }
            )
        ),
    )


__all__ = [
    "ProcessEnvDelta",
    "RuntimeEnvironmentPlan",
    "build_runtime_environment_plan",
]
