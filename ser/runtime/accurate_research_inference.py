"""Accurate-research inference runner backed by Emotion2Vec features."""

from __future__ import annotations

from ser.config import AppConfig
from ser.models.emotion_model import LoadedModel, resolve_accurate_research_model_id
from ser.repr import Emotion2VecBackend, FeatureBackend
from ser.repr.runtime_policy import resolve_feature_runtime_policy
from ser.runtime.accurate_inference import run_accurate_inference
from ser.runtime.contracts import InferenceRequest
from ser.runtime.schema import InferenceResult


def run_accurate_research_inference(
    request: InferenceRequest,
    settings: AppConfig,
    *,
    loaded_model: LoadedModel | None = None,
    backend: FeatureBackend | None = None,
    enforce_timeout: bool = True,
    allow_retries: bool = True,
) -> InferenceResult:
    """Runs accurate-research inference with emotion2vec metadata compatibility checks."""
    accurate_research_model_id = resolve_accurate_research_model_id(settings)
    if backend is not None:
        active_backend: FeatureBackend = backend
    else:
        backend_override = settings.feature_runtime_policy.for_backend("emotion2vec")
        runtime_policy = resolve_feature_runtime_policy(
            backend_id="emotion2vec",
            requested_device=settings.torch_runtime.device,
            requested_dtype=settings.torch_runtime.dtype,
            backend_override_device=(
                backend_override.device if backend_override is not None else None
            ),
            backend_override_dtype=(
                backend_override.dtype if backend_override is not None else None
            ),
        )
        active_backend = Emotion2VecBackend(
            model_id=accurate_research_model_id,
            device=runtime_policy.device,
            modelscope_cache_root=settings.models.modelscope_cache_root,
            huggingface_cache_root=settings.models.huggingface_cache_root,
        )
    return run_accurate_inference(
        request=request,
        settings=settings,
        loaded_model=loaded_model,
        backend=active_backend,
        enforce_timeout=enforce_timeout,
        allow_retries=allow_retries,
        expected_backend_id="emotion2vec",
        expected_profile="accurate-research",
        expected_backend_model_id=accurate_research_model_id,
    )
