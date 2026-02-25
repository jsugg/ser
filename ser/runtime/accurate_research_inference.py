"""Accurate-research inference runner backed by Emotion2Vec features."""

from __future__ import annotations

from ser.config import AppConfig
from ser.models.emotion_model import LoadedModel, resolve_accurate_research_model_id
from ser.repr import Emotion2VecBackend, FeatureBackend
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
    active_backend: FeatureBackend = (
        backend
        if backend is not None
        else Emotion2VecBackend(
            model_id=accurate_research_model_id,
            modelscope_cache_root=settings.models.modelscope_cache_root,
            huggingface_cache_root=settings.models.huggingface_cache_root,
        )
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
