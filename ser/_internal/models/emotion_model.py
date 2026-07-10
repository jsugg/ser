"""Training and inference helpers for the SER emotion classification model."""

from __future__ import annotations

import logging
from functools import partial

import ser._internal.models.model_loading as _model_loading_entrypoints
import ser.models.artifact_envelope as _artifact_envelope
import ser.models.profile_runtime as _profile_runtime
import ser.models.training_support as _training_support
from ser._internal.models.model_loading import ResolveModelFn
from ser._internal.models.model_loading import load_model as _load_model_entrypoint
from ser._internal.runtime.environment_plan import build_runtime_environment_plan
from ser._internal.runtime.process_env import temporary_process_env
from ser.config import AppConfig, reload_settings
from ser.domain import EmotionSegment
from ser.features import extract_feature_frames
from ser.models import training_entrypoints as _training_entrypoints
from ser.models.artifact_envelope import LoadedModel
from ser.models.artifact_loading import (
    load_model_with_resolution,
    resolve_model_for_loading_from_settings,
)
from ser.models.fast_path import (
    predict_emotions_detailed_with_model as _fast_predict_emotions_detailed_with_model,
)
from ser.runtime.schema import OUTPUT_SCHEMA_VERSION, InferenceResult, to_legacy_emotion_segments
from ser.utils.logger import get_logger

logger: logging.Logger = get_logger(__name__)

DEFAULT_BACKEND_ID = _artifact_envelope.DEFAULT_BACKEND_ID
DEFAULT_PROFILE_ID = _artifact_envelope.DEFAULT_PROFILE_ID
MODEL_ARTIFACT_VERSION = _artifact_envelope.MODEL_ARTIFACT_VERSION

MEDIUM_BACKEND_ID = _profile_runtime.MEDIUM_BACKEND_ID
MEDIUM_PROFILE_ID = _profile_runtime.MEDIUM_PROFILE_ID
MEDIUM_MODEL_ID = _profile_runtime.MEDIUM_MODEL_ID
MEDIUM_FRAME_SIZE_SECONDS = _profile_runtime.MEDIUM_FRAME_SIZE_SECONDS
MEDIUM_FRAME_STRIDE_SECONDS = _profile_runtime.MEDIUM_FRAME_STRIDE_SECONDS
MEDIUM_POOLING_STRATEGY = _profile_runtime.MEDIUM_POOLING_STRATEGY
ACCURATE_BACKEND_ID = _profile_runtime.ACCURATE_BACKEND_ID
ACCURATE_PROFILE_ID = _profile_runtime.ACCURATE_PROFILE_ID
ACCURATE_MODEL_ID = _profile_runtime.ACCURATE_MODEL_ID
ACCURATE_POOLING_STRATEGY = _profile_runtime.ACCURATE_POOLING_STRATEGY
ACCURATE_RESEARCH_BACKEND_ID = _profile_runtime.ACCURATE_RESEARCH_BACKEND_ID
ACCURATE_RESEARCH_PROFILE_ID = _profile_runtime.ACCURATE_RESEARCH_PROFILE_ID
ACCURATE_RESEARCH_MODEL_ID = _profile_runtime.ACCURATE_RESEARCH_MODEL_ID


def _resolve_boundary_settings(settings: AppConfig | None) -> AppConfig:
    """Returns explicit settings or reloads a boundary-local settings snapshot."""
    return settings if settings is not None else reload_settings()


def train_medium_model(settings: AppConfig | None = None) -> None:
    """Trains and persists medium-profile model artifacts with XLS-R metadata."""
    _training_entrypoints.train_medium_model(
        settings=_resolve_boundary_settings(settings),
    )


def train_accurate_model(settings: AppConfig | None = None) -> None:
    """Trains and persists accurate-profile model artifacts with Whisper metadata."""
    _training_entrypoints.train_accurate_model(
        settings=_resolve_boundary_settings(settings),
    )


def train_accurate_research_model(settings: AppConfig | None = None) -> None:
    """Trains and persists accurate-research model artifacts with emotion2vec metadata."""
    _training_entrypoints.train_accurate_research_model(
        settings=_resolve_boundary_settings(settings),
    )


def train_model(settings: AppConfig | None = None) -> None:
    """Trains the MLP classifier and persists model + training report artifacts.

    Raises:
        RuntimeError: If no training data could be loaded from the dataset path.
    """

    _training_entrypoints.train_model(
        settings=_resolve_boundary_settings(settings),
    )


def _resolve_model_for_loading(
    settings: AppConfig,
) -> ResolveModelFn:
    """Builds the settings-aware model resolver for artifact loading."""
    return _model_loading_entrypoints.resolve_model_for_loading_from_public_boundary(
        settings,
        resolve_model_for_loading_from_settings_fn=resolve_model_for_loading_from_settings,
        load_secure_model_for_settings_fn=_training_support.load_secure_model,
        load_pickle_model_fn=_training_support.load_pickle_model,
        logger=logger,
    )


def load_model(
    settings: AppConfig | None = None,
    *,
    expected_backend_id: str | None = None,
    expected_profile: str | None = None,
    expected_backend_model_id: str | None = None,
) -> LoadedModel:
    """Loads the serialized SER model from disk.

    Args:
        settings: Optional settings snapshot used to resolve model/report paths.
        expected_backend_id: Optional backend-id compatibility filter.
        expected_profile: Optional profile compatibility filter.
        expected_backend_model_id: Optional backend-model-id compatibility filter.

    Returns:
        The loaded model plus expected feature-vector size.

    Raises:
        FileNotFoundError: If no trained model artifact could be found.
        ValueError: If model artifacts exist but none can be deserialized.
    """
    active_settings = _resolve_boundary_settings(settings)
    return _load_model_entrypoint(
        active_settings,
        settings_resolver=lambda: active_settings,
        resolve_model_factory=_resolve_model_for_loading,
        logger=logger,
        build_runtime_environment_plan_fn=build_runtime_environment_plan,
        temporary_process_env_fn=temporary_process_env,
        load_model_with_resolution_fn=load_model_with_resolution,
        expected_backend_id=expected_backend_id,
        expected_profile=expected_profile,
        expected_backend_model_id=expected_backend_model_id,
    )


def predict_emotions_detailed(
    file: str,
    *,
    loaded_model: LoadedModel | None = None,
    settings: AppConfig | None = None,
) -> InferenceResult:
    """Runs inference and returns detailed frame + segment predictions."""
    active_settings = _resolve_boundary_settings(settings)
    active_loaded_model = (
        loaded_model if loaded_model is not None else load_model(settings=active_settings)
    )
    runtime_environment = build_runtime_environment_plan(active_settings)
    with temporary_process_env(runtime_environment.torch_runtime):
        return _fast_predict_emotions_detailed_with_model(
            file,
            model=active_loaded_model.model,
            expected_feature_size=active_loaded_model.expected_feature_size,
            output_schema_version=OUTPUT_SCHEMA_VERSION,
            extract_feature_frames_fn=partial(
                extract_feature_frames,
                settings=active_settings,
            ),
            logger=logger,
        )


def predict_emotions(
    file: str,
    *,
    loaded_model: LoadedModel | None = None,
    settings: AppConfig | None = None,
) -> list[EmotionSegment]:
    """Returns legacy emotion segments derived from the detailed inference result."""
    return to_legacy_emotion_segments(
        predict_emotions_detailed(
            file,
            loaded_model=loaded_model,
            settings=settings,
        )
    )
