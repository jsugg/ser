"""Training and inference helpers for the SER emotion classification model."""

from __future__ import annotations

import glob
import logging
import os
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Literal, NamedTuple, cast

from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ser.config import (
    AppConfig,
    default_profile_model_id,
    get_settings,
)
from ser.data import (
    Utterance,
    load_data,
    load_utterances,
)
from ser.domain import EmotionSegment
from ser.features import extract_feature_frames
from ser.license_check import (
    build_provenance_metadata,
    ensure_backend_access,
    load_persisted_backend_consents,
    parse_allowed_restricted_backends_env,
)
from ser.models.accurate_feature_dataset import (
    build_accurate_feature_dataset as _build_accurate_feature_dataset_impl,
)
from ser.models.accurate_training_preparation import (
    PreparedAccurateTrainingRunner,
    build_prepared_accurate_training_runner,
    train_accurate_research_profile_entrypoint,
    train_accurate_research_profile_model,
    train_accurate_whisper_profile_entrypoint,
    train_accurate_whisper_profile_model,
)
from ser.models.artifact_envelope import (
    DEFAULT_BACKEND_ID,
    DEFAULT_PROFILE_ID,
    MODEL_ARTIFACT_VERSION,
    LoadedModel,
)
from ser.models.artifact_envelope import build_model_artifact as _build_model_artifact
from ser.models.artifact_envelope import (
    deserialize_model_artifact as _deserialize_model_artifact,
)
from ser.models.artifact_envelope import (
    normalize_model_artifact_metadata as _normalize_v2_artifact_metadata,
)
from ser.models.artifact_loading import (
    load_model_with_resolution,
    load_pickle_model_artifact,
    load_secure_model_artifact,
    resolve_model_for_loading_from_settings,
)
from ser.models.artifact_persistence import (
    persist_model_artifacts_for_settings,
    persist_pickle_artifact,
    persist_secure_artifact,
    persist_training_report,
    read_training_report_feature_size,
)
from ser.models.dataset_controls import (
    build_dataset_controls_for_settings,
)
from ser.models.dataset_controls import (
    resolve_registry_manifest_paths as _resolve_registry_manifest_paths_impl,
)
from ser.models.dataset_splitting import (
    MediumSplitMetadata,
)
from ser.models.dataset_splitting import split_utterances as _split_utterances_impl
from ser.models.dataset_training_consents import (
    ensure_dataset_training_consents as _ensure_dataset_training_consents_impl,
)
from ser.models.fast_path import (
    predict_emotions_detailed_with_model as _fast_predict_emotions_detailed_with_model,
)
from ser.models.fast_training import FastTrainingHooks, train_fast_model
from ser.models.medium_feature_dataset import (
    build_medium_feature_dataset as _build_medium_feature_dataset_impl,
)
from ser.models.medium_feature_dataset import (
    encode_medium_sequence as _encode_medium_sequence_impl,
)
from ser.models.medium_noise_controls import (
    apply_medium_noise_controls as _apply_medium_noise_controls_impl,
)
from ser.models.medium_noise_controls import (
    merge_medium_noise_stats as _merge_medium_noise_stats_impl,
)
from ser.models.medium_training_preparation import (
    train_medium_profile_entrypoint,
    train_medium_profile_model,
)
from ser.models.profile_runtime import (
    build_accurate_backend_for_settings as _build_accurate_backend_for_settings_impl,
)
from ser.models.profile_runtime import (
    build_accurate_research_backend_for_settings as _build_accurate_research_backend_for_settings_impl,
)
from ser.models.profile_runtime import (
    build_medium_backend_for_settings as _build_medium_backend_for_settings_impl,
)
from ser.models.profile_runtime import (
    resolve_accurate_model_id as _resolve_accurate_model_id_impl,
)
from ser.models.profile_runtime import (
    resolve_accurate_research_model_id as _resolve_accurate_research_model_id_impl,
)
from ser.models.profile_runtime import (
    resolve_medium_model_id as _resolve_medium_model_id_impl,
)
from ser.models.profile_runtime import (
    resolve_runtime_selectors_for_backend_id as _resolve_runtime_selectors_for_backend_id_impl,
)
from ser.models.training_orchestration import (
    attach_grouped_metrics,
    evaluate_training_predictions,
    extract_normalized_artifact_metadata,
)
from ser.models.training_reporting import (
    build_grouped_evaluation_controls,
    build_medium_noise_controls,
    build_training_report_for_settings,
)
from ser.repr import (
    Emotion2VecBackend,
    WhisperBackend,
    XLSRBackend,
)
from ser.runtime.schema import (
    ARTIFACT_SCHEMA_VERSION,
    OUTPUT_SCHEMA_VERSION,
    InferenceResult,
    to_legacy_emotion_segments,
)
from ser.utils.logger import get_logger

logger: logging.Logger = get_logger(__name__)

MEDIUM_BACKEND_ID = "hf_xlsr"
MEDIUM_PROFILE_ID = "medium"
MEDIUM_MODEL_ID = default_profile_model_id("medium")
MEDIUM_FRAME_SIZE_SECONDS = 1.0
MEDIUM_FRAME_STRIDE_SECONDS = 1.0
MEDIUM_POOLING_STRATEGY = "mean_std"
ACCURATE_BACKEND_ID = "hf_whisper"
ACCURATE_PROFILE_ID = "accurate"
ACCURATE_MODEL_ID = default_profile_model_id("accurate")
ACCURATE_POOLING_STRATEGY = "mean_std"
ACCURATE_RESEARCH_BACKEND_ID = "emotion2vec"
ACCURATE_RESEARCH_PROFILE_ID = "accurate-research"
ACCURATE_RESEARCH_MODEL_ID = default_profile_model_id("accurate-research")
type EmotionClassifier = MLPClassifier | Pipeline
type ArtifactFormat = Literal["pickle", "skops"]


class ModelCandidate(NamedTuple):
    """A candidate model artifact path and serialization format."""

    path: Path
    artifact_format: ArtifactFormat


@dataclass(frozen=True)
class PersistedArtifacts:
    """Paths to persisted model artifacts from training."""

    pickle_path: Path
    secure_path: Path | None


@dataclass(frozen=True)
class WindowMeta:
    """Window-level metadata for evaluation breakdowns."""

    sample_id: str
    corpus: str
    language: str


def _create_classifier(settings: AppConfig) -> EmotionClassifier:
    """Builds a reproducible scaler+MLP training pipeline."""
    validated_batch_size: int | Literal["auto"] = settings.nn.batch_size
    classifier: MLPClassifier = MLPClassifier(
        alpha=settings.nn.alpha,
        # NOTE: sklearn stubs narrow this to str; runtime accepts int | "auto".
        batch_size=cast(str, validated_batch_size),
        epsilon=settings.nn.epsilon,
        hidden_layer_sizes=settings.nn.hidden_layer_sizes,
        learning_rate=settings.nn.learning_rate,
        max_iter=settings.nn.max_iter,
        random_state=settings.nn.random_state,
    )
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", classifier),
        ]
    )


_persist_training_report = persist_training_report
_read_training_report_feature_size = read_training_report_feature_size
_build_grouped_evaluation_controls = build_grouped_evaluation_controls
_build_medium_noise_controls = build_medium_noise_controls
_evaluate_training_predictions = evaluate_training_predictions


def _build_training_report(
    *,
    accuracy: float,
    macro_f1: float,
    ser_metrics: dict[str, object],
    train_samples: int,
    test_samples: int,
    feature_vector_size: int,
    labels: list[str],
    artifacts: PersistedArtifacts,
    artifact_metadata: dict[str, object],
    data_controls: dict[str, object] | None = None,
    provenance: dict[str, object] | None = None,
    settings: AppConfig,
) -> dict[str, object]:
    """Builds a structured report using the current settings snapshot."""
    return build_training_report_for_settings(
        read_settings=lambda: settings,
        artifact_version=MODEL_ARTIFACT_VERSION,
        artifact_schema_version=ARTIFACT_SCHEMA_VERSION,
        globber=glob.glob,
        accuracy=accuracy,
        macro_f1=macro_f1,
        ser_metrics=ser_metrics,
        train_samples=train_samples,
        test_samples=test_samples,
        feature_vector_size=feature_vector_size,
        labels=labels,
        artifacts=artifacts,
        artifact_metadata=artifact_metadata,
        data_controls=data_controls,
        provenance=provenance,
    )


def _extract_artifact_metadata(artifact: dict[str, object]) -> dict[str, object]:
    """Extracts normalized artifact metadata from a versioned envelope."""
    return extract_normalized_artifact_metadata(
        artifact,
        normalize_metadata=_normalize_v2_artifact_metadata,
    )


def _persist_model_artifacts(
    model: EmotionClassifier,
    artifact: dict[str, object],
    *,
    settings: AppConfig,
) -> PersistedArtifacts:
    """Persists model artifacts using the current settings destinations."""
    return persist_model_artifacts_for_settings(
        model,
        artifact,
        read_settings=lambda: settings,
        persist_pickle=persist_pickle_artifact,
        persist_secure=persist_secure_artifact,
        persisted_artifacts_factory=PersistedArtifacts,
    )


def _build_dataset_controls(
    utterances: list[Utterance],
    *,
    settings: AppConfig,
) -> dict[str, object]:
    """Builds dataset controls using the current manifest/registry settings."""
    return build_dataset_controls_for_settings(
        utterances,
        read_settings=lambda: settings,
        resolve_registry_manifest_paths_for_settings=_resolve_registry_manifest_paths_impl,
    )


def _attach_grouped_training_metrics(
    *,
    ser_metrics: dict[str, object],
    y_true: list[str],
    y_pred: list[str],
    test_meta: list[WindowMeta],
    min_support: int,
) -> dict[str, object]:
    """Attaches grouped corpus/language metrics to SER metric payload."""
    return attach_grouped_metrics(
        ser_metrics=ser_metrics,
        y_true=y_true,
        y_pred=y_pred,
        sample_ids=[item.sample_id for item in test_meta],
        corpus_ids=[item.corpus for item in test_meta],
        language_ids=[item.language for item in test_meta],
        min_support=min_support,
    )


def _load_secure_model(candidate: ModelCandidate, settings: AppConfig) -> LoadedModel:
    """Loads a secure artifact when `skops` is available and trusted."""
    assert candidate.artifact_format == "skops"
    return load_secure_model_artifact(
        candidate_path=candidate.path,
        model_instance_check=lambda payload: isinstance(
            payload, MLPClassifier | Pipeline
        ),
        training_report_file=settings.models.training_report_file,
        read_training_report_feature_size=_read_training_report_feature_size,
        loaded_model_factory=lambda payload, expected_feature_size: LoadedModel(
            model=cast(EmotionClassifier, payload),
            expected_feature_size=expected_feature_size,
        ),
    )


def _load_pickle_model(candidate: ModelCandidate) -> LoadedModel:
    """Loads and validates the compatibility pickle model artifact."""
    assert candidate.artifact_format == "pickle"
    return load_pickle_model_artifact(
        candidate_path=candidate.path,
        deserialize_payload=_deserialize_model_artifact,
    )


def _split_utterances(
    samples: list[Utterance],
    *,
    settings: AppConfig,
) -> tuple[list[Utterance], list[Utterance], MediumSplitMetadata]:
    """Splits utterances deterministically with manifest/speaker/hash policy."""
    return _split_utterances_impl(
        samples=samples,
        settings=settings,
        logger=logger,
    )


def _ensure_dataset_consents_for_training(
    *,
    utterances: list[Utterance],
    settings: AppConfig,
) -> None:
    """Enforces dataset policy/license acknowledgements before training."""
    _ensure_dataset_training_consents_impl(
        utterances=utterances,
        settings=settings,
        logger_warning=logger.warning,
        stdin_isatty=os.isatty,
        prompt_input=input,
        prompt_print=print,
    )


def _group_metrics_min_support() -> int:
    """Minimum sample support required to report per-group metrics."""

    raw = os.getenv("SER_GROUP_METRICS_MIN_SUPPORT", "").strip()
    if not raw:
        return 20
    try:
        value = int(raw)
    except ValueError:
        return 20
    return max(1, value)


def train_medium_model() -> None:
    """Trains and persists medium-profile model artifacts with XLS-R metadata."""
    settings = get_settings()
    train_medium_profile_entrypoint(
        settings=settings,
        logger=logger,
        train_profile_model=train_medium_profile_model,
        load_utterances_for_training=load_utterances,
        ensure_dataset_consents_for_training=partial(
            _ensure_dataset_consents_for_training,
            settings=settings,
        ),
        split_utterances=partial(_split_utterances, settings=settings),
        resolve_model_id_for_settings=_resolve_medium_model_id_impl,
        resolve_runtime_selectors_for_backend=partial(
            _resolve_runtime_selectors_for_backend_id_impl,
            logger=logger,
        ),
        build_backend_for_settings=partial(
            _build_medium_backend_for_settings_impl,
            backend_factory=XLSRBackend,
        ),
        build_feature_dataset=partial(
            _build_medium_feature_dataset_impl,
            settings=settings,
            encode_sequence=partial(
                _encode_medium_sequence_impl,
                settings=settings,
                backend_id=MEDIUM_BACKEND_ID,
                logger=logger,
            ),
            apply_noise_controls=partial(
                _apply_medium_noise_controls_impl,
                min_window_std=settings.medium_training.min_window_std,
                max_windows_per_clip=settings.medium_training.max_windows_per_clip,
            ),
            merge_noise_stats=_merge_medium_noise_stats_impl,
            window_meta_factory=WindowMeta,
        ),
        create_classifier=lambda: _create_classifier(settings),
        min_support=_group_metrics_min_support(),
        evaluate_predictions=_evaluate_training_predictions,
        attach_grouped_metrics=_attach_grouped_training_metrics,
        build_model_artifact=_build_model_artifact,
        extract_artifact_metadata=_extract_artifact_metadata,
        persist_model_artifacts=partial(_persist_model_artifacts, settings=settings),
        build_provenance_metadata=build_provenance_metadata,
        build_dataset_controls=partial(_build_dataset_controls, settings=settings),
        build_medium_noise_controls=_build_medium_noise_controls,
        build_grouped_evaluation_controls=_build_grouped_evaluation_controls,
        build_training_report=partial(_build_training_report, settings=settings),
        persist_training_report=_persist_training_report,
        embedding_cache_name="medium_embeddings",
        profile_label="Medium",
        backend_id=MEDIUM_BACKEND_ID,
        profile_id=MEDIUM_PROFILE_ID,
        pooling_strategy=MEDIUM_POOLING_STRATEGY,
    )


def _build_prepared_accurate_profile_training_runner(
    settings: AppConfig,
) -> PreparedAccurateTrainingRunner:
    """Builds accurate-profile prepared-training runner from current collaborators."""
    return build_prepared_accurate_training_runner(
        logger=logger,
        create_classifier=lambda: _create_classifier(settings),
        min_support_resolver=_group_metrics_min_support,
        evaluate_predictions=_evaluate_training_predictions,
        attach_grouped_metrics=_attach_grouped_training_metrics,
        build_model_artifact=_build_model_artifact,
        extract_artifact_metadata=_extract_artifact_metadata,
        persist_model_artifacts=partial(_persist_model_artifacts, settings=settings),
        build_provenance_metadata=build_provenance_metadata,
        build_dataset_controls=partial(_build_dataset_controls, settings=settings),
        build_grouped_evaluation_controls=_build_grouped_evaluation_controls,
        build_training_report=partial(_build_training_report, settings=settings),
        persist_training_report=_persist_training_report,
    )


def train_accurate_model() -> None:
    """Trains and persists accurate-profile model artifacts with Whisper metadata."""
    settings = get_settings()
    train_accurate_whisper_profile_entrypoint(
        settings=settings,
        logger=logger,
        train_profile_model=train_accurate_whisper_profile_model,
        load_utterances_for_training=load_utterances,
        ensure_dataset_consents_for_training=partial(
            _ensure_dataset_consents_for_training,
            settings=settings,
        ),
        split_utterances=partial(_split_utterances, settings=settings),
        resolve_model_id_for_settings=_resolve_accurate_model_id_impl,
        resolve_runtime_selectors_for_backend=partial(
            _resolve_runtime_selectors_for_backend_id_impl,
            logger=logger,
        ),
        build_backend_for_settings=partial(
            _build_accurate_backend_for_settings_impl,
            backend_factory=WhisperBackend,
        ),
        build_feature_dataset_for_backend=partial(
            _build_accurate_feature_dataset_impl,
            settings=settings,
            accurate_backend_id=ACCURATE_BACKEND_ID,
            accurate_research_backend_id=ACCURATE_RESEARCH_BACKEND_ID,
            logger=logger,
            window_meta_factory=WindowMeta,
        ),
        run_prepared_accurate_profile_training=(
            _build_prepared_accurate_profile_training_runner(settings)
        ),
        backend_id=ACCURATE_BACKEND_ID,
        profile_id=ACCURATE_PROFILE_ID,
        profile_label="Accurate",
        frame_size_seconds=settings.accurate_runtime.pool_window_size_seconds,
        frame_stride_seconds=settings.accurate_runtime.pool_window_stride_seconds,
    )


def train_accurate_research_model() -> None:
    """Trains and persists accurate-research model artifacts with emotion2vec metadata."""
    settings = get_settings()
    train_accurate_research_profile_entrypoint(
        settings=settings,
        logger=logger,
        train_profile_model=train_accurate_research_profile_model,
        parse_allowed_restricted_backends_env=parse_allowed_restricted_backends_env,
        load_persisted_backend_consents=load_persisted_backend_consents,
        ensure_backend_access=ensure_backend_access,
        restricted_backend_id=ACCURATE_RESEARCH_BACKEND_ID,
        load_utterances_for_training=load_utterances,
        ensure_dataset_consents_for_training=partial(
            _ensure_dataset_consents_for_training,
            settings=settings,
        ),
        split_utterances=partial(_split_utterances, settings=settings),
        resolve_model_id_for_settings=_resolve_accurate_research_model_id_impl,
        resolve_runtime_selectors_for_backend=partial(
            _resolve_runtime_selectors_for_backend_id_impl,
            logger=logger,
        ),
        build_backend_for_settings=partial(
            _build_accurate_research_backend_for_settings_impl,
            backend_factory=Emotion2VecBackend,
        ),
        build_feature_dataset_for_backend=partial(
            _build_accurate_feature_dataset_impl,
            settings=settings,
            accurate_backend_id=ACCURATE_BACKEND_ID,
            accurate_research_backend_id=ACCURATE_RESEARCH_BACKEND_ID,
            logger=logger,
            window_meta_factory=WindowMeta,
        ),
        run_prepared_accurate_profile_training=(
            _build_prepared_accurate_profile_training_runner(settings)
        ),
        backend_id=ACCURATE_RESEARCH_BACKEND_ID,
        profile_id=ACCURATE_RESEARCH_PROFILE_ID,
        profile_label="Accurate-research",
        frame_size_seconds=settings.accurate_research_runtime.pool_window_size_seconds,
        frame_stride_seconds=settings.accurate_research_runtime.pool_window_stride_seconds,
    )


def train_model() -> None:
    """Trains the MLP classifier and persists model + training report artifacts.

    Raises:
        RuntimeError: If no training data could be loaded from the dataset path.
    """
    settings = get_settings()
    hooks = FastTrainingHooks(
        logger=logger,
        settings=settings,
        load_utterances=load_utterances,
        ensure_dataset_consents_for_training=partial(
            _ensure_dataset_consents_for_training,
            settings=settings,
        ),
        load_data=partial(load_data, settings=settings),
        create_classifier=lambda: _create_classifier(settings),
        evaluate_training_predictions=_evaluate_training_predictions,
        build_provenance_metadata=build_provenance_metadata,
        build_model_artifact=_build_model_artifact,
        extract_artifact_metadata=_extract_artifact_metadata,
        persist_model_artifacts=partial(_persist_model_artifacts, settings=settings),
        build_training_report=partial(_build_training_report, settings=settings),
        persist_training_report=_persist_training_report,
        default_backend_id=DEFAULT_BACKEND_ID,
        default_profile_id=DEFAULT_PROFILE_ID,
    )
    train_fast_model(hooks=hooks)


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
    active_settings = settings if settings is not None else get_settings()
    return load_model_with_resolution(
        settings=active_settings,
        settings_resolver=get_settings,
        resolve_model=partial(
            resolve_model_for_loading_from_settings,
            folder=active_settings.models.folder,
            secure_model_file=active_settings.models.secure_model_file,
            model_file=active_settings.models.model_file,
            candidate_factory=ModelCandidate,
            load_secure_model_for_settings=_load_secure_model,
            load_pickle_model=_load_pickle_model,
            logger=logger,
        ),
        logger=logger,
        expected_backend_id=expected_backend_id,
        expected_profile=expected_profile,
        expected_backend_model_id=expected_backend_model_id,
    )


def predict_emotions_detailed(
    file: str,
    *,
    loaded_model: LoadedModel | None = None,
) -> InferenceResult:
    """Runs inference and returns detailed frame + segment predictions."""
    active_loaded_model = loaded_model if loaded_model is not None else load_model()
    return _fast_predict_emotions_detailed_with_model(
        file,
        model=active_loaded_model.model,
        expected_feature_size=active_loaded_model.expected_feature_size,
        output_schema_version=OUTPUT_SCHEMA_VERSION,
        extract_feature_frames_fn=extract_feature_frames,
        logger=logger,
    )


def predict_emotions(
    file: str,
    *,
    loaded_model: LoadedModel | None = None,
) -> list[EmotionSegment]:
    """Compatibility wrapper returning legacy emotion segments."""
    inference = (
        predict_emotions_detailed(file)
        if loaded_model is None
        else predict_emotions_detailed(
            file,
            loaded_model=loaded_model,
        )
    )
    return to_legacy_emotion_segments(inference)
