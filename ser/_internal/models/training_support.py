"""Shared support helpers for public training boundary modules."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NamedTuple, cast

from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ser.config import AppConfig
from ser.data import Utterance
from ser.models.artifact_envelope import (
    MODEL_ARTIFACT_VERSION,
    LoadedModel,
)
from ser.models.artifact_envelope import deserialize_model_artifact as _deserialize_model_artifact
from ser.models.artifact_envelope import (
    normalize_model_artifact_metadata as _normalize_v2_artifact_metadata,
)
from ser.models.artifact_loading import (
    load_pickle_model_artifact,
    load_secure_model_artifact,
)
from ser.models.artifact_persistence import (
    persist_model_artifacts_for_settings,
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
from ser.models.training_preparation import (
    attach_grouped_metrics,
    evaluate_training_predictions,
    extract_normalized_artifact_metadata,
)
from ser.models.training_reporting import (
    build_training_report_for_settings,
)
from ser.runtime.schema import ARTIFACT_SCHEMA_VERSION

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


def create_classifier(settings: AppConfig) -> EmotionClassifier:
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


def build_training_report(
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
    globber: Callable[[str], list[str]] | None,
) -> dict[str, object]:
    """Builds a structured report using the current settings snapshot."""

    return build_training_report_for_settings(
        read_settings=lambda: settings,
        artifact_version=MODEL_ARTIFACT_VERSION,
        artifact_schema_version=ARTIFACT_SCHEMA_VERSION,
        globber=globber,
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


def extract_artifact_metadata(artifact: dict[str, object]) -> dict[str, object]:
    """Extracts normalized artifact metadata from one versioned envelope."""

    return extract_normalized_artifact_metadata(
        artifact,
        normalize_metadata=_normalize_v2_artifact_metadata,
    )


def persist_model_artifacts(
    model: EmotionClassifier,
    artifact: dict[str, object],
    *,
    settings: AppConfig,
    persist_pickle: Callable[[Path, dict[str, object]], None],
    persist_secure: Callable[[Path, EmotionClassifier], bool],
) -> PersistedArtifacts:
    """Persists model artifacts using the current settings destinations."""

    return persist_model_artifacts_for_settings(
        model,
        artifact,
        read_settings=lambda: settings,
        persist_pickle=persist_pickle,
        persist_secure=persist_secure,
        persisted_artifacts_factory=PersistedArtifacts,
    )


def build_dataset_controls(
    utterances: list[Utterance],
    *,
    settings: AppConfig,
) -> dict[str, object]:
    """Builds dataset controls using the current manifest and registry settings."""

    return build_dataset_controls_for_settings(
        utterances,
        read_settings=lambda: settings,
        resolve_registry_manifest_paths_for_settings=_resolve_registry_manifest_paths_impl,
    )


def attach_grouped_training_metrics(
    *,
    ser_metrics: dict[str, object],
    y_true: list[str],
    y_pred: list[str],
    test_meta: list[WindowMeta],
    min_support: int,
) -> dict[str, object]:
    """Attaches grouped corpus/language metrics to one SER metric payload."""

    return attach_grouped_metrics(
        ser_metrics=ser_metrics,
        y_true=y_true,
        y_pred=y_pred,
        sample_ids=[item.sample_id for item in test_meta],
        corpus_ids=[item.corpus for item in test_meta],
        language_ids=[item.language for item in test_meta],
        min_support=min_support,
    )


def load_secure_model(candidate: ModelCandidate, settings: AppConfig) -> LoadedModel:
    """Loads a secure artifact when `skops` is available and trusted."""

    assert candidate.artifact_format == "skops"
    return load_secure_model_artifact(
        candidate_path=candidate.path,
        model_instance_check=lambda payload: isinstance(payload, MLPClassifier | Pipeline),
        training_report_file=settings.models.training_report_file,
        read_training_report_feature_size=read_training_report_feature_size,
        loaded_model_factory=lambda payload, expected_feature_size: LoadedModel(
            model=cast(EmotionClassifier, payload),
            expected_feature_size=expected_feature_size,
        ),
    )


def load_pickle_model(candidate: ModelCandidate) -> LoadedModel:
    """Loads and validates the compatibility pickle model artifact."""

    assert candidate.artifact_format == "pickle"
    return load_pickle_model_artifact(
        candidate_path=candidate.path,
        deserialize_payload=_deserialize_model_artifact,
    )


def split_utterances(
    samples: list[Utterance],
    *,
    settings: AppConfig,
    logger: logging.Logger,
) -> tuple[list[Utterance], list[Utterance], MediumSplitMetadata]:
    """Splits utterances deterministically with manifest, speaker, and hash policy."""

    return _split_utterances_impl(
        samples=samples,
        settings=settings,
        logger=logger,
    )


def ensure_dataset_consents_for_training(
    *,
    utterances: list[Utterance],
    settings: AppConfig,
    logger: logging.Logger,
) -> None:
    """Enforces dataset policy and license acknowledgements before training."""

    _ensure_dataset_training_consents_impl(
        utterances=utterances,
        settings=settings,
        logger_warning=logger.warning,
        stdin_isatty=os.isatty,
        prompt_input=input,
        prompt_print=print,
    )


def group_metrics_min_support() -> int:
    """Returns the minimum sample support required for per-group metrics."""

    raw = os.getenv("SER_GROUP_METRICS_MIN_SUPPORT", "").strip()
    if not raw:
        return 20
    try:
        value = int(raw)
    except ValueError:
        return 20
    return max(1, value)


__all__ = [
    "ArtifactFormat",
    "EmotionClassifier",
    "ModelCandidate",
    "PersistedArtifacts",
    "WindowMeta",
    "attach_grouped_training_metrics",
    "build_dataset_controls",
    "build_training_report",
    "create_classifier",
    "ensure_dataset_consents_for_training",
    "evaluate_training_predictions",
    "extract_artifact_metadata",
    "group_metrics_min_support",
    "load_pickle_model",
    "load_secure_model",
    "persist_model_artifacts",
    "split_utterances",
]
