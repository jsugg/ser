"""Accurate-profile prepared-training execution helpers."""

from __future__ import annotations

import glob
import logging
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any, Protocol, TypeVar

from ser.config import AppConfig
from ser.data import Utterance
from ser.license_check import build_provenance_metadata
from ser.models import training_support as _training_support
from ser.models.artifact_envelope import build_model_artifact as _build_model_artifact
from ser.models.artifact_persistence import (
    persist_pickle_artifact,
    persist_secure_artifact,
    persist_training_report,
)
from ser.models.dataset_splitting import MediumSplitMetadata
from ser.models.training_execution import run_accurate_profile_training
from ser.models.training_reporting import build_grouped_evaluation_controls
from ser.models.training_types import (
    AccurateTrainingPreparation,
    PersistedArtifactsLike,
    TrainingEvaluation,
)

_ModelT = TypeVar("_ModelT")


class PreparedAccurateTrainingRunner(Protocol):
    """Callable contract for running one prepared accurate-profile training pass."""

    def __call__(
        self,
        *,
        prepared: AccurateTrainingPreparation[Utterance, MediumSplitMetadata, Any],
        utterances: list[Utterance],
        settings: AppConfig,
        profile_label: str,
        backend_id: str,
        profile_id: str,
        frame_size_seconds: float,
        frame_stride_seconds: float,
    ) -> None: ...


def run_accurate_profile_training_from_prepared(
    *,
    prepared: AccurateTrainingPreparation[Utterance, MediumSplitMetadata, Any],
    utterances: list[Utterance],
    settings: AppConfig,
    logger: logging.Logger,
    profile_label: str,
    backend_id: str,
    profile_id: str,
    pooling_strategy: str,
    frame_size_seconds: float,
    frame_stride_seconds: float,
    create_classifier: Callable[[], _ModelT],
    min_support: int,
    evaluate_predictions: Callable[..., TrainingEvaluation],
    attach_grouped_metrics: Callable[..., dict[str, object]],
    build_model_artifact: Callable[..., dict[str, object]],
    extract_artifact_metadata: Callable[[dict[str, object]], dict[str, object]],
    persist_model_artifacts: Callable[
        [_ModelT, dict[str, object]],
        PersistedArtifactsLike,
    ],
    build_provenance_metadata: Callable[..., dict[str, object]],
    build_dataset_controls: Callable[[list[Utterance]], dict[str, object]],
    build_grouped_evaluation_controls: Callable[
        [MediumSplitMetadata],
        dict[str, object],
    ],
    build_training_report: Callable[..., dict[str, object]],
    persist_training_report: Callable[[dict[str, object], Path], None],
    report_destination: Path,
) -> dict[str, object]:
    """Runs accurate-profile training from one prepared payload."""

    return run_accurate_profile_training(
        prepared=prepared,
        utterances=utterances,
        settings=settings,
        logger=logger,
        profile_label=profile_label,
        backend_id=backend_id,
        profile_id=profile_id,
        pooling_strategy=pooling_strategy,
        frame_size_seconds=frame_size_seconds,
        frame_stride_seconds=frame_stride_seconds,
        create_classifier=create_classifier,
        min_support=min_support,
        evaluate_predictions=evaluate_predictions,
        attach_grouped_metrics=attach_grouped_metrics,
        build_model_artifact=build_model_artifact,
        extract_artifact_metadata=extract_artifact_metadata,
        persist_model_artifacts=persist_model_artifacts,
        build_provenance_metadata=build_provenance_metadata,
        build_dataset_controls=build_dataset_controls,
        build_grouped_evaluation_controls=build_grouped_evaluation_controls,
        build_training_report=build_training_report,
        persist_training_report=persist_training_report,
        report_destination=report_destination,
    )


def build_prepared_accurate_training_runner(
    *,
    logger: logging.Logger,
    create_classifier: Callable[[], _ModelT],
    min_support_resolver: Callable[[], int],
    evaluate_predictions: Callable[..., TrainingEvaluation],
    attach_grouped_metrics: Callable[..., dict[str, object]],
    build_model_artifact: Callable[..., dict[str, object]],
    extract_artifact_metadata: Callable[[dict[str, object]], dict[str, object]],
    persist_model_artifacts: Callable[
        [_ModelT, dict[str, object]],
        PersistedArtifactsLike,
    ],
    build_provenance_metadata: Callable[..., dict[str, object]],
    build_dataset_controls: Callable[[list[Utterance]], dict[str, object]],
    build_grouped_evaluation_controls: Callable[
        [MediumSplitMetadata],
        dict[str, object],
    ],
    build_training_report: Callable[..., dict[str, object]],
    persist_training_report: Callable[[dict[str, object], Path], None],
) -> PreparedAccurateTrainingRunner:
    """Builds a prepared-runner bound to accurate-profile execution hooks."""

    def _run(
        *,
        prepared: AccurateTrainingPreparation[Utterance, MediumSplitMetadata, Any],
        utterances: list[Utterance],
        settings: AppConfig,
        profile_label: str,
        backend_id: str,
        profile_id: str,
        frame_size_seconds: float,
        frame_stride_seconds: float,
    ) -> None:
        _ = run_accurate_profile_training_from_prepared(
            prepared=prepared,
            utterances=utterances,
            settings=settings,
            logger=logger,
            profile_label=profile_label,
            backend_id=backend_id,
            profile_id=profile_id,
            pooling_strategy="mean_std",
            frame_size_seconds=frame_size_seconds,
            frame_stride_seconds=frame_stride_seconds,
            create_classifier=create_classifier,
            min_support=min_support_resolver(),
            evaluate_predictions=evaluate_predictions,
            attach_grouped_metrics=attach_grouped_metrics,
            build_model_artifact=build_model_artifact,
            extract_artifact_metadata=extract_artifact_metadata,
            persist_model_artifacts=persist_model_artifacts,
            build_provenance_metadata=build_provenance_metadata,
            build_dataset_controls=build_dataset_controls,
            build_grouped_evaluation_controls=build_grouped_evaluation_controls,
            build_training_report=build_training_report,
            persist_training_report=persist_training_report,
            report_destination=settings.models.training_report_file,
        )

    return _run


def build_prepared_accurate_profile_training_runner(
    settings: AppConfig,
    *,
    logger: logging.Logger,
) -> PreparedAccurateTrainingRunner:
    """Builds the accurate-profile prepared-runner from canonical owners."""

    return build_prepared_accurate_training_runner(
        logger=logger,
        create_classifier=lambda: _training_support.create_classifier(settings),
        min_support_resolver=_training_support.group_metrics_min_support,
        evaluate_predictions=_training_support.evaluate_training_predictions,
        attach_grouped_metrics=_training_support.attach_grouped_training_metrics,
        build_model_artifact=_build_model_artifact,
        extract_artifact_metadata=_training_support.extract_artifact_metadata,
        persist_model_artifacts=lambda model, artifact: (
            _training_support.persist_model_artifacts(
                model,
                artifact,
                settings=settings,
                persist_pickle=persist_pickle_artifact,
                persist_secure=persist_secure_artifact,
            )
        ),
        build_provenance_metadata=build_provenance_metadata,
        build_dataset_controls=partial(
            _training_support.build_dataset_controls,
            settings=settings,
        ),
        build_grouped_evaluation_controls=build_grouped_evaluation_controls,
        build_training_report=partial(
            _training_support.build_training_report,
            settings=settings,
            globber=glob.glob,
        ),
        persist_training_report=persist_training_report,
    )


__all__ = [
    "build_prepared_accurate_profile_training_runner",
    "build_prepared_accurate_training_runner",
    "PreparedAccurateTrainingRunner",
    "run_accurate_profile_training_from_prepared",
]
