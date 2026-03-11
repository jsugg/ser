"""Public training entrypoints bound to their canonical owner modules."""

from __future__ import annotations

import glob
from functools import partial

import ser._internal.models.fast_training_entrypoints as _fast_training_entrypoints
import ser.data.data_loader as _data_loader
import ser.license_check as _license_check
import ser.models.accurate_feature_dataset as _accurate_feature_dataset
import ser.models.accurate_training_execution as _accurate_training_execution
import ser.models.accurate_training_preparation as _accurate_training_preparation
import ser.models.artifact_envelope as _artifact_envelope
import ser.models.artifact_persistence as _artifact_persistence
import ser.models.fast_training as _fast_training
import ser.models.medium_feature_dataset as _medium_feature_dataset
import ser.models.medium_noise_controls as _medium_noise_controls
import ser.models.medium_training_preparation as _medium_training_preparation
import ser.models.profile_runtime as _profile_runtime
import ser.models.training_reporting as _training_reporting
import ser.models.training_support as _training_support
from ser._internal.runtime.environment_plan import build_runtime_environment_plan
from ser._internal.runtime.process_env import temporary_process_env
from ser.config import AppConfig, get_settings
from ser.utils.logger import get_logger

logger = get_logger(__name__)


def train_model(*, settings: AppConfig) -> None:
    """Runs fast-profile training with one explicit settings snapshot."""

    def _build_hooks(active_settings: AppConfig) -> _fast_training.FastTrainingHooks:
        return _fast_training.FastTrainingHooks(
            logger=logger,
            settings=active_settings,
            load_utterances=partial(_data_loader.load_utterances, settings=active_settings),
            ensure_dataset_consents_for_training=partial(
                _training_support.ensure_dataset_consents_for_training,
                settings=active_settings,
                logger=logger,
            ),
            load_data=partial(_data_loader.load_data, settings=active_settings),
            create_classifier=lambda: _training_support.create_classifier(active_settings),
            evaluate_training_predictions=_training_support.evaluate_training_predictions,
            build_provenance_metadata=_license_check.build_provenance_metadata,
            build_model_artifact=_artifact_envelope.build_model_artifact,
            extract_artifact_metadata=_training_support.extract_artifact_metadata,
            persist_model_artifacts=partial(
                _training_support.persist_model_artifacts,
                settings=active_settings,
                persist_pickle=_artifact_persistence.persist_pickle_artifact,
                persist_secure=_artifact_persistence.persist_secure_artifact,
            ),
            build_training_report=partial(
                _training_support.build_training_report,
                settings=active_settings,
                globber=glob.glob,
            ),
            persist_training_report=_artifact_persistence.persist_training_report,
            default_backend_id=_artifact_envelope.DEFAULT_BACKEND_ID,
            default_profile_id=_artifact_envelope.DEFAULT_PROFILE_ID,
        )

    _fast_training_entrypoints.train_model(
        settings=settings,
        settings_resolver=get_settings,
        build_hooks=_build_hooks,
        build_runtime_environment_plan_fn=build_runtime_environment_plan,
        temporary_process_env_fn=temporary_process_env,
    )


def train_medium_model(*, settings: AppConfig) -> None:
    """Runs medium-profile training with one explicit settings snapshot."""

    _medium_training_preparation.train_medium_profile_entrypoint(
        settings=settings,
        logger=logger,
        train_profile_model=_medium_training_preparation.train_medium_profile_model,
        load_utterances_for_training=partial(_data_loader.load_utterances, settings=settings),
        ensure_dataset_consents_for_training=partial(
            _training_support.ensure_dataset_consents_for_training,
            settings=settings,
            logger=logger,
        ),
        split_utterances=partial(
            _training_support.split_utterances,
            settings=settings,
            logger=logger,
        ),
        resolve_model_id_for_settings=_profile_runtime.resolve_medium_model_id,
        resolve_runtime_selectors_for_backend=partial(
            _profile_runtime.resolve_runtime_selectors_for_backend_id,
            logger=logger,
        ),
        build_backend_for_settings=partial(
            _profile_runtime.build_medium_backend_for_settings,
            backend_factory=_profile_runtime.XLSRBackend,
        ),
        build_feature_dataset=partial(
            _medium_feature_dataset.build_medium_feature_dataset,
            settings=settings,
            encode_sequence=partial(
                _medium_feature_dataset.encode_medium_sequence,
                settings=settings,
                backend_id=_profile_runtime.MEDIUM_BACKEND_ID,
                logger=logger,
            ),
            apply_noise_controls=partial(
                _medium_noise_controls.apply_medium_noise_controls,
                min_window_std=settings.medium_training.min_window_std,
                max_windows_per_clip=settings.medium_training.max_windows_per_clip,
            ),
            merge_noise_stats=_medium_noise_controls.merge_medium_noise_stats,
            window_meta_factory=_training_support.WindowMeta,
        ),
        create_classifier=lambda: _training_support.create_classifier(settings),
        min_support=_training_support.group_metrics_min_support(),
        evaluate_predictions=_training_support.evaluate_training_predictions,
        attach_grouped_metrics=_training_support.attach_grouped_training_metrics,
        build_model_artifact=_artifact_envelope.build_model_artifact,
        extract_artifact_metadata=_training_support.extract_artifact_metadata,
        persist_model_artifacts=partial(
            _training_support.persist_model_artifacts,
            settings=settings,
            persist_pickle=_artifact_persistence.persist_pickle_artifact,
            persist_secure=_artifact_persistence.persist_secure_artifact,
        ),
        build_provenance_metadata=_license_check.build_provenance_metadata,
        build_dataset_controls=partial(_training_support.build_dataset_controls, settings=settings),
        build_medium_noise_controls=_training_reporting.build_medium_noise_controls,
        build_grouped_evaluation_controls=_training_reporting.build_grouped_evaluation_controls,
        build_training_report=partial(
            _training_support.build_training_report,
            settings=settings,
            globber=glob.glob,
        ),
        persist_training_report=_artifact_persistence.persist_training_report,
        embedding_cache_name="medium_embeddings",
        profile_label="Medium",
        backend_id=_profile_runtime.MEDIUM_BACKEND_ID,
        profile_id=_profile_runtime.MEDIUM_PROFILE_ID,
        pooling_strategy=_profile_runtime.MEDIUM_POOLING_STRATEGY,
    )


def train_accurate_model(*, settings: AppConfig) -> None:
    """Runs accurate-profile training with one explicit settings snapshot."""

    _accurate_training_preparation.train_accurate_whisper_profile_entrypoint(
        settings=settings,
        logger=logger,
        train_profile_model=_accurate_training_preparation.train_accurate_whisper_profile_model,
        load_utterances_for_training=partial(_data_loader.load_utterances, settings=settings),
        ensure_dataset_consents_for_training=partial(
            _training_support.ensure_dataset_consents_for_training,
            settings=settings,
            logger=logger,
        ),
        split_utterances=partial(
            _training_support.split_utterances,
            settings=settings,
            logger=logger,
        ),
        resolve_model_id_for_settings=_profile_runtime.resolve_accurate_model_id,
        resolve_runtime_selectors_for_backend=partial(
            _profile_runtime.resolve_runtime_selectors_for_backend_id,
            logger=logger,
        ),
        build_backend_for_settings=partial(
            _profile_runtime.build_accurate_backend_for_settings,
            backend_factory=_profile_runtime.WhisperBackend,
        ),
        build_feature_dataset_for_backend=partial(
            _accurate_feature_dataset.build_accurate_feature_dataset,
            settings=settings,
            accurate_backend_id=_profile_runtime.ACCURATE_BACKEND_ID,
            accurate_research_backend_id=_profile_runtime.ACCURATE_RESEARCH_BACKEND_ID,
            logger=logger,
            window_meta_factory=_training_support.WindowMeta,
        ),
        run_prepared_accurate_profile_training=(
            _accurate_training_execution.build_prepared_accurate_profile_training_runner(
                settings,
                logger=logger,
            )
        ),
        backend_id=_profile_runtime.ACCURATE_BACKEND_ID,
        profile_id=_profile_runtime.ACCURATE_PROFILE_ID,
        profile_label="Accurate",
        frame_size_seconds=settings.accurate_runtime.pool_window_size_seconds,
        frame_stride_seconds=settings.accurate_runtime.pool_window_stride_seconds,
    )


def train_accurate_research_model(*, settings: AppConfig) -> None:
    """Runs accurate-research training with one explicit settings snapshot."""

    _accurate_training_preparation.train_accurate_research_profile_entrypoint(
        settings=settings,
        logger=logger,
        train_profile_model=(_accurate_training_preparation.train_accurate_research_profile_model),
        parse_allowed_restricted_backends_env=(
            _license_check.parse_allowed_restricted_backends_env
        ),
        load_persisted_backend_consents=_license_check.load_persisted_backend_consents,
        ensure_backend_access=_license_check.ensure_backend_access,
        restricted_backend_id=_profile_runtime.ACCURATE_RESEARCH_BACKEND_ID,
        load_utterances_for_training=partial(_data_loader.load_utterances, settings=settings),
        ensure_dataset_consents_for_training=partial(
            _training_support.ensure_dataset_consents_for_training,
            settings=settings,
            logger=logger,
        ),
        split_utterances=partial(
            _training_support.split_utterances,
            settings=settings,
            logger=logger,
        ),
        resolve_model_id_for_settings=_profile_runtime.resolve_accurate_research_model_id,
        resolve_runtime_selectors_for_backend=partial(
            _profile_runtime.resolve_runtime_selectors_for_backend_id,
            logger=logger,
        ),
        build_backend_for_settings=partial(
            _profile_runtime.build_accurate_research_backend_for_settings,
            backend_factory=_profile_runtime.Emotion2VecBackend,
        ),
        build_feature_dataset_for_backend=partial(
            _accurate_feature_dataset.build_accurate_feature_dataset,
            settings=settings,
            accurate_backend_id=_profile_runtime.ACCURATE_BACKEND_ID,
            accurate_research_backend_id=_profile_runtime.ACCURATE_RESEARCH_BACKEND_ID,
            logger=logger,
            window_meta_factory=_training_support.WindowMeta,
        ),
        run_prepared_accurate_profile_training=(
            _accurate_training_execution.build_prepared_accurate_profile_training_runner(
                settings,
                logger=logger,
            )
        ),
        backend_id=_profile_runtime.ACCURATE_RESEARCH_BACKEND_ID,
        profile_id=_profile_runtime.ACCURATE_RESEARCH_PROFILE_ID,
        profile_label="Accurate-research",
        frame_size_seconds=settings.accurate_research_runtime.pool_window_size_seconds,
        frame_stride_seconds=settings.accurate_research_runtime.pool_window_stride_seconds,
    )
