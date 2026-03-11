"""Contract tests for the public emotion-model boundary and its direct owners."""

from __future__ import annotations

import glob
from collections.abc import Callable
from functools import partial
from pathlib import Path
from types import FunctionType, SimpleNamespace
from typing import cast

import numpy as np
import pytest
from sklearn.neural_network import MLPClassifier

import ser.data.data_loader as data_loader
import ser.license_check as license_check
import ser.models.accurate_training_execution as accurate_training_execution
import ser.models.accurate_training_preparation as accurate_training_preparation
import ser.models.artifact_envelope as artifact_envelope
import ser.models.artifact_persistence as artifact_persistence
import ser.models.emotion_model as em
import ser.models.medium_feature_dataset as medium_feature_dataset
import ser.models.medium_training_preparation as medium_training_preparation
import ser.models.profile_runtime as profile_runtime
import ser.models.training_entrypoints as training_entrypoints
import ser.models.training_reporting as training_reporting
import ser.models.training_support as training_support
from ser.features import FeatureFrame
from ser.models.fast_training import FastTrainingHooks
from ser.runtime.schema import InferenceResult


class _PredictOnlyModel(MLPClassifier):
    """Deterministic model stub exposing only `predict`."""

    def __init__(self, predictions: list[str]) -> None:
        super().__init__(hidden_layer_sizes=(1,), max_iter=1, random_state=0)
        self._predictions = predictions

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns deterministic predictions independent of input frame values."""
        del X
        return np.asarray(self._predictions, dtype=object)


def test_train_model_raises_when_dataset_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`train_model` should fail closed when no dataset split can be loaded."""
    settings = SimpleNamespace(
        training=SimpleNamespace(test_size=0.25),
        models=SimpleNamespace(
            huggingface_cache_root=Path("cache/huggingface"),
            modelscope_cache_root=Path("cache/modelscope/hub"),
            torch_cache_root=Path("cache/torch"),
        ),
        torch_runtime=SimpleNamespace(enable_mps_fallback=False),
    )
    monkeypatch.setattr(em, "get_settings", lambda: settings)
    monkeypatch.setattr(data_loader, "load_utterances", lambda *, settings=None: None)
    monkeypatch.setattr(
        data_loader,
        "load_data",
        lambda *, test_size, settings=None: None,
    )

    with pytest.raises(RuntimeError, match="Dataset not loaded"):
        em.train_model()


def test_train_medium_model_wrapper_uses_explicit_settings_without_ambient_lookup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Explicit settings should bypass ambient lookup for the public wrapper."""
    settings = cast(
        em.AppConfig,
        SimpleNamespace(
            models=SimpleNamespace(training_report_file=tmp_path / "report.json"),
            medium_training=SimpleNamespace(
                min_window_std=0.0,
                max_windows_per_clip=0,
            ),
        ),
    )
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        em,
        "get_settings",
        lambda: (_ for _ in ()).throw(AssertionError("wrapper must use explicit settings")),
    )
    monkeypatch.setattr(
        em._training_entrypoints,
        "train_medium_model",
        lambda *, settings: captured.update({"settings": settings}),
    )

    em.train_medium_model(settings=settings)

    assert captured["settings"] is settings


def test_train_accurate_model_wrapper_uses_explicit_settings_without_ambient_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit settings should bypass ambient lookup for the accurate wrapper."""
    settings = cast(em.AppConfig, SimpleNamespace())
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        em,
        "get_settings",
        lambda: (_ for _ in ()).throw(AssertionError("wrapper must use explicit settings")),
    )
    monkeypatch.setattr(
        em._training_entrypoints,
        "train_accurate_model",
        lambda *, settings: captured.update({"settings": settings}),
    )

    em.train_accurate_model(settings=settings)

    assert captured["settings"] is settings


def test_train_accurate_research_model_wrapper_uses_explicit_settings_without_ambient_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit settings should bypass ambient lookup for the research wrapper."""
    settings = cast(em.AppConfig, SimpleNamespace())
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        em,
        "get_settings",
        lambda: (_ for _ in ()).throw(AssertionError("wrapper must use explicit settings")),
    )
    monkeypatch.setattr(
        em._training_entrypoints,
        "train_accurate_research_model",
        lambda *, settings: captured.update({"settings": settings}),
    )

    em.train_accurate_research_model(settings=settings)

    assert captured["settings"] is settings


def test_training_entrypoints_train_model_builds_fast_hooks_from_canonical_owners(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fast training entrypoint should bind canonical collaborators directly."""
    settings = cast(em.AppConfig, SimpleNamespace())
    captured: dict[str, object] = {}

    def _fake_train_model(**kwargs: object) -> None:
        captured.update(kwargs)
        build_hooks = cast(
            Callable[[em.AppConfig], FastTrainingHooks],
            kwargs["build_hooks"],
        )
        hooks = build_hooks(settings)
        captured["hooks"] = hooks

    monkeypatch.setattr(
        training_entrypoints._fast_training_entrypoints,
        "train_model",
        _fake_train_model,
    )

    training_entrypoints.train_model(settings=settings)

    hooks = cast(FastTrainingHooks, captured["hooks"])
    assert hooks.logger is training_entrypoints.logger
    assert hooks.settings is settings
    assert isinstance(hooks.load_utterances, partial)
    assert hooks.load_utterances.func is data_loader.load_utterances
    assert hooks.load_utterances.keywords == {"settings": settings}
    assert isinstance(hooks.ensure_dataset_consents_for_training, partial)
    assert (
        hooks.ensure_dataset_consents_for_training.func
        is training_support.ensure_dataset_consents_for_training
    )
    assert hooks.ensure_dataset_consents_for_training.keywords == {
        "settings": settings,
        "logger": training_entrypoints.logger,
    }
    assert isinstance(hooks.load_data, partial)
    assert hooks.load_data.func is data_loader.load_data
    assert hooks.default_backend_id == artifact_envelope.DEFAULT_BACKEND_ID
    assert hooks.default_profile_id == artifact_envelope.DEFAULT_PROFILE_ID
    persist_model_artifacts = hooks.persist_model_artifacts
    assert isinstance(persist_model_artifacts, partial)
    assert persist_model_artifacts.func is training_support.persist_model_artifacts
    assert persist_model_artifacts.keywords["settings"] is settings
    assert (
        persist_model_artifacts.keywords["persist_pickle"]
        is artifact_persistence.persist_pickle_artifact
    )
    assert (
        persist_model_artifacts.keywords["persist_secure"]
        is artifact_persistence.persist_secure_artifact
    )
    build_training_report = hooks.build_training_report
    assert isinstance(build_training_report, partial)
    assert build_training_report.func is training_support.build_training_report
    assert build_training_report.keywords == {"settings": settings, "globber": glob.glob}


def test_training_entrypoints_train_medium_model_uses_canonical_helpers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Medium training entrypoint should use direct owner modules, not emotion-model aliases."""
    settings = cast(
        em.AppConfig,
        SimpleNamespace(
            models=SimpleNamespace(
                huggingface_cache_root=tmp_path / "hf-cache",
                training_report_file=tmp_path / "training_report.json",
            ),
            medium_training=SimpleNamespace(
                min_window_std=0.0,
                max_windows_per_clip=0,
            ),
            tmp_folder=tmp_path / "tmp",
        ),
    )
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        medium_training_preparation,
        "train_medium_profile_model",
        lambda **kwargs: captured.update(kwargs),
    )

    training_entrypoints.train_medium_model(settings=settings)

    assert captured["settings"] is settings
    assert captured["logger"] is training_entrypoints.logger
    load_utterances = captured["load_utterances_for_training"]
    assert isinstance(load_utterances, partial)
    assert load_utterances.func is data_loader.load_utterances
    assert load_utterances.keywords == {"settings": settings}
    ensure_consents = captured["ensure_dataset_consents_for_training"]
    assert isinstance(ensure_consents, partial)
    assert ensure_consents.func is training_support.ensure_dataset_consents_for_training
    assert ensure_consents.keywords == {
        "settings": settings,
        "logger": training_entrypoints.logger,
    }
    split_utterances = captured["split_utterances"]
    assert isinstance(split_utterances, partial)
    assert split_utterances.func is training_support.split_utterances
    assert split_utterances.keywords == {
        "settings": settings,
        "logger": training_entrypoints.logger,
    }
    assert captured["resolve_model_id_for_settings"] is profile_runtime.resolve_medium_model_id
    assert callable(captured["resolve_runtime_selectors_for_settings"])
    assert callable(captured["build_backend"])
    build_feature_dataset = captured["build_feature_dataset"]
    assert isinstance(build_feature_dataset, partial)
    assert build_feature_dataset.func is medium_feature_dataset.build_medium_feature_dataset
    assert callable(captured["create_classifier"])
    assert captured["min_support"] == training_support.group_metrics_min_support()
    assert captured["evaluate_predictions"] is training_support.evaluate_training_predictions
    assert captured["attach_grouped_metrics"] is training_support.attach_grouped_training_metrics
    assert captured["build_model_artifact"] is artifact_envelope.build_model_artifact
    assert captured["extract_artifact_metadata"] is training_support.extract_artifact_metadata
    assert captured["build_provenance_metadata"] is license_check.build_provenance_metadata
    assert captured["build_medium_noise_controls"] is training_reporting.build_medium_noise_controls
    assert (
        captured["build_grouped_evaluation_controls"]
        is training_reporting.build_grouped_evaluation_controls
    )
    assert captured["persist_training_report"] is artifact_persistence.persist_training_report
    assert captured["backend_id"] == profile_runtime.MEDIUM_BACKEND_ID
    assert captured["profile_id"] == profile_runtime.MEDIUM_PROFILE_ID
    assert captured["pooling_strategy"] == profile_runtime.MEDIUM_POOLING_STRATEGY
    persist_model_artifacts = captured["persist_model_artifacts"]
    assert isinstance(persist_model_artifacts, partial)
    assert persist_model_artifacts.func is training_support.persist_model_artifacts
    assert persist_model_artifacts.keywords["settings"] is settings
    build_dataset_controls = captured["build_dataset_controls"]
    assert isinstance(build_dataset_controls, partial)
    assert build_dataset_controls.func is training_support.build_dataset_controls
    assert build_dataset_controls.keywords == {"settings": settings}
    build_training_report = captured["build_training_report"]
    assert isinstance(build_training_report, partial)
    assert build_training_report.func is training_support.build_training_report
    assert build_training_report.keywords == {"settings": settings, "globber": glob.glob}


def test_training_entrypoints_train_accurate_model_uses_canonical_helpers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Accurate training entrypoint should assemble direct owner collaborators."""
    settings = cast(
        em.AppConfig,
        SimpleNamespace(
            models=SimpleNamespace(huggingface_cache_root=tmp_path / "hf-cache"),
            accurate_runtime=SimpleNamespace(
                pool_window_size_seconds=2.0,
                pool_window_stride_seconds=0.5,
            ),
            tmp_folder=tmp_path / "tmp",
        ),
    )
    captured: dict[str, object] = {}
    sentinel_runner = object()

    monkeypatch.setattr(
        accurate_training_preparation,
        "train_accurate_whisper_profile_model",
        lambda **kwargs: captured.update(kwargs),
    )
    monkeypatch.setattr(
        accurate_training_execution,
        "build_prepared_accurate_profile_training_runner",
        lambda active_settings, *, logger: (
            captured.update(
                {
                    "runner_settings": active_settings,
                    "runner_logger": logger,
                }
            ),
            sentinel_runner,
        )[1],
    )

    training_entrypoints.train_accurate_model(settings=settings)

    assert captured["settings"] is settings
    assert captured["logger"] is training_entrypoints.logger
    assert captured["resolve_model_id_for_settings"] is profile_runtime.resolve_accurate_model_id
    assert callable(captured["resolve_runtime_selectors_for_settings"])
    assert callable(captured["build_backend"])
    build_feature_dataset = captured["build_feature_dataset"]
    assert isinstance(build_feature_dataset, FunctionType)
    assert captured["run_prepared_training"] is not None
    assert captured["runner_settings"] is settings
    assert captured["runner_logger"] is training_entrypoints.logger
    assert build_feature_dataset.__closure__ is not None


def test_training_entrypoints_train_accurate_research_model_uses_canonical_helpers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Research training entrypoint should use direct license and runtime owners."""
    settings = cast(
        em.AppConfig,
        SimpleNamespace(
            models=SimpleNamespace(
                huggingface_cache_root=tmp_path / "hf-cache",
                modelscope_cache_root=tmp_path / "ms-cache",
            ),
            runtime_flags=SimpleNamespace(restricted_backends=True),
            accurate_research_runtime=SimpleNamespace(
                pool_window_size_seconds=2.5,
                pool_window_stride_seconds=0.75,
            ),
            tmp_folder=tmp_path / "tmp",
        ),
    )
    captured: dict[str, object] = {}
    sentinel_runner = object()

    monkeypatch.setattr(
        accurate_training_preparation,
        "train_accurate_research_profile_model",
        lambda **kwargs: captured.update(kwargs),
    )
    monkeypatch.setattr(
        accurate_training_execution,
        "build_prepared_accurate_profile_training_runner",
        lambda active_settings, *, logger: (
            captured.update(
                {
                    "runner_settings": active_settings,
                    "runner_logger": logger,
                }
            ),
            sentinel_runner,
        )[1],
    )

    training_entrypoints.train_accurate_research_model(settings=settings)

    assert captured["settings"] is settings
    assert captured["logger"] is training_entrypoints.logger
    assert (
        captured["parse_allowed_restricted_backends_env"]
        is license_check.parse_allowed_restricted_backends_env
    )
    assert (
        captured["load_persisted_backend_consents"] is license_check.load_persisted_backend_consents
    )
    assert captured["ensure_backend_access"] is license_check.ensure_backend_access
    assert captured["restricted_backend_id"] == profile_runtime.ACCURATE_RESEARCH_BACKEND_ID
    assert (
        captured["resolve_model_id_for_settings"]
        is profile_runtime.resolve_accurate_research_model_id
    )
    assert captured["runner_settings"] is settings
    assert captured["runner_logger"] is training_entrypoints.logger


def test_resolve_model_for_loading_delegates_to_internal_helper(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Model resolver helper should route through the internal model-loading seam."""
    settings = cast(
        em.AppConfig,
        SimpleNamespace(
            models=SimpleNamespace(
                folder=tmp_path / "models",
                secure_model_file=tmp_path / "model.skops",
                model_file=tmp_path / "model.pkl",
            )
        ),
    )
    captured: dict[str, object] = {}
    sentinel = cast(em.ResolveModelFn, object())

    monkeypatch.setattr(
        em._model_loading_entrypoints,
        "resolve_model_for_loading_from_public_boundary",
        lambda *args, **kwargs: (
            captured.update({"args": args, **kwargs}),
            sentinel,
        )[1],
    )

    resolved = em._resolve_model_for_loading(settings)

    assert resolved is sentinel
    assert captured["args"] == (settings,)
    assert (
        captured["resolve_model_for_loading_from_settings_fn"]
        is em.resolve_model_for_loading_from_settings
    )
    assert captured["load_secure_model_for_settings_fn"] is training_support.load_secure_model
    assert captured["load_pickle_model_fn"] is training_support.load_pickle_model
    assert captured["logger"] is em.logger


def test_load_model_delegates_to_internal_entrypoint_with_explicit_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`load_model` should resolve settings once before calling the internal entrypoint."""
    settings = cast(em.AppConfig, SimpleNamespace())
    sentinel = cast(em.LoadedModel, object())
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        em,
        "get_settings",
        lambda: (_ for _ in ()).throw(AssertionError("wrapper must use explicit settings")),
    )
    monkeypatch.setattr(
        em,
        "_load_model_entrypoint",
        lambda *args, **kwargs: (
            captured.update({"args": args, **kwargs}),
            sentinel,
        )[1],
    )

    result = em.load_model(
        settings=settings,
        expected_backend_id="hf_xlsr",
        expected_profile="medium",
        expected_backend_model_id="facebook/wav2vec2-base",
    )

    assert result is sentinel
    assert captured["args"] == (settings,)
    assert captured["settings_resolver"] is em.get_settings
    assert captured["resolve_model_factory"] is em._resolve_model_for_loading
    assert captured["expected_backend_id"] == "hf_xlsr"
    assert captured["expected_profile"] == "medium"
    assert captured["expected_backend_model_id"] == "facebook/wav2vec2-base"


def test_load_model_preserves_file_not_found_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`load_model` must preserve `FileNotFoundError` without remapping."""

    def _raise_file_not_found(*_args: object, **_kwargs: object) -> object:
        raise FileNotFoundError("Train it first.")

    monkeypatch.setattr(
        em,
        "get_settings",
        lambda: SimpleNamespace(
            models=SimpleNamespace(
                folder=Path("."),
                huggingface_cache_root=Path("cache/huggingface"),
                secure_model_file=Path("ser_model.skops"),
                model_file=Path("ser_model.pkl"),
                modelscope_cache_root=Path("cache/modelscope/hub"),
                torch_cache_root=Path("cache/torch"),
            ),
            torch_runtime=SimpleNamespace(enable_mps_fallback=False),
        ),
    )
    monkeypatch.setattr(em, "load_model_with_resolution", _raise_file_not_found)

    with pytest.raises(FileNotFoundError, match="Train it first"):
        em.load_model()


def test_predict_emotions_detailed_rejects_feature_size_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`predict_emotions_detailed` should enforce artifact feature-size contract."""
    frames = [
        FeatureFrame(
            start_seconds=0.0,
            end_seconds=1.0,
            features=np.asarray([0.1, 0.2, 0.3], dtype=np.float64),
        )
    ]
    loaded_model = em.LoadedModel(
        model=_PredictOnlyModel(["neutral"]),
        expected_feature_size=2,
    )
    monkeypatch.setattr(em, "extract_feature_frames", lambda _path, *, settings=None: frames)
    monkeypatch.setattr(
        em,
        "load_model",
        lambda *, settings=None, expected_backend_id=None, expected_profile=None, expected_backend_model_id=None: loaded_model,
    )

    with pytest.raises(ValueError, match="Feature vector size mismatch"):
        em.predict_emotions_detailed("sample.wav")


def test_predict_emotions_detailed_uses_explicit_settings_without_ambient_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Detailed prediction should bind explicit settings into the feature extractor."""
    settings = cast(em.AppConfig, SimpleNamespace(torch_runtime=SimpleNamespace()))
    loaded_model = em.LoadedModel(
        model=_PredictOnlyModel(["neutral"]),
        expected_feature_size=3,
    )
    sentinel = InferenceResult(schema_version="v1", segments=[], frames=[])
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        em,
        "get_settings",
        lambda: (_ for _ in ()).throw(AssertionError("wrapper must use explicit settings")),
    )
    monkeypatch.setattr(
        em,
        "build_runtime_environment_plan",
        lambda _settings: SimpleNamespace(torch_runtime=SimpleNamespace()),
    )

    class _NoopContext:
        def __enter__(self) -> None:
            return None

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            del exc_type, exc, tb
            return None

    monkeypatch.setattr(em, "temporary_process_env", lambda _delta: _NoopContext())

    def _delegate(
        file: str,
        *,
        model: object,
        expected_feature_size: int,
        output_schema_version: str,
        extract_feature_frames_fn: object,
        logger: object,
    ) -> InferenceResult:
        captured["file"] = file
        captured["model"] = model
        captured["expected_feature_size"] = expected_feature_size
        captured["output_schema_version"] = output_schema_version
        captured["extract_feature_frames_fn"] = extract_feature_frames_fn
        captured["logger"] = logger
        return sentinel

    monkeypatch.setattr(em, "_fast_predict_emotions_detailed_with_model", _delegate)

    result = em.predict_emotions_detailed(
        "sample.wav",
        loaded_model=loaded_model,
        settings=settings,
    )

    assert result is sentinel
    assert captured["file"] == "sample.wav"
    assert captured["model"] is loaded_model.model
    assert captured["expected_feature_size"] == loaded_model.expected_feature_size
    assert captured["output_schema_version"] == em.OUTPUT_SCHEMA_VERSION
    extract_feature_frames_fn = captured["extract_feature_frames_fn"]
    assert isinstance(extract_feature_frames_fn, partial)
    assert extract_feature_frames_fn.func is em.extract_feature_frames
    assert extract_feature_frames_fn.keywords == {"settings": settings}


def test_predict_emotions_returns_legacy_segments_from_detailed_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Legacy prediction wrapper should derive its response from the detailed API."""
    inference = InferenceResult(
        schema_version="v1",
        segments=[],
        frames=[],
    )
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        em,
        "predict_emotions_detailed",
        lambda file, *, loaded_model=None, settings=None: (
            captured.update(
                {
                    "file": file,
                    "loaded_model": loaded_model,
                    "settings": settings,
                }
            ),
            inference,
        )[1],
    )
    monkeypatch.setattr(
        em,
        "to_legacy_emotion_segments",
        lambda result: [
            em.EmotionSegment(
                emotion=result.schema_version,
                start_seconds=0.0,
                end_seconds=1.0,
            )
        ],
    )

    result = em.predict_emotions("sample.wav", settings=cast(em.AppConfig, object()))

    assert [segment.emotion for segment in result] == ["v1"]
    assert captured["file"] == "sample.wav"
