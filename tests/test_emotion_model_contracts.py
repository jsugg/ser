"""Contract tests for emotion model public entrypoints."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
from sklearn.neural_network import MLPClassifier

from ser.features import FeatureFrame
from ser.models import emotion_model as em
from ser.models.profile_runtime import (
    resolve_accurate_model_id,
    resolve_accurate_research_model_id,
    resolve_medium_model_id,
)
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
    monkeypatch.setattr(
        em,
        "get_settings",
        lambda: SimpleNamespace(training=SimpleNamespace(test_size=0.25)),
    )
    monkeypatch.setattr(em, "load_utterances", lambda: None)
    monkeypatch.setattr(em, "load_data", lambda *, test_size, settings=None: None)

    with pytest.raises(RuntimeError, match="Dataset not loaded"):
        em.train_model()


def test_train_medium_model_delegates_to_medium_training_helper(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """`train_medium_model` should delegate orchestration to medium helper seam."""
    settings = SimpleNamespace(
        models=SimpleNamespace(
            huggingface_cache_root=tmp_path / "hf-cache",
            training_report_file=tmp_path / "training_report.json",
        ),
        medium_training=SimpleNamespace(
            min_window_std=0.0,
            max_windows_per_clip=0,
        ),
        tmp_folder=tmp_path / "tmp",
    )
    captured: dict[str, object] = {}

    def _fake_train_medium_profile_model(**kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(em, "get_settings", lambda: settings)
    monkeypatch.setattr(
        em,
        "train_medium_profile_model",
        _fake_train_medium_profile_model,
    )

    em.train_medium_model()

    assert captured["settings"] is settings
    assert captured["logger"] is em.logger
    assert captured["load_utterances_for_training"] is em.load_utterances
    ensure_consents = captured["ensure_dataset_consents_for_training"]
    assert isinstance(ensure_consents, partial)
    assert ensure_consents.func is em._ensure_dataset_consents_for_training
    assert ensure_consents.keywords == {"settings": settings}
    split_utterances = captured["split_utterances"]
    assert isinstance(split_utterances, partial)
    assert split_utterances.func is em._split_utterances
    assert split_utterances.keywords == {"settings": settings}
    assert captured["resolve_model_id_for_settings"] is resolve_medium_model_id
    assert callable(captured["resolve_runtime_selectors_for_settings"])
    assert callable(captured["build_backend"])
    assert isinstance(captured["build_feature_dataset"], partial)
    assert captured["embedding_cache_name"] == "medium_embeddings"
    assert captured["profile_label"] == "Medium"
    assert captured["backend_id"] == em.MEDIUM_BACKEND_ID
    assert captured["profile_id"] == em.MEDIUM_PROFILE_ID
    assert captured["pooling_strategy"] == em.MEDIUM_POOLING_STRATEGY
    assert callable(captured["create_classifier"])
    assert captured["min_support"] == em._group_metrics_min_support()
    persist_model_artifacts = captured["persist_model_artifacts"]
    assert isinstance(persist_model_artifacts, partial)
    assert persist_model_artifacts.func is em._persist_model_artifacts
    assert persist_model_artifacts.keywords == {"settings": settings}
    build_dataset_controls = captured["build_dataset_controls"]
    assert isinstance(build_dataset_controls, partial)
    assert build_dataset_controls.func is em._build_dataset_controls
    assert build_dataset_controls.keywords == {"settings": settings}
    build_training_report = captured["build_training_report"]
    assert isinstance(build_training_report, partial)
    assert build_training_report.func is em._build_training_report
    assert build_training_report.keywords == {"settings": settings}


def test_train_accurate_model_delegates_to_whisper_helper(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """`train_accurate_model` should delegate orchestration to helper seam."""
    settings = SimpleNamespace(
        models=SimpleNamespace(
            huggingface_cache_root=tmp_path / "hf-cache",
            training_report_file=tmp_path / "training_report.json",
        ),
        accurate_runtime=SimpleNamespace(
            pool_window_size_seconds=2.0,
            pool_window_stride_seconds=0.5,
        ),
        tmp_folder=tmp_path / "tmp",
    )
    captured: dict[str, object] = {}

    def _fake_train_accurate_whisper_profile_model(**kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(em, "get_settings", lambda: settings)
    monkeypatch.setattr(
        em,
        "train_accurate_whisper_profile_model",
        _fake_train_accurate_whisper_profile_model,
    )

    em.train_accurate_model()

    assert captured["settings"] is settings
    assert captured["logger"] is em.logger
    assert captured["load_utterances_for_training"] is em.load_utterances
    ensure_consents = captured["ensure_dataset_consents_for_training"]
    assert isinstance(ensure_consents, partial)
    assert ensure_consents.func is em._ensure_dataset_consents_for_training
    assert ensure_consents.keywords == {"settings": settings}
    split_utterances = captured["split_utterances"]
    assert isinstance(split_utterances, partial)
    assert split_utterances.func is em._split_utterances
    assert split_utterances.keywords == {"settings": settings}
    assert captured["resolve_model_id_for_settings"] is resolve_accurate_model_id
    assert callable(captured["resolve_runtime_selectors_for_settings"])
    assert callable(captured["build_backend"])
    assert callable(captured["build_feature_dataset"])
    assert callable(captured["run_prepared_training"])


def test_train_accurate_research_model_delegates_to_research_helper(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """`train_accurate_research_model` should delegate orchestration to helper seam."""
    settings = SimpleNamespace(
        models=SimpleNamespace(
            huggingface_cache_root=tmp_path / "hf-cache",
            modelscope_cache_root=tmp_path / "ms-cache",
            training_report_file=tmp_path / "training_report.json",
        ),
        runtime_flags=SimpleNamespace(restricted_backends=True),
        accurate_research_runtime=SimpleNamespace(
            pool_window_size_seconds=2.0,
            pool_window_stride_seconds=0.5,
        ),
        tmp_folder=tmp_path / "tmp",
    )
    captured: dict[str, object] = {}

    def _fake_train_accurate_research_profile_model(**kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(em, "get_settings", lambda: settings)
    monkeypatch.setattr(
        em,
        "train_accurate_research_profile_model",
        _fake_train_accurate_research_profile_model,
    )

    em.train_accurate_research_model()

    assert captured["settings"] is settings
    assert captured["logger"] is em.logger
    assert captured["parse_allowed_restricted_backends_env"] is (
        em.parse_allowed_restricted_backends_env
    )
    assert (
        captured["load_persisted_backend_consents"]
        is em.load_persisted_backend_consents
    )
    assert captured["ensure_backend_access"] is em.ensure_backend_access
    assert captured["restricted_backend_id"] == em.ACCURATE_RESEARCH_BACKEND_ID
    assert captured["load_utterances_for_training"] is em.load_utterances
    ensure_consents = captured["ensure_dataset_consents_for_training"]
    assert isinstance(ensure_consents, partial)
    assert ensure_consents.func is em._ensure_dataset_consents_for_training
    assert ensure_consents.keywords == {"settings": settings}
    split_utterances = captured["split_utterances"]
    assert isinstance(split_utterances, partial)
    assert split_utterances.func is em._split_utterances
    assert split_utterances.keywords == {"settings": settings}
    assert (
        captured["resolve_model_id_for_settings"] is resolve_accurate_research_model_id
    )
    assert callable(captured["resolve_runtime_selectors_for_settings"])
    assert callable(captured["build_backend"])
    assert callable(captured["build_feature_dataset"])
    assert callable(captured["run_prepared_training"])


def test_build_prepared_accurate_runner_uses_canonical_artifact_persistor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prepared accurate runner should reuse the canonical artifact persistor."""
    captured: dict[str, object] = {}
    runner = object()
    settings = cast(em.AppConfig, SimpleNamespace())

    def _fake_build_prepared_accurate_training_runner(**kwargs: object) -> object:
        captured.update(kwargs)
        return runner

    monkeypatch.setattr(
        em,
        "build_prepared_accurate_training_runner",
        _fake_build_prepared_accurate_training_runner,
    )

    resolved_runner = em._build_prepared_accurate_profile_training_runner(settings)

    assert resolved_runner is runner
    persist_model_artifacts = captured["persist_model_artifacts"]
    assert isinstance(persist_model_artifacts, partial)
    assert persist_model_artifacts.func is em._persist_model_artifacts
    assert persist_model_artifacts.keywords == {"settings": settings}
    build_dataset_controls = captured["build_dataset_controls"]
    assert isinstance(build_dataset_controls, partial)
    assert build_dataset_controls.func is em._build_dataset_controls
    assert build_dataset_controls.keywords == {"settings": settings}
    build_training_report = captured["build_training_report"]
    assert isinstance(build_training_report, partial)
    assert build_training_report.func is em._build_training_report
    assert build_training_report.keywords == {"settings": settings}


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
                secure_model_file=Path("ser_model.skops"),
                model_file=Path("ser_model.pkl"),
            )
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
    monkeypatch.setattr(em, "extract_feature_frames", lambda _path: frames)
    monkeypatch.setattr(em, "load_model", lambda: loaded_model)

    with pytest.raises(ValueError, match="Feature vector size mismatch"):
        em.predict_emotions_detailed("sample.wav")


def test_train_model_success_keeps_persistence_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """`train_model` should persist artifact and report on successful training."""
    report_path = tmp_path / "training_report.json"
    x_train = np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)
    x_test = np.asarray([[0.5, 0.6]], dtype=np.float64)
    y_train = ["happy", "sad"]
    y_test = ["happy"]

    class _DummyClassifier:
        def fit(self, X: np.ndarray, y: list[str]) -> None:
            del X, y

        def predict(self, X: np.ndarray) -> np.ndarray:
            del X
            return np.asarray(["happy"], dtype=object)

    artifact_calls: dict[str, object] = {}
    report_calls: dict[str, object] = {}

    monkeypatch.setattr(
        em,
        "get_settings",
        lambda: SimpleNamespace(
            training=SimpleNamespace(test_size=0.25),
            models=SimpleNamespace(training_report_file=report_path),
        ),
    )
    monkeypatch.setattr(em, "load_utterances", lambda: None)
    monkeypatch.setattr(
        em,
        "load_data",
        lambda *, test_size, settings=None: (x_train, x_test, y_train, y_test),
    )
    monkeypatch.setattr(
        em,
        "_create_classifier",
        lambda _settings=None: _DummyClassifier(),
    )
    monkeypatch.setattr(
        em,
        "_evaluate_training_predictions",
        lambda *, y_true, y_pred: SimpleNamespace(
            accuracy=1.0,
            macro_f1=1.0,
            uar=1.0,
            ser_metrics={"uar": 1.0, "macro_f1": 1.0},
        ),
    )
    monkeypatch.setattr(em, "build_provenance_metadata", lambda **_kwargs: {})
    monkeypatch.setattr(
        em,
        "_build_model_artifact",
        lambda **kwargs: {
            "artifact_version": em.MODEL_ARTIFACT_VERSION,
            "metadata": {"feature_vector_size": kwargs["feature_vector_size"]},
        },
    )
    monkeypatch.setattr(em, "_extract_artifact_metadata", lambda _artifact: {})

    def _persist_model_artifacts(
        *,
        model: object,
        artifact: dict[str, object],
        settings: object,
    ) -> em.PersistedArtifacts:
        artifact_calls["model"] = model
        artifact_calls["artifact"] = artifact
        artifact_calls["settings"] = settings
        return em.PersistedArtifacts(
            pickle_path=tmp_path / "ser_model.pkl",
            secure_path=None,
        )

    def _persist_training_report(report: dict[str, object], path: Path) -> None:
        report_calls["report"] = report
        report_calls["path"] = path

    monkeypatch.setattr(em, "_persist_model_artifacts", _persist_model_artifacts)
    monkeypatch.setattr(
        em,
        "_build_training_report",
        lambda **kwargs: {"accuracy": kwargs["accuracy"]},
    )
    monkeypatch.setattr(em, "_persist_training_report", _persist_training_report)

    em.train_model()

    assert "artifact" in artifact_calls
    assert artifact_calls["settings"] is not None
    assert report_calls["path"] == report_path


def test_predict_emotions_detailed_delegates_to_fast_path_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`predict_emotions_detailed` should delegate to fast-path helper with loaded model."""
    loaded_model = em.LoadedModel(
        model=_PredictOnlyModel(["neutral"]),
        expected_feature_size=3,
    )
    sentinel = InferenceResult(schema_version="v1", segments=[], frames=[])
    captured: dict[str, object] = {}

    def _delegate(
        file: str,
        *,
        model: em.EmotionClassifier,
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

    result = em.predict_emotions_detailed("sample.wav", loaded_model=loaded_model)

    assert result is sentinel
    assert captured["file"] == "sample.wav"
    assert captured["model"] is loaded_model.model
    assert captured["expected_feature_size"] == loaded_model.expected_feature_size
    assert captured["output_schema_version"] == em.OUTPUT_SCHEMA_VERSION
    assert captured["extract_feature_frames_fn"] is em.extract_feature_frames
