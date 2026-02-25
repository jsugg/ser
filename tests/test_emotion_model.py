"""Behavior tests for model artifact serialization and report persistence."""

import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.neural_network import MLPClassifier

from ser.features import FeatureFrame
from ser.models import emotion_model as em
from ser.runtime.schema import OUTPUT_SCHEMA_VERSION


class PredictOnlyModel(MLPClassifier):
    """Deterministic model stub exposing `predict` only."""

    def __init__(self, predictions: list[str]) -> None:
        super().__init__(hidden_layer_sizes=(1,), max_iter=1, random_state=0)
        self._predictions = predictions

    def predict(self, X: np.ndarray) -> np.ndarray:
        del X
        return np.asarray(self._predictions, dtype=object)


class PredictProbaModel(PredictOnlyModel):
    """Deterministic model stub exposing both `predict` and `predict_proba`."""

    def __init__(
        self,
        predictions: list[str],
        probabilities: list[list[float]],
        classes: list[str],
    ) -> None:
        super().__init__(predictions)
        self._probabilities = np.asarray(probabilities, dtype=np.float64)
        self.classes_ = np.asarray(classes, dtype=object)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        del X
        return self._probabilities


def _build_classifier() -> MLPClassifier:
    """Creates a classifier instance for artifact tests."""
    return MLPClassifier(
        hidden_layer_sizes=(2,),
        max_iter=20,
        solver="lbfgs",
        random_state=0,
    )


def _set_model_settings(
    monkeypatch: pytest.MonkeyPatch,
    *,
    model_path: Path,
    secure_model_path: Path | None = None,
    training_report_path: Path | None = None,
) -> None:
    """Injects model settings for load-model tests."""
    resolved_secure_path = (
        model_path.with_suffix(".skops")
        if secure_model_path is None
        else secure_model_path
    )
    resolved_report_path = (
        model_path.parent / "training_report.json"
        if training_report_path is None
        else training_report_path
    )

    monkeypatch.setattr(
        em,
        "get_settings",
        lambda: SimpleNamespace(
            dataset=SimpleNamespace(glob_pattern="unused"),
            training=SimpleNamespace(test_size=0.5),
            models=SimpleNamespace(
                folder=model_path.parent,
                model_file=model_path,
                model_file_name=model_path.name,
                secure_model_file=resolved_secure_path,
                secure_model_file_name=resolved_secure_path.name,
                training_report_file=resolved_report_path,
            ),
        ),
    )


def test_load_model_reads_versioned_artifact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Versioned model envelopes should load with metadata validation."""
    model_path = tmp_path / "ser_model.pkl"
    classifier = _build_classifier()
    artifact = {
        "artifact_version": em.MODEL_ARTIFACT_VERSION,
        "model": classifier,
        "metadata": {
            "artifact_version": em.MODEL_ARTIFACT_VERSION,
            "artifact_schema_version": "v2",
            "created_at_utc": "2026-01-01T00:00:00+00:00",
            "feature_vector_size": 2,
            "training_samples": 4,
            "labels": ["happy", "sad"],
            "backend_id": "handcrafted",
            "profile": "fast",
            "feature_dim": 2,
            "frame_size_seconds": 3.0,
            "frame_stride_seconds": 1.0,
            "pooling_strategy": "mean",
        },
    }
    with model_path.open("wb") as handle:
        pickle.dump(artifact, handle, protocol=pickle.HIGHEST_PROTOCOL)

    _set_model_settings(monkeypatch, model_path=model_path)
    loaded = em.load_model()

    assert isinstance(loaded.model, MLPClassifier)
    assert loaded.expected_feature_size == 2


def test_load_model_rejects_legacy_pickled_classifier(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Legacy pickled classifiers should fail without envelope metadata."""
    model_path = tmp_path / "legacy.pkl"
    classifier = _build_classifier()
    with model_path.open("wb") as handle:
        pickle.dump(classifier, handle, protocol=pickle.HIGHEST_PROTOCOL)

    _set_model_settings(monkeypatch, model_path=model_path)
    with pytest.raises(ValueError, match="configured locations"):
        em.load_model()


def test_load_model_requires_primary_storage_when_default_paths_are_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Loader should fail closed when no primary artifacts exist."""
    primary_model_path = tmp_path / "runtime" / "models" / "ser_model.pkl"

    _set_model_settings(monkeypatch, model_path=primary_model_path)
    with pytest.raises(FileNotFoundError, match="Train it first"):
        em.load_model()


def test_load_model_rejects_unknown_artifact_version(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Artifacts with unsupported versions should fail predictably."""
    model_path = tmp_path / "invalid.pkl"
    artifact = {
        "artifact_version": 999,
        "model": _build_classifier(),
        "metadata": {
            "artifact_version": 999,
            "created_at_utc": "2026-01-01T00:00:00+00:00",
            "feature_vector_size": 2,
            "training_samples": 4,
            "labels": ["happy", "sad"],
        },
    }
    with model_path.open("wb") as handle:
        pickle.dump(artifact, handle, protocol=pickle.HIGHEST_PROTOCOL)

    _set_model_settings(monkeypatch, model_path=model_path)
    with pytest.raises(ValueError, match="configured locations"):
        em.load_model()


def test_load_model_falls_back_to_pickle_when_secure_loader_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """If secure loading fails, pickle should still be attempted."""
    model_path = tmp_path / "ser_model.pkl"
    secure_model_path = tmp_path / "ser_model.skops"
    model_path.write_bytes(b"pickle-model")
    secure_model_path.write_bytes(b"secure-model")

    _set_model_settings(
        monkeypatch,
        model_path=model_path,
        secure_model_path=secure_model_path,
    )
    monkeypatch.setattr(
        em,
        "_load_secure_model",
        lambda _candidate, _settings: (_ for _ in ()).throw(ValueError),
    )
    monkeypatch.setattr(
        em,
        "_load_pickle_model",
        lambda _candidate: em.LoadedModel(
            model=_build_classifier(), expected_feature_size=5
        ),
    )

    loaded = em.load_model()

    assert isinstance(loaded.model, MLPClassifier)
    assert loaded.expected_feature_size == 5


def test_load_model_selects_compatible_profile_artifact_from_models_folder(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Profile-filtered loads should skip incompatible artifacts and select matches."""
    models_folder = tmp_path / "models"
    models_folder.mkdir(parents=True, exist_ok=True)
    primary_path = models_folder / "ser_model.pkl"
    medium_path = models_folder / "ser_model_medium_full.pkl"

    fast_artifact = {
        "artifact_version": em.MODEL_ARTIFACT_VERSION,
        "model": _build_classifier(),
        "metadata": {
            "artifact_version": em.MODEL_ARTIFACT_VERSION,
            "artifact_schema_version": "v2",
            "created_at_utc": "2026-01-01T00:00:00+00:00",
            "feature_vector_size": 2,
            "training_samples": 4,
            "labels": ["happy", "sad"],
            "backend_id": "handcrafted",
            "profile": "fast",
            "feature_dim": 2,
            "frame_size_seconds": 3.0,
            "frame_stride_seconds": 1.0,
            "pooling_strategy": "mean",
        },
    }
    medium_artifact = {
        "artifact_version": em.MODEL_ARTIFACT_VERSION,
        "model": _build_classifier(),
        "metadata": {
            "artifact_version": em.MODEL_ARTIFACT_VERSION,
            "artifact_schema_version": "v2",
            "created_at_utc": "2026-01-01T00:00:00+00:00",
            "feature_vector_size": 2,
            "training_samples": 4,
            "labels": ["happy", "sad"],
            "backend_id": "hf_xlsr",
            "profile": "medium",
            "feature_dim": 2,
            "frame_size_seconds": 1.0,
            "frame_stride_seconds": 1.0,
            "pooling_strategy": "mean_std",
        },
    }

    with primary_path.open("wb") as handle:
        pickle.dump(fast_artifact, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with medium_path.open("wb") as handle:
        pickle.dump(medium_artifact, handle, protocol=pickle.HIGHEST_PROTOCOL)

    _set_model_settings(monkeypatch, model_path=primary_path)
    loaded = em.load_model(
        expected_backend_id="hf_xlsr",
        expected_profile="medium",
    )

    assert isinstance(loaded.model, MLPClassifier)
    assert loaded.artifact_metadata is not None
    assert loaded.artifact_metadata["backend_id"] == "hf_xlsr"
    assert loaded.artifact_metadata["profile"] == "medium"


def test_load_model_filters_by_backend_model_id(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Model loader should honor backend_model_id compatibility filter."""
    models_folder = tmp_path / "models"
    models_folder.mkdir(parents=True, exist_ok=True)
    small_path = models_folder / "ser_model_medium_small.pkl"
    large_path = models_folder / "ser_model_medium_large.pkl"

    small_artifact = {
        "artifact_version": em.MODEL_ARTIFACT_VERSION,
        "model": _build_classifier(),
        "metadata": {
            "artifact_version": em.MODEL_ARTIFACT_VERSION,
            "artifact_schema_version": "v2",
            "created_at_utc": "2026-01-01T00:00:00+00:00",
            "feature_vector_size": 2,
            "training_samples": 4,
            "labels": ["happy", "sad"],
            "backend_id": "hf_xlsr",
            "profile": "medium",
            "feature_dim": 2,
            "frame_size_seconds": 1.0,
            "frame_stride_seconds": 1.0,
            "pooling_strategy": "mean_std",
            "backend_model_id": "unit-test/xlsr-small",
        },
    }
    large_artifact = {
        "artifact_version": em.MODEL_ARTIFACT_VERSION,
        "model": _build_classifier(),
        "metadata": {
            "artifact_version": em.MODEL_ARTIFACT_VERSION,
            "artifact_schema_version": "v2",
            "created_at_utc": "2026-01-01T00:00:00+00:00",
            "feature_vector_size": 2,
            "training_samples": 4,
            "labels": ["happy", "sad"],
            "backend_id": "hf_xlsr",
            "profile": "medium",
            "feature_dim": 2,
            "frame_size_seconds": 1.0,
            "frame_stride_seconds": 1.0,
            "pooling_strategy": "mean_std",
            "backend_model_id": "unit-test/xlsr-large",
        },
    }
    with small_path.open("wb") as handle:
        pickle.dump(small_artifact, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with large_path.open("wb") as handle:
        pickle.dump(large_artifact, handle, protocol=pickle.HIGHEST_PROTOCOL)

    _set_model_settings(monkeypatch, model_path=small_path)
    loaded = em.load_model(
        expected_backend_id="hf_xlsr",
        expected_profile="medium",
        expected_backend_model_id="unit-test/xlsr-large",
    )

    assert isinstance(loaded.model, MLPClassifier)
    assert loaded.artifact_metadata is not None
    assert loaded.artifact_metadata["backend_model_id"] == "unit-test/xlsr-large"


def test_read_training_report_feature_size(
    tmp_path: Path,
) -> None:
    """Feature-vector size should be read from the persisted training report."""
    report_path = tmp_path / "training_report.json"
    report_path.write_text(
        json.dumps({"feature_vector_size": 193}),
        encoding="utf-8",
    )

    feature_size = em._read_training_report_feature_size(report_path)

    assert feature_size == 193


def test_build_training_report_tracks_corpus_vs_effective_samples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Report should capture corpus and effective sample counts for traceability."""
    monkeypatch.setattr(
        em,
        "get_settings",
        lambda: SimpleNamespace(dataset=SimpleNamespace(glob_pattern="unused")),
    )
    monkeypatch.setattr(
        em.glob,
        "glob",
        lambda _pattern: [f"file_{idx}.wav" for idx in range(10)],
    )

    report = em._build_training_report(
        accuracy=0.95,
        macro_f1=0.91,
        ser_metrics={
            "labels": ["happy", "sad"],
            "uar": 0.75,
            "macro_f1": 0.91,
            "per_class_recall": {"happy": 1.0, "sad": 0.5},
            "confusion_matrix": [[2, 0], [1, 1]],
        },
        train_samples=6,
        test_samples=3,
        feature_vector_size=193,
        labels=["happy", "happy", "sad"],
        artifacts=em.PersistedArtifacts(
            pickle_path=Path("ser_model.pkl"),
            secure_path=None,
        ),
        artifact_metadata={
            "artifact_version": em.MODEL_ARTIFACT_VERSION,
            "artifact_schema_version": "v2",
            "created_at_utc": "2026-01-01T00:00:00+00:00",
            "feature_vector_size": 193,
            "training_samples": 6,
            "labels": ["happy", "sad"],
            "backend_id": "handcrafted",
            "profile": "fast",
            "feature_dim": 193,
            "frame_size_seconds": 3.0,
            "frame_stride_seconds": 1.0,
            "pooling_strategy": "mean",
        },
    )

    assert report["dataset_corpus_samples"] == 10
    assert report["dataset_effective_samples"] == 9
    assert report["dataset_skipped_samples"] == 1
    assert report["feature_vector_size"] == 193
    assert report["artifact_schema_version"] == "v2"
    assert report["metrics"] == {
        "labels": ["happy", "sad"],
        "uar": 0.75,
        "macro_f1": 0.91,
        "per_class_recall": {"happy": 1.0, "sad": 0.5},
        "confusion_matrix": [[2, 0], [1, 1]],
    }
    assert report["artifact_metadata"] == {
        "artifact_version": em.MODEL_ARTIFACT_VERSION,
        "artifact_schema_version": "v2",
        "created_at_utc": "2026-01-01T00:00:00+00:00",
        "feature_vector_size": 193,
        "training_samples": 6,
        "labels": ["happy", "sad"],
        "backend_id": "handcrafted",
        "profile": "fast",
        "feature_dim": 193,
        "frame_size_seconds": 3.0,
        "frame_stride_seconds": 1.0,
        "pooling_strategy": "mean",
    }


def test_build_training_report_includes_optional_data_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Report should include optional data-controls block when provided."""
    monkeypatch.setattr(
        em,
        "get_settings",
        lambda: SimpleNamespace(dataset=SimpleNamespace(glob_pattern="unused")),
    )
    monkeypatch.setattr(em.glob, "glob", lambda _pattern: ["file_0.wav"])

    report = em._build_training_report(
        accuracy=0.5,
        macro_f1=0.5,
        ser_metrics={
            "labels": ["happy", "sad"],
            "uar": 0.5,
            "macro_f1": 0.5,
            "per_class_recall": {"happy": 0.5, "sad": 0.5},
            "confusion_matrix": [[1, 1], [1, 1]],
        },
        train_samples=2,
        test_samples=2,
        feature_vector_size=4,
        labels=["happy", "sad", "happy", "sad"],
        artifacts=em.PersistedArtifacts(
            pickle_path=Path("ser_model.pkl"),
            secure_path=None,
        ),
        artifact_metadata={
            "artifact_version": em.MODEL_ARTIFACT_VERSION,
            "artifact_schema_version": "v2",
            "created_at_utc": "2026-01-01T00:00:00+00:00",
            "feature_vector_size": 4,
            "training_samples": 2,
            "labels": ["happy", "sad"],
            "backend_id": "hf_xlsr",
            "profile": "medium",
            "feature_dim": 4,
            "frame_size_seconds": 1.0,
            "frame_stride_seconds": 1.0,
            "pooling_strategy": "mean_std",
        },
        data_controls={"medium_noise_controls": {"min_window_std": 0.1}},
    )

    assert report["data_controls"] == {"medium_noise_controls": {"min_window_std": 0.1}}


def test_predict_emotions_detailed_uses_predict_proba(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Detailed inference should carry per-frame confidence/probabilities."""
    frames = [
        FeatureFrame(
            start_seconds=0.0,
            end_seconds=1.0,
            features=np.asarray([1.0, 1.0], dtype=np.float64),
        ),
        FeatureFrame(
            start_seconds=1.0,
            end_seconds=2.0,
            features=np.asarray([2.0, 2.0], dtype=np.float64),
        ),
    ]
    model = PredictProbaModel(
        predictions=["happy", "sad"],
        probabilities=[[0.8, 0.2], [0.4, 0.6]],
        classes=["happy", "sad"],
    )

    monkeypatch.setattr(em, "extract_feature_frames", lambda _file: frames)
    monkeypatch.setattr(
        em,
        "load_model",
        lambda: em.LoadedModel(
            model=model,
            expected_feature_size=2,
        ),
    )

    detailed = em.predict_emotions_detailed("sample.wav")
    legacy = em.predict_emotions("sample.wav")

    assert detailed.schema_version == OUTPUT_SCHEMA_VERSION
    assert [item.confidence for item in detailed.frames] == pytest.approx([0.8, 0.6])
    assert detailed.frames[0].probabilities == {"happy": 0.8, "sad": 0.2}
    assert detailed.frames[1].probabilities == {"happy": 0.4, "sad": 0.6}
    assert [
        (seg.emotion, seg.start_seconds, seg.end_seconds) for seg in detailed.segments
    ] == [
        ("happy", 0.0, 1.0),
        ("sad", 1.0, 2.0),
    ]
    assert legacy == [
        em.EmotionSegment("happy", 0.0, 1.0),
        em.EmotionSegment("sad", 1.0, 2.0),
    ]


def test_predict_emotions_detailed_falls_back_without_predict_proba(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Models without probabilities should use deterministic confidence fallback."""
    frames = [
        FeatureFrame(
            start_seconds=0.0,
            end_seconds=1.5,
            features=np.asarray([1.0, 1.0], dtype=np.float64),
        ),
        FeatureFrame(
            start_seconds=1.0,
            end_seconds=2.0,
            features=np.asarray([2.0, 2.0], dtype=np.float64),
        ),
    ]
    model = PredictOnlyModel(predictions=["neutral", "neutral"])

    monkeypatch.setattr(em, "extract_feature_frames", lambda _file: frames)
    monkeypatch.setattr(
        em,
        "load_model",
        lambda: em.LoadedModel(
            model=model,
            expected_feature_size=2,
        ),
    )

    detailed = em.predict_emotions_detailed("sample.wav")

    assert [frame.confidence for frame in detailed.frames] == [1.0, 1.0]
    assert [frame.probabilities for frame in detailed.frames] == [None, None]
    assert detailed.segments == [
        em.SegmentPrediction(
            emotion="neutral",
            start_seconds=0.0,
            end_seconds=2.0,
            confidence=1.0,
            probabilities=None,
        )
    ]
