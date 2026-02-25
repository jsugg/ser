"""Tests for accurate-profile training artifact persistence and metadata guards."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ser.models import emotion_model as em


def _settings_stub(tmp_path: Path) -> SimpleNamespace:
    """Builds minimal settings stub required by accurate training helpers."""
    return SimpleNamespace(
        dataset=SimpleNamespace(glob_pattern=str(tmp_path / "*.wav")),
        tmp_folder=tmp_path / "tmp",
        models=SimpleNamespace(
            folder=tmp_path / "models",
            training_report_file=tmp_path / "training_report_accurate.json",
            accurate_model_id="openai/whisper-large-v3",
            accurate_research_model_id="iic/emotion2vec_plus_large",
            huggingface_cache_root=tmp_path / "model-cache" / "huggingface",
            modelscope_cache_root=tmp_path / "model-cache" / "modelscope" / "hub",
        ),
        runtime_flags=SimpleNamespace(restricted_backends=True),
        medium_runtime=SimpleNamespace(
            pool_window_size_seconds=1.0,
            pool_window_stride_seconds=1.0,
        ),
    )


def _dummy_classifier() -> Pipeline:
    """Creates deterministic classifier for artifact-persistence tests."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", DummyClassifier(strategy="most_frequent")),
        ]
    )


def test_train_accurate_model_requires_labeled_dataset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Accurate training should fail fast when dataset loading returns no samples."""
    monkeypatch.setattr(em, "get_settings", lambda: _settings_stub(tmp_path))
    monkeypatch.setattr(em, "load_labeled_audio_paths", lambda: None)

    with pytest.raises(RuntimeError, match="Dataset not loaded"):
        em.train_accurate_model()


def test_train_accurate_model_persists_whisper_profile_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Accurate training should persist explicit hf_whisper/accurate metadata guards."""
    settings = _settings_stub(tmp_path)
    train_samples = [("train_0.wav", "happy"), ("train_1.wav", "sad")]
    test_samples = [("test_0.wav", "happy"), ("test_1.wav", "sad")]
    split_metadata = em.MediumSplitMetadata(
        split_strategy="group_shuffle_split",
        speaker_grouped=True,
        speaker_id_coverage=1.0,
        train_unique_speakers=2,
        test_unique_speakers=2,
        speaker_overlap_count=0,
    )
    x_train = np.asarray(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.2, 0.3, 0.4, 0.5],
            [0.3, 0.4, 0.5, 0.6],
            [0.4, 0.5, 0.6, 0.7],
        ],
        dtype=np.float64,
    )
    y_train = ["happy", "sad", "happy", "sad"]
    x_test = np.asarray(
        [
            [0.5, 0.6, 0.7, 0.8],
            [0.6, 0.7, 0.8, 0.9],
        ],
        dtype=np.float64,
    )
    y_test = ["happy", "sad"]
    captured: dict[str, object] = {}

    monkeypatch.setattr(em, "get_settings", lambda: settings)
    monkeypatch.setattr(
        em,
        "load_labeled_audio_paths",
        lambda: [*train_samples, *test_samples],
    )
    monkeypatch.setattr(
        em,
        "_split_labeled_audio_samples",
        lambda _samples: (train_samples, test_samples, split_metadata),
    )

    def _build_dataset(
        *,
        samples: list[em.LabeledAudioSample],
        backend: em.WhisperBackend,
        cache: em.EmbeddingCache,
        model_id: str | None = None,
        backend_id: str = em.ACCURATE_BACKEND_ID,
    ) -> tuple[np.ndarray, list[str]]:
        del backend, cache, model_id
        assert backend_id == em.ACCURATE_BACKEND_ID
        if samples == train_samples:
            return x_train, y_train
        if samples == test_samples:
            return x_test, y_test
        raise AssertionError(f"Unexpected sample partition: {samples!r}")

    monkeypatch.setattr(em, "_build_accurate_feature_dataset", _build_dataset)
    monkeypatch.setattr(em, "_create_classifier", _dummy_classifier)
    monkeypatch.setattr(
        em.glob,
        "glob",
        lambda _pattern: [f"corpus_{index}.wav" for index in range(6)],
    )

    def _persist_artifacts(
        *,
        model: em.EmotionClassifier,
        artifact: dict[str, object],
    ) -> em.PersistedArtifacts:
        del model
        captured["artifact"] = artifact
        return em.PersistedArtifacts(
            pickle_path=tmp_path / "ser_model_accurate.pkl",
            secure_path=None,
        )

    def _persist_report(report: dict[str, object], path: Path) -> None:
        captured["report"] = report
        captured["report_path"] = path

    monkeypatch.setattr(em, "_persist_model_artifacts", _persist_artifacts)
    monkeypatch.setattr(em, "_persist_training_report", _persist_report)

    em.train_accurate_model()

    artifact = captured.get("artifact")
    assert isinstance(artifact, dict)
    metadata = artifact.get("metadata")
    assert isinstance(metadata, dict)
    assert metadata["backend_id"] == "hf_whisper"
    assert metadata["profile"] == "accurate"
    assert metadata["backend_model_id"] == settings.models.accurate_model_id
    assert metadata["pooling_strategy"] == "mean_std"
    assert metadata["feature_dim"] == 4

    report = captured.get("report")
    assert isinstance(report, dict)
    artifact_metadata = report.get("artifact_metadata")
    assert isinstance(artifact_metadata, dict)
    assert artifact_metadata["backend_id"] == "hf_whisper"
    assert artifact_metadata["profile"] == "accurate"
    assert artifact_metadata["backend_model_id"] == settings.models.accurate_model_id
    assert artifact_metadata["pooling_strategy"] == "mean_std"
    data_controls = report.get("data_controls")
    assert isinstance(data_controls, dict)
    grouped = data_controls.get("accurate_grouped_evaluation")
    assert isinstance(grouped, dict)
    assert grouped["split_strategy"] == "group_shuffle_split"
    assert grouped["speaker_overlap_count"] == 0

    report_path = captured.get("report_path")
    assert report_path == settings.models.training_report_file


def test_train_accurate_model_uses_configured_model_id(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Accurate training should initialize backend/dataset with configured model id."""
    settings = _settings_stub(tmp_path)
    settings.models.accurate_model_id = "unit-test/whisper-tiny"
    train_samples = [("train_0.wav", "happy"), ("train_1.wav", "sad")]
    test_samples = [("test_0.wav", "happy"), ("test_1.wav", "sad")]
    split_metadata = em.MediumSplitMetadata(
        split_strategy="group_shuffle_split",
        speaker_grouped=True,
        speaker_id_coverage=1.0,
        train_unique_speakers=2,
        test_unique_speakers=2,
        speaker_overlap_count=0,
    )
    x_train = np.asarray([[0.1, 0.2], [0.2, 0.3]], dtype=np.float64)
    y_train = ["happy", "sad"]
    x_test = np.asarray([[0.3, 0.4], [0.4, 0.5]], dtype=np.float64)
    y_test = ["happy", "sad"]
    captured: dict[str, object] = {"dataset_model_ids": []}

    class _BackendStub:
        def __init__(self, *, model_id: str, cache_dir: Path) -> None:
            captured["backend_model_id"] = model_id
            captured["backend_cache_dir"] = cache_dir

    monkeypatch.setattr(em, "WhisperBackend", _BackendStub)
    monkeypatch.setattr(em, "get_settings", lambda: settings)
    monkeypatch.setattr(
        em,
        "load_labeled_audio_paths",
        lambda: [*train_samples, *test_samples],
    )
    monkeypatch.setattr(
        em,
        "_split_labeled_audio_samples",
        lambda _samples: (train_samples, test_samples, split_metadata),
    )

    def _build_dataset(
        *,
        samples: list[em.LabeledAudioSample],
        backend: _BackendStub,
        cache: em.EmbeddingCache,
        model_id: str | None = None,
        backend_id: str = em.ACCURATE_BACKEND_ID,
    ) -> tuple[np.ndarray, list[str]]:
        del backend, cache
        assert model_id is not None
        assert backend_id == em.ACCURATE_BACKEND_ID
        dataset_model_ids = captured["dataset_model_ids"]
        assert isinstance(dataset_model_ids, list)
        dataset_model_ids.append(model_id)
        if samples == train_samples:
            return x_train, y_train
        if samples == test_samples:
            return x_test, y_test
        raise AssertionError(f"Unexpected sample partition: {samples!r}")

    monkeypatch.setattr(em, "_build_accurate_feature_dataset", _build_dataset)
    monkeypatch.setattr(em, "_create_classifier", _dummy_classifier)
    monkeypatch.setattr(
        em.glob,
        "glob",
        lambda _pattern: [f"corpus_{index}.wav" for index in range(4)],
    )
    monkeypatch.setattr(
        em,
        "_persist_model_artifacts",
        lambda **_kwargs: em.PersistedArtifacts(
            pickle_path=tmp_path / "ser_model_accurate.pkl",
            secure_path=None,
        ),
    )
    monkeypatch.setattr(em, "_persist_training_report", lambda *_args, **_kwargs: None)

    em.train_accurate_model()

    assert captured["backend_model_id"] == "unit-test/whisper-tiny"
    assert captured["backend_cache_dir"] == settings.models.huggingface_cache_root
    dataset_model_ids = captured["dataset_model_ids"]
    assert isinstance(dataset_model_ids, list)
    assert dataset_model_ids == ["unit-test/whisper-tiny", "unit-test/whisper-tiny"]


def test_resolve_accurate_research_model_id_uses_configured_value(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Accurate-research model id resolver should honor configured settings value."""
    settings = _settings_stub(tmp_path)
    settings.models.accurate_research_model_id = "unit-test/emotion2vec-plus"
    monkeypatch.setattr(em, "get_settings", lambda: settings)

    assert em.resolve_accurate_research_model_id() == "unit-test/emotion2vec-plus"


def test_train_accurate_research_model_persists_emotion2vec_profile_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Accurate-research training should persist emotion2vec/accurate-research metadata."""
    settings = _settings_stub(tmp_path)
    settings.models.accurate_research_model_id = "unit-test/emotion2vec-plus"
    train_samples = [("train_0.wav", "happy"), ("train_1.wav", "sad")]
    test_samples = [("test_0.wav", "happy"), ("test_1.wav", "sad")]
    split_metadata = em.MediumSplitMetadata(
        split_strategy="group_shuffle_split",
        speaker_grouped=True,
        speaker_id_coverage=1.0,
        train_unique_speakers=2,
        test_unique_speakers=2,
        speaker_overlap_count=0,
    )
    x_train = np.asarray([[0.1, 0.2], [0.2, 0.3]], dtype=np.float64)
    y_train = ["happy", "sad"]
    x_test = np.asarray([[0.3, 0.4], [0.4, 0.5]], dtype=np.float64)
    y_test = ["happy", "sad"]
    captured: dict[str, object] = {"dataset_model_ids": []}

    class _BackendStub:
        def __init__(
            self,
            *,
            model_id: str,
            modelscope_cache_root: Path,
            huggingface_cache_root: Path,
        ) -> None:
            captured["backend_model_id"] = model_id
            captured["backend_modelscope_cache_root"] = modelscope_cache_root
            captured["backend_huggingface_cache_root"] = huggingface_cache_root

    monkeypatch.setattr(em, "Emotion2VecBackend", _BackendStub)
    monkeypatch.setattr(em, "get_settings", lambda: settings)
    monkeypatch.setattr(
        em,
        "load_labeled_audio_paths",
        lambda: [*train_samples, *test_samples],
    )
    monkeypatch.setattr(
        em,
        "_split_labeled_audio_samples",
        lambda _samples: (train_samples, test_samples, split_metadata),
    )

    def _build_dataset(
        *,
        samples: list[em.LabeledAudioSample],
        backend: _BackendStub,
        cache: em.EmbeddingCache,
        model_id: str | None = None,
        backend_id: str = em.ACCURATE_BACKEND_ID,
    ) -> tuple[np.ndarray, list[str]]:
        del backend, cache
        assert model_id is not None
        assert backend_id == em.ACCURATE_RESEARCH_BACKEND_ID
        dataset_model_ids = captured["dataset_model_ids"]
        assert isinstance(dataset_model_ids, list)
        dataset_model_ids.append(model_id)
        if samples == train_samples:
            return x_train, y_train
        if samples == test_samples:
            return x_test, y_test
        raise AssertionError(f"Unexpected sample partition: {samples!r}")

    monkeypatch.setattr(em, "_build_accurate_feature_dataset", _build_dataset)
    monkeypatch.setattr(em, "_create_classifier", _dummy_classifier)
    monkeypatch.setattr(
        em.glob,
        "glob",
        lambda _pattern: [f"corpus_{index}.wav" for index in range(4)],
    )
    monkeypatch.setattr(
        em,
        "_persist_model_artifacts",
        lambda **_kwargs: em.PersistedArtifacts(
            pickle_path=tmp_path / "ser_model_accurate_research.pkl",
            secure_path=None,
        ),
    )

    def _persist_report(report: dict[str, object], path: Path) -> None:
        captured["report"] = report
        captured["report_path"] = path

    monkeypatch.setattr(em, "_persist_training_report", _persist_report)

    em.train_accurate_research_model()

    assert captured["backend_model_id"] == "unit-test/emotion2vec-plus"
    assert (
        captured["backend_modelscope_cache_root"]
        == settings.models.modelscope_cache_root
    )
    assert (
        captured["backend_huggingface_cache_root"]
        == settings.models.huggingface_cache_root
    )
    dataset_model_ids = captured["dataset_model_ids"]
    assert isinstance(dataset_model_ids, list)
    assert dataset_model_ids == [
        "unit-test/emotion2vec-plus",
        "unit-test/emotion2vec-plus",
    ]

    report = captured.get("report")
    assert isinstance(report, dict)
    artifact_metadata = report.get("artifact_metadata")
    assert isinstance(artifact_metadata, dict)
    assert artifact_metadata["backend_id"] == "emotion2vec"
    assert artifact_metadata["profile"] == "accurate-research"
    assert (
        artifact_metadata["backend_model_id"]
        == settings.models.accurate_research_model_id
    )
    assert artifact_metadata["pooling_strategy"] == "mean_std"

    report_path = captured.get("report_path")
    assert report_path == settings.models.training_report_file
