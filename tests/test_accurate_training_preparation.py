"""Contracts for accurate-profile training preparation helper extraction."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

import ser.models.accurate_training_preparation as atp
from ser.config import AppConfig
from ser.data.manifest import MANIFEST_SCHEMA_VERSION, Utterance
from ser.models.dataset_splitting import MediumSplitMetadata


def _settings_stub(tmp_path: Path) -> AppConfig:
    return cast(
        AppConfig,
        SimpleNamespace(
            tmp_folder=tmp_path / "tmp",
            models=SimpleNamespace(
                huggingface_cache_root=tmp_path / "model-cache" / "huggingface",
                modelscope_cache_root=tmp_path / "model-cache" / "modelscope" / "hub",
            ),
        ),
    )


def _utterance(sample_id: str, label: str) -> Utterance:
    return Utterance(
        schema_version=MANIFEST_SCHEMA_VERSION,
        sample_id=f"ravdess:{sample_id}",
        corpus="ravdess",
        audio_path=Path(f"{sample_id}.wav"),
        label=label,
        speaker_id=f"ravdess:{sample_id}",
        language="en",
    )


def _split_metadata() -> MediumSplitMetadata:
    return MediumSplitMetadata(
        split_strategy="group_shuffle_split",
        speaker_grouped=True,
        speaker_id_coverage=1.0,
        train_unique_speakers=1,
        test_unique_speakers=1,
        speaker_overlap_count=0,
    )


def test_prepare_accurate_whisper_training_wires_backend_and_cache(
    tmp_path: Path,
) -> None:
    """Whisper preparation should wire model/runtime selectors and cache directory."""
    settings = _settings_stub(tmp_path)
    utterances = [_utterance("train", "happy"), _utterance("test", "sad")]
    train_utterances = [utterances[0]]
    test_utterances = [utterances[1]]
    captured: dict[str, object] = {"cache_dirs": []}

    def _build_feature_dataset(
        partition: list[Utterance],
        backend: object,
        cache: atp.EmbeddingCache,
        model_id: str,
    ) -> tuple[np.ndarray, list[str], list[dict[str, str]]]:
        del backend
        assert model_id == "unit-test/whisper-tiny"
        cache_dirs = captured["cache_dirs"]
        assert isinstance(cache_dirs, list)
        cache_dirs.append(cache._cache_dir)
        if partition == train_utterances:
            return (
                np.asarray([[0.1, 0.2]], dtype=np.float64),
                ["happy"],
                [{"sample_id": "train"}],
            )
        if partition == test_utterances:
            return (
                np.asarray([[0.3, 0.4]], dtype=np.float64),
                ["sad"],
                [{"sample_id": "test"}],
            )
        raise AssertionError(f"Unexpected partition: {partition!r}")

    resolved_utterances, prepared = atp.prepare_accurate_whisper_training(
        settings=settings,
        logger=atp.logging.getLogger("tests.accurate_training_preparation.whisper"),
        load_utterances_for_training=lambda: utterances,
        ensure_dataset_consents_for_training=lambda loaded: captured.update(
            {"consented": list(loaded)}
        ),
        split_utterances=lambda _loaded: (
            train_utterances,
            test_utterances,
            _split_metadata(),
        ),
        resolve_model_id=lambda: "unit-test/whisper-tiny",
        resolve_runtime_selectors=lambda: ("cpu", "float32"),
        build_backend=lambda model_id, runtime_device, runtime_dtype: (
            captured.update(
                {
                    "backend_model_id": model_id,
                    "backend_cache_dir": settings.models.huggingface_cache_root,
                    "backend_device": runtime_device,
                    "backend_dtype": runtime_dtype,
                }
            ),
            cast(atp.WhisperBackend, object()),
        )[1],
        build_feature_dataset=_build_feature_dataset,
        embedding_cache_path=settings.tmp_folder / "accurate_embeddings",
    )

    assert resolved_utterances == utterances
    assert prepared.model_id == "unit-test/whisper-tiny"
    assert prepared.runtime_device == "cpu"
    assert prepared.runtime_dtype == "float32"
    assert prepared.y_train == ["happy"]
    assert prepared.y_test == ["sad"]
    assert captured["consented"] == utterances
    assert captured["backend_model_id"] == "unit-test/whisper-tiny"
    assert captured["backend_cache_dir"] == settings.models.huggingface_cache_root
    assert captured["backend_device"] == "cpu"
    assert captured["backend_dtype"] == "float32"
    assert captured["cache_dirs"] == [
        settings.tmp_folder / "accurate_embeddings",
        settings.tmp_folder / "accurate_embeddings",
    ]


def test_prepare_accurate_research_training_wires_backend_and_cache(
    tmp_path: Path,
) -> None:
    """Research preparation should wire emotion2vec backend and cache roots."""
    settings = _settings_stub(tmp_path)
    utterances = [_utterance("train", "happy"), _utterance("test", "sad")]
    train_utterances = [utterances[0]]
    test_utterances = [utterances[1]]
    captured: dict[str, object] = {"cache_dirs": []}

    def _build_feature_dataset(
        partition: list[Utterance],
        backend: object,
        cache: atp.EmbeddingCache,
        model_id: str,
    ) -> tuple[np.ndarray, list[str], list[dict[str, str]]]:
        del backend
        assert model_id == "unit-test/emotion2vec-plus"
        cache_dirs = captured["cache_dirs"]
        assert isinstance(cache_dirs, list)
        cache_dirs.append(cache._cache_dir)
        if partition == train_utterances:
            return (
                np.asarray([[0.1, 0.2]], dtype=np.float64),
                ["happy"],
                [{"sample_id": "train"}],
            )
        if partition == test_utterances:
            return (
                np.asarray([[0.3, 0.4]], dtype=np.float64),
                ["sad"],
                [{"sample_id": "test"}],
            )
        raise AssertionError(f"Unexpected partition: {partition!r}")

    resolved_utterances, prepared = atp.prepare_accurate_research_training(
        settings=settings,
        logger=atp.logging.getLogger("tests.accurate_training_preparation.research"),
        load_utterances_for_training=lambda: utterances,
        ensure_dataset_consents_for_training=lambda loaded: captured.update(
            {"consented": list(loaded)}
        ),
        split_utterances=lambda _loaded: (
            train_utterances,
            test_utterances,
            _split_metadata(),
        ),
        resolve_model_id=lambda: "unit-test/emotion2vec-plus",
        resolve_runtime_selectors=lambda: ("mps", "float16"),
        build_backend=lambda model_id, runtime_device, _runtime_dtype: (
            captured.update(
                {
                    "backend_model_id": model_id,
                    "backend_device": runtime_device,
                    "backend_modelscope_cache_root": (
                        settings.models.modelscope_cache_root
                    ),
                    "backend_huggingface_cache_root": (
                        settings.models.huggingface_cache_root
                    ),
                }
            ),
            cast(atp.Emotion2VecBackend, object()),
        )[1],
        build_feature_dataset=_build_feature_dataset,
        embedding_cache_path=settings.tmp_folder / "accurate_research_embeddings",
    )

    assert resolved_utterances == utterances
    assert prepared.model_id == "unit-test/emotion2vec-plus"
    assert prepared.runtime_device == "mps"
    assert prepared.runtime_dtype == "float16"
    assert prepared.y_train == ["happy"]
    assert prepared.y_test == ["sad"]
    assert captured["consented"] == utterances
    assert captured["backend_model_id"] == "unit-test/emotion2vec-plus"
    assert captured["backend_device"] == "mps"
    assert (
        captured["backend_modelscope_cache_root"]
        == settings.models.modelscope_cache_root
    )
    assert (
        captured["backend_huggingface_cache_root"]
        == settings.models.huggingface_cache_root
    )
    assert captured["cache_dirs"] == [
        settings.tmp_folder / "accurate_research_embeddings",
        settings.tmp_folder / "accurate_research_embeddings",
    ]


def test_train_accurate_research_profile_model_enforces_access_and_delegates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Research training helper should guard access before prepare/run delegation."""
    settings = cast(
        AppConfig,
        SimpleNamespace(
            runtime_flags=SimpleNamespace(restricted_backends=True),
            tmp_folder=tmp_path / "tmp",
            models=SimpleNamespace(
                huggingface_cache_root=tmp_path / "hf-cache",
                modelscope_cache_root=tmp_path / "ms-cache",
            ),
        ),
    )
    utterances = [_utterance("train", "happy"), _utterance("test", "sad")]
    prepared = cast(
        atp.AccurateTrainingPreparation[Utterance, MediumSplitMetadata, dict[str, str]],
        object(),
    )
    access_call: dict[str, object] = {}
    prepare_call: dict[str, object] = {}
    run_call: dict[str, object] = {}

    def _fake_prepare_accurate_research_training(**kwargs: object) -> object:
        prepare_call.update(kwargs)
        return utterances, prepared

    monkeypatch.setattr(
        atp,
        "prepare_accurate_research_training",
        _fake_prepare_accurate_research_training,
    )

    atp.train_accurate_research_profile_model(
        settings=settings,
        logger=atp.logging.getLogger("tests.accurate_training_preparation.train"),
        parse_allowed_restricted_backends_env=lambda: {"emotion2vec"},
        load_persisted_backend_consents=lambda *, settings: {
            "emotion2vec": {"consented": True}
        },
        ensure_backend_access=lambda **kwargs: access_call.update(kwargs),
        restricted_backend_id="emotion2vec",
        load_utterances_for_training=lambda: utterances,
        ensure_dataset_consents_for_training=lambda *, utterances: None,
        split_utterances=lambda _loaded: (
            [_loaded[0]],
            [_loaded[1]],
            _split_metadata(),
        ),
        resolve_model_id_for_settings=lambda _settings: "unit-test/emotion2vec-plus",
        resolve_runtime_selectors_for_settings=lambda _settings: ("cpu", "float32"),
        build_backend=lambda model_id, runtime_device, runtime_dtype, _settings: cast(
            atp.Emotion2VecBackend,
            (
                model_id,
                runtime_device,
                runtime_dtype,
                object(),
            )[3],
        ),
        build_feature_dataset=lambda **_kwargs: (
            np.asarray([[0.1, 0.2]], dtype=np.float64),
            ["happy"],
            [{"sample_id": "train"}],
        ),
        run_prepared_training=lambda _prepared, _utterances, _settings: run_call.update(
            {
                "prepared": _prepared,
                "utterances": _utterances,
                "settings": _settings,
            }
        ),
    )

    assert access_call["backend_id"] == "emotion2vec"
    assert access_call["restricted_backends_enabled"] is True
    assert access_call["allowed_restricted_backends"] == {"emotion2vec"}
    assert access_call["persisted_consents"] == {"emotion2vec": {"consented": True}}
    assert prepare_call["settings"] is settings
    assert prepare_call["load_utterances_for_training"] is not None
    assert prepare_call["embedding_cache_path"] == (
        settings.tmp_folder / "accurate_research_embeddings"
    )
    assert run_call == {
        "prepared": prepared,
        "utterances": utterances,
        "settings": settings,
    }


def test_train_accurate_whisper_profile_model_delegates_prepare_and_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Whisper training helper should prepare and run through delegated hooks."""
    settings = cast(
        AppConfig,
        SimpleNamespace(
            tmp_folder=tmp_path / "tmp",
            models=SimpleNamespace(huggingface_cache_root=tmp_path / "hf-cache"),
        ),
    )
    utterances = [_utterance("train", "happy"), _utterance("test", "sad")]
    prepared = cast(
        atp.AccurateTrainingPreparation[Utterance, MediumSplitMetadata, dict[str, str]],
        object(),
    )
    prepare_call: dict[str, object] = {}
    run_call: dict[str, object] = {}

    def _fake_prepare_accurate_whisper_training(**kwargs: object) -> object:
        prepare_call.update(kwargs)
        return utterances, prepared

    monkeypatch.setattr(
        atp,
        "prepare_accurate_whisper_training",
        _fake_prepare_accurate_whisper_training,
    )

    atp.train_accurate_whisper_profile_model(
        settings=settings,
        logger=atp.logging.getLogger("tests.accurate_training_preparation.whisper_run"),
        load_utterances_for_training=lambda: utterances,
        ensure_dataset_consents_for_training=lambda *, utterances: None,
        split_utterances=lambda _loaded: (
            [_loaded[0]],
            [_loaded[1]],
            _split_metadata(),
        ),
        resolve_model_id_for_settings=lambda _settings: "unit-test/whisper-large",
        resolve_runtime_selectors_for_settings=lambda _settings: ("cpu", "float32"),
        build_backend=lambda model_id, runtime_device, runtime_dtype, _settings: cast(
            atp.WhisperBackend,
            (
                model_id,
                runtime_device,
                runtime_dtype,
                object(),
            )[3],
        ),
        build_feature_dataset=lambda **_kwargs: (
            np.asarray([[0.1, 0.2]], dtype=np.float64),
            ["happy"],
            [{"sample_id": "train"}],
        ),
        run_prepared_training=lambda _prepared, _utterances, _settings: run_call.update(
            {
                "prepared": _prepared,
                "utterances": _utterances,
                "settings": _settings,
            }
        ),
    )

    assert prepare_call["settings"] is settings
    assert prepare_call["load_utterances_for_training"] is not None
    assert (
        prepare_call["embedding_cache_path"]
        == settings.tmp_folder / "accurate_embeddings"
    )
    assert run_call == {
        "prepared": prepared,
        "utterances": utterances,
        "settings": settings,
    }


def test_run_accurate_profile_training_from_prepared_delegates_to_orchestration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Prepared accurate payload should delegate to orchestration helper unchanged."""
    utterances = [_utterance("train", "happy"), _utterance("test", "sad")]
    split_metadata = _split_metadata()
    prepared = atp.AccurateTrainingPreparation[
        Utterance, MediumSplitMetadata, dict[str, str]
    ](
        train_utterances=[utterances[0]],
        test_utterances=[utterances[1]],
        split_metadata=split_metadata,
        model_id="unit-test/whisper-tiny",
        runtime_device="cpu",
        runtime_dtype="float32",
        x_train=np.asarray([[0.1, 0.2]], dtype=np.float64),
        y_train=["happy"],
        x_test=np.asarray([[0.3, 0.4]], dtype=np.float64),
        y_test=["sad"],
        test_meta=[{"sample_id": utterances[1].sample_id}],
    )
    captured: dict[str, object] = {}

    def _fake_run_accurate_profile_training(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"status": "ok"}

    monkeypatch.setattr(
        atp,
        "run_accurate_profile_training",
        _fake_run_accurate_profile_training,
    )

    report_destination = tmp_path / "training_report.json"
    result = atp.run_accurate_profile_training_from_prepared(
        prepared=prepared,
        utterances=utterances,
        settings=_settings_stub(tmp_path),
        logger=atp.logging.getLogger("tests.accurate_training_preparation.run"),
        profile_label="Accurate",
        backend_id="hf_whisper",
        profile_id="accurate",
        pooling_strategy="mean_std",
        frame_size_seconds=2.0,
        frame_stride_seconds=0.5,
        create_classifier=lambda: object(),
        min_support=3,
        evaluate_predictions=lambda **_kwargs: atp.TrainingEvaluation(
            accuracy=1.0,
            macro_f1=1.0,
            uar=1.0,
            ser_metrics={"uar": 1.0},
        ),
        attach_grouped_metrics=lambda **_kwargs: {"group_metrics": {}},
        build_model_artifact=lambda **_kwargs: {"metadata": {}},
        extract_artifact_metadata=lambda _artifact: {"profile": "accurate"},
        persist_model_artifacts=lambda _model, _artifact: cast(
            atp.PersistedArtifactsLike,
            SimpleNamespace(
                pickle_path=tmp_path / "model.pkl",
                secure_path=None,
            ),
        ),
        build_provenance_metadata=lambda **_kwargs: {"source": "unit-test"},
        build_dataset_controls=lambda _utterances: {"dataset": "ok"},
        build_grouped_evaluation_controls=lambda _split: {
            "split_strategy": _split.split_strategy
        },
        build_training_report=lambda **_kwargs: {"report": "ok"},
        persist_training_report=lambda _report, _path: None,
        report_destination=report_destination,
    )

    assert result == {"status": "ok"}
    assert captured["prepared"] is prepared
    assert captured["utterances"] == utterances
    assert captured["profile_label"] == "Accurate"
    assert captured["backend_id"] == "hf_whisper"
    assert captured["profile_id"] == "accurate"
    assert captured["pooling_strategy"] == "mean_std"
    assert captured["frame_size_seconds"] == 2.0
    assert captured["frame_stride_seconds"] == 0.5
    assert captured["report_destination"] == report_destination
