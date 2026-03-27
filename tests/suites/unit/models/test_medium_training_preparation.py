"""Contracts for medium-profile training preparation helper extraction."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

import ser.models.medium_training_preparation as mtp
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


def test_prepare_medium_xlsr_training_wires_backend_and_cache(tmp_path: Path) -> None:
    """Medium preparation should wire model/runtime selectors and cache directory."""
    settings = _settings_stub(tmp_path)
    utterances = [_utterance("train", "happy"), _utterance("test", "sad")]
    train_utterances = [utterances[0]]
    test_utterances = [utterances[1]]
    captured: dict[str, object] = {"cache_dirs": []}
    train_noise_stats = {"clipped": 1}
    test_noise_stats = {"clipped": 0}

    def _build_feature_dataset(
        partition: list[Utterance],
        backend: object,
        cache: mtp.EmbeddingCache,
        model_id: str,
    ) -> tuple[np.ndarray, list[str], list[dict[str, str]], dict[str, int]]:
        del backend
        assert model_id == "unit-test/xlsr-medium"
        cache_dirs = captured["cache_dirs"]
        assert isinstance(cache_dirs, list)
        cache_dirs.append(cache._cache_dir)
        if partition == train_utterances:
            return (
                np.asarray([[0.1, 0.2]], dtype=np.float64),
                ["happy"],
                [{"sample_id": "train"}],
                train_noise_stats,
            )
        if partition == test_utterances:
            return (
                np.asarray([[0.3, 0.4]], dtype=np.float64),
                ["sad"],
                [{"sample_id": "test"}],
                test_noise_stats,
            )
        raise AssertionError(f"Unexpected partition: {partition!r}")

    resolved_utterances, prepared = mtp.prepare_medium_xlsr_training(
        settings=settings,
        logger=mtp.logging.getLogger("tests.medium_training_preparation.prepare"),
        load_utterances_for_training=lambda: utterances,
        ensure_dataset_consents_for_training=lambda loaded: captured.update(
            {"consented": list(loaded)}
        ),
        split_utterances=lambda _loaded: (
            train_utterances,
            test_utterances,
            _split_metadata(),
        ),
        resolve_model_id=lambda: "unit-test/xlsr-medium",
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
            cast(mtp.XLSRBackend, object()),
        )[1],
        build_feature_dataset=_build_feature_dataset,
        embedding_cache_path=settings.tmp_folder / "medium_embeddings",
    )

    assert resolved_utterances == utterances
    assert prepared.model_id == "unit-test/xlsr-medium"
    assert prepared.runtime_device == "cpu"
    assert prepared.runtime_dtype == "float32"
    assert prepared.y_train == ["happy"]
    assert prepared.y_test == ["sad"]
    assert prepared.train_noise_stats == train_noise_stats
    assert prepared.test_noise_stats == test_noise_stats
    assert captured["consented"] == utterances
    assert captured["backend_model_id"] == "unit-test/xlsr-medium"
    assert captured["backend_cache_dir"] == settings.models.huggingface_cache_root
    assert captured["backend_device"] == "cpu"
    assert captured["backend_dtype"] == "float32"
    assert captured["cache_dirs"] == [
        settings.tmp_folder / "medium_embeddings",
        settings.tmp_folder / "medium_embeddings",
    ]


def test_run_medium_profile_training_from_prepared_delegates_to_orchestration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Prepared medium payload should delegate to orchestration helper unchanged."""
    utterances = [_utterance("train", "happy"), _utterance("test", "sad")]
    prepared = mtp.MediumTrainingPreparation[
        Utterance, MediumSplitMetadata, dict[str, str], dict[str, int]
    ](
        train_utterances=[utterances[0]],
        test_utterances=[utterances[1]],
        split_metadata=_split_metadata(),
        model_id="unit-test/xlsr-medium",
        runtime_device="cpu",
        runtime_dtype="float32",
        x_train=np.asarray([[0.1, 0.2]], dtype=np.float64),
        y_train=["happy"],
        x_test=np.asarray([[0.3, 0.4]], dtype=np.float64),
        y_test=["sad"],
        test_meta=[{"sample_id": utterances[1].sample_id}],
        train_noise_stats={"clipped": 1},
        test_noise_stats={"clipped": 0},
    )
    captured: dict[str, object] = {}

    def _fake_run_medium_profile_training(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"status": "ok"}

    monkeypatch.setattr(
        mtp,
        "run_medium_profile_training",
        _fake_run_medium_profile_training,
    )

    result = mtp.run_medium_profile_training_from_prepared(
        prepared=prepared,
        utterances=utterances,
        settings=_settings_stub(tmp_path),
        logger=mtp.logging.getLogger("tests.medium_training_preparation.run"),
        profile_label="Medium",
        backend_id="hf_xlsr",
        profile_id="medium",
        pooling_strategy="mean_std",
        create_classifier=lambda: object(),
        min_support=3,
        evaluate_predictions=lambda **_kwargs: mtp.TrainingEvaluation(
            accuracy=1.0,
            macro_f1=1.0,
            uar=1.0,
            ser_metrics={"uar": 1.0},
        ),
        attach_grouped_metrics=lambda **_kwargs: {"group_metrics": {}},
        build_model_artifact=lambda **_kwargs: {"metadata": {}},
        extract_artifact_metadata=lambda _artifact: {"profile": "medium"},
        persist_model_artifacts=lambda _model, _artifact: cast(
            mtp.PersistedArtifactsLike,
            SimpleNamespace(
                pickle_path=tmp_path / "model.pkl",
                secure_path=None,
            ),
        ),
        build_provenance_metadata=lambda **_kwargs: {"source": "unit-test"},
        build_dataset_controls=lambda _utterances: {"dataset": "ok"},
        build_medium_noise_controls=lambda **_kwargs: {"noise": "ok"},
        build_grouped_evaluation_controls=lambda _split: {"split_strategy": _split.split_strategy},
        build_training_report=lambda **_kwargs: {"report": "ok"},
        persist_training_report=lambda _report, _path: None,
    )

    assert result == {"status": "ok"}
    assert captured["prepared"] is prepared
    assert captured["utterances"] == utterances
    assert captured["profile_label"] == "Medium"
    assert captured["backend_id"] == "hf_xlsr"
    assert captured["profile_id"] == "medium"
    assert captured["pooling_strategy"] == "mean_std"


def test_train_medium_profile_model_delegates_to_prepare_and_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Medium model entry helper should compose prepare + run seams."""
    settings = _settings_stub(tmp_path)
    utterances = [_utterance("train", "happy")]
    fake_prepared = object()
    captured: dict[str, object] = {}

    def _fake_prepare_medium_xlsr_training(
        **kwargs: object,
    ) -> tuple[list[Utterance], object]:  # noqa: E501
        captured["prepare_kwargs"] = kwargs
        return utterances, fake_prepared

    def _fake_run_medium_profile_training_from_prepared(
        **kwargs: object,
    ) -> dict[str, object]:
        captured["run_kwargs"] = kwargs
        return {"status": "ok"}

    monkeypatch.setattr(
        mtp,
        "prepare_medium_xlsr_training",
        _fake_prepare_medium_xlsr_training,
    )
    monkeypatch.setattr(
        mtp,
        "run_medium_profile_training_from_prepared",
        _fake_run_medium_profile_training_from_prepared,
    )

    def _ensure_dataset_consents_for_training(*, utterances: list[Utterance]) -> None:
        captured["consents_utterances"] = utterances

    def _resolve_model_id_for_settings(active_settings: AppConfig) -> str:
        assert active_settings is settings
        return "unit-test/xlsr-medium"

    def _resolve_runtime_selectors_for_settings(
        active_settings: AppConfig,
    ) -> tuple[str, str]:
        assert active_settings is settings
        return "cpu", "float32"

    def _build_backend(
        model_id: str,
        runtime_device: str,
        runtime_dtype: str,
        active_settings: AppConfig,
    ) -> mtp.XLSRBackend:
        del model_id, runtime_device, runtime_dtype
        assert active_settings is settings
        return cast(mtp.XLSRBackend, object())

    def _build_feature_dataset(
        *,
        utterances: list[Utterance],
        backend: mtp.XLSRBackend,
        cache: mtp.EmbeddingCache,
        model_id: str,
    ) -> tuple[np.ndarray, list[str], list[dict[str, str]], dict[str, int]]:
        del utterances, backend, cache, model_id
        return (
            np.asarray([[0.1, 0.2]], dtype=np.float64),
            ["happy"],
            [{"sample_id": "train"}],
            {"clipped": 0},
        )

    mtp.train_medium_profile_model(
        settings=settings,
        logger=mtp.logging.getLogger("tests.medium_training_preparation.entry"),
        load_utterances_for_training=lambda: utterances,
        ensure_dataset_consents_for_training=_ensure_dataset_consents_for_training,
        split_utterances=lambda loaded: (loaded, loaded, _split_metadata()),
        resolve_model_id_for_settings=_resolve_model_id_for_settings,
        resolve_runtime_selectors_for_settings=_resolve_runtime_selectors_for_settings,
        build_backend=_build_backend,
        build_feature_dataset=_build_feature_dataset,
        create_classifier=lambda: object(),
        min_support=2,
        evaluate_predictions=lambda **_kwargs: mtp.TrainingEvaluation(
            accuracy=1.0,
            macro_f1=1.0,
            uar=1.0,
            ser_metrics={"uar": 1.0},
        ),
        attach_grouped_metrics=lambda **_kwargs: {"group_metrics": {}},
        build_model_artifact=lambda **_kwargs: {"metadata": {}},
        extract_artifact_metadata=lambda _artifact: {"profile": "medium"},
        persist_model_artifacts=lambda _model, _artifact: cast(
            mtp.PersistedArtifactsLike,
            SimpleNamespace(
                pickle_path=tmp_path / "model.pkl",
                secure_path=None,
            ),
        ),
        build_provenance_metadata=lambda **_kwargs: {"source": "unit-test"},
        build_dataset_controls=lambda _utterances: {"dataset": "ok"},
        build_medium_noise_controls=lambda **_kwargs: {"noise": "ok"},
        build_grouped_evaluation_controls=lambda _split: {"split_strategy": _split.split_strategy},
        build_training_report=lambda **_kwargs: {"report": "ok"},
        persist_training_report=lambda _report, _path: None,
        profile_label="Medium",
        backend_id="hf_xlsr",
        profile_id="medium",
        pooling_strategy="mean_std",
        embedding_cache_name="medium_embeddings",
    )

    prepare_kwargs = captured["prepare_kwargs"]
    assert isinstance(prepare_kwargs, dict)
    assert prepare_kwargs["settings"] is settings
    assert (
        prepare_kwargs["embedding_cache_path"] == settings.tmp_folder / "medium_embeddings"
    )  # noqa: E501
    assert callable(prepare_kwargs["ensure_dataset_consents_for_training"])
    assert callable(prepare_kwargs["resolve_model_id"])
    assert callable(prepare_kwargs["resolve_runtime_selectors"])
    assert callable(prepare_kwargs["build_backend"])
    assert callable(prepare_kwargs["build_feature_dataset"])

    run_kwargs = captured["run_kwargs"]
    assert isinstance(run_kwargs, dict)
    assert run_kwargs["prepared"] is fake_prepared
    assert run_kwargs["utterances"] == utterances
    assert run_kwargs["settings"] is settings
    assert run_kwargs["backend_id"] == "hf_xlsr"
    assert run_kwargs["profile_id"] == "medium"


def test_train_medium_profile_entrypoint_preserves_training_contract(
    tmp_path: Path,
) -> None:
    """Entrypoint helper should forward the same medium training contract keys."""
    settings = _settings_stub(tmp_path)
    captured: dict[str, object] = {}

    def _fake_train_medium_profile_model(**kwargs: object) -> None:
        captured.update(kwargs)

    mtp.train_medium_profile_entrypoint(
        settings=settings,
        logger=mtp.logging.getLogger("tests.medium_training_preparation.entrypoint"),
        train_profile_model=_fake_train_medium_profile_model,
        load_utterances_for_training=lambda: [],
        ensure_dataset_consents_for_training=lambda **_kwargs: None,
        split_utterances=lambda loaded: (loaded, loaded, _split_metadata()),
        resolve_model_id_for_settings=lambda _settings: "unit-test/xlsr-medium",
        resolve_runtime_selectors_for_backend=lambda *, settings, backend_id: (
            "cpu",
            "float32",
        ),
        build_backend_for_settings=lambda _model_id, _device, _dtype, _settings: cast(
            mtp.XLSRBackend,
            object(),
        ),
        build_feature_dataset=lambda **_kwargs: (
            np.asarray([[0.1, 0.2]], dtype=np.float64),
            ["happy"],
            [{"sample_id": "train"}],
            {"clipped": 0},
        ),
        create_classifier=lambda: object(),
        min_support=2,
        evaluate_predictions=lambda **_kwargs: mtp.TrainingEvaluation(
            accuracy=1.0,
            macro_f1=1.0,
            uar=1.0,
            ser_metrics={"uar": 1.0},
        ),
        attach_grouped_metrics=lambda **_kwargs: {"group_metrics": {}},
        build_model_artifact=lambda **_kwargs: {"metadata": {}},
        extract_artifact_metadata=lambda _artifact: {"profile": "medium"},
        persist_model_artifacts=lambda _model, _artifact: cast(
            mtp.PersistedArtifactsLike,
            SimpleNamespace(
                pickle_path=tmp_path / "model.pkl",
                secure_path=None,
            ),
        ),
        build_provenance_metadata=lambda **_kwargs: {"source": "unit-test"},
        build_dataset_controls=lambda _utterances: {"dataset": "ok"},
        build_medium_noise_controls=lambda **_kwargs: {"noise": "ok"},
        build_grouped_evaluation_controls=lambda _split: {"split_strategy": _split.split_strategy},
        build_training_report=lambda **_kwargs: {"report": "ok"},
        persist_training_report=lambda _report, _path: None,
        profile_label="Medium",
        backend_id="hf_xlsr",
        profile_id="medium",
        pooling_strategy="mean_std",
        embedding_cache_name="medium_embeddings",
    )

    assert captured["settings"] is settings
    assert callable(captured["resolve_runtime_selectors_for_settings"])
    assert callable(captured["build_backend"])
    assert captured["profile_label"] == "Medium"
    assert captured["backend_id"] == "hf_xlsr"
    assert captured["profile_id"] == "medium"
    assert captured["embedding_cache_name"] == "medium_embeddings"
