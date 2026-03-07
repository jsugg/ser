"""Contract tests for shared training orchestration helpers."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from ser.models.training_orchestration import (
    AccurateTrainingPreparation,
    MediumTrainingPreparation,
    TrainingEvaluation,
    execute_default_profile_training,
    finalize_profile_training_report,
    prepare_accurate_training_payload,
    prepare_medium_training_payload,
    run_accurate_profile_training,
    run_medium_profile_training,
)


class _ArtifactsStub:
    """Persisted artifact stub for training-report orchestration tests."""

    def __init__(self, *, pickle_path: Path, secure_path: Path | None) -> None:
        self.pickle_path = pickle_path
        self.secure_path = secure_path


def test_finalize_profile_training_report_builds_and_persists_payload(
    tmp_path: Path,
) -> None:
    """Finalization helper should build and persist report payload deterministically."""
    evaluation = TrainingEvaluation(
        accuracy=0.8,
        macro_f1=0.7,
        uar=0.6,
        ser_metrics={"uar": 0.6},
    )
    artifacts = _ArtifactsStub(
        pickle_path=tmp_path / "ser_model.pkl",
        secure_path=tmp_path / "ser_model.skops",
    )
    captured_build: dict[str, object] = {}
    captured_persist: dict[str, object] = {}

    def _build_training_report(**kwargs: object) -> dict[str, object]:
        captured_build.update(kwargs)
        return {"accuracy": kwargs["accuracy"], "labels": kwargs["labels"]}

    def _persist_training_report(report: dict[str, object]) -> None:
        captured_persist["report"] = report

    result = finalize_profile_training_report(
        profile_label="Accurate",
        logger=logging.getLogger("tests.training_orchestration"),
        evaluation=evaluation,
        ser_metrics=evaluation.ser_metrics,
        artifact_metadata={"backend_id": "hf_whisper"},
        persisted_artifacts=artifacts,
        x_train=np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64),
        x_test=np.asarray([[0.5, 0.6]], dtype=np.float64),
        y_train=["happy", "sad"],
        y_test=["happy"],
        provenance={"source": "unit-test"},
        data_controls={"dataset": {"name": "synthetic"}},
        build_training_report=_build_training_report,
        persist_training_report=_persist_training_report,
        report_destination=tmp_path / "report.json",
    )

    assert result == {"accuracy": 0.8, "labels": ["happy", "sad", "happy"]}
    assert captured_build["train_samples"] == 2
    assert captured_build["test_samples"] == 1
    assert captured_build["feature_vector_size"] == 2
    assert captured_build["artifacts"] is artifacts
    assert captured_persist["report"] == result


def test_execute_default_profile_training_runs_fit_predict_and_persistence() -> None:
    """Default training helper should run fit/predict/eval/persist in order."""

    class _ClassifierStub:
        def __init__(self) -> None:
            self.fitted: tuple[np.ndarray, list[str]] | None = None

        def fit(
            self,
            x_train: np.ndarray,
            y_train: Sequence[str],
        ) -> object | None:
            self.fitted = (x_train, [str(item) for item in y_train])
            return None

        def predict(self, x_test: np.ndarray) -> Sequence[object]:
            return ["angry" for _ in range(int(x_test.shape[0]))]

    def _extract_artifact_metadata(artifact: dict[str, object]) -> dict[str, object]:
        metadata = artifact.get("metadata")
        assert isinstance(metadata, dict)
        return metadata

    classifier = _ClassifierStub()
    captured_attach: dict[str, object] = {}
    captured_persist: dict[str, object] = {}

    execution = execute_default_profile_training(
        create_classifier=lambda: classifier,
        x_train=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
        y_train=["happy", "sad"],
        x_test=np.asarray([[5.0, 6.0]], dtype=np.float64),
        y_test=["angry"],
        test_meta=[{"sample_id": "s1"}],
        min_support=3,
        evaluate_predictions=lambda y_true, y_pred: TrainingEvaluation(
            accuracy=1.0,
            macro_f1=1.0,
            uar=1.0,
            ser_metrics={"uar": 1.0, "y_true": y_true, "y_pred": y_pred},
        ),
        attach_grouped_metrics=(
            lambda ser_metrics, y_true, y_pred, test_meta, min_support: (
                captured_attach.update(
                    {
                        "ser_metrics": ser_metrics,
                        "y_true": y_true,
                        "y_pred": y_pred,
                        "test_meta": list(test_meta),
                        "min_support": min_support,
                    }
                ),
                ser_metrics,
            )[1]
        ),
        build_model_artifact=lambda model: {
            "metadata": {"model_type": model.__class__.__name__}
        },
        extract_artifact_metadata=_extract_artifact_metadata,
        persist_model_artifacts=lambda model, artifact: (
            captured_persist.update({"model": model, "artifact": artifact}),
            {"saved": True},
        )[1],
    )

    assert classifier.fitted is not None
    _, fitted_labels = classifier.fitted
    assert fitted_labels == ["happy", "sad"]
    assert execution.evaluation.accuracy == 1.0
    assert execution.ser_metrics["uar"] == 1.0
    assert execution.artifact_metadata == {"model_type": "_ClassifierStub"}
    assert execution.persisted_artifacts == {"saved": True}
    assert captured_attach["y_true"] == ["angry"]
    assert captured_attach["y_pred"] == ["angry"]
    assert captured_attach["test_meta"] == [{"sample_id": "s1"}]
    assert captured_attach["min_support"] == 3
    assert captured_persist["model"] is classifier


def test_prepare_accurate_training_payload_requires_loaded_dataset() -> None:
    """Preparation helper should fail fast when no training dataset is available."""
    logger = logging.getLogger("tests.training_orchestration.prepare_dataset_required")

    try:
        _ = prepare_accurate_training_payload(
            logger=logger,
            load_utterances_for_training=lambda: None,
            ensure_dataset_consents_for_training=lambda _utterances: None,
            prepare_training_features=lambda _utterances: (_ for _ in ()).throw(
                AssertionError("prepare_training_features should not be called")
            ),
        )
        raise AssertionError("prepare_accurate_training_payload should have raised")
    except RuntimeError as exc:
        assert str(exc) == "Dataset not loaded. Please load the dataset first."


def test_prepare_accurate_training_payload_builds_prepared_output() -> None:
    """Preparation helper should enforce consent and return prepared payload."""
    logger = logging.getLogger("tests.training_orchestration.prepare_dataset_success")
    utterances = ["u1", "u2"]
    captured: dict[str, object] = {}
    prepared = AccurateTrainingPreparation[str, str, dict[str, str]](
        train_utterances=["u1"],
        test_utterances=["u2"],
        split_metadata="group_shuffle_split",
        model_id="openai/whisper-large-v3",
        runtime_device="cpu",
        runtime_dtype="float32",
        x_train=np.asarray([[0.1, 0.2]], dtype=np.float64),
        y_train=["happy"],
        x_test=np.asarray([[0.3, 0.4]], dtype=np.float64),
        y_test=["sad"],
        test_meta=[{"sample_id": "u2"}],
    )

    resolved_utterances, resolved_prepared = prepare_accurate_training_payload(
        logger=logger,
        load_utterances_for_training=lambda: utterances,
        ensure_dataset_consents_for_training=lambda loaded_utterances: captured.update(
            {"consents": list(loaded_utterances)}
        ),
        prepare_training_features=lambda loaded_utterances: (
            captured.update({"prepared_input": list(loaded_utterances)}),
            prepared,
        )[1],
    )

    assert resolved_utterances == utterances
    assert resolved_prepared is prepared
    assert captured["consents"] == utterances
    assert captured["prepared_input"] == utterances


def test_prepare_medium_training_payload_requires_loaded_dataset() -> None:
    """Medium preparation helper should fail fast when no dataset is available."""
    logger = logging.getLogger("tests.training_orchestration.prepare_medium_required")

    try:
        _ = prepare_medium_training_payload(
            logger=logger,
            load_utterances_for_training=lambda: None,
            ensure_dataset_consents_for_training=lambda _utterances: None,
            prepare_training_features=lambda _utterances: (_ for _ in ()).throw(
                AssertionError("prepare_training_features should not be called")
            ),
        )
        raise AssertionError("prepare_medium_training_payload should have raised")
    except RuntimeError as exc:
        assert str(exc) == "Dataset not loaded. Please load the dataset first."


def test_prepare_medium_training_payload_builds_prepared_output() -> None:
    """Medium preparation helper should enforce consent and return prepared payload."""
    logger = logging.getLogger("tests.training_orchestration.prepare_medium_success")
    utterances = ["u1", "u2"]
    captured: dict[str, object] = {}
    prepared = MediumTrainingPreparation[str, str, dict[str, str], dict[str, int]](
        train_utterances=["u1"],
        test_utterances=["u2"],
        split_metadata="group_shuffle_split",
        model_id="facebook/wav2vec2-xls-r-300m",
        runtime_device="cpu",
        runtime_dtype="float32",
        x_train=np.asarray([[0.1, 0.2]], dtype=np.float64),
        y_train=["happy"],
        x_test=np.asarray([[0.3, 0.4]], dtype=np.float64),
        y_test=["sad"],
        test_meta=[{"sample_id": "u2"}],
        train_noise_stats={"filtered_windows": 1},
        test_noise_stats={"filtered_windows": 0},
    )

    resolved_utterances, resolved_prepared = prepare_medium_training_payload(
        logger=logger,
        load_utterances_for_training=lambda: utterances,
        ensure_dataset_consents_for_training=lambda loaded_utterances: captured.update(
            {"consents": list(loaded_utterances)}
        ),
        prepare_training_features=lambda loaded_utterances: (
            captured.update({"prepared_input": list(loaded_utterances)}),
            prepared,
        )[1],
    )

    assert resolved_utterances == utterances
    assert resolved_prepared is prepared
    assert captured["consents"] == utterances
    assert captured["prepared_input"] == utterances


def test_run_medium_profile_training_executes_and_persists_report(
    tmp_path: Path,
) -> None:
    """Medium profile helper should execute fit/eval/persist/report with controls."""

    class _SplitMeta:
        split_strategy = "speaker_grouped"

    class _RuntimeConfig:
        pool_window_size_seconds = 2.0
        pool_window_stride_seconds = 1.0

    class _TrainingConfig:
        min_window_std = 0.1
        max_windows_per_clip = 3

    class _ModelsConfig:
        training_report_file = tmp_path / "training_report.json"

    class _Settings:
        medium_runtime = _RuntimeConfig()
        medium_training = _TrainingConfig()
        models = _ModelsConfig()

    class _Classifier:
        def fit(self, x_train: np.ndarray, y_train: Sequence[str]) -> object | None:
            del x_train, y_train
            return None

        def predict(self, x_test: np.ndarray) -> Sequence[str]:
            return ["happy" for _ in range(int(x_test.shape[0]))]

    prepared = MediumTrainingPreparation[
        str, _SplitMeta, dict[str, str], dict[str, int]
    ](
        train_utterances=["u1", "u2"],
        test_utterances=["u3"],
        split_metadata=_SplitMeta(),
        model_id="facebook/wav2vec2-large-xlsr-53",
        runtime_device="cpu",
        runtime_dtype="float32",
        x_train=np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64),
        y_train=["happy", "sad"],
        x_test=np.asarray([[0.5, 0.6]], dtype=np.float64),
        y_test=["happy"],
        test_meta=[{"sample_id": "s3", "corpus": "ravdess", "language": "en"}],
        train_noise_stats={"dropped": 1},
        test_noise_stats={"dropped": 0},
    )
    captured: dict[str, object] = {}

    def _build_model_artifact(**kwargs: object) -> dict[str, object]:
        captured["artifact_build"] = kwargs
        return {"metadata": {"backend_id": kwargs["backend_id"]}}

    def _extract_artifact_metadata(artifact: dict[str, object]) -> dict[str, object]:
        metadata = artifact.get("metadata")
        assert isinstance(metadata, dict)
        return metadata

    def _persist_model_artifacts(
        model: _Classifier,
        artifact: dict[str, object],
    ) -> _ArtifactsStub:
        del model
        captured["artifact_payload"] = artifact
        return _ArtifactsStub(
            pickle_path=tmp_path / "ser_model.pkl",
            secure_path=tmp_path / "ser_model.skops",
        )

    def _build_training_report(**kwargs: object) -> dict[str, object]:
        captured["report_build"] = kwargs
        return {"ok": True}

    def _persist_training_report(report: dict[str, object], path: Path) -> None:
        captured["report"] = report
        captured["report_path"] = path

    _ = run_medium_profile_training(
        prepared=prepared,
        utterances=["u1", "u2", "u3"],
        settings=_Settings(),
        logger=logging.getLogger("tests.training_orchestration.medium"),
        profile_label="Medium",
        backend_id="hf_xlsr",
        profile_id="medium",
        pooling_strategy="mean_std_pool_v1",
        create_classifier=_Classifier,
        min_support=2,
        evaluate_predictions=lambda y_true, y_pred: TrainingEvaluation(
            accuracy=1.0,
            macro_f1=1.0,
            uar=1.0,
            ser_metrics={"uar": 1.0, "y_true": y_true, "y_pred": y_pred},
        ),
        attach_grouped_metrics=(
            lambda ser_metrics, y_true, y_pred, test_meta, min_support: (
                captured.update(
                    {
                        "attach": {
                            "y_true": y_true,
                            "y_pred": y_pred,
                            "test_meta": list(test_meta),
                            "min_support": min_support,
                        }
                    }
                ),
                ser_metrics,
            )[1]
        ),
        build_model_artifact=_build_model_artifact,
        extract_artifact_metadata=_extract_artifact_metadata,
        persist_model_artifacts=_persist_model_artifacts,
        build_provenance_metadata=lambda **kwargs: {"backend_id": kwargs["backend_id"]},
        build_dataset_controls=lambda utterances: {"count": len(utterances)},
        build_medium_noise_controls=lambda **kwargs: {"noise": kwargs},
        build_grouped_evaluation_controls=lambda split_meta: {
            "split_strategy": split_meta.split_strategy
        },
        build_training_report=_build_training_report,
        persist_training_report=_persist_training_report,
    )

    artifact_build = captured["artifact_build"]
    assert isinstance(artifact_build, dict)
    assert artifact_build["backend_id"] == "hf_xlsr"
    assert artifact_build["profile"] == "medium"
    assert artifact_build["feature_vector_size"] == 2
    report_build = captured["report_build"]
    assert isinstance(report_build, dict)
    data_controls = report_build["data_controls"]
    assert isinstance(data_controls, dict)
    assert data_controls["dataset"] == {"count": 3}
    assert data_controls["medium_grouped_evaluation"] == {
        "split_strategy": "speaker_grouped"
    }
    assert captured["report"] == {"ok": True}
    assert captured["report_path"] == tmp_path / "training_report.json"


def test_run_accurate_profile_training_executes_and_persists_report(
    tmp_path: Path,
) -> None:
    """Accurate profile helper should execute fit/eval/persist/report with controls."""

    class _SplitMeta:
        split_strategy = "group_shuffle_split"

    class _Classifier:
        def fit(self, x_train: np.ndarray, y_train: Sequence[str]) -> object | None:
            del x_train, y_train
            return None

        def predict(self, x_test: np.ndarray) -> Sequence[str]:
            return ["calm" for _ in range(int(x_test.shape[0]))]

    prepared = AccurateTrainingPreparation[str, _SplitMeta, dict[str, str]](
        train_utterances=["u1", "u2"],
        test_utterances=["u3"],
        split_metadata=_SplitMeta(),
        model_id="openai/whisper-large-v3",
        runtime_device="cpu",
        runtime_dtype="float32",
        x_train=np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64),
        y_train=["calm", "happy"],
        x_test=np.asarray([[0.5, 0.6]], dtype=np.float64),
        y_test=["calm"],
        test_meta=[{"sample_id": "s3", "corpus": "ravdess", "language": "en"}],
    )
    captured: dict[str, object] = {}
    report_destination = tmp_path / "training_report_accurate.json"

    def _build_model_artifact(**kwargs: object) -> dict[str, object]:
        captured["artifact_build"] = kwargs
        return {"metadata": {"backend_id": kwargs["backend_id"]}}

    def _extract_artifact_metadata(artifact: dict[str, object]) -> dict[str, object]:
        metadata = artifact.get("metadata")
        assert isinstance(metadata, dict)
        return metadata

    def _persist_model_artifacts(
        model: _Classifier,
        artifact: dict[str, object],
    ) -> _ArtifactsStub:
        del model
        captured["artifact_payload"] = artifact
        return _ArtifactsStub(
            pickle_path=tmp_path / "ser_model_accurate.pkl",
            secure_path=tmp_path / "ser_model_accurate.skops",
        )

    def _build_training_report(**kwargs: object) -> dict[str, object]:
        captured["report_build"] = kwargs
        return {"ok": True}

    def _persist_training_report(report: dict[str, object], path: Path) -> None:
        captured["report"] = report
        captured["report_path"] = path

    _ = run_accurate_profile_training(
        prepared=prepared,
        utterances=["u1", "u2", "u3"],
        settings={"runtime": "settings"},
        logger=logging.getLogger("tests.training_orchestration.accurate"),
        profile_label="Accurate",
        backend_id="hf_whisper",
        profile_id="accurate",
        pooling_strategy="mean_std_pool_v1",
        frame_size_seconds=2.0,
        frame_stride_seconds=0.5,
        create_classifier=_Classifier,
        min_support=2,
        evaluate_predictions=lambda y_true, y_pred: TrainingEvaluation(
            accuracy=1.0,
            macro_f1=1.0,
            uar=1.0,
            ser_metrics={"uar": 1.0, "y_true": y_true, "y_pred": y_pred},
        ),
        attach_grouped_metrics=(
            lambda ser_metrics, y_true, y_pred, test_meta, min_support: (
                captured.update(
                    {
                        "attach": {
                            "y_true": y_true,
                            "y_pred": y_pred,
                            "test_meta": list(test_meta),
                            "min_support": min_support,
                        }
                    }
                ),
                ser_metrics,
            )[1]
        ),
        build_model_artifact=_build_model_artifact,
        extract_artifact_metadata=_extract_artifact_metadata,
        persist_model_artifacts=_persist_model_artifacts,
        build_provenance_metadata=lambda **kwargs: {
            "backend_id": kwargs["backend_id"],
            "profile": kwargs["profile"],
        },
        build_dataset_controls=lambda utterances: {"count": len(utterances)},
        build_grouped_evaluation_controls=lambda split_meta: {
            "split_strategy": split_meta.split_strategy
        },
        build_training_report=_build_training_report,
        persist_training_report=_persist_training_report,
        report_destination=report_destination,
    )

    artifact_build = captured["artifact_build"]
    assert isinstance(artifact_build, dict)
    assert artifact_build["backend_id"] == "hf_whisper"
    assert artifact_build["profile"] == "accurate"
    assert artifact_build["feature_vector_size"] == 2
    assert artifact_build["frame_size_seconds"] == 2.0
    assert artifact_build["frame_stride_seconds"] == 0.5
    report_build = captured["report_build"]
    assert isinstance(report_build, dict)
    data_controls = report_build["data_controls"]
    assert isinstance(data_controls, dict)
    assert data_controls["dataset"] == {"count": 3}
    assert data_controls["accurate_grouped_evaluation"] == {
        "split_strategy": "group_shuffle_split"
    }
    assert captured["report"] == {"ok": True}
    assert captured["report_path"] == report_destination
