"""Training and inference helpers for the SER emotion classification model."""

from __future__ import annotations

import glob
import importlib
import json
import logging
import os
import pickle
import warnings
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean
from typing import Any, Literal, NamedTuple

import numpy as np
from halo import Halo
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ser.config import LEGACY_MODEL_FOLDER, get_settings
from ser.data import load_data
from ser.domain import EmotionSegment
from ser.features import extract_feature_frames
from ser.runtime.schema import (
    OUTPUT_SCHEMA_VERSION,
    FramePrediction,
    InferenceResult,
    SegmentPrediction,
    to_legacy_emotion_segments,
)
from ser.train.metrics import compute_ser_metrics
from ser.utils.logger import get_logger

logger: logging.Logger = get_logger(__name__)

MODEL_ARTIFACT_VERSION = 1
type EmotionClassifier = MLPClassifier | Pipeline
type ArtifactFormat = Literal["pickle", "skops"]


class LoadedModel(NamedTuple):
    """Loaded model object and optional expected feature-vector length."""

    model: EmotionClassifier
    expected_feature_size: int | None


class ModelCandidate(NamedTuple):
    """A candidate model artifact path and its source."""

    path: Path
    artifact_format: ArtifactFormat
    origin: Literal["primary", "legacy"]


@dataclass(frozen=True)
class PersistedArtifacts:
    """Paths to persisted model artifacts from training."""

    pickle_path: Path
    secure_path: Path | None


def _create_classifier() -> EmotionClassifier:
    """Builds a reproducible scaler+MLP training pipeline."""
    settings = get_settings()
    classifier: MLPClassifier = MLPClassifier(
        alpha=settings.nn.alpha,
        batch_size=settings.nn.batch_size,  # pyright: ignore[reportArgumentType]
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


def _build_model_artifact(
    model: EmotionClassifier,
    feature_vector_size: int,
    training_samples: int,
    labels: list[str],
) -> dict[str, object]:
    """Constructs a versioned model artifact envelope for safer loading."""
    return {
        "artifact_version": MODEL_ARTIFACT_VERSION,
        "model": model,
        "metadata": {
            "artifact_version": MODEL_ARTIFACT_VERSION,
            "created_at_utc": datetime.now(tz=UTC).isoformat(),
            "feature_vector_size": feature_vector_size,
            "training_samples": training_samples,
            "labels": sorted(set(labels)),
        },
    }


def _deserialize_model_artifact(payload: object) -> LoadedModel:
    """Validates and unwraps persisted model payloads, including legacy formats."""
    if isinstance(payload, (MLPClassifier, Pipeline)):
        logger.warning(
            "Loaded legacy model artifact without metadata validation envelope."
        )
        return LoadedModel(model=payload, expected_feature_size=None)

    if not isinstance(payload, dict):
        raise ValueError(
            "Unexpected model artifact payload type: "
            f"{type(payload).__name__}. Expected dict envelope."
        )

    artifact_version = payload.get("artifact_version")
    if artifact_version != MODEL_ARTIFACT_VERSION:
        raise ValueError(
            "Unsupported model artifact version "
            f"{artifact_version!r}; expected {MODEL_ARTIFACT_VERSION}."
        )

    model = payload.get("model")
    if not isinstance(model, (MLPClassifier, Pipeline)):
        raise ValueError(
            "Unexpected model object type in artifact envelope: "
            f"{type(model).__name__}."
        )

    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Model artifact metadata is missing or invalid.")

    expected_feature_size = metadata.get("feature_vector_size")
    if not isinstance(expected_feature_size, int) or expected_feature_size <= 0:
        raise ValueError(
            "Model artifact metadata contains invalid 'feature_vector_size'."
        )

    return LoadedModel(model=model, expected_feature_size=expected_feature_size)


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Writes binary payload atomically to avoid partial artifacts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    try:
        with tmp_path.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def _atomic_write_text(path: Path, payload: str) -> None:
    """Writes UTF-8 text atomically to avoid truncated JSON output."""
    _atomic_write_bytes(path, payload.encode("utf-8"))


def _persist_pickle_artifact(path: Path, artifact: dict[str, object]) -> None:
    """Serializes and stores the pickle envelope for broad compatibility."""
    serialized = pickle.dumps(artifact, protocol=pickle.HIGHEST_PROTOCOL)
    _atomic_write_bytes(path, serialized)


def _persist_secure_artifact(path: Path, model: EmotionClassifier) -> bool:
    """Attempts to persist a secure model format via `skops`, if available."""
    try:
        skops_io: Any = importlib.import_module("skops.io")
    except ModuleNotFoundError:
        logger.info(
            "Optional dependency `skops` is not installed; skipping secure model "
            "artifact generation."
        )
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    try:
        skops_io.dump(model, str(tmp_path))
        os.replace(tmp_path, path)
        return True
    except Exception as err:
        logger.warning(
            "Failed to persist secure model artifact at %s: %s. Continuing with "
            "pickle artifact.",
            path,
            err,
        )
        return False
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def _persist_training_report(report: dict[str, object], path: Path) -> None:
    """Persists a deterministic JSON training report to disk."""
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    _atomic_write_text(path, serialized)


def _read_training_report_feature_size(path: Path) -> int | None:
    """Reads expected feature size from the training report when available."""
    if not path.exists():
        return None

    try:
        with path.open("r", encoding="utf-8") as report_fh:
            payload = json.load(report_fh)
    except Exception as err:
        logger.warning("Failed to parse training report at %s: %s", path, err)
        return None

    if not isinstance(payload, dict):
        return None

    feature_size = payload.get("feature_vector_size")
    return feature_size if isinstance(feature_size, int) and feature_size > 0 else None


def _build_training_report(
    *,
    accuracy: float,
    macro_f1: float,
    ser_metrics: dict[str, object],
    train_samples: int,
    test_samples: int,
    feature_vector_size: int,
    labels: list[str],
    artifacts: PersistedArtifacts,
) -> dict[str, object]:
    """Builds a structured report for training quality and artifact traceability."""
    settings = get_settings()
    corpus_samples = len(glob.glob(settings.dataset.glob_pattern))
    effective_samples = train_samples + test_samples
    label_distribution = dict(Counter(labels))
    model_artifacts: dict[str, str] = {"pickle": str(artifacts.pickle_path)}
    if artifacts.secure_path is not None:
        model_artifacts["secure"] = str(artifacts.secure_path)

    return {
        "artifact_version": MODEL_ARTIFACT_VERSION,
        "created_at_utc": datetime.now(tz=UTC).isoformat(),
        "dataset_glob_pattern": settings.dataset.glob_pattern,
        "dataset_corpus_samples": corpus_samples,
        "dataset_effective_samples": effective_samples,
        "dataset_skipped_samples": max(0, corpus_samples - effective_samples),
        "train_samples": train_samples,
        "test_samples": test_samples,
        "feature_vector_size": feature_vector_size,
        "labels": sorted(set(labels)),
        "label_distribution": label_distribution,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "metrics": ser_metrics,
        "model_artifacts": model_artifacts,
    }


def _persist_model_artifacts(
    model: EmotionClassifier, artifact: dict[str, object]
) -> PersistedArtifacts:
    """Persists model artifacts in compatibility-first and secure formats."""
    settings = get_settings()
    pickle_path = settings.models.model_file
    secure_path = settings.models.secure_model_file

    _persist_pickle_artifact(pickle_path, artifact)
    secure_saved = _persist_secure_artifact(secure_path, model)
    return PersistedArtifacts(
        pickle_path=pickle_path,
        secure_path=secure_path if secure_saved else None,
    )


def _model_load_candidates() -> tuple[ModelCandidate, ...]:
    """Returns model artifacts in preferred load order with legacy fallback."""
    settings = get_settings()
    primary_secure = settings.models.secure_model_file
    primary_pickle = settings.models.model_file
    legacy_secure = LEGACY_MODEL_FOLDER / settings.models.secure_model_file_name
    legacy_pickle = LEGACY_MODEL_FOLDER / settings.models.model_file_name

    ordered = (
        ModelCandidate(primary_secure, "skops", "primary"),
        ModelCandidate(primary_pickle, "pickle", "primary"),
        ModelCandidate(legacy_secure, "skops", "legacy"),
        ModelCandidate(legacy_pickle, "pickle", "legacy"),
    )

    deduped: list[ModelCandidate] = []
    seen: set[tuple[str, ArtifactFormat]] = set()
    for candidate in ordered:
        key: tuple[str, ArtifactFormat] = (
            str(candidate.path),
            candidate.artifact_format,
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return tuple(deduped)


def _load_secure_model(candidate: ModelCandidate) -> LoadedModel:
    """Loads a secure artifact when `skops` is available and trusted."""
    assert candidate.artifact_format == "skops"
    try:
        skops_io: Any = importlib.import_module("skops.io")
    except ModuleNotFoundError as err:
        raise RuntimeError(
            "Secure model artifact found but `skops` is not installed."
        ) from err

    untrusted_types = set(skops_io.get_untrusted_types(file=str(candidate.path)))
    if untrusted_types:
        raise ValueError(
            "Secure model artifact contains untrusted types; refusing automatic "
            f"trust for {candidate.path}."
        )

    payload = skops_io.load(str(candidate.path), trusted=[])
    if not isinstance(payload, (MLPClassifier, Pipeline)):
        raise ValueError(
            "Unexpected secure model payload type: "
            f"{type(payload).__name__}. Expected sklearn classifier/pipeline."
        )

    settings = get_settings()
    feature_size = _read_training_report_feature_size(
        settings.models.training_report_file
    )
    return LoadedModel(model=payload, expected_feature_size=feature_size)


def _load_pickle_model(candidate: ModelCandidate) -> LoadedModel:
    """Loads and validates the compatibility pickle model artifact."""
    assert candidate.artifact_format == "pickle"
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        with candidate.path.open("rb") as model_fh:
            payload = pickle.load(model_fh)
    return _deserialize_model_artifact(payload)


def _resolve_model_for_loading() -> tuple[ModelCandidate, LoadedModel]:
    """Finds and loads the first valid model artifact candidate."""
    candidates = _model_load_candidates()
    existing_candidates = [
        candidate for candidate in candidates if candidate.path.exists()
    ]
    if not existing_candidates:
        candidate_list = ", ".join(str(candidate.path) for candidate in candidates)
        raise FileNotFoundError(
            "Model not found. Checked: "
            f"{candidate_list}. Train it first with `ser --train`."
        )

    last_error: Exception | None = None
    for candidate in existing_candidates:
        try:
            loaded_model = (
                _load_secure_model(candidate)
                if candidate.artifact_format == "skops"
                else _load_pickle_model(candidate)
            )
            return candidate, loaded_model
        except Exception as err:
            last_error = err
            logger.warning(
                "Failed to load %s model artifact at %s: %s",
                candidate.artifact_format,
                candidate.path,
                err,
            )

    candidate_list = ", ".join(str(candidate.path) for candidate in existing_candidates)
    raise ValueError(
        f"Failed to deserialize model from any candidate path: {candidate_list}."
    ) from last_error


def train_model() -> None:
    """Trains the MLP classifier and persists model + training report artifacts.

    Raises:
        RuntimeError: If no training data could be loaded from the dataset path.
    """
    settings = get_settings()
    with Halo(text="Loading dataset... ", spinner="dots", text_color="green"):
        if data := load_data(test_size=settings.training.test_size):
            x_train, x_test, y_train, y_test = data
            model: EmotionClassifier = _create_classifier()
            logger.info(msg="Dataset loaded successfully.")
        else:
            logger.error("Dataset not loaded. Please load the dataset first.")
            raise RuntimeError("Dataset not loaded. Please load the dataset first.")

    with Halo(text="Training the model... ", spinner="dots", text_color="green"):
        model.fit(x_train, y_train)
    logger.info(msg=f"Model trained with {len(x_train)} samples")

    with Halo(text="Measuring accuracy... ", spinner="dots", text_color="green"):
        y_pred = [str(item) for item in model.predict(x_test)]
        accuracy: float = float(accuracy_score(y_true=y_test, y_pred=y_pred))
        macro_f1: float = float(f1_score(y_test, y_pred, average="macro"))
        ser_metrics = compute_ser_metrics(y_true=y_test, y_pred=y_pred)
    uar = ser_metrics.get("uar")
    if not isinstance(uar, float):
        raise RuntimeError("SER metrics payload missing float 'uar'.")
    logger.info(msg=f"Accuracy: {accuracy * 100:.2f}%")
    logger.info(msg=f"Macro F1 score: {macro_f1:.4f}")
    logger.info(msg=f"UAR: {uar:.4f}")

    with Halo(text="Saving the model... ", spinner="dots", text_color="green"):
        artifact = _build_model_artifact(
            model=model,
            feature_vector_size=int(x_train.shape[1]),
            training_samples=int(x_train.shape[0]),
            labels=y_train,
        )
        persisted_artifacts = _persist_model_artifacts(model=model, artifact=artifact)
    logger.info(msg=f"Model saved to {persisted_artifacts.pickle_path}")
    if persisted_artifacts.secure_path is not None:
        logger.info(msg=f"Secure model saved to {persisted_artifacts.secure_path}")

    report = _build_training_report(
        accuracy=accuracy,
        macro_f1=macro_f1,
        ser_metrics=ser_metrics,
        train_samples=int(x_train.shape[0]),
        test_samples=int(x_test.shape[0]),
        feature_vector_size=int(x_train.shape[1]),
        labels=[*y_train, *y_test],
        artifacts=persisted_artifacts,
    )
    _persist_training_report(report, settings.models.training_report_file)
    logger.info(msg=f"Training report saved to {settings.models.training_report_file}")


def load_model() -> LoadedModel:
    """Loads the serialized SER model from disk.

    Returns:
        The loaded model plus expected feature-vector size.

    Raises:
        FileNotFoundError: If no trained model artifact could be found.
        ValueError: If model artifacts exist but none can be deserialized.
    """
    try:
        with Halo(
            text="Loading SER model... ",
            spinner="dots",
            text_color="green",
        ):
            candidate, loaded_model = _resolve_model_for_loading()

        if candidate.origin == "legacy":
            settings = get_settings()
            logger.warning(
                "Primary model not found under %s; using legacy model path %s. "
                "Set SER_MODELS_DIR to migrate location explicitly.",
                settings.models.folder,
                candidate.path,
            )

        logger.info(
            "Model loaded from %s (%s).",
            candidate.path,
            candidate.artifact_format,
        )
        return loaded_model
    except FileNotFoundError:
        raise
    except Exception as err:
        logger.error("Failed to load model: %s", err)
        raise ValueError("Failed to load model from configured locations.") from err


def _frame_confidence_and_probabilities(
    model: EmotionClassifier,
    feature_matrix: np.ndarray,
    frame_count: int,
) -> tuple[list[float], list[dict[str, float] | None]]:
    """Returns per-frame confidence and optional class-probability maps."""
    fallback_confidence = [1.0] * frame_count
    fallback_probabilities: list[dict[str, float] | None] = [None] * frame_count

    predict_proba = getattr(model, "predict_proba", None)
    if not callable(predict_proba):
        logger.warning(
            "Loaded model does not expose predict_proba; using confidence=1.0 fallback."
        )
        return fallback_confidence, fallback_probabilities

    classes_attr = getattr(model, "classes_", None)
    if not isinstance(classes_attr, (list, tuple, np.ndarray)):
        logger.warning(
            "Loaded model predict_proba path missing classes_; using confidence fallback."
        )
        return fallback_confidence, fallback_probabilities

    class_labels = [str(item) for item in classes_attr]
    raw_probabilities = np.asarray(predict_proba(feature_matrix), dtype=np.float64)
    if raw_probabilities.ndim != 2:
        logger.warning(
            "Unexpected predict_proba output shape %s; using confidence fallback.",
            raw_probabilities.shape,
        )
        return fallback_confidence, fallback_probabilities
    if raw_probabilities.shape[0] != frame_count:
        logger.warning(
            "predict_proba frame count mismatch (expected=%s, got=%s); using fallback.",
            frame_count,
            raw_probabilities.shape[0],
        )
        return fallback_confidence, fallback_probabilities
    if raw_probabilities.shape[1] != len(class_labels):
        logger.warning(
            "predict_proba class count mismatch (classes=%s, probs=%s); using fallback.",
            len(class_labels),
            raw_probabilities.shape[1],
        )
        return fallback_confidence, fallback_probabilities

    confidences = [float(np.max(row)) for row in raw_probabilities]
    probabilities: list[dict[str, float] | None] = [
        {class_labels[idx]: float(row[idx]) for idx in range(len(class_labels))}
        for row in raw_probabilities
    ]
    return confidences, probabilities


def _aggregate_probabilities(
    probabilities: list[dict[str, float] | None],
) -> dict[str, float] | None:
    """Averages per-frame probabilities when all frames provide full maps."""
    if not probabilities or any(item is None for item in probabilities):
        return None

    first = probabilities[0]
    if first is None:
        return None
    labels = list(first.keys())
    if any(item is None or set(item.keys()) != set(labels) for item in probabilities[1:]):
        return None

    aggregates: dict[str, float] = {}
    for label in labels:
        values = [item[label] for item in probabilities if item is not None]
        aggregates[label] = float(fmean(values))
    return aggregates


def _segment_predictions(
    frame_predictions: list[FramePrediction],
) -> list[SegmentPrediction]:
    """Merges adjacent equal frame labels into segment-level predictions."""
    if not frame_predictions:
        return []

    segments: list[SegmentPrediction] = []
    active_emotion = frame_predictions[0].emotion
    active_start = frame_predictions[0].start_seconds
    active_end = frame_predictions[0].end_seconds
    active_confidences = [frame_predictions[0].confidence]
    active_probabilities = [frame_predictions[0].probabilities]

    for frame in frame_predictions[1:]:
        if frame.emotion == active_emotion:
            active_end = frame.end_seconds
            active_confidences.append(frame.confidence)
            active_probabilities.append(frame.probabilities)
            continue

        segments.append(
            SegmentPrediction(
                emotion=active_emotion,
                start_seconds=active_start,
                end_seconds=active_end,
                confidence=float(fmean(active_confidences)),
                probabilities=_aggregate_probabilities(active_probabilities),
            )
        )
        active_emotion = frame.emotion
        active_start = frame.start_seconds
        active_end = frame.end_seconds
        active_confidences = [frame.confidence]
        active_probabilities = [frame.probabilities]

    segments.append(
        SegmentPrediction(
            emotion=active_emotion,
            start_seconds=active_start,
            end_seconds=active_end,
            confidence=float(fmean(active_confidences)),
            probabilities=_aggregate_probabilities(active_probabilities),
        )
    )
    return segments


def predict_emotions_detailed(file: str) -> InferenceResult:
    """Runs inference and returns detailed frame + segment predictions."""
    loaded_model: LoadedModel = load_model()
    model = loaded_model.model

    with Halo(
        text="Inferring Emotions from Audio File... ",
        spinner="dots",
        text_color="green",
    ):
        feature_frames = extract_feature_frames(file)
        if not feature_frames:
            logger.warning("No features extracted for file %s.", file)
            return InferenceResult(
                schema_version=OUTPUT_SCHEMA_VERSION,
                segments=[],
                frames=[],
            )

        feature_vectors: list[np.ndarray] = [frame.features for frame in feature_frames]
        if loaded_model.expected_feature_size is not None:
            invalid_sizes = {
                vector.shape[0]
                for vector in feature_vectors
                if vector.shape[0] != loaded_model.expected_feature_size
            }
            if invalid_sizes:
                raise ValueError(
                    "Feature vector size mismatch for loaded model. "
                    f"Expected {loaded_model.expected_feature_size}, "
                    f"got {sorted(invalid_sizes)}."
                )

        feature_matrix = np.asarray(feature_vectors, dtype=np.float64)
        predicted_emotions: list[str] = [
            str(item) for item in model.predict(feature_matrix)
        ]
        if len(predicted_emotions) != len(feature_frames):
            raise RuntimeError(
                "Frame/prediction length mismatch. "
                f"Got {len(feature_frames)} frames and {len(predicted_emotions)} predictions."
            )
        confidences, probabilities = _frame_confidence_and_probabilities(
            model=model,
            feature_matrix=feature_matrix,
            frame_count=len(feature_frames),
        )

    logger.info(msg="Emotion inference completed.")
    if not predicted_emotions:
        logger.warning("No emotions predicted for file %s.", file)
        return InferenceResult(
            schema_version=OUTPUT_SCHEMA_VERSION,
            segments=[],
            frames=[],
        )

    frame_predictions = [
        FramePrediction(
            start_seconds=feature_frames[idx].start_seconds,
            end_seconds=feature_frames[idx].end_seconds,
            emotion=predicted_emotions[idx],
            confidence=confidences[idx],
            probabilities=probabilities[idx],
        )
        for idx in range(len(feature_frames))
    ]
    segment_predictions = _segment_predictions(frame_predictions)
    logger.info("Emotion prediction and timestamp extraction completed.")
    return InferenceResult(
        schema_version=OUTPUT_SCHEMA_VERSION,
        segments=segment_predictions,
        frames=frame_predictions,
    )


def predict_emotions(file: str) -> list[EmotionSegment]:
    """Compatibility wrapper returning legacy emotion segments."""
    return to_legacy_emotion_segments(predict_emotions_detailed(file))
