"""Dataset loading and feature extraction helpers for model training."""

import glob
import logging
import multiprocessing as mp
import os
from collections.abc import Collection
from functools import partial
from typing import Any, NamedTuple

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split

from ser.config import AppConfig, get_settings
from ser.features import extract_feature
from ser.utils.logger import get_logger

logger: logging.Logger = get_logger(__name__)

type FeatureVector = NDArray[np.float64]
type FeatureMatrix = NDArray[np.float64]
type LabelList = list[str]
type DataSplit = tuple[FeatureMatrix, FeatureMatrix, LabelList, LabelList]
type LabeledAudioSample = tuple[str, str]


class ProcessFileResult(NamedTuple):
    """Container for per-file extraction success or failure."""

    sample: tuple[FeatureVector, str] | None
    error: str | None


def _extract_emotion_code(file_name: str) -> str | None:
    """Extracts the RAVDESS emotion code from a dataset filename."""
    parts: list[str] = file_name.split("-")
    return None if len(parts) < 3 else parts[2]


def extract_ravdess_speaker_id_from_path(file_path: str) -> str | None:
    """Extracts actor ID from a RAVDESS-style file path when present."""
    file_name: str = os.path.basename(file_path)
    parts: list[str] = file_name.split("-")
    if len(parts) < 7:
        return None
    speaker_id: str = parts[6].split(".")[0].strip()
    return speaker_id or None


def _resolve_worker_count(max_cores: int, max_workers: int, file_count: int) -> int:
    """Returns a bounded worker count based on configured and runtime limits."""
    if file_count <= 0:
        return 1
    return max(1, min(max_cores, max_workers, file_count))


def process_file(
    file: str,
    observed_emotions: Collection[str],
    emotion_map: dict[str, str],
) -> ProcessFileResult:
    """Extracts features for a file when its label is in the target emotion set.

    Args:
        file: Path to an audio file.
        observed_emotions: Emotion labels accepted for training.
        emotion_map: Mapping from dataset emotion codes to labels.

    Returns:
        A result object containing the extracted `(feature_vector, emotion_label)` or
        an error string for reporting.
    """
    try:
        file_name: str = os.path.basename(file)
        emotion_code: str | None = _extract_emotion_code(file_name)
        if emotion_code is None:
            return ProcessFileResult(
                sample=None,
                error=(
                    "Skipping file with unexpected name format "
                    f"(missing emotion code): {file_name}"
                ),
            )

        emotion: str | None = emotion_map.get(emotion_code)

        if not emotion or emotion not in observed_emotions:
            return ProcessFileResult(sample=None, error=None)
        features: FeatureVector = np.asarray(extract_feature(file), dtype=np.float64)

        return ProcessFileResult(sample=(features, emotion), error=None)
    except Exception as err:
        return ProcessFileResult(sample=None, error=f"Failed to process file {file}: {err}")


def load_labeled_audio_paths() -> list[LabeledAudioSample] | None:
    """Loads dataset file paths paired with labels for backend-driven training.

    Returns:
        A list of ``(audio_path, label)`` tuples or ``None`` if no usable data exists.

    Raises:
        RuntimeError: If filename-format failures exceed configured threshold.
    """
    settings: AppConfig = get_settings()
    observed_emotions: set[str] = set(settings.emotions.values())
    files: list[str] = sorted(glob.glob(settings.dataset.glob_pattern))
    if not files:
        logger.warning("No dataset files found for pattern: %s", settings.dataset.glob_pattern)
        return None

    samples: list[LabeledAudioSample] = []
    parse_errors: list[str] = []
    for file_path in files:
        file_name: str = os.path.basename(file_path)
        emotion_code: str | None = _extract_emotion_code(file_name)
        if emotion_code is None:
            parse_errors.append(
                "Skipping file with unexpected name format " f"(missing emotion code): {file_name}"
            )
            continue
        emotion: str | None = settings.emotions.get(emotion_code)
        if not emotion or emotion not in observed_emotions:
            continue
        samples.append((file_path, emotion))

    if parse_errors:
        logger.warning(
            "Skipped %s/%s files while resolving labels.",
            len(parse_errors),
            len(files),
        )
        for error in parse_errors[:5]:
            logger.warning("%s", error)

    failure_ratio: float = len(parse_errors) / float(len(files))
    if failure_ratio > settings.data_loader.max_failed_file_ratio:
        raise RuntimeError(
            "Aborting data load: "
            f"{failure_ratio * 100.0:.1f}% file failures exceeded configured "
            "limit "
            f"{settings.data_loader.max_failed_file_ratio * 100.0:.1f}%."
        )

    if not samples:
        logger.warning("No labeled audio samples matched configured emotions.")
        return None

    labels: list[str] = [label for _, label in samples]
    if len(set(labels)) < 2:
        logger.warning("At least two emotion classes are required to train the model.")
        return None
    return samples


def load_data(test_size: float | None = None) -> DataSplit | None:
    """Loads the configured dataset, extracts features, and splits train/test sets.

    Args:
        test_size: Optional fraction of examples reserved for the test split.

    Returns:
        The `train_test_split` output `(x_train, x_test, y_train, y_test)` when
        data is available; otherwise `None`.
    """
    settings: AppConfig = get_settings()
    observed_emotions: set[str] = set(settings.emotions.values())
    resolved_test_size: float = settings.training.test_size if test_size is None else test_size
    if not 0.0 < resolved_test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1.")

    raw_data: list[ProcessFileResult]
    data_path_pattern: str = settings.dataset.glob_pattern
    files: list[str] = sorted(glob.glob(data_path_pattern))
    if not files:
        logger.warning("No dataset files found for pattern: %s", data_path_pattern)
        return None

    process_fn: partial[ProcessFileResult] = partial(
        process_file,
        observed_emotions=observed_emotions,
        emotion_map=dict(settings.emotions),
    )
    worker_count: int = _resolve_worker_count(
        max_cores=settings.models.num_cores,
        max_workers=settings.data_loader.max_workers,
        file_count=len(files),
    )

    if worker_count == 1:
        raw_data = [process_fn(file) for file in files]
    else:
        chunk_size: int = max(1, len(files) // (worker_count * 4))
        with mp.Pool(worker_count) as pool:
            raw_data = list(pool.imap_unordered(process_fn, files, chunksize=chunk_size))

    errors: list[str] = [item.error for item in raw_data if item.error is not None]
    if errors:
        logger.warning(
            "Skipped %s/%s files during feature extraction.",
            len(errors),
            len(files),
        )
        for error in errors[:5]:
            logger.warning("%s", error)

    failure_ratio: float = len(errors) / float(len(files))
    if failure_ratio > settings.data_loader.max_failed_file_ratio:
        raise RuntimeError(
            "Aborting data load: "
            f"{failure_ratio * 100.0:.1f}% file failures exceeded configured "
            "limit "
            f"{settings.data_loader.max_failed_file_ratio * 100.0:.1f}%. "
            "You can relax this limit by increasing the SER_MAX_FAILED_FILE_RATIO "
            "environment variable."
        )

    data: list[tuple[FeatureVector, str]] = [
        item.sample for item in raw_data if item.sample is not None
    ]
    if not data:
        logger.warning("No data found or processed.")
        return None

    features: tuple[FeatureVector, ...]
    labels: tuple[str, ...]
    features, labels = zip(*data, strict=False)
    labels_list: list[str] = list(labels)
    unique_labels: set[str] = set(labels_list)
    if len(unique_labels) < 2:
        logger.warning("At least two emotion classes are required to train the model.")
        return None

    stratify_labels: list[str] | None = labels_list if settings.training.stratify_split else None
    split: list[Any | list[Any]]
    try:
        split = train_test_split(
            np.asarray(features, dtype=np.float64),
            labels_list,
            test_size=resolved_test_size,
            random_state=settings.training.random_state,
            stratify=stratify_labels,
        )
    except ValueError as err:
        logger.warning(
            "Stratified split failed (%s). Falling back to non-stratified split.",
            err,
        )
        split = train_test_split(
            np.asarray(features, dtype=np.float64),
            labels_list,
            test_size=resolved_test_size,
            random_state=settings.training.random_state,
            stratify=None,
        )
    x_train: NDArray = np.asarray(split[0], dtype=np.float64)
    x_test: NDArray = np.asarray(split[1], dtype=np.float64)
    y_train: list[str] = [str(label) for label in split[2]]
    y_test: list[str] = [str(label) for label in split[3]]
    return x_train, x_test, y_train, y_test
