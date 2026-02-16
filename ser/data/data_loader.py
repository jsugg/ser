"""Dataset loading and feature extraction helpers for model training."""

import glob
import logging
import multiprocessing as mp
import os
from collections.abc import Collection
from functools import partial

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split

from ser.config import get_settings
from ser.features import extract_feature
from ser.utils.logger import get_logger

logger: logging.Logger = get_logger(__name__)

type FeatureVector = NDArray[np.float64]
type FeatureMatrix = NDArray[np.float64]
type LabelList = list[str]
type DataSplit = tuple[FeatureMatrix, FeatureMatrix, LabelList, LabelList]


def process_file(
    file: str, observed_emotions: Collection[str]
) -> tuple[FeatureVector, str] | None:
    """Extracts features for a file when its label is in the target emotion set.

    Args:
        file: Path to an audio file.
        observed_emotions: Emotion labels accepted for training.

    Returns:
        A tuple of `(feature_vector, emotion_label)` when the file matches one of
        `observed_emotions`; otherwise `None`.
    """
    settings = get_settings()
    try:
        file_name: str = os.path.basename(file)
        emotion: str | None = settings.emotions.get(file_name.split("-")[2])

        if not emotion or emotion not in observed_emotions:
            return None
        features: FeatureVector = np.asarray(extract_feature(file), dtype=np.float64)

        return (features, emotion)
    except Exception as err:
        logger.error(msg=f"Failed to process file {file}: {err}", exc_info=True)
        raise


def load_data(test_size: float = 0.2) -> DataSplit | None:
    """Loads the configured dataset, extracts features, and splits train/test sets.

    Args:
        test_size: Fraction of examples reserved for the test split.

    Returns:
        The `train_test_split` output `(x_train, x_test, y_train, y_test)` when
        data is available; otherwise `None`.
    """
    settings = get_settings()
    observed_emotions: set[str] = set(settings.emotions.values())
    raw_data: list[tuple[FeatureVector, str] | None]
    data_path_pattern: str = settings.dataset.glob_pattern
    files: list[str] = glob.glob(data_path_pattern)

    with mp.Pool(settings.models.num_cores) as pool:
        raw_data = pool.map(
            partial(process_file, observed_emotions=observed_emotions), files
        )

    data: list[tuple[FeatureVector, str]] = [
        item for item in raw_data if item is not None
    ]
    if not data:
        logger.warning("No data found or processed.")
        return None

    features: tuple[FeatureVector, ...]
    labels: tuple[str, ...]
    features, labels = zip(*data, strict=False)
    split = train_test_split(
        np.asarray(features, dtype=np.float64),
        list(labels),
        test_size=test_size,
        random_state=42,
    )
    x_train = np.asarray(split[0], dtype=np.float64)
    x_test = np.asarray(split[1], dtype=np.float64)
    y_train = [str(label) for label in split[2]]
    y_test = [str(label) for label in split[3]]
    return x_train, x_test, y_train, y_test
