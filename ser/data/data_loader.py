"""Dataset loading and feature extraction helpers for model training."""

import glob
import logging
import multiprocessing as mp
import os
from functools import partial

import numpy as np
from sklearn.model_selection import train_test_split

from ser.config import Config
from ser.features import extract_feature
from ser.utils.logger import get_logger

logger: logging.Logger = get_logger(__name__)


def process_file(
    file: str, observed_emotions: list[str]
) -> tuple[np.ndarray, str] | None:
    """Extracts features for a file when its label is in the target emotion set.

    Args:
        file: Path to an audio file.
        observed_emotions: Emotion labels accepted for training.

    Returns:
        A tuple of `(feature_vector, emotion_label)` when the file matches one of
        `observed_emotions`; otherwise `None`.
    """
    try:
        file_name: str = os.path.basename(file)
        emotion: str | None = Config.EMOTIONS.get(file_name.split("-")[2])

        if not emotion or emotion not in observed_emotions:
            return None
        features: np.ndarray = extract_feature(file)

        return (features, emotion)
    except Exception as e:
        logger.error(msg=f"Failed to process file {file}: {e}")
        raise e


def load_data(test_size: float = 0.2) -> list | None:
    """Loads the configured dataset, extracts features, and splits train/test sets.

    Args:
        test_size: Fraction of examples reserved for the test split.

    Returns:
        The `train_test_split` output `(x_train, x_test, y_train, y_test)` when
        data is available; otherwise `None`.
    """
    observed_emotions: list[str] = list(Config.EMOTIONS.values())
    raw_data: list[tuple[np.ndarray, str] | None]
    data_path_pattern: str = (
        f"{Config.DATASET['folder']}/"
        f"{Config.DATASET['subfolder_prefix']}/"
        f"{Config.DATASET['extension']}"
    )
    files: list[str] = glob.glob(data_path_pattern)

    with mp.Pool(int(Config.MODELS_CONFIG["num_cores"])) as pool:
        raw_data = pool.map(
            partial(process_file, observed_emotions=observed_emotions), files
        )

    data: list[tuple[np.ndarray, str]] = [item for item in raw_data if item is not None]
    if not data:
        logger.warning("No data found or processed.")
        return None

    features: tuple[np.ndarray, ...]
    labels: tuple[str, ...]
    features, labels = zip(*data, strict=False)
    return train_test_split(
        np.array(features), labels, test_size=test_size, random_state=42
    )
