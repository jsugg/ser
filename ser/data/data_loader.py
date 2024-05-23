"""
Data Loader for Speech Emotion Recognition (SER) Tool

This module provides functions to load and process audio data for the SER tool. It
handles feature extraction from audio files, splitting the dataset into training and
testing sets, and parallel processing of audio files.

Functions:
    - process_file: Processes an audio file to extract features and associated emotion label.
    - load_data: Loads data from the dataset directory and splits it into training and testing sets.

Author: Juan Sugg (juanpedrosugg [at] gmail.com)
Version: 1.0
License: MIT
"""

import os
import glob
import logging
from typing import List, Tuple, Optional
import multiprocessing as mp
from functools import partial

import numpy as np
from sklearn.model_selection import train_test_split

from ser.utils import get_logger
from ser.features import extract_feature
from ser import Config


logger: logging.Logger = get_logger(__name__)


def process_file(
    file: str, observed_emotions: List[str]
) -> Tuple[np.ndarray, str]:
    """
    Process an audio file to extract features and the associated emotion label.

    Arguments:
        file (str): Path to the audio file.
        observed_emotions (List[str]): List of observed emotions.

    Returns:
        Optional[Tuple[np.ndarray, str]]: Extracted features and associated
            emotion label for the audio file.
        Returns None if the emotion is not in observed_emotions.
    """
    try:
        file_name: str = os.path.basename(file)
        emotion: Optional[str] = Config.EMOTIONS.get(file_name.split("-")[2])

        if not emotion or emotion not in observed_emotions:
            return (np.array([]), "")
        features: np.ndarray = extract_feature(file)

        return (features, emotion)
    except Exception as e:
        logger.error(msg=f"Failed to process file {file}: {e}")
        raise e


def load_data(test_size: float = 0.2) -> Optional[List]:
    """
    Load data from the dataset directory and split into training and testing sets.

    Arguments:
        test_size (float): Fraction of the dataset to be used as test set.

    Returns:
        Tuple containing training features, training labels, test features,
        and test labels.
    """
    observed_emotions: List[str] = list(Config.EMOTIONS.values())
    data: List[Tuple[np.ndarray, str]]
    data_path_pattern: str = (
        f"{Config.DATASET['folder']}/"
        f"{Config.DATASET['subfolder_prefix']}/"
        f"{Config.DATASET['extension']}"
    )
    files: List[str] = glob.glob(data_path_pattern)

    with mp.Pool(int(Config.MODELS_CONFIG["num_cores"])) as pool:
        data = pool.map(
            partial(process_file, observed_emotions=observed_emotions), files
        )

    # Remove None entries from data list
    data = [item for item in data if item is not None]
    if not data:
        logger.warning("No data found or processed.")
        return None

    features: Tuple[np.ndarray, ...]
    labels: Tuple[str, ...]
    features, labels = zip(*data)
    return train_test_split(
        np.array(features), labels, test_size=test_size, random_state=42
    )
