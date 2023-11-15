import glob
import os
import warnings
from typing import List, Tuple, Optional
import numpy as np
from sklearn.model_selection import train_test_split
from functools import partial
import multiprocessing as mp
from ser.features.feature_extractor import extract_feature
from ser.config import EMOTIONS, MODELS_CONFIG, DATASET

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

def process_file(
        file: str,
        observed_emotions: List[str],
    ) -> Optional[Tuple[np.ndarray, str]]:
    """
    Process an audio file to extract features and the associated emotion label.

    Parameters
    ----------
    file : str
        Path to the audio file.
    observed_emotions : List[str]
        List of observed emotions.

    Returns
    -------
    Optional[Tuple[np.ndarray, str]]
        Extracted features and associated emotion label for the audio file.
        Returns None if the emotion is not in observed_emotions.
    """
    file_name: str = os.path.basename(file)
    emotion: str = EMOTIONS[file_name.split("-")[2]]

    if emotion not in observed_emotions:
        return None

    feature: Optional[np.ndarray] = extract_feature(file)
    return (feature, emotion) if feature is not None else None


def load_data(
        test_size: float = 0.2,
    ) -> List:
    """
    Loads the data and extracts features for each sound file.

    Parameters
    ----------
    test_size : float, optional
        Ratio of test data, by default 0.2.

    Returns
    -------
    Tuple[np.ndarray, List[str], np.ndarray, List[str]]
        Tuple containing train and test data and labels.
    """
    process_pool = mp.Pool(MODELS_CONFIG['num_cores'])
    observed_emotions: List[str] = list(EMOTIONS.values())
    folder, subfolder_prefix, extension = DATASET.values()
    print(f"{folder}/{subfolder_prefix}/{extension}")

    data: List = process_pool.map(
        partial(
            process_file,
            observed_emotions=observed_emotions
        ),
        glob.glob(f"{folder}/{subfolder_prefix}/{extension}"))

    process_pool.close()

    # Filter out None values
    data = [item for item in data if item is not None]
    x, y = zip(*data) if data else ([], [])

    return train_test_split(np.array(x), y, test_size=test_size, random_state=9)
