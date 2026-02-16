"""
Emotion Classification Model for Speech Emotion Recognition (SER) System

This module provides functions for training and using the emotion classification model
in the SER system. It includes functions to train the model, load the trained model, and
predict emotions from audio files.

Functions:
    - train_model: Trains the emotion classification model.
    - load_model: Loads the trained emotion classification model.
    - predict_emotions: Predicts emotions from an audio file.

Author: Juan Sugg (juanpedrosugg [at] gmail.com)
Version: 1.0
License: MIT
"""

import logging
import os
import pickle
import warnings

import librosa
import numpy as np
from halo import Halo
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

from ser.config import Config
from ser.data import load_data
from ser.features import extended_extract_feature
from ser.utils import get_logger, read_audio_file

logger: logging.Logger = get_logger(__name__)


def train_model() -> None:
    """
    Train the emotion classification model.

    This function loads the dataset, trains an MLPClassifier on the training data,
    measures the model's accuracy on the test data, and saves the trained model to a file.

    Raises:
        Exception: If the dataset is not loaded successfully.
    """
    with Halo(text="Loading dataset... ", spinner="dots", text_color="green"):
        if data := load_data(test_size=0.25):
            x_train, x_test, y_train, y_test = data
            model: MLPClassifier = MLPClassifier(**Config.NN_PARAMS)
            logger.info(msg="Dataset loaded successfully.")
        else:
            logger.error("Dataset not loaded. Please load the dataset first.")
            raise RuntimeError("Dataset not loaded. Please load the dataset first.")

    with Halo(text="Training the model... ", spinner="dots", text_color="green"):
        model.fit(x_train, y_train)
    logger.info(msg=f"Model trained with {len(x_train)} samples")

    with Halo(text="Measuring accuracy... ", spinner="dots", text_color="green"):
        y_pred = model.predict(x_test)
        accuracy: float = float(accuracy_score(y_true=y_test, y_pred=y_pred))
        model_file: str = f"{Config.MODELS_CONFIG['models_folder']}/ser_model.pkl"
    logger.info(msg=f"Accuracy: {accuracy * 100:.2f}%")

    with Halo(text="Saving the model... ", spinner="dots", text_color="green"):
        pickle.dump(model, open(model_file, "wb"))
    logger.info(msg=f"Model saved to {model_file}")


def load_model() -> MLPClassifier:
    """
    Load the trained emotion classification model.

    This function loads the trained MLPClassifier model from a file.

    Returns:
        MLPClassifier: The trained emotion classification model.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    model_path: str = f"{Config.MODELS_CONFIG['models_folder']}/ser_model.pkl"
    model: MLPClassifier | None = None
    try:
        with Halo(
            text=f"Loading SER model from {model_path}... ",
            spinner="dots",
            text_color="green",
        ):
            if os.path.exists(model_path):
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    model = pickle.load(open(model_path, "rb"))

        if model:
            logger.info(msg=f"Model loaded from {model_path}")
            return model
    except FileNotFoundError as err:
        logger.error(
            msg="Model not found. Please train the model first. "
            "If you already trained the model, please ensure that "
            f"{model_path} file exists and is a valid .pkl file."
        )
        raise err

    logger.error(
        msg=(
            'Failed to load the model. Ensure `MODELS_CONFIG["models_folder"]` '
            "is defined in the Config class in the config file, and that the "
            f"file {model_path} actually exists and is a valid .pkl file."
        )
    )
    raise ValueError("Failed to load the model.")


def predict_emotions(file: str) -> list[tuple[str, float, float]]:
    """
    Predict emotions from an audio file.

    This function loads a trained model, extracts features from the audio file,
    predicts emotions at each timestamp, and returns a list of predicted emotions
    with their start and end timestamps.

    Arguments:
        file (str): Path to the audio file.

    Returns:
        List[Tuple[str, float, float]]: A list of tuples where each tuple contains
        the predicted emotion, start time, and end time.

    Raises:
        Exception: If the model is not loaded.
    """
    model: MLPClassifier = load_model()
    if model is None:
        raise RuntimeError("Model not loaded.")

    with Halo(
        text="Inferring Emotions from Audio File... ",
        spinner="dots",
        text_color="green",
    ):
        feature: list[np.ndarray] = extended_extract_feature(file)
        predicted_emotions = model.predict(feature)
    logger.info(msg="Emotion inference completed.")

    audio_duration: float = librosa.get_duration(y=read_audio_file(file)[0])
    emotion_timestamps: list[tuple[str, float, float]] = []
    prev_emotion: str | None = None
    start_time: float = 0

    for timestamp, emotion in enumerate(predicted_emotions):
        if emotion != prev_emotion:
            if prev_emotion is not None:
                end_time: float = timestamp * audio_duration / len(predicted_emotions)
                emotion_timestamps.append((prev_emotion, start_time, end_time))
            (
                prev_emotion,
                start_time,
            ) = emotion, timestamp * audio_duration / len(predicted_emotions)

    if prev_emotion is not None:
        emotion_timestamps.append((prev_emotion, start_time, audio_duration))

    logger.info("Emotion prediction and timestamp extraction completed.")
    return emotion_timestamps
