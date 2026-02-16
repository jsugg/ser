"""Training and inference helpers for the SER emotion classification model."""

import logging
import os
import pickle
import warnings

import librosa
import numpy as np
from halo import Halo
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

from ser.config import get_settings
from ser.data import load_data
from ser.domain import EmotionSegment
from ser.features import extended_extract_feature
from ser.utils.audio_utils import read_audio_file
from ser.utils.logger import get_logger

logger: logging.Logger = get_logger(__name__)


def train_model() -> None:
    """Trains the MLP classifier and persists the resulting model artifact.

    Raises:
        RuntimeError: If no training data could be loaded from the dataset path.
    """
    settings = get_settings()
    with Halo(text="Loading dataset... ", spinner="dots", text_color="green"):
        if data := load_data(test_size=0.25):
            x_train, x_test, y_train, y_test = data
            # sklearn's stubs currently mis-type parts of MLPClassifier constructor
            # (notably `batch_size`), so pyright flags valid runtime arguments here.
            model: MLPClassifier = MLPClassifier(
                alpha=settings.nn.alpha,
                batch_size=settings.nn.batch_size,  # pyright: ignore[reportArgumentType]
                epsilon=settings.nn.epsilon,
                hidden_layer_sizes=settings.nn.hidden_layer_sizes,
                learning_rate=settings.nn.learning_rate,
                max_iter=settings.nn.max_iter,
            )
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
        os.makedirs(settings.models.folder, exist_ok=True)
        model_file: str = str(settings.models.model_file)
    logger.info(msg=f"Accuracy: {accuracy * 100:.2f}%")

    with Halo(text="Saving the model... ", spinner="dots", text_color="green"):
        with open(model_file, "wb") as model_fh:
            pickle.dump(model, model_fh)
    logger.info(msg=f"Model saved to {model_file}")


def load_model() -> MLPClassifier:
    """Loads the serialized SER model from disk.

    Returns:
        The trained `MLPClassifier` instance.

    Raises:
        FileNotFoundError: If the trained model file is missing.
        ValueError: If the model file exists but cannot be deserialized.
    """
    settings = get_settings()
    model_path: str = str(settings.models.model_file)
    if not os.path.exists(model_path):
        logger.error(
            "Model not found at %s. Train it first with `ser --train`.",
            model_path,
        )
        raise FileNotFoundError(
            f"Model not found at {model_path}. Train it first with `ser --train`."
        )

    try:
        with Halo(
            text=f"Loading SER model from {model_path}... ",
            spinner="dots",
            text_color="green",
        ):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                with open(model_path, "rb") as model_fh:
                    loaded_model = pickle.load(model_fh)
                    if not isinstance(loaded_model, MLPClassifier):
                        raise ValueError(
                            "Unexpected object type in model artifact: "
                            f"{type(loaded_model).__name__}"
                        )
                    model = loaded_model
        logger.info(msg=f"Model loaded from {model_path}")
        return model
    except Exception as err:
        logger.error("Failed to deserialize model from %s: %s", model_path, err)
        raise ValueError(f"Failed to load model from {model_path}.") from err


def predict_emotions(file: str) -> list[EmotionSegment]:
    """Runs frame-level inference and merges adjacent equal labels into segments.

    Args:
        file: Path to the audio file.

    Returns:
        A list of emotion segments with start/end timing.

    """
    model: MLPClassifier = load_model()

    with Halo(
        text="Inferring Emotions from Audio File... ",
        spinner="dots",
        text_color="green",
    ):
        feature: list[np.ndarray] = extended_extract_feature(file)
        predicted_emotions: list[str] = [str(item) for item in model.predict(feature)]
    logger.info(msg="Emotion inference completed.")

    if not predicted_emotions:
        logger.warning("No emotions predicted for file %s.", file)
        return []

    audio_duration: float = librosa.get_duration(y=read_audio_file(file)[0])
    emotion_timestamps: list[EmotionSegment] = []
    prev_emotion: str | None = None
    start_time: float = 0
    segment_count: int = len(predicted_emotions)

    for timestamp, emotion in enumerate(predicted_emotions):
        if emotion != prev_emotion:
            if prev_emotion is not None:
                end_time: float = timestamp * audio_duration / segment_count
                emotion_timestamps.append(
                    EmotionSegment(prev_emotion, start_time, end_time)
                )
            (
                prev_emotion,
                start_time,
            ) = (
                emotion,
                timestamp * audio_duration / segment_count,
            )

    if prev_emotion is not None:
        emotion_timestamps.append(
            EmotionSegment(prev_emotion, start_time, audio_duration)
        )

    logger.info("Emotion prediction and timestamp extraction completed.")
    return emotion_timestamps
