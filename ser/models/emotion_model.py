import os
import pickle
from typing import Optional, Tuple, List
import numpy as np
import librosa
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from halo import Halo
import warnings

from ser.config import NN_PARAMS, MODELS_CONFIG
from ser.data.data_loader import load_data
from ser.features.feature_extractor import extended_extract_feature
from ser.utils.audio_utils import read_audio_file

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

def train_model() -> None:
    """
    Train the emotion classification model.
    """
    with Halo(text='Loading dataset... ', spinner='dots', text_color='green'):
        x_train, x_test, y_train, y_test = load_data(test_size=0.25)
        model: MLPClassifier = MLPClassifier(**NN_PARAMS)
    with Halo(text='Training the model... ', spinner='dots', text_color='green'):
        model.fit(x_train, y_train)
        print(f"Model trained with {len(x_train)} samples")
    with Halo(text='Measuring accuracy... ', spinner='dots', text_color='green'):
        y_pred: np.ndarray = model.predict(x_test)
        accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
        print(f"Accuracy: {accuracy * 100:.2f}%")
        model_file: str = f"{MODELS_CONFIG['models_folder']}/ser_model.pkl"
    with Halo(text='Saving the model... ', spinner='dots', text_color='green'):
        pickle.dump(model, open(model_file, 'wb'))
        print(f'Model saved to {model_file}')


def load_model() -> MLPClassifier:
    """
    Load the trained emotion classification model.

    Returns
    -------
    MLPClassifier
        Trained model.
    """
    with Halo(text='Loading the model... ', spinner='dots', text_color='green'):
        model_path: str = f'{MODELS_CONFIG["models_folder"]}/ser_model.pkl'
        if os.path.exists(model_path):
            return pickle.load(open(model_path, 'rb'))
        raise FileNotFoundError("Model not found. Please train the model first.")


def predict_emotions(file: str) -> List[Tuple[str, float, float]]:
    """
    Predict emotions from an audio file.

    Parameters
    ----------
    file : str
        Path to the audio file.
    feature_config : dict
        Configuration for feature extraction.

    Returns
    -------
    List[Tuple[str, float, float]]
        List of predicted emotions with their start and end timestamps.
    """
    model: MLPClassifier = load_model()
    if model is None:
        raise Exception("Model not loaded. Please train the model first.")

    with Halo(text='Inferring Emotions from Audio File... ', spinner='dots', text_color='green'):
        feature: List[np.ndarray] = extended_extract_feature(file)
        predicted_emotions: np.ndarray = model.predict(feature)

    audio_duration: float = librosa.get_duration(y=read_audio_file(file)[0])
    emotion_timestamps: List = []
    prev_emotion: Optional[str] = None
    start_time: float = 0

    for timestamp, emotion in enumerate(predicted_emotions):
        if emotion != prev_emotion:
            if prev_emotion is not None:
                end_time = timestamp * audio_duration / len(predicted_emotions)
                emotion_timestamps.append((prev_emotion, start_time, end_time))
            prev_emotion, start_time = emotion, timestamp * audio_duration / len(predicted_emotions)

    if prev_emotion is not None:
        emotion_timestamps.append((prev_emotion, start_time, audio_duration))

    return emotion_timestamps
