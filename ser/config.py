"""Application-level configuration values for the SER package."""

import os
from typing import Any

from dotenv import load_dotenv

load_dotenv()


class Config:
    """Holds static configuration values used across training and inference."""

    EMOTIONS: dict[str, str] = {
        "01": "neutral",
        "02": "calm",
        "03": "happy",
        "04": "sad",
        "05": "angry",
        "06": "fearful",
        "07": "disgust",
        "08": "surprised",
    }

    TMP_FOLDER: str = "./tmp"

    DEFAULT_FEATURE_CONFIG: dict[str, bool] = {
        "mfcc": True,
        "chroma": True,
        "mel": True,
        "contrast": True,
        "tonnetz": True,
    }

    NN_PARAMS: dict[str, Any] = {
        "alpha": 0.01,
        "batch_size": 256,
        "epsilon": 1e-08,
        "hidden_layer_sizes": (300,),
        "learning_rate": "adaptive",
        "max_iter": 500,
    }

    AUDIO_READ_CONFIG: dict[str, int] = {
        "max_retries": 3,
        "retry_delay": 1,
    }

    DATASET: dict[str, str] = {
        "folder": os.getenv("DATASET_FOLDER", "ser/dataset/ravdess"),
        "subfolder_prefix": "Actor_*",
        "extension": "*.wav",
    }

    MODELS_CONFIG: dict[str, Any] = {
        "models_folder": "./ser/models",
        "whisper_model": {"name": "large-v2", "path": "OpenAI/whisper/"},
        "num_cores": os.cpu_count(),
    }

    TIMELINE_CONFIG: dict[str, str] = {"folder": "./transcripts"}

    DEFAULT_LANGUAGE: str = os.getenv("DEFAULT_LANGUAGE", "en")
    FILE_SETTING: None | str = None
    TRAIN_MODE: bool = False
