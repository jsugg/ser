"""
Configuration Module for Speech Emotion Recognition (SER) Tool

This module provides a central configuration class for the SER application. It defines
various settings and parameters used throughout the tool, including emotions, feature
extraction configuration, neural network parameters, audio file read parameters, dataset
configuration, model configuration, transcript configuration, and default language settings.

Classes:
    - Config: Contains all configuration settings for the SER application.

Author: Juan Sugg (juanpedrosugg [at] gmail.com)
Version: 1.0
License: MIT
"""

import os
from typing import Any

from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()


class Config:
    """
    Central configuration class for the SER application.
    """

    # Emotions supported by the dataset
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

    # Temporary folder for processing
    TMP_FOLDER: str = "./tmp"

    # Default feature extraction configuration
    DEFAULT_FEATURE_CONFIG: dict[str, bool] = {
        "mfcc": True,
        "chroma": True,
        "mel": True,
        "contrast": True,
        "tonnetz": True,
    }

    # Neural network parameters for MLP Classifier
    NN_PARAMS: dict[str, Any] = {
        "alpha": 0.01,
        "batch_size": 256,
        "epsilon": 1e-08,
        "hidden_layer_sizes": (300,),
        "learning_rate": "adaptive",
        "max_iter": 500,
    }

    # Audio file read parameters
    AUDIO_READ_CONFIG: dict[str, int] = {
        "max_retries": 3,
        "retry_delay": 1,  # in seconds
    }

    # Dataset configuration
    DATASET: dict[str, str] = {
        "folder": os.getenv("DATASET_FOLDER", "ser/dataset/ravdess"),
        "subfolder_prefix": "Actor_*",
        "extension": "*.wav",
    }

    # Model configuration
    MODELS_CONFIG: dict[str, Any] = {
        "models_folder": "./ser/models",
        "whisper_model": {"name": "large-v2", "path": "OpenAI/whisper/"},
        "num_cores": os.cpu_count(),
    }

    # Transcript configuration
    TIMELINE_CONFIG: dict[str, str] = {"folder": "./transcripts"}

    # Language settings
    DEFAULT_LANGUAGE: str = os.getenv("DEFAULT_LANGUAGE", "en")
    FILE_SETTING: None | str = None
    TRAIN_MODE: bool = False
