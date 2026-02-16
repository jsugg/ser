"""Typed application configuration for SER."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Literal


@dataclass(frozen=True)
class FeatureFlags:
    """Feature extraction toggles."""

    mfcc: bool = True
    chroma: bool = True
    mel: bool = True
    contrast: bool = True
    tonnetz: bool = True


@dataclass(frozen=True)
class NeuralNetConfig:
    """Training parameters for the MLP classifier."""

    alpha: float = 0.01
    batch_size: int | Literal["auto"] = 256
    epsilon: float = 1e-08
    hidden_layer_sizes: tuple[int, ...] = (300,)
    learning_rate: Literal["constant", "invscaling", "adaptive"] = "adaptive"
    max_iter: int = 500


@dataclass(frozen=True)
class AudioReadConfig:
    """Retry behavior for audio loading."""

    max_retries: int = 3
    retry_delay_seconds: float = 1.0


@dataclass(frozen=True)
class DatasetConfig:
    """Dataset location and glob settings."""

    folder: Path
    subfolder_prefix: str = "Actor_*"
    extension: str = "*.wav"

    @property
    def glob_pattern(self) -> str:
        """Returns the on-disk glob pattern for dataset audio files."""
        return str(self.folder / self.subfolder_prefix / self.extension)


@dataclass(frozen=True)
class WhisperModelConfig:
    """Whisper model selection and storage location."""

    name: str = "large-v2"
    relative_path: Path = Path("OpenAI/whisper")


@dataclass(frozen=True)
class ModelsConfig:
    """Storage and runtime settings for trained artifacts."""

    folder: Path = Path("ser/models")
    whisper_model: WhisperModelConfig = field(default_factory=WhisperModelConfig)
    num_cores: int = 1
    model_file_name: str = "ser_model.pkl"

    @property
    def model_file(self) -> Path:
        """Returns the expected path for the trained SER model."""
        return self.folder / self.model_file_name

    @property
    def whisper_download_root(self) -> Path:
        """Returns model cache/download root for Whisper assets."""
        return self.folder / self.whisper_model.relative_path


@dataclass(frozen=True)
class TimelineConfig:
    """Output settings for transcript timeline exports."""

    folder: Path = Path("transcripts")


@dataclass(frozen=True)
class AppConfig:
    """Immutable runtime configuration for the application."""

    emotions: Mapping[str, str]
    tmp_folder: Path = Path("tmp")
    feature_flags: FeatureFlags = field(default_factory=FeatureFlags)
    nn: NeuralNetConfig = field(default_factory=NeuralNetConfig)
    audio_read: AudioReadConfig = field(default_factory=AudioReadConfig)
    dataset: DatasetConfig = field(
        default_factory=lambda: DatasetConfig(folder=Path("ser/dataset/ravdess"))
    )
    models: ModelsConfig = field(default_factory=ModelsConfig)
    timeline: TimelineConfig = field(default_factory=TimelineConfig)
    default_language: str = "en"


def _build_settings() -> AppConfig:
    """Builds settings from process environment."""
    emotions: Mapping[str, str] = MappingProxyType(
        {
            "01": "neutral",
            "02": "calm",
            "03": "happy",
            "04": "sad",
            "05": "angry",
            "06": "fearful",
            "07": "disgust",
            "08": "surprised",
        }
    )
    dataset_folder: Path = Path(os.getenv("DATASET_FOLDER", "ser/dataset/ravdess"))
    default_language: str = os.getenv("DEFAULT_LANGUAGE", "en")
    cores: int = os.cpu_count() or 1
    return AppConfig(
        emotions=emotions,
        dataset=DatasetConfig(folder=dataset_folder),
        models=ModelsConfig(num_cores=cores),
        default_language=default_language,
    )


_SETTINGS: AppConfig = _build_settings()


def get_settings() -> AppConfig:
    """Returns current immutable settings."""
    return _SETTINGS


def reload_settings() -> AppConfig:
    """Reloads settings from environment and returns the new snapshot."""
    global _SETTINGS
    _SETTINGS = _build_settings()
    return _SETTINGS
