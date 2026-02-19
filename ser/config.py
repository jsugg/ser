"""Application configuration for SER."""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Literal

APP_NAME = "ser"
LEGACY_MODEL_FOLDER = Path("ser/models")
LEGACY_TRANSCRIPTS_FOLDER = Path("transcripts")
LEGACY_TMP_FOLDER = Path("tmp")


def _platform_cache_base_dir() -> Path:
    """Returns the platform-native cache base directory."""
    if sys.platform == "win32":
        return Path(os.getenv("LOCALAPPDATA", Path.home() / "AppData/Local"))
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches"
    return Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache"))


def _platform_data_base_dir() -> Path:
    """Returns the platform-native data base directory."""
    if sys.platform == "win32":
        return Path(os.getenv("APPDATA", Path.home() / "AppData/Roaming"))
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support"
    return Path(os.getenv("XDG_DATA_HOME", Path.home() / ".local/share"))


def _default_cache_root() -> Path:
    """Returns the default cache root for SER runtime artifacts."""
    return _platform_cache_base_dir() / APP_NAME


def _default_data_root() -> Path:
    """Returns the default data root for SER runtime artifacts."""
    return _platform_data_base_dir() / APP_NAME


def _default_tmp_folder() -> Path:
    """Returns the default temporary-file location."""
    return _default_cache_root() / "tmp"


def _default_models_folder() -> Path:
    """Returns the default persisted-model location."""
    return _default_data_root() / "models"


def _default_transcripts_folder() -> Path:
    """Returns the default transcript export location."""
    return _default_data_root() / "transcripts"


def _read_path_env(name: str, default: Path) -> Path:
    """Reads a path env var and returns an expanded path."""
    raw_value = os.getenv(name)
    selected = Path(raw_value) if raw_value else default
    return selected.expanduser()


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
    random_state: int = 42


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
class DataLoaderConfig:
    """Parallelism and error-threshold controls for dataset loading."""

    max_workers: int = 8
    max_failed_file_ratio: float = 0.01


@dataclass(frozen=True)
class TrainingConfig:
    """Dataset split controls for model training."""

    test_size: float = 0.25
    random_state: int = 42
    stratify_split: bool = True


@dataclass(frozen=True)
class WhisperModelConfig:
    """Whisper model selection and storage location."""

    name: str = "large-v2"
    relative_path: Path = Path("OpenAI/whisper")


@dataclass(frozen=True)
class ModelsConfig:
    """Storage and runtime settings for trained artifacts."""

    folder: Path = field(default_factory=_default_models_folder)
    whisper_model: WhisperModelConfig = field(default_factory=WhisperModelConfig)
    num_cores: int = 1
    model_file_name: str = "ser_model.pkl"
    secure_model_file_name: str = "ser_model.skops"
    training_report_file_name: str = "training_report.json"

    @property
    def model_file(self) -> Path:
        """Returns the expected path for the trained SER model."""
        return self.folder / self.model_file_name

    @property
    def secure_model_file(self) -> Path:
        """Returns the path for secure model serialization when available."""
        return self.folder / self.secure_model_file_name

    @property
    def training_report_file(self) -> Path:
        """Returns the expected path for persisted training metrics."""
        return self.folder / self.training_report_file_name

    @property
    def whisper_download_root(self) -> Path:
        """Returns model cache/download root for Whisper assets."""
        return self.folder / self.whisper_model.relative_path


@dataclass(frozen=True)
class TimelineConfig:
    """Output settings for transcript timeline exports."""

    folder: Path = field(default_factory=_default_transcripts_folder)


@dataclass(frozen=True)
class TranscriptionConfig:
    """Runtime controls for Whisper transcription behavior."""

    use_demucs: bool = True
    use_vad: bool = True


@dataclass(frozen=True)
class AppConfig:
    """Immutable runtime configuration for the application."""

    emotions: Mapping[str, str]
    tmp_folder: Path = field(default_factory=_default_tmp_folder)
    feature_flags: FeatureFlags = field(default_factory=FeatureFlags)
    nn: NeuralNetConfig = field(default_factory=NeuralNetConfig)
    audio_read: AudioReadConfig = field(default_factory=AudioReadConfig)
    dataset: DatasetConfig = field(
        default_factory=lambda: DatasetConfig(folder=Path("ser/dataset/ravdess"))
    )
    data_loader: DataLoaderConfig = field(default_factory=DataLoaderConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    timeline: TimelineConfig = field(default_factory=TimelineConfig)
    transcription: TranscriptionConfig = field(default_factory=TranscriptionConfig)
    default_language: str = "en"


def _read_bool_env(name: str, default: bool) -> bool:
    """Reads a boolean env var and returns a validated value."""
    raw_value: str | None = os.getenv(name)
    if raw_value is None:
        return default

    normalized: str = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _read_int_env(name: str, default: int, minimum: int | None = None) -> int:
    """Reads an integer env var and enforces an optional lower bound."""
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    try:
        parsed = int(raw_value)
    except ValueError:
        return default

    return default if minimum is not None and parsed < minimum else parsed


def _read_float_env(
    name: str,
    default: float,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    """Reads a float env var and enforces optional bounds."""
    raw_value: str | None = os.getenv(name)
    if raw_value is None:
        return default

    try:
        parsed: float = float(raw_value)
    except ValueError:
        return default

    if minimum is not None and parsed < minimum:
        return default
    return default if maximum is not None and parsed > maximum else parsed


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
    whisper_model_name: str = os.getenv("WHISPER_MODEL", "large-v2")
    max_workers: int = _read_int_env("SER_MAX_WORKERS", 8, minimum=1)
    max_failed_file_ratio: float = _read_float_env(
        "SER_MAX_FAILED_FILE_RATIO",
        0.01,
        minimum=0.0,
        maximum=1.0,
    )
    test_size: float = _read_float_env(
        "SER_TEST_SIZE", 0.25, minimum=0.05, maximum=0.95
    )
    random_state: int = _read_int_env("SER_RANDOM_STATE", 42, minimum=0)
    use_demucs: bool = _read_bool_env("WHISPER_DEMUCS", True)
    use_vad: bool = _read_bool_env("WHISPER_VAD", True)
    model_file_name: str = os.getenv("SER_MODEL_FILE_NAME", "ser_model.pkl")
    secure_model_file_name: str = os.getenv(
        "SER_SECURE_MODEL_FILE_NAME", "ser_model.skops"
    )
    training_report_file_name: str = os.getenv(
        "SER_TRAINING_REPORT_FILE_NAME", "training_report.json"
    )
    cache_root: Path = _read_path_env("SER_CACHE_DIR", _default_cache_root())
    data_root: Path = _read_path_env("SER_DATA_DIR", _default_data_root())
    tmp_folder: Path = _read_path_env("SER_TMP_DIR", cache_root / "tmp")
    models_folder: Path = _read_path_env("SER_MODELS_DIR", data_root / "models")
    transcripts_folder: Path = _read_path_env(
        "SER_TRANSCRIPTS_DIR", data_root / "transcripts"
    )
    cores: int = os.cpu_count() or 1
    return AppConfig(
        emotions=emotions,
        tmp_folder=tmp_folder,
        nn=NeuralNetConfig(random_state=random_state),
        dataset=DatasetConfig(folder=dataset_folder),
        data_loader=DataLoaderConfig(
            max_workers=max_workers,
            max_failed_file_ratio=max_failed_file_ratio,
        ),
        training=TrainingConfig(test_size=test_size, random_state=random_state),
        models=ModelsConfig(
            folder=models_folder,
            num_cores=cores,
            model_file_name=model_file_name,
            secure_model_file_name=secure_model_file_name,
            training_report_file_name=training_report_file_name,
            whisper_model=WhisperModelConfig(name=whisper_model_name),
        ),
        timeline=TimelineConfig(folder=transcripts_folder),
        transcription=TranscriptionConfig(use_demucs=use_demucs, use_vad=use_vad),
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
