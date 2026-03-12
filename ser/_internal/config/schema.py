"""Pure public configuration schema and default helpers."""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from ser._internal.config import artifact_naming as artifact_naming_helpers
from ser.profiles import (
    ProfileRuntimeDefaults,
    ProfileTranscriptionDefaults,
    TranscriptionBackendId,
    get_profile_catalog,
)

APP_NAME = "ser"
DEFAULT_FAST_MODEL_FILE_NAME = "ser_model.pkl"
DEFAULT_FAST_SECURE_MODEL_FILE_NAME = "ser_model.skops"
DEFAULT_FAST_TRAINING_REPORT_FILE_NAME = "training_report.json"

type ArtifactProfileName = Literal["fast", "medium", "accurate", "accurate-research"]


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


def _default_model_cache_root() -> Path:
    """Returns the default third-party foundation model cache root."""
    return _default_cache_root() / "model-cache"


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


def _artifact_profile_from_runtime_flags(
    *,
    medium_profile: bool,
    accurate_profile: bool,
    accurate_research_profile: bool,
) -> ArtifactProfileName:
    """Resolves artifact profile from runtime flags using runtime precedence."""
    return artifact_naming_helpers.artifact_profile_from_runtime_flags(
        medium_profile=medium_profile,
        accurate_profile=accurate_profile,
        accurate_research_profile=accurate_research_profile,
    )


def _artifact_model_id_suffix(model_id: str) -> str:
    """Builds a stable, filename-safe suffix for backend model ids."""
    return artifact_naming_helpers.artifact_model_id_suffix(model_id)


def default_profile_model_id(profile: ArtifactProfileName) -> str:
    """Returns the catalog-defined default model id for one model-backed profile."""
    default_model_id = get_profile_catalog()[profile].model.default_model_id
    if isinstance(default_model_id, str) and default_model_id.strip():
        return default_model_id.strip()
    raise RuntimeError(f"Profile {profile!r} does not define a default model id.")


def _default_profile_runtime_defaults(
    profile: ArtifactProfileName,
) -> ProfileRuntimeDefaults:
    """Returns catalog-owned runtime defaults for one profile."""
    return get_profile_catalog()[profile].runtime_defaults


def _default_profile_transcription_defaults(
    profile: ArtifactProfileName,
) -> ProfileTranscriptionDefaults:
    """Returns catalog-owned transcription defaults for one profile."""
    return get_profile_catalog()[profile].transcription_defaults


def _default_fast_whisper_model_name() -> str:
    """Returns the fast-profile default transcription model name."""
    return _default_profile_transcription_defaults("fast").model_name


def _default_fast_transcription_backend_id() -> TranscriptionBackendId:
    """Returns the fast-profile default transcription backend identifier."""
    return _default_profile_transcription_defaults("fast").backend_id


def _default_fast_transcription_use_demucs() -> bool:
    """Returns the fast-profile default demucs policy."""
    return _default_profile_transcription_defaults("fast").use_demucs


def _default_fast_transcription_use_vad() -> bool:
    """Returns the fast-profile default VAD policy."""
    return _default_profile_transcription_defaults("fast").use_vad


def _default_runtime_timeout_seconds(profile: ArtifactProfileName) -> float:
    """Returns the catalog-owned timeout budget for one profile."""
    return _default_profile_runtime_defaults(profile).timeout_seconds


def _default_runtime_max_timeout_retries(profile: ArtifactProfileName) -> int:
    """Returns the catalog-owned timeout retry budget for one profile."""
    return _default_profile_runtime_defaults(profile).max_timeout_retries


def _default_runtime_max_transient_retries(profile: ArtifactProfileName) -> int:
    """Returns the catalog-owned transient retry budget for one profile."""
    return _default_profile_runtime_defaults(profile).max_transient_retries


def _default_runtime_retry_backoff_seconds(profile: ArtifactProfileName) -> float:
    """Returns the catalog-owned retry backoff for one profile."""
    return _default_profile_runtime_defaults(profile).retry_backoff_seconds


def _default_runtime_pool_window_size_seconds(profile: ArtifactProfileName) -> float:
    """Returns the catalog-owned pooling window size for one profile."""
    return _default_profile_runtime_defaults(profile).pool_window_size_seconds


def _default_runtime_pool_window_stride_seconds(profile: ArtifactProfileName) -> float:
    """Returns the catalog-owned pooling window stride for one profile."""
    return _default_profile_runtime_defaults(profile).pool_window_stride_seconds


def _default_runtime_post_smoothing_window_frames(profile: ArtifactProfileName) -> int:
    """Returns the catalog-owned smoothing window size for one profile."""
    return _default_profile_runtime_defaults(profile).post_smoothing_window_frames


def _default_runtime_post_hysteresis_enter_confidence(profile: ArtifactProfileName) -> float:
    """Returns the catalog-owned enter-confidence threshold for one profile."""
    return _default_profile_runtime_defaults(profile).post_hysteresis_enter_confidence


def _default_runtime_post_hysteresis_exit_confidence(profile: ArtifactProfileName) -> float:
    """Returns the catalog-owned exit-confidence threshold for one profile."""
    return _default_profile_runtime_defaults(profile).post_hysteresis_exit_confidence


def _default_runtime_post_min_segment_duration_seconds(profile: ArtifactProfileName) -> float:
    """Returns the catalog-owned minimum segment duration for one profile."""
    return _default_profile_runtime_defaults(profile).post_min_segment_duration_seconds


def _default_runtime_process_isolation(profile: ArtifactProfileName) -> bool:
    """Returns the catalog-owned process-isolation default for one profile."""
    return _default_profile_runtime_defaults(profile).process_isolation


def profile_artifact_file_names(
    *,
    profile: ArtifactProfileName,
    medium_model_id: str | None = None,
    accurate_model_id: str | None = None,
    accurate_research_model_id: str | None = None,
) -> tuple[str, str, str]:
    """Returns default artifact filenames for one profile/backend-model tuple."""
    return artifact_naming_helpers.profile_artifact_file_names(
        profile=profile,
        medium_model_id=(
            default_profile_model_id("medium") if medium_model_id is None else medium_model_id
        ),
        accurate_model_id=(
            default_profile_model_id("accurate") if accurate_model_id is None else accurate_model_id
        ),
        accurate_research_model_id=(
            default_profile_model_id("accurate-research")
            if accurate_research_model_id is None
            else accurate_research_model_id
        ),
        default_fast_model_file_name=DEFAULT_FAST_MODEL_FILE_NAME,
        default_fast_secure_model_file_name=DEFAULT_FAST_SECURE_MODEL_FILE_NAME,
        default_fast_training_report_file_name=DEFAULT_FAST_TRAINING_REPORT_FILE_NAME,
    )


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
    manifest_paths: tuple[Path, ...] = ()

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

    name: str = field(default_factory=_default_fast_whisper_model_name)
    relative_path: Path = Path("OpenAI/whisper")


@dataclass(frozen=True)
class ModelsConfig:
    """Storage and runtime settings for trained artifacts."""

    folder: Path = field(default_factory=_default_models_folder)
    model_cache_dir: Path = field(default_factory=_default_model_cache_root)
    whisper_model: WhisperModelConfig = field(default_factory=WhisperModelConfig)
    medium_model_id: str = field(default_factory=lambda: default_profile_model_id("medium"))
    accurate_model_id: str = field(default_factory=lambda: default_profile_model_id("accurate"))
    accurate_research_model_id: str = field(
        default_factory=lambda: default_profile_model_id("accurate-research")
    )
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
        return self.model_cache_dir / self.whisper_model.relative_path

    @property
    def modelscope_cache_root(self) -> Path:
        """Returns ModelScope cache root used by Emotion2Vec/FunASR paths."""
        return self.model_cache_dir / "modelscope" / "hub"

    @property
    def huggingface_cache_root(self) -> Path:
        """Returns Hugging Face cache root used by optional backend downloads."""
        return self.model_cache_dir / "huggingface"

    @property
    def torch_cache_root(self) -> Path:
        """Returns Torch cache root used by Demucs/VAD and other torch-hub assets."""
        return self.model_cache_dir / "torch"


@dataclass(frozen=True)
class TimelineConfig:
    """Output settings for transcript timeline exports."""

    folder: Path = field(default_factory=_default_transcripts_folder)


@dataclass(frozen=True)
class TranscriptionConfig:
    """Runtime controls for Whisper transcription behavior."""

    backend_id: TranscriptionBackendId = field(
        default_factory=_default_fast_transcription_backend_id
    )
    use_demucs: bool = field(default_factory=_default_fast_transcription_use_demucs)
    use_vad: bool = field(default_factory=_default_fast_transcription_use_vad)
    mps_low_memory_threshold_gb: float = 16.0
    mps_admission_control_enabled: bool = True
    mps_hard_oom_shortcut_enabled: bool = True
    mps_admission_min_headroom_mb: float = 64.0
    mps_admission_safety_margin_mb: float = 64.0
    mps_admission_calibration_overrides_enabled: bool = True
    mps_admission_calibration_min_confidence: Literal["high", "medium", "low"] = "high"
    mps_admission_calibration_report_max_age_hours: float = 168.0
    mps_admission_calibration_report_path: Path | None = None


@dataclass(frozen=True)
class RuntimeFlags:
    """Feature flags for staged runtime rollout."""

    profile_pipeline: bool = False
    medium_profile: bool = False
    accurate_profile: bool = False
    accurate_research_profile: bool = False
    restricted_backends: bool = False
    new_output_schema: bool = False


@dataclass(frozen=True)
class ProfileRuntimeConfig:
    """Execution budgets and postprocessing controls for one runtime profile."""

    timeout_seconds: float
    max_timeout_retries: int
    max_transient_retries: int
    retry_backoff_seconds: float
    pool_window_size_seconds: float
    pool_window_stride_seconds: float
    post_smoothing_window_frames: int
    post_hysteresis_enter_confidence: float
    post_hysteresis_exit_confidence: float
    post_min_segment_duration_seconds: float
    process_isolation: bool


@dataclass(frozen=True)
class FastRuntimeConfig(ProfileRuntimeConfig):
    """Execution budgets and retry controls for fast profile runtime."""

    timeout_seconds: float = field(default_factory=lambda: _default_runtime_timeout_seconds("fast"))
    max_timeout_retries: int = field(
        default_factory=lambda: _default_runtime_max_timeout_retries("fast")
    )
    max_transient_retries: int = field(
        default_factory=lambda: _default_runtime_max_transient_retries("fast")
    )
    retry_backoff_seconds: float = field(
        default_factory=lambda: _default_runtime_retry_backoff_seconds("fast")
    )
    pool_window_size_seconds: float = field(
        default_factory=lambda: _default_runtime_pool_window_size_seconds("fast")
    )
    pool_window_stride_seconds: float = field(
        default_factory=lambda: _default_runtime_pool_window_stride_seconds("fast")
    )
    post_smoothing_window_frames: int = field(
        default_factory=lambda: _default_runtime_post_smoothing_window_frames("fast")
    )
    post_hysteresis_enter_confidence: float = field(
        default_factory=lambda: _default_runtime_post_hysteresis_enter_confidence("fast")
    )
    post_hysteresis_exit_confidence: float = field(
        default_factory=lambda: _default_runtime_post_hysteresis_exit_confidence("fast")
    )
    post_min_segment_duration_seconds: float = field(
        default_factory=lambda: _default_runtime_post_min_segment_duration_seconds("fast")
    )
    process_isolation: bool = field(
        default_factory=lambda: _default_runtime_process_isolation("fast")
    )


@dataclass(frozen=True)
class MediumRuntimeConfig(ProfileRuntimeConfig):
    """Execution budgets and retry controls for medium profile runtime."""

    timeout_seconds: float = field(
        default_factory=lambda: _default_runtime_timeout_seconds("medium")
    )
    max_timeout_retries: int = field(
        default_factory=lambda: _default_runtime_max_timeout_retries("medium")
    )
    max_transient_retries: int = field(
        default_factory=lambda: _default_runtime_max_transient_retries("medium")
    )
    retry_backoff_seconds: float = field(
        default_factory=lambda: _default_runtime_retry_backoff_seconds("medium")
    )
    pool_window_size_seconds: float = field(
        default_factory=lambda: _default_runtime_pool_window_size_seconds("medium")
    )
    pool_window_stride_seconds: float = field(
        default_factory=lambda: _default_runtime_pool_window_stride_seconds("medium")
    )
    post_smoothing_window_frames: int = field(
        default_factory=lambda: _default_runtime_post_smoothing_window_frames("medium")
    )
    post_hysteresis_enter_confidence: float = field(
        default_factory=lambda: _default_runtime_post_hysteresis_enter_confidence("medium")
    )
    post_hysteresis_exit_confidence: float = field(
        default_factory=lambda: _default_runtime_post_hysteresis_exit_confidence("medium")
    )
    post_min_segment_duration_seconds: float = field(
        default_factory=lambda: _default_runtime_post_min_segment_duration_seconds("medium")
    )
    process_isolation: bool = field(
        default_factory=lambda: _default_runtime_process_isolation("medium")
    )


@dataclass(frozen=True)
class AccurateRuntimeConfig(ProfileRuntimeConfig):
    """Execution budgets and retry controls for accurate profile runtime."""

    timeout_seconds: float = field(
        default_factory=lambda: _default_runtime_timeout_seconds("accurate")
    )
    max_timeout_retries: int = field(
        default_factory=lambda: _default_runtime_max_timeout_retries("accurate")
    )
    max_transient_retries: int = field(
        default_factory=lambda: _default_runtime_max_transient_retries("accurate")
    )
    retry_backoff_seconds: float = field(
        default_factory=lambda: _default_runtime_retry_backoff_seconds("accurate")
    )
    pool_window_size_seconds: float = field(
        default_factory=lambda: _default_runtime_pool_window_size_seconds("accurate")
    )
    pool_window_stride_seconds: float = field(
        default_factory=lambda: _default_runtime_pool_window_stride_seconds("accurate")
    )
    post_smoothing_window_frames: int = field(
        default_factory=lambda: _default_runtime_post_smoothing_window_frames("accurate")
    )
    post_hysteresis_enter_confidence: float = field(
        default_factory=lambda: _default_runtime_post_hysteresis_enter_confidence("accurate")
    )
    post_hysteresis_exit_confidence: float = field(
        default_factory=lambda: _default_runtime_post_hysteresis_exit_confidence("accurate")
    )
    post_min_segment_duration_seconds: float = field(
        default_factory=lambda: _default_runtime_post_min_segment_duration_seconds("accurate")
    )
    process_isolation: bool = field(
        default_factory=lambda: _default_runtime_process_isolation("accurate")
    )


@dataclass(frozen=True)
class AccurateResearchRuntimeConfig(ProfileRuntimeConfig):
    """Execution budgets and retry controls for accurate-research runtime."""

    timeout_seconds: float = field(
        default_factory=lambda: _default_runtime_timeout_seconds("accurate-research")
    )
    max_timeout_retries: int = field(
        default_factory=lambda: _default_runtime_max_timeout_retries("accurate-research")
    )
    max_transient_retries: int = field(
        default_factory=lambda: _default_runtime_max_transient_retries("accurate-research")
    )
    retry_backoff_seconds: float = field(
        default_factory=lambda: _default_runtime_retry_backoff_seconds("accurate-research")
    )
    pool_window_size_seconds: float = field(
        default_factory=lambda: _default_runtime_pool_window_size_seconds("accurate-research")
    )
    pool_window_stride_seconds: float = field(
        default_factory=lambda: _default_runtime_pool_window_stride_seconds("accurate-research")
    )
    post_smoothing_window_frames: int = field(
        default_factory=lambda: _default_runtime_post_smoothing_window_frames("accurate-research")
    )
    post_hysteresis_enter_confidence: float = field(
        default_factory=lambda: _default_runtime_post_hysteresis_enter_confidence(
            "accurate-research"
        )
    )
    post_hysteresis_exit_confidence: float = field(
        default_factory=lambda: _default_runtime_post_hysteresis_exit_confidence(
            "accurate-research"
        )
    )
    post_min_segment_duration_seconds: float = field(
        default_factory=lambda: _default_runtime_post_min_segment_duration_seconds(
            "accurate-research"
        )
    )
    process_isolation: bool = field(
        default_factory=lambda: _default_runtime_process_isolation("accurate-research")
    )


@dataclass(frozen=True)
class MediumTrainingConfig:
    """Noise-control settings for medium training dataset construction."""

    min_window_std: float = 0.0
    max_windows_per_clip: int = 0


@dataclass(frozen=True)
class QualityGateConfig:
    """Default rollout thresholds for fast-versus-medium quality gates."""

    min_uar_delta: float = 0.0025
    min_macro_f1_delta: float = 0.0025
    max_medium_segments_per_minute: float = 25.0
    min_medium_median_segment_duration_seconds: float = 2.5


@dataclass(frozen=True)
class SchemaConfig:
    """Version controls for runtime and artifact schema compatibility."""

    output_schema_version: str = "v1"
    artifact_schema_version: str = "v2"


@dataclass(frozen=True)
class TorchRuntimeConfig:
    """Torch runtime selection controls for optional HF backend acceleration."""

    device: str = "auto"
    dtype: str = "auto"
    enable_mps_fallback: bool = False


@dataclass(frozen=True)
class FeatureRuntimeBackendOverride:
    """Backend-scoped runtime selector override used by feature policy resolution."""

    device: str | None = None
    dtype: str | None = None


@dataclass(frozen=True)
class FeatureRuntimePolicyConfig:
    """Optional backend-specific runtime selector overrides."""

    backend_overrides: tuple[tuple[str, FeatureRuntimeBackendOverride], ...] = ()

    def for_backend(self, backend_id: str) -> FeatureRuntimeBackendOverride | None:
        """Returns one backend override when present."""
        normalized_backend_id = backend_id.strip().lower()
        if not normalized_backend_id:
            return None
        for candidate_backend_id, override in self.backend_overrides:
            if candidate_backend_id == normalized_backend_id:
                return override
        return None


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
    runtime_flags: RuntimeFlags = field(default_factory=RuntimeFlags)
    fast_runtime: FastRuntimeConfig = field(default_factory=FastRuntimeConfig)
    medium_runtime: MediumRuntimeConfig = field(default_factory=MediumRuntimeConfig)
    accurate_runtime: AccurateRuntimeConfig = field(default_factory=AccurateRuntimeConfig)
    accurate_research_runtime: AccurateResearchRuntimeConfig = field(
        default_factory=AccurateResearchRuntimeConfig
    )
    medium_training: MediumTrainingConfig = field(default_factory=MediumTrainingConfig)
    quality_gate: QualityGateConfig = field(default_factory=QualityGateConfig)
    schema: SchemaConfig = field(default_factory=SchemaConfig)
    torch_runtime: TorchRuntimeConfig = field(default_factory=TorchRuntimeConfig)
    feature_runtime_policy: FeatureRuntimePolicyConfig = field(
        default_factory=FeatureRuntimePolicyConfig
    )
    default_language: str = "en"


__all__ = [
    "APP_NAME",
    "DEFAULT_FAST_MODEL_FILE_NAME",
    "DEFAULT_FAST_SECURE_MODEL_FILE_NAME",
    "DEFAULT_FAST_TRAINING_REPORT_FILE_NAME",
    "AccurateResearchRuntimeConfig",
    "AccurateRuntimeConfig",
    "AppConfig",
    "ArtifactProfileName",
    "AudioReadConfig",
    "DataLoaderConfig",
    "DatasetConfig",
    "FastRuntimeConfig",
    "FeatureFlags",
    "FeatureRuntimeBackendOverride",
    "FeatureRuntimePolicyConfig",
    "MediumRuntimeConfig",
    "MediumTrainingConfig",
    "ModelsConfig",
    "NeuralNetConfig",
    "ProfileRuntimeConfig",
    "QualityGateConfig",
    "RuntimeFlags",
    "SchemaConfig",
    "TimelineConfig",
    "TorchRuntimeConfig",
    "TrainingConfig",
    "TranscriptionConfig",
    "WhisperModelConfig",
    "default_profile_model_id",
    "profile_artifact_file_names",
]
