"""Application configuration for SER."""

from __future__ import annotations

import os
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from hashlib import sha1
from pathlib import Path
from types import MappingProxyType
from typing import Literal, cast

from ser.profiles import (
    ProfileCatalogEntry,
    TranscriptionBackendId,
    get_profile_catalog,
)

APP_NAME = "ser"
DEFAULT_ACCURATE_MODEL_ID = "openai/whisper-large-v3"
DEFAULT_MEDIUM_MODEL_ID = "facebook/wav2vec2-xls-r-300m"
DEFAULT_ACCURATE_RESEARCH_MODEL_ID = "iic/emotion2vec_plus_large"
DEFAULT_FAST_MODEL_FILE_NAME = "ser_model.pkl"
DEFAULT_FAST_SECURE_MODEL_FILE_NAME = "ser_model.skops"
DEFAULT_FAST_TRAINING_REPORT_FILE_NAME = "training_report.json"
_SER_TORCH_ENABLE_MPS_FALLBACK_ENV = "SER_TORCH_ENABLE_MPS_FALLBACK"
_PYTORCH_ENABLE_MPS_FALLBACK_ENV = "PYTORCH_ENABLE_MPS_FALLBACK"

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
    if accurate_research_profile:
        return "accurate-research"
    if accurate_profile:
        return "accurate"
    if medium_profile:
        return "medium"
    return "fast"


def _artifact_model_id_suffix(model_id: str) -> str:
    """Builds a stable, filename-safe suffix for backend model ids."""
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", model_id.strip().lower())
    cleaned = cleaned.strip("._-")
    if not cleaned:
        cleaned = "model"
    trimmed = cleaned[:48]
    digest = sha1(model_id.encode("utf-8")).hexdigest()[:10]
    return f"{trimmed}_{digest}"


def profile_artifact_file_names(
    *,
    profile: ArtifactProfileName,
    medium_model_id: str = DEFAULT_MEDIUM_MODEL_ID,
    accurate_model_id: str = DEFAULT_ACCURATE_MODEL_ID,
    accurate_research_model_id: str = DEFAULT_ACCURATE_RESEARCH_MODEL_ID,
) -> tuple[str, str, str]:
    """Returns default artifact filenames for one profile/backend-model tuple."""
    if profile == "fast":
        return (
            DEFAULT_FAST_MODEL_FILE_NAME,
            DEFAULT_FAST_SECURE_MODEL_FILE_NAME,
            DEFAULT_FAST_TRAINING_REPORT_FILE_NAME,
        )

    if profile == "medium":
        backend_model_id = medium_model_id
    elif profile == "accurate":
        backend_model_id = accurate_model_id
    else:
        backend_model_id = accurate_research_model_id

    model_suffix = _artifact_model_id_suffix(backend_model_id)
    profile_token = profile.replace("-", "_")
    model_stem = f"ser_model_{profile_token}_{model_suffix}"
    report_stem = f"training_report_{profile_token}_{model_suffix}"
    return (f"{model_stem}.pkl", f"{model_stem}.skops", f"{report_stem}.json")


def _read_path_env(name: str, default: Path) -> Path:
    """Reads a path env var and returns an expanded path."""
    raw_value: str | None = os.getenv(name)
    selected: Path = Path(raw_value) if raw_value else default
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

    name: str = "large-v2"
    relative_path: Path = Path("OpenAI/whisper")


@dataclass(frozen=True)
class ModelsConfig:
    """Storage and runtime settings for trained artifacts."""

    folder: Path = field(default_factory=_default_models_folder)
    model_cache_dir: Path = field(default_factory=_default_model_cache_root)
    whisper_model: WhisperModelConfig = field(default_factory=WhisperModelConfig)
    medium_model_id: str = DEFAULT_MEDIUM_MODEL_ID
    accurate_model_id: str = DEFAULT_ACCURATE_MODEL_ID
    accurate_research_model_id: str = DEFAULT_ACCURATE_RESEARCH_MODEL_ID
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

    backend_id: TranscriptionBackendId = "stable_whisper"
    use_demucs: bool = True
    use_vad: bool = True
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
    pool_window_size_seconds: float = 1.0
    pool_window_stride_seconds: float = 1.0
    post_smoothing_window_frames: int = 3
    post_hysteresis_enter_confidence: float = 0.60
    post_hysteresis_exit_confidence: float = 0.45
    post_min_segment_duration_seconds: float = 0.40
    process_isolation: bool = True


@dataclass(frozen=True)
class FastRuntimeConfig(ProfileRuntimeConfig):
    """Execution budgets and retry controls for fast profile runtime."""

    timeout_seconds: float = 0.0
    max_timeout_retries: int = 0
    max_transient_retries: int = 0
    retry_backoff_seconds: float = 0.0
    process_isolation: bool = False


@dataclass(frozen=True)
class MediumRuntimeConfig(ProfileRuntimeConfig):
    """Execution budgets and retry controls for medium profile runtime."""

    timeout_seconds: float = 60.0
    max_timeout_retries: int = 1
    max_transient_retries: int = 1
    retry_backoff_seconds: float = 0.25


@dataclass(frozen=True)
class AccurateRuntimeConfig(ProfileRuntimeConfig):
    """Execution budgets and retry controls for accurate profile runtime."""

    timeout_seconds: float = 120.0
    max_timeout_retries: int = 0
    max_transient_retries: int = 1
    retry_backoff_seconds: float = 0.25


@dataclass(frozen=True)
class AccurateResearchRuntimeConfig(ProfileRuntimeConfig):
    """Execution budgets and retry controls for accurate-research runtime."""

    timeout_seconds: float = 120.0
    max_timeout_retries: int = 0
    max_transient_retries: int = 1
    retry_backoff_seconds: float = 0.25


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
    accurate_runtime: AccurateRuntimeConfig = field(
        default_factory=AccurateRuntimeConfig
    )
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


def _read_bool_env(name: str, default: bool) -> bool:
    """Reads a boolean env var and returns a validated value."""
    raw_value: str | None = os.getenv(name)
    if raw_value is None:
        return default

    normalized: str = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    return False if normalized in {"0", "false", "no", "off"} else default


def _sync_torch_runtime_environment(torch_runtime: TorchRuntimeConfig) -> None:
    """Synchronizes torch runtime compatibility env vars from settings."""
    os.environ[_PYTORCH_ENABLE_MPS_FALLBACK_ENV] = (
        "1" if torch_runtime.enable_mps_fallback else "0"
    )


def _read_int_env(name: str, default: int, minimum: int | None = None) -> int:
    """Reads an integer env var and enforces an optional lower bound."""
    raw_value: str | None = os.getenv(name)
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


def _read_bool_env_optional(name: str | None, default: bool) -> bool:
    """Reads a nullable boolean env var and returns a validated value."""
    if not isinstance(name, str) or not name.strip():
        return default
    return _read_bool_env(name, default)


def _read_int_env_optional(
    name: str | None,
    default: int,
    *,
    minimum: int | None = None,
) -> int:
    """Reads a nullable integer env var and enforces an optional lower bound."""
    if not isinstance(name, str) or not name.strip():
        return default
    return _read_int_env(name, default, minimum=minimum)


def _read_float_env_optional(
    name: str | None,
    default: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    """Reads a nullable float env var and enforces optional bounds."""
    if not isinstance(name, str) or not name.strip():
        return default
    return _read_float_env(name, default, minimum=minimum, maximum=maximum)


def _resolve_profile_enabled(entry: ProfileCatalogEntry) -> bool:
    """Returns whether one profile is enabled after env/default resolution."""
    return _read_bool_env_optional(entry.enable_flag, entry.enabled_by_default)


def _resolve_profile_model_id(
    entry: ProfileCatalogEntry,
    *,
    fallback_default_model_id: str,
) -> str:
    """Resolves one profile model id from definition defaults and environment."""
    default_model_id = (
        entry.model.default_model_id
        if isinstance(entry.model.default_model_id, str)
        and entry.model.default_model_id.strip()
        else fallback_default_model_id
    )
    env_var = entry.model.env_var
    if not isinstance(env_var, str) or not env_var.strip():
        return default_model_id
    resolved = os.getenv(env_var, default_model_id).strip()
    return resolved if resolved else default_model_id


def _read_torch_device_env(name: str, default: str = "auto") -> str:
    """Reads and validates torch device selector from environment."""
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    normalized = raw_value.strip().lower()
    if normalized in {"auto", "cpu", "mps", "cuda"}:
        return normalized
    if re.fullmatch(r"cuda:\d+", normalized):
        return normalized
    return default


def _read_torch_dtype_env(name: str, default: str = "auto") -> str:
    """Reads and validates torch dtype selector from environment."""
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    normalized = raw_value.strip().lower()
    return (
        normalized
        if normalized in {"auto", "float32", "float16", "bfloat16"}
        else default
    )


def _read_confidence_level_env(
    name: str,
    default: Literal["high", "medium", "low"],
) -> Literal["high", "medium", "low"]:
    """Reads one confidence level env var with strict validation."""
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    normalized = raw_value.strip().lower()
    if normalized in {"high", "medium", "low"}:
        return cast(Literal["high", "medium", "low"], normalized)
    return default


def _build_feature_runtime_policy_config(
    profile_catalog: Mapping[ArtifactProfileName, ProfileCatalogEntry],
) -> FeatureRuntimePolicyConfig:
    """Builds backend-level feature-runtime overrides from profile definitions."""
    overrides: dict[str, FeatureRuntimeBackendOverride] = {}
    source_profiles: dict[str, ArtifactProfileName] = {}
    for profile_name, entry in profile_catalog.items():
        defaults = entry.feature_runtime_defaults
        if defaults is None:
            continue
        backend_id = entry.backend_id.strip().lower()
        candidate = FeatureRuntimeBackendOverride(
            device=defaults.torch_device,
            dtype=defaults.torch_dtype,
        )
        existing = overrides.get(backend_id)
        if existing is not None and existing != candidate:
            previous_profile = source_profiles[backend_id]
            raise RuntimeError(
                "Profile definitions contain conflicting feature runtime defaults for "
                f"backend_id={backend_id!r} across profiles "
                f"{previous_profile!r} and {profile_name!r}."
            )
        overrides[backend_id] = candidate
        source_profiles[backend_id] = profile_name
    return FeatureRuntimePolicyConfig(
        backend_overrides=tuple(sorted(overrides.items())),
    )


def _resolve_runtime_config_from_profile(
    entry: ProfileCatalogEntry,
) -> ProfileRuntimeConfig:
    """Resolves profile runtime config from YAML defaults + env overrides."""
    runtime_defaults = entry.runtime_defaults
    runtime_env = entry.runtime_env

    timeout_seconds = _read_float_env_optional(
        runtime_env.timeout_seconds,
        runtime_defaults.timeout_seconds,
        minimum=0.0,
    )
    max_timeout_retries = _read_int_env_optional(
        runtime_env.max_timeout_retries,
        runtime_defaults.max_timeout_retries,
        minimum=0,
    )
    max_transient_retries = _read_int_env_optional(
        runtime_env.max_transient_retries,
        runtime_defaults.max_transient_retries,
        minimum=0,
    )
    retry_backoff_seconds = _read_float_env_optional(
        runtime_env.retry_backoff_seconds,
        runtime_defaults.retry_backoff_seconds,
        minimum=0.0,
    )
    pool_window_size_seconds = _read_float_env_optional(
        runtime_env.pool_window_size_seconds,
        runtime_defaults.pool_window_size_seconds,
        minimum=0.05,
    )
    pool_window_stride_seconds = _read_float_env_optional(
        runtime_env.pool_window_stride_seconds,
        runtime_defaults.pool_window_stride_seconds,
        minimum=0.05,
    )
    post_smoothing_window_frames = _read_int_env_optional(
        runtime_env.post_smoothing_window_frames,
        runtime_defaults.post_smoothing_window_frames,
        minimum=1,
    )
    post_hysteresis_enter_confidence = _read_float_env_optional(
        runtime_env.post_hysteresis_enter_confidence,
        runtime_defaults.post_hysteresis_enter_confidence,
        minimum=0.0,
        maximum=1.0,
    )
    post_hysteresis_exit_confidence = _read_float_env_optional(
        runtime_env.post_hysteresis_exit_confidence,
        runtime_defaults.post_hysteresis_exit_confidence,
        minimum=0.0,
        maximum=1.0,
    )
    if post_hysteresis_enter_confidence < post_hysteresis_exit_confidence:
        post_hysteresis_enter_confidence = (
            runtime_defaults.post_hysteresis_enter_confidence
        )
        post_hysteresis_exit_confidence = (
            runtime_defaults.post_hysteresis_exit_confidence
        )
    post_min_segment_duration_seconds = _read_float_env_optional(
        runtime_env.post_min_segment_duration_seconds,
        runtime_defaults.post_min_segment_duration_seconds,
        minimum=0.0,
    )
    process_isolation = _read_bool_env_optional(
        runtime_env.process_isolation,
        runtime_defaults.process_isolation,
    )

    return ProfileRuntimeConfig(
        timeout_seconds=timeout_seconds,
        max_timeout_retries=max_timeout_retries,
        max_transient_retries=max_transient_retries,
        retry_backoff_seconds=retry_backoff_seconds,
        pool_window_size_seconds=pool_window_size_seconds,
        pool_window_stride_seconds=pool_window_stride_seconds,
        post_smoothing_window_frames=post_smoothing_window_frames,
        post_hysteresis_enter_confidence=post_hysteresis_enter_confidence,
        post_hysteresis_exit_confidence=post_hysteresis_exit_confidence,
        post_min_segment_duration_seconds=post_min_segment_duration_seconds,
        process_isolation=process_isolation,
    )


def resolve_profile_transcription_config(
    profile: ArtifactProfileName,
) -> tuple[TranscriptionBackendId, str, bool, bool]:
    """Resolves transcription defaults for one profile with env overrides."""
    entry = get_profile_catalog()[profile]
    defaults = entry.transcription_defaults
    backend_override = os.getenv("WHISPER_BACKEND")
    backend_id_raw = (
        defaults.backend_id
        if backend_override is None
        else backend_override.strip().lower()
    )
    if backend_id_raw in {"stable_whisper", "faster_whisper"}:
        backend_id = cast(TranscriptionBackendId, backend_id_raw)
    else:
        backend_id = defaults.backend_id
    model_name = os.getenv("WHISPER_MODEL", defaults.model_name).strip()
    if not model_name:
        model_name = defaults.model_name
    use_demucs = _read_bool_env("WHISPER_DEMUCS", defaults.use_demucs)
    use_vad = _read_bool_env("WHISPER_VAD", defaults.use_vad)
    return backend_id, model_name, use_demucs, use_vad


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
    dataset_manifests_raw = os.getenv("SER_DATASET_MANIFESTS", "").strip()
    manifest_paths: tuple[Path, ...] = ()
    if dataset_manifests_raw:
        manifest_paths = tuple(
            Path(item.strip()).expanduser()
            for item in dataset_manifests_raw.split(",")
            if item.strip()
        )
    default_language: str = os.getenv("DEFAULT_LANGUAGE", "en")
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
    profile_catalog = get_profile_catalog()
    feature_runtime_policy = _build_feature_runtime_policy_config(profile_catalog)
    fast_profile_entry = profile_catalog["fast"]
    medium_profile_entry = profile_catalog["medium"]
    accurate_profile_entry = profile_catalog["accurate"]
    accurate_research_profile_entry = profile_catalog["accurate-research"]
    profile_pipeline: bool = _read_bool_env("SER_ENABLE_PROFILE_PIPELINE", False)
    medium_profile: bool = _resolve_profile_enabled(medium_profile_entry)
    accurate_profile: bool = _resolve_profile_enabled(accurate_profile_entry)
    accurate_research_profile: bool = _resolve_profile_enabled(
        accurate_research_profile_entry
    )
    restricted_backends: bool = _read_bool_env("SER_ENABLE_RESTRICTED_BACKENDS", False)
    new_output_schema: bool = _read_bool_env("SER_ENABLE_NEW_OUTPUT_SCHEMA", False)
    fast_runtime_defaults = _resolve_runtime_config_from_profile(fast_profile_entry)
    medium_runtime_defaults = _resolve_runtime_config_from_profile(medium_profile_entry)
    accurate_runtime_defaults = _resolve_runtime_config_from_profile(
        accurate_profile_entry
    )
    accurate_research_runtime_defaults = _resolve_runtime_config_from_profile(
        accurate_research_profile_entry
    )
    medium_min_window_std: float = _read_float_env(
        "SER_MEDIUM_MIN_WINDOW_STD",
        0.0,
        minimum=0.0,
    )
    medium_max_windows_per_clip: int = _read_int_env(
        "SER_MEDIUM_MAX_WINDOWS_PER_CLIP",
        0,
        minimum=0,
    )
    quality_gate_min_uar_delta: float = _read_float_env(
        "SER_QUALITY_GATE_MIN_UAR_DELTA",
        0.0025,
        minimum=0.0,
    )
    quality_gate_min_macro_f1_delta: float = _read_float_env(
        "SER_QUALITY_GATE_MIN_MACRO_F1_DELTA",
        0.0025,
        minimum=0.0,
    )
    quality_gate_max_medium_segments_per_minute: float = _read_float_env(
        "SER_QUALITY_GATE_MAX_MEDIUM_SEGMENTS_PER_MINUTE",
        25.0,
        minimum=0.1,
    )
    quality_gate_min_medium_median_segment_duration_seconds: float = _read_float_env(
        "SER_QUALITY_GATE_MIN_MEDIUM_MEDIAN_SEGMENT_DURATION_SECONDS",
        2.5,
        minimum=0.0,
    )
    output_schema_version: str = os.getenv("SER_OUTPUT_SCHEMA_VERSION", "v1")
    artifact_schema_version: str = os.getenv("SER_ARTIFACT_SCHEMA_VERSION", "v2")
    torch_device: str = _read_torch_device_env("SER_TORCH_DEVICE", "auto")
    torch_dtype: str = _read_torch_dtype_env("SER_TORCH_DTYPE", "auto")
    torch_enable_mps_fallback: bool = _read_bool_env(
        _SER_TORCH_ENABLE_MPS_FALLBACK_ENV,
        _read_bool_env(_PYTORCH_ENABLE_MPS_FALLBACK_ENV, False),
    )
    transcription_mps_low_memory_threshold_gb: float = _read_float_env(
        "SER_TRANSCRIPTION_MPS_LOW_MEMORY_GB",
        16.0,
        minimum=1.0,
    )
    transcription_mps_admission_control_enabled: bool = _read_bool_env(
        "SER_TRANSCRIPTION_MPS_ADMISSION_CONTROL",
        True,
    )
    transcription_mps_hard_oom_shortcut_enabled: bool = _read_bool_env(
        "SER_TRANSCRIPTION_MPS_HARD_OOM_SHORTCUT",
        True,
    )
    transcription_mps_admission_min_headroom_mb: float = _read_float_env(
        "SER_TRANSCRIPTION_MPS_MIN_HEADROOM_MB",
        64.0,
        minimum=0.0,
    )
    transcription_mps_admission_safety_margin_mb: float = _read_float_env(
        "SER_TRANSCRIPTION_MPS_SAFETY_MARGIN_MB",
        64.0,
        minimum=0.0,
    )
    transcription_mps_admission_calibration_overrides_enabled: bool = _read_bool_env(
        "SER_TRANSCRIPTION_MPS_CALIBRATION_OVERRIDES",
        True,
    )
    transcription_mps_admission_calibration_min_confidence = _read_confidence_level_env(
        "SER_TRANSCRIPTION_MPS_CALIBRATION_MIN_CONFIDENCE",
        "high",
    )
    transcription_mps_admission_calibration_report_max_age_hours: float = (
        _read_float_env(
            "SER_TRANSCRIPTION_MPS_CALIBRATION_REPORT_MAX_AGE_HOURS",
            168.0,
            minimum=1.0,
        )
    )
    transcription_mps_admission_calibration_report_path_raw = os.getenv(
        "SER_TRANSCRIPTION_MPS_CALIBRATION_REPORT_PATH",
        "",
    ).strip()
    transcription_mps_admission_calibration_report_path = (
        Path(transcription_mps_admission_calibration_report_path_raw).expanduser()
        if transcription_mps_admission_calibration_report_path_raw
        else None
    )
    torch_runtime = TorchRuntimeConfig(
        device=torch_device,
        dtype=torch_dtype,
        enable_mps_fallback=torch_enable_mps_fallback,
    )
    _sync_torch_runtime_environment(torch_runtime)
    medium_model_id = _resolve_profile_model_id(
        medium_profile_entry,
        fallback_default_model_id=DEFAULT_MEDIUM_MODEL_ID,
    )
    accurate_model_id = _resolve_profile_model_id(
        accurate_profile_entry,
        fallback_default_model_id=DEFAULT_ACCURATE_MODEL_ID,
    )
    accurate_research_model_id = _resolve_profile_model_id(
        accurate_research_profile_entry,
        fallback_default_model_id=DEFAULT_ACCURATE_RESEARCH_MODEL_ID,
    )
    default_artifact_profile = _artifact_profile_from_runtime_flags(
        medium_profile=medium_profile,
        accurate_profile=accurate_profile,
        accurate_research_profile=accurate_research_profile,
    )
    (
        whisper_backend_id,
        whisper_model_name,
        use_demucs,
        use_vad,
    ) = resolve_profile_transcription_config(default_artifact_profile)
    (
        default_model_file_name,
        default_secure_model_file_name,
        default_training_report_file_name,
    ) = profile_artifact_file_names(
        profile=default_artifact_profile,
        medium_model_id=medium_model_id,
        accurate_model_id=accurate_model_id,
        accurate_research_model_id=accurate_research_model_id,
    )
    model_file_name: str = os.getenv("SER_MODEL_FILE_NAME", default_model_file_name)
    secure_model_file_name: str = os.getenv(
        "SER_SECURE_MODEL_FILE_NAME",
        default_secure_model_file_name,
    )
    training_report_file_name: str = os.getenv(
        "SER_TRAINING_REPORT_FILE_NAME",
        default_training_report_file_name,
    )
    cache_root: Path = _read_path_env("SER_CACHE_DIR", _default_cache_root())
    data_root: Path = _read_path_env("SER_DATA_DIR", _default_data_root())
    model_cache_dir: Path = _read_path_env(
        "SER_MODEL_CACHE_DIR",
        cache_root / "model-cache",
    )
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
        dataset=DatasetConfig(folder=dataset_folder, manifest_paths=manifest_paths),
        data_loader=DataLoaderConfig(
            max_workers=max_workers,
            max_failed_file_ratio=max_failed_file_ratio,
        ),
        training=TrainingConfig(test_size=test_size, random_state=random_state),
        models=ModelsConfig(
            folder=models_folder,
            model_cache_dir=model_cache_dir,
            num_cores=cores,
            model_file_name=model_file_name,
            secure_model_file_name=secure_model_file_name,
            training_report_file_name=training_report_file_name,
            whisper_model=WhisperModelConfig(name=whisper_model_name),
            medium_model_id=medium_model_id,
            accurate_model_id=accurate_model_id,
            accurate_research_model_id=accurate_research_model_id,
        ),
        timeline=TimelineConfig(folder=transcripts_folder),
        transcription=TranscriptionConfig(
            backend_id=whisper_backend_id,
            use_demucs=use_demucs,
            use_vad=use_vad,
            mps_low_memory_threshold_gb=transcription_mps_low_memory_threshold_gb,
            mps_admission_control_enabled=transcription_mps_admission_control_enabled,
            mps_hard_oom_shortcut_enabled=transcription_mps_hard_oom_shortcut_enabled,
            mps_admission_min_headroom_mb=transcription_mps_admission_min_headroom_mb,
            mps_admission_safety_margin_mb=transcription_mps_admission_safety_margin_mb,
            mps_admission_calibration_overrides_enabled=(
                transcription_mps_admission_calibration_overrides_enabled
            ),
            mps_admission_calibration_min_confidence=(
                transcription_mps_admission_calibration_min_confidence
            ),
            mps_admission_calibration_report_max_age_hours=(
                transcription_mps_admission_calibration_report_max_age_hours
            ),
            mps_admission_calibration_report_path=(
                transcription_mps_admission_calibration_report_path
            ),
        ),
        runtime_flags=RuntimeFlags(
            profile_pipeline=profile_pipeline,
            medium_profile=medium_profile,
            accurate_profile=accurate_profile,
            accurate_research_profile=accurate_research_profile,
            restricted_backends=restricted_backends,
            new_output_schema=new_output_schema,
        ),
        fast_runtime=FastRuntimeConfig(
            timeout_seconds=fast_runtime_defaults.timeout_seconds,
            max_timeout_retries=fast_runtime_defaults.max_timeout_retries,
            max_transient_retries=fast_runtime_defaults.max_transient_retries,
            retry_backoff_seconds=fast_runtime_defaults.retry_backoff_seconds,
            pool_window_size_seconds=fast_runtime_defaults.pool_window_size_seconds,
            pool_window_stride_seconds=fast_runtime_defaults.pool_window_stride_seconds,
            post_smoothing_window_frames=(
                fast_runtime_defaults.post_smoothing_window_frames
            ),
            post_hysteresis_enter_confidence=(
                fast_runtime_defaults.post_hysteresis_enter_confidence
            ),
            post_hysteresis_exit_confidence=(
                fast_runtime_defaults.post_hysteresis_exit_confidence
            ),
            post_min_segment_duration_seconds=(
                fast_runtime_defaults.post_min_segment_duration_seconds
            ),
            process_isolation=fast_runtime_defaults.process_isolation,
        ),
        medium_runtime=MediumRuntimeConfig(
            timeout_seconds=medium_runtime_defaults.timeout_seconds,
            max_timeout_retries=medium_runtime_defaults.max_timeout_retries,
            max_transient_retries=medium_runtime_defaults.max_transient_retries,
            retry_backoff_seconds=medium_runtime_defaults.retry_backoff_seconds,
            pool_window_size_seconds=medium_runtime_defaults.pool_window_size_seconds,
            pool_window_stride_seconds=(
                medium_runtime_defaults.pool_window_stride_seconds
            ),
            post_smoothing_window_frames=(
                medium_runtime_defaults.post_smoothing_window_frames
            ),
            post_hysteresis_enter_confidence=(
                medium_runtime_defaults.post_hysteresis_enter_confidence
            ),
            post_hysteresis_exit_confidence=(
                medium_runtime_defaults.post_hysteresis_exit_confidence
            ),
            post_min_segment_duration_seconds=(
                medium_runtime_defaults.post_min_segment_duration_seconds
            ),
            process_isolation=medium_runtime_defaults.process_isolation,
        ),
        accurate_runtime=AccurateRuntimeConfig(
            timeout_seconds=accurate_runtime_defaults.timeout_seconds,
            max_timeout_retries=accurate_runtime_defaults.max_timeout_retries,
            max_transient_retries=accurate_runtime_defaults.max_transient_retries,
            retry_backoff_seconds=accurate_runtime_defaults.retry_backoff_seconds,
            pool_window_size_seconds=(
                accurate_runtime_defaults.pool_window_size_seconds
            ),
            pool_window_stride_seconds=(
                accurate_runtime_defaults.pool_window_stride_seconds
            ),
            post_smoothing_window_frames=(
                accurate_runtime_defaults.post_smoothing_window_frames
            ),
            post_hysteresis_enter_confidence=(
                accurate_runtime_defaults.post_hysteresis_enter_confidence
            ),
            post_hysteresis_exit_confidence=(
                accurate_runtime_defaults.post_hysteresis_exit_confidence
            ),
            post_min_segment_duration_seconds=(
                accurate_runtime_defaults.post_min_segment_duration_seconds
            ),
            process_isolation=accurate_runtime_defaults.process_isolation,
        ),
        accurate_research_runtime=AccurateResearchRuntimeConfig(
            timeout_seconds=accurate_research_runtime_defaults.timeout_seconds,
            max_timeout_retries=(
                accurate_research_runtime_defaults.max_timeout_retries
            ),
            max_transient_retries=(
                accurate_research_runtime_defaults.max_transient_retries
            ),
            retry_backoff_seconds=(
                accurate_research_runtime_defaults.retry_backoff_seconds
            ),
            pool_window_size_seconds=(
                accurate_research_runtime_defaults.pool_window_size_seconds
            ),
            pool_window_stride_seconds=(
                accurate_research_runtime_defaults.pool_window_stride_seconds
            ),
            post_smoothing_window_frames=(
                accurate_research_runtime_defaults.post_smoothing_window_frames
            ),
            post_hysteresis_enter_confidence=(
                accurate_research_runtime_defaults.post_hysteresis_enter_confidence
            ),
            post_hysteresis_exit_confidence=(
                accurate_research_runtime_defaults.post_hysteresis_exit_confidence
            ),
            post_min_segment_duration_seconds=(
                accurate_research_runtime_defaults.post_min_segment_duration_seconds
            ),
            process_isolation=accurate_research_runtime_defaults.process_isolation,
        ),
        medium_training=MediumTrainingConfig(
            min_window_std=medium_min_window_std,
            max_windows_per_clip=medium_max_windows_per_clip,
        ),
        quality_gate=QualityGateConfig(
            min_uar_delta=quality_gate_min_uar_delta,
            min_macro_f1_delta=quality_gate_min_macro_f1_delta,
            max_medium_segments_per_minute=(
                quality_gate_max_medium_segments_per_minute
            ),
            min_medium_median_segment_duration_seconds=(
                quality_gate_min_medium_median_segment_duration_seconds
            ),
        ),
        schema=SchemaConfig(
            output_schema_version=output_schema_version,
            artifact_schema_version=artifact_schema_version,
        ),
        torch_runtime=torch_runtime,
        feature_runtime_policy=feature_runtime_policy,
        default_language=default_language,
    )


_SETTINGS: AppConfig = _build_settings()


def get_settings() -> AppConfig:
    """Returns current immutable settings."""
    return _SETTINGS


def apply_settings(settings: AppConfig) -> AppConfig:
    """Applies an explicit settings snapshot as the active process settings."""
    global _SETTINGS
    _SETTINGS = settings
    _sync_torch_runtime_environment(_SETTINGS.torch_runtime)
    return _SETTINGS


def reload_settings() -> AppConfig:
    """Reloads settings from environment and returns the new snapshot."""
    global _SETTINGS
    _SETTINGS = _build_settings()
    return _SETTINGS
