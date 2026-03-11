"""Immutable input contract for internal settings assembly."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ser._internal.config.profile_inputs import (
    ArtifactProfileName,
    ConfidenceLevel,
    FeatureRuntimeOverrideInput,
    ReadBoolEnvOptional,
    ReadFloatEnvOptional,
    ReadIntEnvOptional,
    RuntimeProfileSettingsInput,
    build_feature_runtime_overrides,
    resolve_runtime_profile_settings_input,
)
from ser.profiles import ProfileCatalogEntry, TranscriptionBackendId


@dataclass(frozen=True, slots=True)
class ResolvedSettingsInputs:
    """Resolved environment/profile inputs for deterministic AppConfig assembly."""

    dataset_folder: Path
    manifest_paths: tuple[Path, ...]
    default_language: str
    max_workers: int
    max_failed_file_ratio: float
    test_size: float
    random_state: int
    profile_pipeline: bool
    medium_profile: bool
    accurate_profile: bool
    accurate_research_profile: bool
    restricted_backends: bool
    new_output_schema: bool
    fast_runtime_defaults: RuntimeProfileSettingsInput
    medium_runtime_defaults: RuntimeProfileSettingsInput
    accurate_runtime_defaults: RuntimeProfileSettingsInput
    accurate_research_runtime_defaults: RuntimeProfileSettingsInput
    medium_min_window_std: float
    medium_max_windows_per_clip: int
    quality_gate_min_uar_delta: float
    quality_gate_min_macro_f1_delta: float
    quality_gate_max_medium_segments_per_minute: float
    quality_gate_min_medium_median_segment_duration_seconds: float
    output_schema_version: str
    artifact_schema_version: str
    torch_device: str
    torch_dtype: str
    torch_enable_mps_fallback: bool
    transcription_mps_low_memory_threshold_gb: float
    transcription_mps_admission_control_enabled: bool
    transcription_mps_hard_oom_shortcut_enabled: bool
    transcription_mps_admission_min_headroom_mb: float
    transcription_mps_admission_safety_margin_mb: float
    transcription_mps_admission_calibration_overrides_enabled: bool
    transcription_mps_admission_calibration_min_confidence: ConfidenceLevel
    transcription_mps_admission_calibration_report_max_age_hours: float
    transcription_mps_admission_calibration_report_path: Path | None
    medium_model_id: str
    accurate_model_id: str
    accurate_research_model_id: str
    default_artifact_profile: ArtifactProfileName
    whisper_backend_id: TranscriptionBackendId
    whisper_model_name: str
    use_demucs: bool
    use_vad: bool
    model_file_name: str
    secure_model_file_name: str
    training_report_file_name: str
    cache_root: Path
    data_root: Path
    model_cache_dir: Path
    tmp_folder: Path
    models_folder: Path
    transcripts_folder: Path
    cores: int
    feature_runtime_overrides: tuple[FeatureRuntimeOverrideInput, ...]


class ReadIntEnv(Protocol):
    """Typed callable contract for integer env parsing."""

    def __call__(
        self,
        name: str,
        default: int,
        minimum: int | None = None,
    ) -> int: ...


class ReadFloatEnv(Protocol):
    """Typed callable contract for float env parsing."""

    def __call__(
        self,
        name: str,
        default: float,
        minimum: float | None = None,
        maximum: float | None = None,
    ) -> float: ...


class ResolveProfileModelId(Protocol):
    """Typed callable contract for profile model-id resolution."""

    def __call__(self, entry: ProfileCatalogEntry) -> str: ...


class ArtifactProfileResolver(Protocol):
    """Typed callable contract for artifact-profile selection from runtime flags."""

    def __call__(
        self,
        *,
        medium_profile: bool,
        accurate_profile: bool,
        accurate_research_profile: bool,
    ) -> ArtifactProfileName: ...


@dataclass(frozen=True, slots=True)
class SettingsInputDeps:
    """Dependency contract for resolving settings inputs outside `ser.config`."""

    getenv: Callable[[str, str | None], str | None]
    cpu_count: Callable[[], int | None]
    parse_manifest_paths: Callable[[str], tuple[Path, ...]]
    resolve_optional_path: Callable[[str], Path | None]
    read_path_env: Callable[[str, Path], Path]
    read_bool_env: Callable[[str, bool], bool]
    read_int_env: ReadIntEnv
    read_float_env: ReadFloatEnv
    read_bool_env_optional: ReadBoolEnvOptional
    read_int_env_optional: ReadIntEnvOptional
    read_float_env_optional: ReadFloatEnvOptional
    read_torch_device_env: Callable[[str, str], str]
    read_torch_dtype_env: Callable[[str, str], str]
    read_confidence_level_env: Callable[[str, ConfidenceLevel], ConfidenceLevel]
    get_profile_catalog: Callable[[], Mapping[ArtifactProfileName, ProfileCatalogEntry]]
    resolve_profile_enabled: Callable[[ProfileCatalogEntry], bool]
    resolve_profile_model_id: ResolveProfileModelId
    artifact_profile_from_runtime_flags: ArtifactProfileResolver
    resolve_profile_transcription_config: Callable[
        [ArtifactProfileName],
        tuple[TranscriptionBackendId, str, bool, bool],
    ]
    profile_artifact_file_names: Callable[..., tuple[str, str, str]]
    default_cache_root: Callable[[], Path]
    default_data_root: Callable[[], Path]
    ser_torch_enable_mps_fallback_env: str
    pytorch_enable_mps_fallback_env: str


def resolve_settings_inputs(deps: SettingsInputDeps) -> ResolvedSettingsInputs:
    """Resolves environment/profile values into immutable builder inputs."""
    dataset_folder = Path(
        deps.getenv("DATASET_FOLDER", "ser/dataset/ravdess") or "ser/dataset/ravdess"
    )
    manifest_paths = deps.parse_manifest_paths(
        (deps.getenv("SER_DATASET_MANIFESTS", "") or "").strip()
    )
    default_language = deps.getenv("DEFAULT_LANGUAGE", "en") or "en"

    max_workers = deps.read_int_env("SER_MAX_WORKERS", 8, minimum=1)
    max_failed_file_ratio = deps.read_float_env(
        "SER_MAX_FAILED_FILE_RATIO",
        0.01,
        minimum=0.0,
        maximum=1.0,
    )
    test_size = deps.read_float_env("SER_TEST_SIZE", 0.25, minimum=0.05, maximum=0.95)
    random_state = deps.read_int_env("SER_RANDOM_STATE", 42, minimum=0)

    profile_catalog = deps.get_profile_catalog()
    feature_runtime_overrides = build_feature_runtime_overrides(profile_catalog)
    fast_profile_entry = profile_catalog["fast"]
    medium_profile_entry = profile_catalog["medium"]
    accurate_profile_entry = profile_catalog["accurate"]
    accurate_research_profile_entry = profile_catalog["accurate-research"]

    profile_pipeline = deps.read_bool_env("SER_ENABLE_PROFILE_PIPELINE", False)
    medium_profile = deps.resolve_profile_enabled(medium_profile_entry)
    accurate_profile = deps.resolve_profile_enabled(accurate_profile_entry)
    accurate_research_profile = deps.resolve_profile_enabled(accurate_research_profile_entry)
    restricted_backends = deps.read_bool_env("SER_ENABLE_RESTRICTED_BACKENDS", False)
    new_output_schema = deps.read_bool_env("SER_ENABLE_NEW_OUTPUT_SCHEMA", False)

    fast_runtime_defaults = resolve_runtime_profile_settings_input(
        fast_profile_entry,
        read_bool_env_optional=deps.read_bool_env_optional,
        read_int_env_optional=deps.read_int_env_optional,
        read_float_env_optional=deps.read_float_env_optional,
    )
    medium_runtime_defaults = resolve_runtime_profile_settings_input(
        medium_profile_entry,
        read_bool_env_optional=deps.read_bool_env_optional,
        read_int_env_optional=deps.read_int_env_optional,
        read_float_env_optional=deps.read_float_env_optional,
    )
    accurate_runtime_defaults = resolve_runtime_profile_settings_input(
        accurate_profile_entry,
        read_bool_env_optional=deps.read_bool_env_optional,
        read_int_env_optional=deps.read_int_env_optional,
        read_float_env_optional=deps.read_float_env_optional,
    )
    accurate_research_runtime_defaults = resolve_runtime_profile_settings_input(
        accurate_research_profile_entry,
        read_bool_env_optional=deps.read_bool_env_optional,
        read_int_env_optional=deps.read_int_env_optional,
        read_float_env_optional=deps.read_float_env_optional,
    )

    medium_min_window_std = deps.read_float_env(
        "SER_MEDIUM_MIN_WINDOW_STD",
        0.0,
        minimum=0.0,
    )
    medium_max_windows_per_clip = deps.read_int_env(
        "SER_MEDIUM_MAX_WINDOWS_PER_CLIP",
        0,
        minimum=0,
    )

    quality_gate_min_uar_delta = deps.read_float_env(
        "SER_QUALITY_GATE_MIN_UAR_DELTA",
        0.0025,
        minimum=0.0,
    )
    quality_gate_min_macro_f1_delta = deps.read_float_env(
        "SER_QUALITY_GATE_MIN_MACRO_F1_DELTA",
        0.0025,
        minimum=0.0,
    )
    quality_gate_max_medium_segments_per_minute = deps.read_float_env(
        "SER_QUALITY_GATE_MAX_MEDIUM_SEGMENTS_PER_MINUTE",
        25.0,
        minimum=0.1,
    )
    quality_gate_min_medium_median_segment_duration_seconds = deps.read_float_env(
        "SER_QUALITY_GATE_MIN_MEDIUM_MEDIAN_SEGMENT_DURATION_SECONDS",
        2.5,
        minimum=0.0,
    )

    output_schema_version = deps.getenv("SER_OUTPUT_SCHEMA_VERSION", "v1") or "v1"
    artifact_schema_version = deps.getenv("SER_ARTIFACT_SCHEMA_VERSION", "v2") or "v2"

    torch_device = deps.read_torch_device_env("SER_TORCH_DEVICE", "auto")
    torch_dtype = deps.read_torch_dtype_env("SER_TORCH_DTYPE", "auto")
    torch_enable_mps_fallback = deps.read_bool_env(
        deps.ser_torch_enable_mps_fallback_env,
        deps.read_bool_env(deps.pytorch_enable_mps_fallback_env, False),
    )

    transcription_mps_low_memory_threshold_gb = deps.read_float_env(
        "SER_TRANSCRIPTION_MPS_LOW_MEMORY_GB",
        16.0,
        minimum=1.0,
    )
    transcription_mps_admission_control_enabled = deps.read_bool_env(
        "SER_TRANSCRIPTION_MPS_ADMISSION_CONTROL",
        True,
    )
    transcription_mps_hard_oom_shortcut_enabled = deps.read_bool_env(
        "SER_TRANSCRIPTION_MPS_HARD_OOM_SHORTCUT",
        True,
    )
    transcription_mps_admission_min_headroom_mb = deps.read_float_env(
        "SER_TRANSCRIPTION_MPS_MIN_HEADROOM_MB",
        64.0,
        minimum=0.0,
    )
    transcription_mps_admission_safety_margin_mb = deps.read_float_env(
        "SER_TRANSCRIPTION_MPS_SAFETY_MARGIN_MB",
        64.0,
        minimum=0.0,
    )
    transcription_mps_admission_calibration_overrides_enabled = deps.read_bool_env(
        "SER_TRANSCRIPTION_MPS_CALIBRATION_OVERRIDES",
        True,
    )
    transcription_mps_admission_calibration_min_confidence = deps.read_confidence_level_env(
        "SER_TRANSCRIPTION_MPS_CALIBRATION_MIN_CONFIDENCE",
        "high",
    )
    transcription_mps_admission_calibration_report_max_age_hours = deps.read_float_env(
        "SER_TRANSCRIPTION_MPS_CALIBRATION_REPORT_MAX_AGE_HOURS",
        168.0,
        minimum=1.0,
    )
    transcription_mps_admission_calibration_report_path = deps.resolve_optional_path(
        deps.getenv("SER_TRANSCRIPTION_MPS_CALIBRATION_REPORT_PATH", "") or ""
    )

    medium_model_id = deps.resolve_profile_model_id(medium_profile_entry)
    accurate_model_id = deps.resolve_profile_model_id(accurate_profile_entry)
    accurate_research_model_id = deps.resolve_profile_model_id(accurate_research_profile_entry)

    default_artifact_profile = deps.artifact_profile_from_runtime_flags(
        medium_profile=medium_profile,
        accurate_profile=accurate_profile,
        accurate_research_profile=accurate_research_profile,
    )
    whisper_backend_id, whisper_model_name, use_demucs, use_vad = (
        deps.resolve_profile_transcription_config(default_artifact_profile)
    )
    (
        default_model_file_name,
        default_secure_model_file_name,
        default_training_report_file_name,
    ) = deps.profile_artifact_file_names(
        profile=default_artifact_profile,
        medium_model_id=medium_model_id,
        accurate_model_id=accurate_model_id,
        accurate_research_model_id=accurate_research_model_id,
    )
    model_file_name = deps.getenv("SER_MODEL_FILE_NAME", default_model_file_name) or (
        default_model_file_name
    )
    secure_model_file_name = (
        deps.getenv(
            "SER_SECURE_MODEL_FILE_NAME",
            default_secure_model_file_name,
        )
        or default_secure_model_file_name
    )
    training_report_file_name = (
        deps.getenv(
            "SER_TRAINING_REPORT_FILE_NAME",
            default_training_report_file_name,
        )
        or default_training_report_file_name
    )

    cache_root = deps.read_path_env("SER_CACHE_DIR", deps.default_cache_root())
    data_root = deps.read_path_env("SER_DATA_DIR", deps.default_data_root())
    model_cache_dir = deps.read_path_env(
        "SER_MODEL_CACHE_DIR",
        cache_root / "model-cache",
    )
    tmp_folder = deps.read_path_env("SER_TMP_DIR", cache_root / "tmp")
    models_folder = deps.read_path_env("SER_MODELS_DIR", data_root / "models")
    transcripts_folder = deps.read_path_env(
        "SER_TRANSCRIPTS_DIR",
        data_root / "transcripts",
    )

    return ResolvedSettingsInputs(
        dataset_folder=dataset_folder,
        manifest_paths=manifest_paths,
        default_language=default_language,
        max_workers=max_workers,
        max_failed_file_ratio=max_failed_file_ratio,
        test_size=test_size,
        random_state=random_state,
        profile_pipeline=profile_pipeline,
        medium_profile=medium_profile,
        accurate_profile=accurate_profile,
        accurate_research_profile=accurate_research_profile,
        restricted_backends=restricted_backends,
        new_output_schema=new_output_schema,
        fast_runtime_defaults=fast_runtime_defaults,
        medium_runtime_defaults=medium_runtime_defaults,
        accurate_runtime_defaults=accurate_runtime_defaults,
        accurate_research_runtime_defaults=accurate_research_runtime_defaults,
        medium_min_window_std=medium_min_window_std,
        medium_max_windows_per_clip=medium_max_windows_per_clip,
        quality_gate_min_uar_delta=quality_gate_min_uar_delta,
        quality_gate_min_macro_f1_delta=quality_gate_min_macro_f1_delta,
        quality_gate_max_medium_segments_per_minute=(quality_gate_max_medium_segments_per_minute),
        quality_gate_min_medium_median_segment_duration_seconds=(
            quality_gate_min_medium_median_segment_duration_seconds
        ),
        output_schema_version=output_schema_version,
        artifact_schema_version=artifact_schema_version,
        torch_device=torch_device,
        torch_dtype=torch_dtype,
        torch_enable_mps_fallback=torch_enable_mps_fallback,
        transcription_mps_low_memory_threshold_gb=(transcription_mps_low_memory_threshold_gb),
        transcription_mps_admission_control_enabled=(transcription_mps_admission_control_enabled),
        transcription_mps_hard_oom_shortcut_enabled=(transcription_mps_hard_oom_shortcut_enabled),
        transcription_mps_admission_min_headroom_mb=(transcription_mps_admission_min_headroom_mb),
        transcription_mps_admission_safety_margin_mb=(transcription_mps_admission_safety_margin_mb),
        transcription_mps_admission_calibration_overrides_enabled=(
            transcription_mps_admission_calibration_overrides_enabled
        ),
        transcription_mps_admission_calibration_min_confidence=(
            transcription_mps_admission_calibration_min_confidence
        ),
        transcription_mps_admission_calibration_report_max_age_hours=(
            transcription_mps_admission_calibration_report_max_age_hours
        ),
        transcription_mps_admission_calibration_report_path=(
            transcription_mps_admission_calibration_report_path
        ),
        medium_model_id=medium_model_id,
        accurate_model_id=accurate_model_id,
        accurate_research_model_id=accurate_research_model_id,
        default_artifact_profile=default_artifact_profile,
        whisper_backend_id=whisper_backend_id,
        whisper_model_name=whisper_model_name,
        use_demucs=use_demucs,
        use_vad=use_vad,
        model_file_name=model_file_name,
        secure_model_file_name=secure_model_file_name,
        training_report_file_name=training_report_file_name,
        cache_root=cache_root,
        data_root=data_root,
        model_cache_dir=model_cache_dir,
        tmp_folder=tmp_folder,
        models_folder=models_folder,
        transcripts_folder=transcripts_folder,
        cores=deps.cpu_count() or 1,
        feature_runtime_overrides=feature_runtime_overrides,
    )
