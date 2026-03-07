"""Pure AppConfig assembly from resolved settings inputs."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import TypeVar

from ser._internal.config import schema as config_schema
from ser._internal.config.profile_inputs import (
    FeatureRuntimeOverrideInput,
    RuntimeProfileSettingsInput,
)
from ser._internal.config.settings_inputs import ResolvedSettingsInputs

_EMOTIONS: Mapping[str, str] = MappingProxyType(
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

_RuntimeConfigT = TypeVar(
    "_RuntimeConfigT",
    bound=config_schema.ProfileRuntimeConfig,
)


def _build_profile_runtime_config(
    *,
    defaults: RuntimeProfileSettingsInput,
    factory: type[_RuntimeConfigT],
) -> _RuntimeConfigT:
    """Builds one concrete profile runtime config from resolved defaults."""
    return factory(
        timeout_seconds=defaults.timeout_seconds,
        max_timeout_retries=defaults.max_timeout_retries,
        max_transient_retries=defaults.max_transient_retries,
        retry_backoff_seconds=defaults.retry_backoff_seconds,
        pool_window_size_seconds=defaults.pool_window_size_seconds,
        pool_window_stride_seconds=defaults.pool_window_stride_seconds,
        post_smoothing_window_frames=defaults.post_smoothing_window_frames,
        post_hysteresis_enter_confidence=defaults.post_hysteresis_enter_confidence,
        post_hysteresis_exit_confidence=defaults.post_hysteresis_exit_confidence,
        post_min_segment_duration_seconds=defaults.post_min_segment_duration_seconds,
        process_isolation=defaults.process_isolation,
    )


def _build_feature_runtime_policy(
    overrides: tuple[FeatureRuntimeOverrideInput, ...],
) -> config_schema.FeatureRuntimePolicyConfig:
    """Builds runtime policy config from flattened backend overrides."""
    return config_schema.FeatureRuntimePolicyConfig(
        backend_overrides=tuple(
            (
                override.backend_id,
                config_schema.FeatureRuntimeBackendOverride(
                    device=override.device,
                    dtype=override.dtype,
                ),
            )
            for override in overrides
        ),
    )


def build_settings_from_inputs(
    inputs: ResolvedSettingsInputs,
) -> config_schema.AppConfig:
    """Builds one immutable app settings snapshot from resolved inputs."""
    torch_runtime = config_schema.TorchRuntimeConfig(
        device=inputs.torch_device,
        dtype=inputs.torch_dtype,
        enable_mps_fallback=inputs.torch_enable_mps_fallback,
    )
    return config_schema.AppConfig(
        emotions=_EMOTIONS,
        tmp_folder=inputs.tmp_folder,
        nn=config_schema.NeuralNetConfig(random_state=inputs.random_state),
        dataset=config_schema.DatasetConfig(
            folder=inputs.dataset_folder,
            manifest_paths=inputs.manifest_paths,
        ),
        data_loader=config_schema.DataLoaderConfig(
            max_workers=inputs.max_workers,
            max_failed_file_ratio=inputs.max_failed_file_ratio,
        ),
        training=config_schema.TrainingConfig(
            test_size=inputs.test_size,
            random_state=inputs.random_state,
        ),
        models=config_schema.ModelsConfig(
            folder=inputs.models_folder,
            model_cache_dir=inputs.model_cache_dir,
            num_cores=inputs.cores,
            model_file_name=inputs.model_file_name,
            secure_model_file_name=inputs.secure_model_file_name,
            training_report_file_name=inputs.training_report_file_name,
            whisper_model=config_schema.WhisperModelConfig(
                name=inputs.whisper_model_name
            ),
            medium_model_id=inputs.medium_model_id,
            accurate_model_id=inputs.accurate_model_id,
            accurate_research_model_id=inputs.accurate_research_model_id,
        ),
        timeline=config_schema.TimelineConfig(folder=inputs.transcripts_folder),
        transcription=config_schema.TranscriptionConfig(
            backend_id=inputs.whisper_backend_id,
            use_demucs=inputs.use_demucs,
            use_vad=inputs.use_vad,
            mps_low_memory_threshold_gb=(
                inputs.transcription_mps_low_memory_threshold_gb
            ),
            mps_admission_control_enabled=(
                inputs.transcription_mps_admission_control_enabled
            ),
            mps_hard_oom_shortcut_enabled=(
                inputs.transcription_mps_hard_oom_shortcut_enabled
            ),
            mps_admission_min_headroom_mb=inputs.transcription_mps_admission_min_headroom_mb,
            mps_admission_safety_margin_mb=(
                inputs.transcription_mps_admission_safety_margin_mb
            ),
            mps_admission_calibration_overrides_enabled=(
                inputs.transcription_mps_admission_calibration_overrides_enabled
            ),
            mps_admission_calibration_min_confidence=(
                inputs.transcription_mps_admission_calibration_min_confidence
            ),
            mps_admission_calibration_report_max_age_hours=(
                inputs.transcription_mps_admission_calibration_report_max_age_hours
            ),
            mps_admission_calibration_report_path=(
                inputs.transcription_mps_admission_calibration_report_path
            ),
        ),
        runtime_flags=config_schema.RuntimeFlags(
            profile_pipeline=inputs.profile_pipeline,
            medium_profile=inputs.medium_profile,
            accurate_profile=inputs.accurate_profile,
            accurate_research_profile=inputs.accurate_research_profile,
            restricted_backends=inputs.restricted_backends,
            new_output_schema=inputs.new_output_schema,
        ),
        fast_runtime=_build_profile_runtime_config(
            defaults=inputs.fast_runtime_defaults,
            factory=config_schema.FastRuntimeConfig,
        ),
        medium_runtime=_build_profile_runtime_config(
            defaults=inputs.medium_runtime_defaults,
            factory=config_schema.MediumRuntimeConfig,
        ),
        accurate_runtime=_build_profile_runtime_config(
            defaults=inputs.accurate_runtime_defaults,
            factory=config_schema.AccurateRuntimeConfig,
        ),
        accurate_research_runtime=_build_profile_runtime_config(
            defaults=inputs.accurate_research_runtime_defaults,
            factory=config_schema.AccurateResearchRuntimeConfig,
        ),
        medium_training=config_schema.MediumTrainingConfig(
            min_window_std=inputs.medium_min_window_std,
            max_windows_per_clip=inputs.medium_max_windows_per_clip,
        ),
        quality_gate=config_schema.QualityGateConfig(
            min_uar_delta=inputs.quality_gate_min_uar_delta,
            min_macro_f1_delta=inputs.quality_gate_min_macro_f1_delta,
            max_medium_segments_per_minute=(
                inputs.quality_gate_max_medium_segments_per_minute
            ),
            min_medium_median_segment_duration_seconds=(
                inputs.quality_gate_min_medium_median_segment_duration_seconds
            ),
        ),
        schema=config_schema.SchemaConfig(
            output_schema_version=inputs.output_schema_version,
            artifact_schema_version=inputs.artifact_schema_version,
        ),
        torch_runtime=torch_runtime,
        feature_runtime_policy=_build_feature_runtime_policy(
            inputs.feature_runtime_overrides
        ),
        default_language=inputs.default_language,
    )
