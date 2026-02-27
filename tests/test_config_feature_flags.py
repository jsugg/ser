"""Tests for runtime feature-flag and schema version configuration."""

import os
from collections.abc import Generator
from dataclasses import replace
from pathlib import Path

import pytest

import ser.config as config


@pytest.fixture(autouse=True)
def _reset_settings(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    """Keeps global settings stable across tests."""
    config.reload_settings()
    yield
    monkeypatch.delenv("WHISPER_BACKEND", raising=False)
    config.reload_settings()


def test_runtime_flags_and_schema_defaults() -> None:
    """Feature flags and schema versions should default to conservative values."""
    settings = config.reload_settings()

    assert settings.runtime_flags.profile_pipeline is False
    assert settings.runtime_flags.medium_profile is False
    assert settings.runtime_flags.accurate_profile is False
    assert settings.runtime_flags.accurate_research_profile is False
    assert settings.runtime_flags.restricted_backends is False
    assert settings.runtime_flags.new_output_schema is False
    assert settings.fast_runtime.timeout_seconds == pytest.approx(0.0)
    assert settings.fast_runtime.max_timeout_retries == 0
    assert settings.fast_runtime.max_transient_retries == 0
    assert settings.fast_runtime.retry_backoff_seconds == pytest.approx(0.0)
    assert settings.fast_runtime.process_isolation is False
    assert settings.medium_runtime.timeout_seconds == pytest.approx(60.0)
    assert settings.medium_runtime.max_timeout_retries == 1
    assert settings.medium_runtime.max_transient_retries == 1
    assert settings.medium_runtime.retry_backoff_seconds == pytest.approx(0.25)
    assert settings.medium_runtime.pool_window_size_seconds == pytest.approx(1.0)
    assert settings.medium_runtime.pool_window_stride_seconds == pytest.approx(1.0)
    assert settings.medium_runtime.post_smoothing_window_frames == 3
    assert settings.medium_runtime.post_hysteresis_enter_confidence == pytest.approx(
        0.60
    )
    assert settings.medium_runtime.post_hysteresis_exit_confidence == pytest.approx(
        0.45
    )
    assert settings.medium_runtime.post_min_segment_duration_seconds == pytest.approx(
        0.40
    )
    assert settings.medium_runtime.process_isolation is True
    assert settings.accurate_runtime.timeout_seconds == pytest.approx(120.0)
    assert settings.accurate_runtime.max_timeout_retries == 0
    assert settings.accurate_runtime.max_transient_retries == 1
    assert settings.accurate_runtime.retry_backoff_seconds == pytest.approx(0.25)
    assert settings.accurate_runtime.process_isolation is True
    assert settings.accurate_research_runtime.timeout_seconds == pytest.approx(120.0)
    assert settings.accurate_research_runtime.max_timeout_retries == 0
    assert settings.accurate_research_runtime.max_transient_retries == 1
    assert settings.accurate_research_runtime.retry_backoff_seconds == pytest.approx(
        0.25
    )
    assert settings.accurate_research_runtime.process_isolation is True
    assert settings.medium_training.min_window_std == pytest.approx(0.0)
    assert settings.medium_training.max_windows_per_clip == 0
    assert settings.quality_gate.min_uar_delta == pytest.approx(0.0025)
    assert settings.quality_gate.min_macro_f1_delta == pytest.approx(0.0025)
    assert settings.quality_gate.max_medium_segments_per_minute == pytest.approx(25.0)
    assert settings.quality_gate.min_medium_median_segment_duration_seconds == (
        pytest.approx(2.5)
    )
    assert settings.schema.output_schema_version == "v1"
    assert settings.schema.artifact_schema_version == "v2"
    assert settings.torch_runtime.device == "auto"
    assert settings.torch_runtime.dtype == "auto"
    assert settings.torch_runtime.enable_mps_fallback is False
    xlsr_override = settings.feature_runtime_policy.for_backend("hf_xlsr")
    assert xlsr_override is not None
    assert xlsr_override.device is None
    assert xlsr_override.dtype == "float32"
    assert settings.feature_runtime_policy.for_backend("hf_whisper") is None
    assert os.getenv("PYTORCH_ENABLE_MPS_FALLBACK") == "0"
    assert settings.models.medium_model_id == "facebook/wav2vec2-xls-r-300m"
    assert settings.models.accurate_model_id == "openai/whisper-large-v3"
    assert settings.models.accurate_research_model_id == "iic/emotion2vec_plus_large"
    assert settings.models.whisper_model.name == "distil-large-v3"
    assert settings.transcription.backend_id == "faster_whisper"
    assert settings.transcription.use_demucs is False
    assert settings.transcription.use_vad is True
    assert settings.transcription.mps_low_memory_threshold_gb == pytest.approx(16.0)
    assert settings.transcription.mps_admission_control_enabled is True
    assert settings.transcription.mps_hard_oom_shortcut_enabled is True
    assert settings.transcription.mps_admission_min_headroom_mb == pytest.approx(64.0)
    assert settings.transcription.mps_admission_safety_margin_mb == pytest.approx(64.0)
    assert settings.transcription.mps_admission_calibration_overrides_enabled is True
    assert settings.transcription.mps_admission_calibration_min_confidence == "high"
    assert settings.transcription.mps_admission_calibration_report_max_age_hours == (
        pytest.approx(168.0)
    )
    assert settings.transcription.mps_admission_calibration_report_path is None
    assert settings.models.model_cache_dir.name == "model-cache"
    assert settings.models.huggingface_cache_root == (
        settings.models.model_cache_dir / "huggingface"
    )
    assert settings.models.modelscope_cache_root == (
        settings.models.model_cache_dir / "modelscope" / "hub"
    )
    assert settings.models.torch_cache_root == (
        settings.models.model_cache_dir / "torch"
    )
    assert settings.models.model_file_name == "ser_model.pkl"
    assert settings.models.secure_model_file_name == "ser_model.skops"
    assert settings.models.training_report_file_name == "training_report.json"
    assert settings.dataset.manifest_paths == ()


def test_runtime_flags_and_schema_env_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Environment variables should override rollout flags and schema versions."""
    monkeypatch.setenv("SER_ENABLE_PROFILE_PIPELINE", "true")
    monkeypatch.setenv("SER_ENABLE_MEDIUM_PROFILE", "1")
    monkeypatch.setenv("SER_ENABLE_ACCURATE_PROFILE", "yes")
    monkeypatch.setenv("SER_ENABLE_ACCURATE_RESEARCH_PROFILE", "yes")
    monkeypatch.setenv("SER_ENABLE_RESTRICTED_BACKENDS", "on")
    monkeypatch.setenv("SER_ENABLE_NEW_OUTPUT_SCHEMA", "true")
    monkeypatch.setenv("SER_FAST_TIMEOUT_SECONDS", "15.0")
    monkeypatch.setenv("SER_FAST_MAX_TIMEOUT_RETRIES", "2")
    monkeypatch.setenv("SER_FAST_MAX_TRANSIENT_RETRIES", "1")
    monkeypatch.setenv("SER_FAST_RETRY_BACKOFF_SECONDS", "0.2")
    monkeypatch.setenv("SER_FAST_PROCESS_ISOLATION", "true")
    monkeypatch.setenv("SER_MEDIUM_TIMEOUT_SECONDS", "45.5")
    monkeypatch.setenv("SER_MEDIUM_MAX_TIMEOUT_RETRIES", "3")
    monkeypatch.setenv("SER_MEDIUM_MAX_TRANSIENT_RETRIES", "4")
    monkeypatch.setenv("SER_MEDIUM_RETRY_BACKOFF_SECONDS", "0.6")
    monkeypatch.setenv("SER_MEDIUM_POOL_WINDOW_SIZE_SECONDS", "2.5")
    monkeypatch.setenv("SER_MEDIUM_POOL_WINDOW_STRIDE_SECONDS", "0.75")
    monkeypatch.setenv("SER_MEDIUM_POST_SMOOTHING_WINDOW_FRAMES", "5")
    monkeypatch.setenv("SER_MEDIUM_POST_HYSTERESIS_ENTER_CONFIDENCE", "0.72")
    monkeypatch.setenv("SER_MEDIUM_POST_HYSTERESIS_EXIT_CONFIDENCE", "0.41")
    monkeypatch.setenv("SER_MEDIUM_POST_MIN_SEGMENT_DURATION_SECONDS", "0.55")
    monkeypatch.setenv("SER_MEDIUM_PROCESS_ISOLATION", "false")
    monkeypatch.setenv("SER_ACCURATE_TIMEOUT_SECONDS", "80.0")
    monkeypatch.setenv("SER_ACCURATE_MAX_TIMEOUT_RETRIES", "1")
    monkeypatch.setenv("SER_ACCURATE_MAX_TRANSIENT_RETRIES", "2")
    monkeypatch.setenv("SER_ACCURATE_RETRY_BACKOFF_SECONDS", "0.4")
    monkeypatch.setenv("SER_ACCURATE_PROCESS_ISOLATION", "false")
    monkeypatch.setenv("SER_ACCURATE_RESEARCH_TIMEOUT_SECONDS", "70.0")
    monkeypatch.setenv("SER_ACCURATE_RESEARCH_MAX_TIMEOUT_RETRIES", "2")
    monkeypatch.setenv("SER_ACCURATE_RESEARCH_MAX_TRANSIENT_RETRIES", "3")
    monkeypatch.setenv("SER_ACCURATE_RESEARCH_RETRY_BACKOFF_SECONDS", "0.9")
    monkeypatch.setenv("SER_ACCURATE_RESEARCH_PROCESS_ISOLATION", "false")
    monkeypatch.setenv("SER_MEDIUM_MIN_WINDOW_STD", "0.12")
    monkeypatch.setenv("SER_MEDIUM_MAX_WINDOWS_PER_CLIP", "5")
    monkeypatch.setenv("SER_QUALITY_GATE_MIN_UAR_DELTA", "0.015")
    monkeypatch.setenv("SER_QUALITY_GATE_MIN_MACRO_F1_DELTA", "0.02")
    monkeypatch.setenv("SER_QUALITY_GATE_MAX_MEDIUM_SEGMENTS_PER_MINUTE", "22.5")
    monkeypatch.setenv(
        "SER_QUALITY_GATE_MIN_MEDIUM_MEDIAN_SEGMENT_DURATION_SECONDS",
        "2.2",
    )
    monkeypatch.setenv("SER_OUTPUT_SCHEMA_VERSION", "v2")
    monkeypatch.setenv("SER_ARTIFACT_SCHEMA_VERSION", "v3")
    monkeypatch.setenv("SER_TORCH_DEVICE", "cuda:0")
    monkeypatch.setenv("SER_TORCH_DTYPE", "bfloat16")
    monkeypatch.setenv("SER_TORCH_ENABLE_MPS_FALLBACK", "true")
    monkeypatch.setenv("SER_TRANSCRIPTION_MPS_LOW_MEMORY_GB", "24.0")
    monkeypatch.setenv("SER_TRANSCRIPTION_MPS_ADMISSION_CONTROL", "false")
    monkeypatch.setenv("SER_TRANSCRIPTION_MPS_HARD_OOM_SHORTCUT", "false")
    monkeypatch.setenv("SER_TRANSCRIPTION_MPS_MIN_HEADROOM_MB", "96")
    monkeypatch.setenv("SER_TRANSCRIPTION_MPS_SAFETY_MARGIN_MB", "48")
    monkeypatch.setenv("SER_TRANSCRIPTION_MPS_CALIBRATION_OVERRIDES", "false")
    monkeypatch.setenv("SER_TRANSCRIPTION_MPS_CALIBRATION_MIN_CONFIDENCE", "medium")
    monkeypatch.setenv(
        "SER_TRANSCRIPTION_MPS_CALIBRATION_REPORT_MAX_AGE_HOURS",
        "72",
    )
    monkeypatch.setenv(
        "SER_TRANSCRIPTION_MPS_CALIBRATION_REPORT_PATH",
        "unit-test/transcription_runtime_calibration_report.json",
    )
    monkeypatch.setenv(
        "SER_DATASET_MANIFESTS",
        "unit-test/manifest_a.jsonl,unit-test/manifest_b.jsonl",
    )
    monkeypatch.setenv("SER_MODEL_CACHE_DIR", "unit-test/model-cache")
    monkeypatch.setenv("SER_MEDIUM_MODEL_ID", "unit-test/xlsr")
    monkeypatch.setenv("SER_ACCURATE_MODEL_ID", "unit-test/whisper-tiny")
    monkeypatch.setenv("SER_ACCURATE_RESEARCH_MODEL_ID", "unit-test/emotion2vec-plus")
    monkeypatch.setenv("WHISPER_BACKEND", "faster_whisper")
    monkeypatch.setenv("WHISPER_MODEL", "base")
    monkeypatch.setenv("WHISPER_DEMUCS", "false")
    monkeypatch.setenv("WHISPER_VAD", "false")

    settings = config.reload_settings()

    assert settings.runtime_flags.profile_pipeline is True
    assert settings.runtime_flags.medium_profile is True
    assert settings.runtime_flags.accurate_profile is True
    assert settings.runtime_flags.accurate_research_profile is True
    assert settings.runtime_flags.restricted_backends is True
    assert settings.runtime_flags.new_output_schema is True
    assert settings.fast_runtime.timeout_seconds == pytest.approx(15.0)
    assert settings.fast_runtime.max_timeout_retries == 2
    assert settings.fast_runtime.max_transient_retries == 1
    assert settings.fast_runtime.retry_backoff_seconds == pytest.approx(0.2)
    assert settings.fast_runtime.process_isolation is True
    assert settings.medium_runtime.timeout_seconds == pytest.approx(45.5)
    assert settings.medium_runtime.max_timeout_retries == 3
    assert settings.medium_runtime.max_transient_retries == 4
    assert settings.medium_runtime.retry_backoff_seconds == pytest.approx(0.6)
    assert settings.medium_runtime.pool_window_size_seconds == pytest.approx(2.5)
    assert settings.medium_runtime.pool_window_stride_seconds == pytest.approx(0.75)
    assert settings.medium_runtime.post_smoothing_window_frames == 5
    assert settings.medium_runtime.post_hysteresis_enter_confidence == pytest.approx(
        0.72
    )
    assert settings.medium_runtime.post_hysteresis_exit_confidence == pytest.approx(
        0.41
    )
    assert settings.medium_runtime.post_min_segment_duration_seconds == pytest.approx(
        0.55
    )
    assert settings.medium_runtime.process_isolation is False
    assert settings.accurate_runtime.timeout_seconds == pytest.approx(80.0)
    assert settings.accurate_runtime.max_timeout_retries == 1
    assert settings.accurate_runtime.max_transient_retries == 2
    assert settings.accurate_runtime.retry_backoff_seconds == pytest.approx(0.4)
    assert settings.accurate_runtime.process_isolation is False
    assert settings.accurate_research_runtime.timeout_seconds == pytest.approx(70.0)
    assert settings.accurate_research_runtime.max_timeout_retries == 2
    assert settings.accurate_research_runtime.max_transient_retries == 3
    assert settings.accurate_research_runtime.retry_backoff_seconds == pytest.approx(
        0.9
    )
    assert settings.accurate_research_runtime.process_isolation is False
    assert settings.medium_training.min_window_std == pytest.approx(0.12)
    assert settings.medium_training.max_windows_per_clip == 5
    assert settings.quality_gate.min_uar_delta == pytest.approx(0.015)
    assert settings.quality_gate.min_macro_f1_delta == pytest.approx(0.02)
    assert settings.quality_gate.max_medium_segments_per_minute == pytest.approx(22.5)
    assert settings.quality_gate.min_medium_median_segment_duration_seconds == (
        pytest.approx(2.2)
    )
    assert settings.schema.output_schema_version == "v2"
    assert settings.schema.artifact_schema_version == "v3"
    assert settings.torch_runtime.device == "cuda:0"
    assert settings.torch_runtime.dtype == "bfloat16"
    assert settings.torch_runtime.enable_mps_fallback is True
    xlsr_override = settings.feature_runtime_policy.for_backend("hf_xlsr")
    assert xlsr_override is not None
    assert xlsr_override.device is None
    assert xlsr_override.dtype == "float32"
    assert settings.feature_runtime_policy.for_backend("hf_whisper") is None
    assert os.getenv("PYTORCH_ENABLE_MPS_FALLBACK") == "1"
    assert settings.dataset.manifest_paths == (
        Path("unit-test/manifest_a.jsonl"),
        Path("unit-test/manifest_b.jsonl"),
    )
    assert settings.models.model_cache_dir == Path("unit-test/model-cache")
    assert settings.models.huggingface_cache_root == Path(
        "unit-test/model-cache/huggingface"
    )
    assert settings.models.modelscope_cache_root == Path(
        "unit-test/model-cache/modelscope/hub"
    )
    assert settings.models.torch_cache_root == Path("unit-test/model-cache/torch")
    assert settings.models.medium_model_id == "unit-test/xlsr"
    assert settings.models.accurate_model_id == "unit-test/whisper-tiny"
    assert settings.models.accurate_research_model_id == "unit-test/emotion2vec-plus"
    assert settings.transcription.backend_id == "faster_whisper"
    assert settings.models.whisper_model.name == "base"
    assert settings.transcription.use_demucs is False
    assert settings.transcription.use_vad is False
    assert settings.transcription.mps_low_memory_threshold_gb == pytest.approx(24.0)
    assert settings.transcription.mps_admission_control_enabled is False
    assert settings.transcription.mps_hard_oom_shortcut_enabled is False
    assert settings.transcription.mps_admission_min_headroom_mb == pytest.approx(96.0)
    assert settings.transcription.mps_admission_safety_margin_mb == pytest.approx(48.0)
    assert settings.transcription.mps_admission_calibration_overrides_enabled is False
    assert settings.transcription.mps_admission_calibration_min_confidence == "medium"
    assert settings.transcription.mps_admission_calibration_report_max_age_hours == (
        pytest.approx(72.0)
    )
    assert settings.transcription.mps_admission_calibration_report_path == Path(
        "unit-test/transcription_runtime_calibration_report.json"
    )


@pytest.mark.parametrize(
    ("env", "expected_backend_id", "expected_model_name"),
    [
        ({"SER_ENABLE_MEDIUM_PROFILE": "true"}, "stable_whisper", "turbo"),
        ({"SER_ENABLE_ACCURATE_PROFILE": "true"}, "stable_whisper", "large"),
        (
            {
                "SER_ENABLE_ACCURATE_PROFILE": "true",
                "SER_ENABLE_ACCURATE_RESEARCH_PROFILE": "true",
            },
            "stable_whisper",
            "large",
        ),
    ],
)
def test_profile_selection_controls_transcription_defaults(
    monkeypatch: pytest.MonkeyPatch,
    env: dict[str, str],
    expected_backend_id: str,
    expected_model_name: str,
) -> None:
    """Selected runtime profile should drive default Whisper transcription model."""
    monkeypatch.delenv("WHISPER_MODEL", raising=False)
    monkeypatch.delenv("WHISPER_DEMUCS", raising=False)
    monkeypatch.delenv("WHISPER_VAD", raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)

    settings = config.reload_settings()

    assert settings.transcription.backend_id == expected_backend_id
    assert settings.models.whisper_model.name == expected_model_name
    assert settings.transcription.use_demucs is True
    assert settings.transcription.use_vad is True


def test_invalid_whisper_backend_env_falls_back_to_profile_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invalid backend override should not replace profile-defined backend id."""
    monkeypatch.setenv("WHISPER_BACKEND", "not-a-real-backend")

    settings = config.reload_settings()

    assert settings.transcription.backend_id == "faster_whisper"


def test_profile_default_artifact_names_include_backend_model_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-fast profile default filenames should be tuple-scoped by model id."""
    monkeypatch.delenv("SER_MODEL_FILE_NAME", raising=False)
    monkeypatch.delenv("SER_SECURE_MODEL_FILE_NAME", raising=False)
    monkeypatch.delenv("SER_TRAINING_REPORT_FILE_NAME", raising=False)
    monkeypatch.setenv("SER_ENABLE_PROFILE_PIPELINE", "true")
    monkeypatch.setenv("SER_ENABLE_ACCURATE_PROFILE", "true")
    monkeypatch.setenv("SER_ACCURATE_MODEL_ID", "unit-test/whisper-large")

    settings = config.reload_settings()
    expected = config.profile_artifact_file_names(
        profile="accurate",
        medium_model_id=settings.models.medium_model_id,
        accurate_model_id=settings.models.accurate_model_id,
        accurate_research_model_id=settings.models.accurate_research_model_id,
    )

    assert settings.models.model_file_name == expected[0]
    assert settings.models.secure_model_file_name == expected[1]
    assert settings.models.training_report_file_name == expected[2]


def test_runtime_timeout_zero_disables_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Zero timeout env overrides should be accepted to disable timeout budgets."""
    monkeypatch.setenv("SER_MEDIUM_TIMEOUT_SECONDS", "0")
    monkeypatch.setenv("SER_FAST_TIMEOUT_SECONDS", "0")
    monkeypatch.setenv("SER_ACCURATE_TIMEOUT_SECONDS", "0")
    monkeypatch.setenv("SER_ACCURATE_RESEARCH_TIMEOUT_SECONDS", "0")

    settings = config.reload_settings()

    assert settings.fast_runtime.timeout_seconds == pytest.approx(0.0)
    assert settings.medium_runtime.timeout_seconds == pytest.approx(0.0)
    assert settings.accurate_runtime.timeout_seconds == pytest.approx(0.0)
    assert settings.accurate_research_runtime.timeout_seconds == pytest.approx(0.0)


def test_invalid_torch_runtime_env_falls_back_to_auto(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invalid torch runtime selectors should fall back to conservative defaults."""
    monkeypatch.setenv("SER_TORCH_DEVICE", "quantum:9")
    monkeypatch.setenv("SER_TORCH_DTYPE", "float128")

    settings = config.reload_settings()

    assert settings.torch_runtime.device == "auto"
    assert settings.torch_runtime.dtype == "auto"


def test_torch_runtime_inherits_pytorch_mps_fallback_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Torch fallback setting should inherit from PYTORCH env when SER override is absent."""
    monkeypatch.delenv("SER_TORCH_ENABLE_MPS_FALLBACK", raising=False)
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    settings = config.reload_settings()

    assert settings.torch_runtime.enable_mps_fallback is True
    assert os.getenv("PYTORCH_ENABLE_MPS_FALLBACK") == "1"


def test_apply_settings_syncs_pytorch_mps_fallback_env() -> None:
    """Applying a settings snapshot should synchronize torch fallback environment."""
    base_settings = config.reload_settings()
    enabled_settings = replace(
        base_settings,
        torch_runtime=replace(
            base_settings.torch_runtime,
            enable_mps_fallback=True,
        ),
    )

    config.apply_settings(enabled_settings)

    assert config.get_settings().torch_runtime.enable_mps_fallback is True
    assert os.getenv("PYTORCH_ENABLE_MPS_FALLBACK") == "1"


def test_invalid_transcription_mps_threshold_env_falls_back_to_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invalid or non-positive threshold should fall back to default value."""
    monkeypatch.setenv("SER_TRANSCRIPTION_MPS_LOW_MEMORY_GB", "-2")
    monkeypatch.setenv("SER_TRANSCRIPTION_MPS_MIN_HEADROOM_MB", "-10")
    monkeypatch.setenv("SER_TRANSCRIPTION_MPS_SAFETY_MARGIN_MB", "-20")

    settings = config.reload_settings()

    assert settings.transcription.mps_low_memory_threshold_gb == pytest.approx(16.0)
    assert settings.transcription.mps_admission_min_headroom_mb == pytest.approx(64.0)
    assert settings.transcription.mps_admission_safety_margin_mb == pytest.approx(64.0)
