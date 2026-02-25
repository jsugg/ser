"""Tests for runtime feature-flag and schema version configuration."""

from collections.abc import Generator
from pathlib import Path

import pytest

import ser.config as config


@pytest.fixture(autouse=True)
def _reset_settings() -> Generator[None, None, None]:
    """Keeps global settings stable across tests."""
    config.reload_settings()
    yield
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
    assert settings.models.medium_model_id == "facebook/wav2vec2-xls-r-300m"
    assert settings.models.accurate_model_id == "openai/whisper-large-v3"
    assert settings.models.accurate_research_model_id == "iic/emotion2vec_plus_large"
    assert settings.models.whisper_model.name == "turbo"
    assert settings.transcription.use_demucs is True
    assert settings.transcription.use_vad is True
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
    monkeypatch.setenv("SER_MODEL_CACHE_DIR", "unit-test/model-cache")
    monkeypatch.setenv("SER_MEDIUM_MODEL_ID", "unit-test/xlsr")
    monkeypatch.setenv("SER_ACCURATE_MODEL_ID", "unit-test/whisper-tiny")
    monkeypatch.setenv("SER_ACCURATE_RESEARCH_MODEL_ID", "unit-test/emotion2vec-plus")
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
    assert settings.models.whisper_model.name == "base"
    assert settings.transcription.use_demucs is False
    assert settings.transcription.use_vad is False


@pytest.mark.parametrize(
    ("env", "expected_model_name"),
    [
        ({"SER_ENABLE_MEDIUM_PROFILE": "true"}, "turbo"),
        ({"SER_ENABLE_ACCURATE_PROFILE": "true"}, "large"),
        (
            {
                "SER_ENABLE_ACCURATE_PROFILE": "true",
                "SER_ENABLE_ACCURATE_RESEARCH_PROFILE": "true",
            },
            "large",
        ),
    ],
)
def test_profile_selection_controls_transcription_defaults(
    monkeypatch: pytest.MonkeyPatch,
    env: dict[str, str],
    expected_model_name: str,
) -> None:
    """Selected runtime profile should drive default Whisper transcription model."""
    monkeypatch.delenv("WHISPER_MODEL", raising=False)
    monkeypatch.delenv("WHISPER_DEMUCS", raising=False)
    monkeypatch.delenv("WHISPER_VAD", raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)

    settings = config.reload_settings()

    assert settings.models.whisper_model.name == expected_model_name
    assert settings.transcription.use_demucs is True
    assert settings.transcription.use_vad is True


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
