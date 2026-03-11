"""Tests for typed configuration loading and environment refresh."""

from collections.abc import Generator
from pathlib import Path

import pytest

import ser.config as config
from ser._internal.config import bootstrap
from ser._internal.config.settings_inputs import ResolvedSettingsInputs


@pytest.fixture(autouse=True)
def _reset_settings() -> Generator[None, None, None]:
    """Keeps global settings stable across tests."""
    config.reload_settings()
    yield
    config.reload_settings()


def test_reload_settings_reads_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Environment variables should be reflected in loaded settings."""
    monkeypatch.setenv("DATASET_FOLDER", "custom/dataset")
    monkeypatch.setenv("DEFAULT_LANGUAGE", "es")
    monkeypatch.setenv("SER_CACHE_DIR", "custom/cache")
    monkeypatch.setenv("SER_DATA_DIR", "custom/data")
    monkeypatch.setenv("SER_MODEL_CACHE_DIR", "custom/model-cache")
    monkeypatch.setenv("SER_TMP_DIR", "custom/tmp")
    monkeypatch.setenv("SER_MODELS_DIR", "custom/models")
    monkeypatch.setenv("SER_TRANSCRIPTS_DIR", "custom/transcripts")
    monkeypatch.setenv("SER_MODEL_FILE_NAME", "custom_model.pkl")
    monkeypatch.setenv("SER_SECURE_MODEL_FILE_NAME", "custom_model.skops")
    monkeypatch.setenv("SER_TRAINING_REPORT_FILE_NAME", "custom_training_report.json")
    monkeypatch.setenv("SER_MAX_WORKERS", "6")
    monkeypatch.setenv("SER_MAX_FAILED_FILE_RATIO", "0.45")
    monkeypatch.setenv("SER_TEST_SIZE", "0.3")
    monkeypatch.setenv("SER_RANDOM_STATE", "7")
    monkeypatch.setenv("WHISPER_MODEL", "base")
    monkeypatch.setenv("WHISPER_DEMUCS", "false")
    monkeypatch.setenv("WHISPER_VAD", "true")
    monkeypatch.setattr(bootstrap.os, "cpu_count", lambda: 4)

    settings = config.reload_settings()

    assert settings.dataset.folder == Path("custom/dataset")
    assert settings.default_language == "es"
    assert settings.models.num_cores == 4
    assert settings.tmp_folder == Path("custom/tmp")
    assert settings.models.folder == Path("custom/models")
    assert settings.models.model_cache_dir == Path("custom/model-cache")
    assert settings.models.huggingface_cache_root == Path("custom/model-cache/huggingface")
    assert settings.models.modelscope_cache_root == Path("custom/model-cache/modelscope/hub")
    assert settings.models.whisper_download_root == Path("custom/model-cache/OpenAI/whisper")
    assert settings.models.torch_cache_root == Path("custom/model-cache/torch")
    assert settings.timeline.folder == Path("custom/transcripts")
    assert settings.models.model_file == Path("custom/models/custom_model.pkl")
    assert settings.models.secure_model_file == Path("custom/models/custom_model.skops")
    assert settings.models.training_report_file == Path("custom/models/custom_training_report.json")
    assert settings.models.whisper_model.name == "base"
    assert settings.data_loader.max_workers == 6
    assert settings.data_loader.max_failed_file_ratio == pytest.approx(0.45)
    assert settings.training.test_size == pytest.approx(0.3)
    assert settings.training.random_state == 7
    assert settings.nn.random_state == 7
    assert settings.transcription.use_demucs is False
    assert settings.transcription.use_vad is True


def test_reload_settings_uses_cpu_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """When cpu_count is unavailable, configuration should default to one core."""
    monkeypatch.setattr(bootstrap.os, "cpu_count", lambda: None)

    settings = config.reload_settings()

    assert settings.models.num_cores == 1


def test_bootstrap_build_settings_delegates_to_internal_settings_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bootstrap should keep a single delegation path for settings construction."""
    expected = config.reload_settings()
    call_count = 0

    def _fake_build_settings_from_inputs(
        inputs: ResolvedSettingsInputs,
    ) -> config.AppConfig:
        nonlocal call_count
        assert inputs.default_language
        call_count += 1
        return expected

    monkeypatch.setattr(
        "ser._internal.config.settings_builder.build_settings_from_inputs",
        _fake_build_settings_from_inputs,
    )

    assert bootstrap._build_settings() is expected
    assert call_count == 1


def test_bootstrap_resolve_settings_inputs_delegates_to_internal_resolver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bootstrap should keep a single delegation path for settings input resolution."""
    expected = bootstrap._resolve_settings_inputs()
    call_count = 0

    def _fake_resolve_settings_inputs_from_internal(
        deps: object,
    ) -> ResolvedSettingsInputs:
        nonlocal call_count
        assert callable(getattr(deps, "resolve_profile_model_id", None))
        assert callable(getattr(deps, "get_profile_catalog", None))
        call_count += 1
        return expected

    monkeypatch.setattr(
        bootstrap,
        "_resolve_settings_inputs_from_internal",
        _fake_resolve_settings_inputs_from_internal,
    )

    assert bootstrap._resolve_settings_inputs() is expected
    assert call_count == 1


def test_public_config_facade_does_not_expose_private_builder_helpers() -> None:
    """Public config facade should not expose private bootstrap builder helpers."""
    for private_name in (
        "_build_settings",
        "_resolve_settings_inputs",
        "_resolve_settings_inputs_from_internal",
        "os",
        "sys",
    ):
        assert not hasattr(config, private_name)
