"""Behavior tests for CLI argument dispatch and exit semantics."""

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

import ser.__main__ as cli
import ser.config as config_module
import ser.models.emotion_model as emotion_model
from ser.runtime import InferenceRequest
from ser.runtime.accurate_inference import AccurateRuntimeDependencyError
from ser.runtime.medium_inference import MediumRuntimeDependencyError
from ser.runtime.registry import UnsupportedProfileError


def _patch_common_cli_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patches shared CLI dependencies to keep tests deterministic."""
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    monkeypatch.setattr(
        cli,
        "reload_settings",
        lambda: SimpleNamespace(default_language="en"),
    )


def test_cli_log_level_flag_overrides_environment_level(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`--log-level` should override LOG_LEVEL for the command invocation."""
    _patch_common_cli_dependencies(monkeypatch)
    monkeypatch.setenv("LOG_LEVEL", "ERROR")
    monkeypatch.setattr(
        cli.sys,
        "argv",
        ["ser", "--file", "sample.wav", "--log-level", "DEBUG"],
    )
    configured_levels: list[str | int | None] = []

    def _capture_log_level(level: str | int | None = None) -> int:
        configured_levels.append(level)
        return 0

    class FakePipeline:
        def run_inference(self, _request: object) -> object:
            return SimpleNamespace(timeline_csv_path=None)

    monkeypatch.setattr(cli, "configure_logging", _capture_log_level)
    monkeypatch.setattr(
        cli, "_build_runtime_pipeline", lambda _settings: FakePipeline()
    )

    cli.main()

    assert configured_levels[-1] == "DEBUG"


def test_cli_exits_with_error_when_file_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CLI should return exit code 1 when no prediction file is provided."""
    _patch_common_cli_dependencies(monkeypatch)
    monkeypatch.setattr(cli.sys, "argv", ["ser"])

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 1


def test_cli_calibration_requires_file_argument(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Calibration mode should fail fast when `--file` is missing."""
    _patch_common_cli_dependencies(monkeypatch)
    monkeypatch.setattr(
        cli.sys,
        "argv",
        ["ser", "--calibrate-transcription-runtime"],
    )

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 2


def test_cli_calibration_dispatches_runtime_calibration_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Calibration mode should call profiling runtime calibration and exit zero."""
    _patch_common_cli_dependencies(monkeypatch)
    monkeypatch.setattr(
        cli.sys,
        "argv",
        [
            "ser",
            "--calibrate-transcription-runtime",
            "--file",
            "sample.wav",
            "--calibration-iterations",
            "3",
            "--calibration-profiles",
            "medium,accurate",
        ],
    )
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "ser.transcript.profiling.parse_calibration_profiles",
        lambda raw: (
            captured.setdefault("profiles_raw", raw),
            ("medium", "accurate"),
        )[1],
    )
    monkeypatch.setattr(
        "ser.transcript.profiling.run_transcription_runtime_calibration",
        lambda **kwargs: (
            captured.setdefault("kwargs", kwargs),
            SimpleNamespace(
                report_path=Path("runtime_calibration.json"),
                recommendations=(),
            ),
        )[1],
    )

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 0
    assert captured["profiles_raw"] == "medium,accurate"
    kwargs = cast(dict[str, object], captured["kwargs"])
    assert kwargs["calibration_file"] == Path("sample.wav")
    assert kwargs["iterations_per_profile"] == 3
    assert kwargs["profile_names"] == ("medium", "accurate")


def test_cli_train_option_invokes_training_and_exits_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`--train` should dispatch to training workflow and exit successfully."""
    _patch_common_cli_dependencies(monkeypatch)
    monkeypatch.setattr(cli.sys, "argv", ["ser", "--train"])

    called = {"train": False}

    def fake_train_model() -> None:
        called["train"] = True

    monkeypatch.setattr(emotion_model, "train_model", fake_train_model)

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 0
    assert called["train"] is True


def test_cli_prediction_passes_language_and_saves_transcript(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prediction path should pass language and transcript-save flags to pipeline."""
    _patch_common_cli_dependencies(monkeypatch)
    monkeypatch.setattr(
        cli.sys,
        "argv",
        ["ser", "--file", "sample.wav", "--language", "es", "--save_transcript"],
    )

    calls: dict[str, object] = {}

    class FakePipeline:
        def run_inference(self, request: object) -> object:
            calls["request"] = request
            return SimpleNamespace(timeline_csv_path="out.csv")

    monkeypatch.setattr(
        cli, "_build_runtime_pipeline", lambda _settings: FakePipeline()
    )

    cli.main()

    request = cast(InferenceRequest, calls["request"])
    assert request.file_path == "sample.wav"
    assert request.language == "es"
    assert request.save_transcript is True
    assert request.include_transcript is True


def test_cli_prediction_with_no_transcript_disables_transcript_in_pipeline_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prediction path should disable transcript extraction when --no-transcript is set."""
    _patch_common_cli_dependencies(monkeypatch)
    monkeypatch.setattr(
        cli.sys, "argv", ["ser", "--file", "sample.wav", "--no-transcript"]
    )

    calls: dict[str, object] = {}

    class FakePipeline:
        def run_inference(self, request: object) -> object:
            calls["request"] = request
            return SimpleNamespace(timeline_csv_path=None)

    monkeypatch.setattr(
        cli, "_build_runtime_pipeline", lambda _settings: FakePipeline()
    )

    cli.main()

    request = cast(InferenceRequest, calls["request"])
    assert request.file_path == "sample.wav"
    assert request.save_transcript is False
    assert request.include_transcript is False


def test_cli_prediction_does_not_save_when_flag_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prediction path should pass save_transcript=False when flag is absent."""
    _patch_common_cli_dependencies(monkeypatch)
    monkeypatch.setattr(cli.sys, "argv", ["ser", "--file", "sample.wav"])
    calls: dict[str, object] = {}

    class FakePipeline:
        def run_inference(self, request: object) -> object:
            calls["request"] = request
            return SimpleNamespace(timeline_csv_path=None)

    monkeypatch.setattr(
        cli, "_build_runtime_pipeline", lambda _settings: FakePipeline()
    )

    cli.main()

    request = cast(InferenceRequest, calls["request"])
    assert request.file_path == "sample.wav"
    assert request.save_transcript is False
    assert request.include_transcript is True


def test_cli_train_option_uses_runtime_pipeline_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`--train` should dispatch through runtime pipeline when flag is enabled."""
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    monkeypatch.setattr(
        cli,
        "reload_settings",
        lambda: SimpleNamespace(
            default_language="en",
            runtime_flags=SimpleNamespace(profile_pipeline=True),
        ),
    )
    monkeypatch.setattr(cli.sys, "argv", ["ser", "--train"])

    called = {"train": False}

    class FakePipeline:
        def run_training(self) -> None:
            called["train"] = True

    monkeypatch.setattr(
        cli, "_build_runtime_pipeline", lambda _settings: FakePipeline()
    )

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 0
    assert called["train"] is True


def test_cli_prediction_uses_runtime_pipeline_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prediction should route through runtime pipeline when flag is enabled."""
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    monkeypatch.setattr(
        cli,
        "reload_settings",
        lambda: SimpleNamespace(
            default_language="en",
            runtime_flags=SimpleNamespace(profile_pipeline=True),
        ),
    )
    monkeypatch.setattr(
        cli.sys,
        "argv",
        ["ser", "--file", "sample.wav", "--language", "pt", "--save_transcript"],
    )

    calls: dict[str, object] = {}

    class FakePipeline:
        def run_training(self) -> None:
            raise AssertionError("Training path should not run for prediction command.")

        def run_inference(self, request: object) -> object:
            calls["request"] = request
            return SimpleNamespace(timeline_csv_path="timeline.csv")

    monkeypatch.setattr(
        cli, "_build_runtime_pipeline", lambda _settings: FakePipeline()
    )

    cli.main()

    request = cast(InferenceRequest, calls["request"])
    assert request.file_path == "sample.wav"
    assert request.language == "pt"
    assert request.save_transcript is True
    assert request.include_transcript is True


def test_cli_prediction_pipeline_honors_no_transcript_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pipeline request should disable transcript extraction when --no-transcript is set."""
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    monkeypatch.setattr(
        cli,
        "reload_settings",
        lambda: SimpleNamespace(
            default_language="en",
            runtime_flags=SimpleNamespace(profile_pipeline=True),
        ),
    )
    monkeypatch.setattr(
        cli.sys,
        "argv",
        ["ser", "--file", "sample.wav", "--language", "pt", "--no-transcript"],
    )

    calls: dict[str, object] = {}

    class FakePipeline:
        def run_training(self) -> None:
            raise AssertionError("Training path should not run for prediction command.")

        def run_inference(self, request: object) -> object:
            calls["request"] = request
            return SimpleNamespace(timeline_csv_path=None)

    monkeypatch.setattr(
        cli, "_build_runtime_pipeline", lambda _settings: FakePipeline()
    )

    cli.main()

    request = cast(InferenceRequest, calls["request"])
    assert request.file_path == "sample.wav"
    assert request.language == "pt"
    assert request.save_transcript is False
    assert request.include_transcript is False


def test_cli_prediction_pipeline_unsupported_profile_exits_two(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pipeline capability failures should map to exit code 2."""
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    monkeypatch.setattr(
        cli,
        "reload_settings",
        lambda: SimpleNamespace(
            default_language="en",
            runtime_flags=SimpleNamespace(profile_pipeline=True),
        ),
    )
    monkeypatch.setattr(cli.sys, "argv", ["ser", "--file", "sample.wav"])

    class FakePipeline:
        def run_training(self) -> None:
            raise AssertionError("Training path should not run for prediction command.")

        def run_inference(self, _request: object) -> object:
            raise UnsupportedProfileError(
                "Runtime profile 'medium' is not implemented."
            )

    monkeypatch.setattr(
        cli, "_build_runtime_pipeline", lambda _settings: FakePipeline()
    )

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 2


def test_cli_prediction_pipeline_medium_dependency_error_exits_two(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Medium dependency/runtime readiness errors should map to exit code 2."""
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    monkeypatch.setattr(
        cli,
        "reload_settings",
        lambda: SimpleNamespace(
            default_language="en",
            runtime_flags=SimpleNamespace(profile_pipeline=True),
        ),
    )
    monkeypatch.setattr(cli.sys, "argv", ["ser", "--file", "sample.wav"])

    class FakePipeline:
        def run_training(self) -> None:
            raise AssertionError("Training path should not run for prediction command.")

        def run_inference(self, _request: object) -> object:
            raise MediumRuntimeDependencyError(
                "Runtime profile 'medium' requires missing dependency: transformers."
            )

    monkeypatch.setattr(
        cli, "_build_runtime_pipeline", lambda _settings: FakePipeline()
    )

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 2


def test_cli_prediction_pipeline_accurate_dependency_error_exits_two(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Accurate dependency/runtime readiness errors should map to exit code 2."""
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    monkeypatch.setattr(
        cli,
        "reload_settings",
        lambda: SimpleNamespace(
            default_language="en",
            runtime_flags=SimpleNamespace(profile_pipeline=True),
        ),
    )
    monkeypatch.setattr(cli.sys, "argv", ["ser", "--file", "sample.wav"])

    class FakePipeline:
        def run_training(self) -> None:
            raise AssertionError("Training path should not run for prediction command.")

        def run_inference(self, _request: object) -> object:
            raise AccurateRuntimeDependencyError(
                "Runtime profile 'accurate' requires missing dependency: transformers."
            )

    monkeypatch.setattr(
        cli, "_build_runtime_pipeline", lambda _settings: FakePipeline()
    )

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 2


def test_cli_profile_option_enables_pipeline_and_overrides_runtime_flags_for_training(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`--profile` should enable pipeline mode and override runtime flags."""
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    monkeypatch.setattr(
        cli,
        "reload_settings",
        lambda: SimpleNamespace(
            default_language="en",
            runtime_flags=SimpleNamespace(
                profile_pipeline=False,
                medium_profile=False,
                accurate_profile=False,
                accurate_research_profile=False,
                restricted_backends=False,
            ),
        ),
    )
    monkeypatch.setattr(
        cli.sys,
        "argv",
        ["ser", "--train", "--profile", "accurate-research"],
    )

    captured: dict[str, object] = {"called": False}

    class FakePipeline:
        def run_training(self) -> None:
            captured["called"] = True

    def _build_pipeline(settings: object) -> FakePipeline:
        runtime_flags = cast(
            SimpleNamespace, cast(SimpleNamespace, settings).runtime_flags
        )
        captured["profile_pipeline"] = runtime_flags.profile_pipeline
        captured["medium_profile"] = runtime_flags.medium_profile
        captured["accurate_profile"] = runtime_flags.accurate_profile
        captured["accurate_research_profile"] = runtime_flags.accurate_research_profile
        captured["restricted_backends"] = runtime_flags.restricted_backends
        return FakePipeline()

    monkeypatch.setattr(cli, "_build_runtime_pipeline", _build_pipeline)

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 0
    assert captured["called"] is True
    assert captured["profile_pipeline"] is True
    assert captured["medium_profile"] is False
    assert captured["accurate_profile"] is False
    assert captured["accurate_research_profile"] is True
    assert captured["restricted_backends"] is False


def test_cli_profile_option_routes_prediction_with_selected_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prediction should route through pipeline when `--profile` is provided."""
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    monkeypatch.setattr(
        cli,
        "reload_settings",
        lambda: SimpleNamespace(
            default_language="en",
            runtime_flags=SimpleNamespace(
                profile_pipeline=False,
                medium_profile=False,
                accurate_profile=False,
                accurate_research_profile=False,
                restricted_backends=False,
            ),
        ),
    )
    monkeypatch.setattr(
        cli.sys,
        "argv",
        ["ser", "--file", "sample.wav", "--profile", "medium"],
    )

    captured: dict[str, object] = {}

    class FakePipeline:
        def run_training(self) -> None:
            raise AssertionError("Training path should not run for prediction command.")

        def run_inference(self, request: object) -> object:
            captured["request"] = request
            return SimpleNamespace(timeline_csv_path=None)

    def _build_pipeline(settings: object) -> FakePipeline:
        runtime_flags = cast(
            SimpleNamespace, cast(SimpleNamespace, settings).runtime_flags
        )
        captured["profile_pipeline"] = runtime_flags.profile_pipeline
        captured["medium_profile"] = runtime_flags.medium_profile
        captured["accurate_profile"] = runtime_flags.accurate_profile
        captured["accurate_research_profile"] = runtime_flags.accurate_research_profile
        return FakePipeline()

    monkeypatch.setattr(cli, "_build_runtime_pipeline", _build_pipeline)

    cli.main()

    request = cast(InferenceRequest, captured["request"])
    assert request.file_path == "sample.wav"
    assert request.language == "en"
    assert request.save_transcript is False
    assert request.include_transcript is True
    assert captured["profile_pipeline"] is True
    assert captured["medium_profile"] is True
    assert captured["accurate_profile"] is False
    assert captured["accurate_research_profile"] is False


def test_cli_profile_override_sets_profile_specific_artifact_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`--profile` should select profile/model-id specific artifact names."""
    monkeypatch.delenv("SER_MODEL_FILE_NAME", raising=False)
    monkeypatch.delenv("SER_SECURE_MODEL_FILE_NAME", raising=False)
    monkeypatch.delenv("SER_TRAINING_REPORT_FILE_NAME", raising=False)
    settings = config_module.reload_settings()
    expected_names = config_module.profile_artifact_file_names(
        profile="medium",
        medium_model_id=settings.models.medium_model_id,
        accurate_model_id=settings.models.accurate_model_id,
        accurate_research_model_id=settings.models.accurate_research_model_id,
    )

    resolved = cast(
        config_module.AppConfig,
        cli._apply_cli_profile_override(settings, "medium"),
    )

    assert resolved.models.model_file_name == expected_names[0]
    assert resolved.models.secure_model_file_name == expected_names[1]
    assert resolved.models.training_report_file_name == expected_names[2]


def test_cli_profile_override_preserves_explicit_artifact_env_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit artifact environment overrides must win over profile defaults."""
    monkeypatch.setenv("SER_MODEL_FILE_NAME", "custom_model.pkl")
    monkeypatch.setenv("SER_SECURE_MODEL_FILE_NAME", "custom_model.skops")
    monkeypatch.setenv("SER_TRAINING_REPORT_FILE_NAME", "custom_report.json")
    settings = config_module.reload_settings()

    resolved = cast(
        config_module.AppConfig,
        cli._apply_cli_profile_override(settings, "accurate"),
    )

    assert resolved.models.model_file_name == "custom_model.pkl"
    assert resolved.models.secure_model_file_name == "custom_model.skops"
    assert resolved.models.training_report_file_name == "custom_report.json"


def test_cli_profile_override_accurate_name_tracks_backend_model_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Accurate defaults should include backend model id in artifact filenames."""
    monkeypatch.delenv("SER_MODEL_FILE_NAME", raising=False)
    monkeypatch.delenv("SER_SECURE_MODEL_FILE_NAME", raising=False)
    monkeypatch.delenv("SER_TRAINING_REPORT_FILE_NAME", raising=False)
    monkeypatch.setenv("SER_ACCURATE_MODEL_ID", "unit-test/whisper-large")
    settings_a = config_module.reload_settings()
    resolved_a = cast(
        config_module.AppConfig,
        cli._apply_cli_profile_override(settings_a, "accurate"),
    )

    monkeypatch.setenv("SER_ACCURATE_MODEL_ID", "unit-test/whisper-tiny")
    settings_b = config_module.reload_settings()
    resolved_b = cast(
        config_module.AppConfig,
        cli._apply_cli_profile_override(settings_b, "accurate"),
    )

    assert resolved_a.models.model_file_name != resolved_b.models.model_file_name
    assert (
        resolved_a.models.secure_model_file_name
        != resolved_b.models.secure_model_file_name
    )
    assert (
        resolved_a.models.training_report_file_name
        != resolved_b.models.training_report_file_name
    )


def test_cli_profile_override_updates_transcription_defaults_by_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CLI profile override should switch Whisper transcription defaults."""
    monkeypatch.delenv("WHISPER_MODEL", raising=False)
    monkeypatch.delenv("WHISPER_DEMUCS", raising=False)
    monkeypatch.delenv("WHISPER_VAD", raising=False)
    settings = config_module.reload_settings()

    resolved_fast = cast(
        config_module.AppConfig,
        cli._apply_cli_profile_override(settings, "fast"),
    )
    resolved_medium = cast(
        config_module.AppConfig,
        cli._apply_cli_profile_override(settings, "medium"),
    )
    resolved_accurate = cast(
        config_module.AppConfig,
        cli._apply_cli_profile_override(settings, "accurate"),
    )
    resolved_accurate_research = cast(
        config_module.AppConfig,
        cli._apply_cli_profile_override(settings, "accurate-research"),
    )

    assert resolved_fast.transcription.backend_id == "faster_whisper"
    assert resolved_fast.models.whisper_model.name == "distil-large-v3"
    assert resolved_fast.transcription.use_demucs is False
    assert resolved_fast.transcription.use_vad is True
    assert resolved_medium.transcription.backend_id == "stable_whisper"
    assert resolved_medium.models.whisper_model.name == "turbo"
    assert resolved_medium.transcription.use_demucs is True
    assert resolved_medium.transcription.use_vad is True
    assert resolved_accurate.transcription.backend_id == "stable_whisper"
    assert resolved_accurate.models.whisper_model.name == "large"
    assert resolved_accurate_research.transcription.backend_id == "stable_whisper"
    assert resolved_accurate_research.models.whisper_model.name == "large"


def test_cli_timeout_override_sets_all_profile_timeouts_to_zero() -> None:
    """Timeout override should force all profile timeout budgets to zero."""
    settings = config_module.reload_settings()

    resolved = cast(
        config_module.AppConfig,
        cli._apply_cli_timeout_override(settings, disable_timeouts=True),
    )

    assert resolved.fast_runtime.timeout_seconds == 0.0
    assert resolved.medium_runtime.timeout_seconds == 0.0
    assert resolved.accurate_runtime.timeout_seconds == 0.0
    assert resolved.accurate_research_runtime.timeout_seconds == 0.0


def test_cli_disable_timeouts_flag_applies_to_selected_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`--disable-timeouts` should override selected profile timeout budget."""
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    settings = config_module.reload_settings()
    monkeypatch.setattr(cli, "reload_settings", lambda: settings)
    monkeypatch.setattr(
        cli.sys,
        "argv",
        ["ser", "--file", "sample.wav", "--profile", "accurate", "--disable-timeouts"],
    )

    captured: dict[str, object] = {}

    class FakePipeline:
        def run_training(self) -> None:
            raise AssertionError("Training path should not run for prediction command.")

        def run_inference(self, request: object) -> object:
            captured["request"] = request
            return SimpleNamespace(timeline_csv_path=None)

    def _build_pipeline(runtime_settings: object) -> FakePipeline:
        app_settings = cast(config_module.AppConfig, runtime_settings)
        captured["timeout_seconds"] = app_settings.accurate_runtime.timeout_seconds
        return FakePipeline()

    monkeypatch.setattr(cli, "_build_runtime_pipeline", _build_pipeline)

    cli.main()

    assert captured["timeout_seconds"] == 0.0


def test_cli_accept_restricted_backends_persists_active_profile_consent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """CLI opt-in flag should persist consent for active restricted profile backend."""
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    monkeypatch.setenv(
        "SER_RESTRICTED_BACKENDS_CONSENT_FILE",
        str(tmp_path / "restricted-backend-consent.json"),
    )
    monkeypatch.delenv("SER_ALLOWED_RESTRICTED_BACKENDS", raising=False)
    monkeypatch.delenv("SER_ENABLE_RESTRICTED_BACKENDS", raising=False)
    settings = config_module.reload_settings()
    monkeypatch.setattr(cli, "reload_settings", lambda: settings)
    monkeypatch.setattr(
        cli.sys,
        "argv",
        [
            "ser",
            "--profile",
            "accurate-research",
            "--accept-restricted-backends",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 0
    persisted = cli.load_persisted_backend_consents(settings=settings)
    assert "emotion2vec" in persisted
    assert persisted["emotion2vec"].consent_source == "cli_flag_accept_restricted"


def test_cli_accept_all_restricted_backends_persists_all_and_exits_zero(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Bulk opt-in flag should persist consent for all known restricted backends."""
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    monkeypatch.setenv(
        "SER_RESTRICTED_BACKENDS_CONSENT_FILE",
        str(tmp_path / "restricted-backend-consent.json"),
    )
    settings = config_module.reload_settings()
    monkeypatch.setattr(cli, "reload_settings", lambda: settings)
    monkeypatch.setattr(cli.sys, "argv", ["ser", "--accept-all-restricted-backends"])

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 0
    persisted = cli.load_persisted_backend_consents(settings=settings)
    assert "emotion2vec" in persisted
    assert persisted["emotion2vec"].consent_source == "cli_flag_accept_all"


def test_cli_interactive_restricted_backend_prompt_persists_consent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Interactive prompt should persist consent and continue command execution."""
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    monkeypatch.setenv(
        "SER_RESTRICTED_BACKENDS_CONSENT_FILE",
        str(tmp_path / "restricted-backend-consent.json"),
    )
    monkeypatch.delenv("SER_ALLOWED_RESTRICTED_BACKENDS", raising=False)
    monkeypatch.delenv("SER_ENABLE_RESTRICTED_BACKENDS", raising=False)
    settings = config_module.reload_settings()
    monkeypatch.setattr(cli, "reload_settings", lambda: settings)
    monkeypatch.setattr(
        cli.sys,
        "argv",
        ["ser", "--file", "sample.wav", "--profile", "accurate-research"],
    )
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(cli.sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _prompt="": "y")

    class FakePipeline:
        def run_training(self) -> None:
            raise AssertionError("Training path should not run for prediction command.")

        def run_inference(self, _request: object) -> object:
            return SimpleNamespace(timeline_csv_path=None)

    monkeypatch.setattr(
        cli, "_build_runtime_pipeline", lambda _settings: FakePipeline()
    )

    cli.main()

    persisted = cli.load_persisted_backend_consents(settings=settings)
    assert "emotion2vec" in persisted
    assert persisted["emotion2vec"].consent_source == "interactive_prompt"


def test_cli_dispatches_configure_subcommand_with_global_log_level(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dataset configure subcommand should dispatch after pre-arg parsing."""
    _patch_common_cli_dependencies(monkeypatch)
    monkeypatch.setattr(
        cli.sys,
        "argv",
        ["ser", "--log-level", "DEBUG", "configure", "--show"],
    )
    captured: dict[str, object] = {}

    def _run_configure_command(argv: list[str]) -> int:
        captured["argv"] = argv
        return 0

    monkeypatch.setattr(
        "ser.data.cli.run_configure_command",
        _run_configure_command,
    )

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 0
    assert captured["argv"] == ["--show"]


def test_cli_dispatches_data_subcommand_after_global_log_level(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dataset data subcommand should dispatch when global flags appear first."""
    _patch_common_cli_dependencies(monkeypatch)
    monkeypatch.setattr(
        cli.sys,
        "argv",
        ["ser", "--log-level", "INFO", "data", "download", "--dataset", "ravdess"],
    )
    captured: dict[str, object] = {}

    def _run_data_command(argv: list[str]) -> int:
        captured["argv"] = argv
        return 0

    monkeypatch.setattr(
        "ser.data.cli.run_data_command",
        _run_data_command,
    )

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 0
    assert captured["argv"] == ["download", "--dataset", "ravdess"]


def test_cli_dataset_workflow_configure_download_and_registry_load(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """End-to-end dataset workflow should persist consent and load registry manifests."""
    import ser.data.cli as data_cli_module
    import ser.data.data_loader as data_loader_module
    from ser.data.dataset_consents import load_persisted_dataset_consents
    from ser.data.dataset_registry import load_dataset_registry

    dataset_root = tmp_path / "datasets" / "ravdess"
    (dataset_root / "Actor_01").mkdir(parents=True, exist_ok=True)
    (dataset_root / "Actor_02").mkdir(parents=True, exist_ok=True)
    (dataset_root / "Actor_01" / "03-01-03-01-01-01-01.wav").write_bytes(b"")
    (dataset_root / "Actor_02" / "03-01-04-01-01-01-02.wav").write_bytes(b"")
    manifest_path = tmp_path / "manifests" / "ravdess.jsonl"

    settings = cast(
        config_module.AppConfig,
        SimpleNamespace(
            default_language="en",
            emotions={"03": "happy", "04": "sad"},
            dataset=SimpleNamespace(
                folder=dataset_root,
                subfolder_prefix="Actor_*",
                extension="*.wav",
                manifest_paths=(),
                glob_pattern=str(dataset_root / "Actor_*" / "*.wav"),
            ),
            models=SimpleNamespace(
                folder=tmp_path / "data" / "models",
                num_cores=1,
            ),
            data_loader=SimpleNamespace(
                max_failed_file_ratio=1.0,
                max_workers=1,
            ),
        ),
    )

    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    monkeypatch.setattr(cli, "configure_logging", lambda _level=None: 0)
    monkeypatch.setattr(cli, "reload_settings", lambda: settings)
    monkeypatch.setattr(data_cli_module, "get_settings", lambda: settings)
    monkeypatch.setattr(data_loader_module, "get_settings", lambda: settings)

    monkeypatch.setattr(
        cli.sys,
        "argv",
        [
            "ser",
            "configure",
            "--accept-dataset-policy",
            "noncommercial",
            "--accept-dataset-license",
            "cc-by-nc-sa-4.0",
            "--persist",
        ],
    )
    with pytest.raises(SystemExit) as configure_exit:
        cli.main()
    assert configure_exit.value.code == 0

    persisted_consents = load_persisted_dataset_consents(settings=settings)
    assert persisted_consents.policy_consents["noncommercial"] == "ser configure"
    assert persisted_consents.license_consents["cc-by-nc-sa-4.0"] == "ser configure"

    monkeypatch.setattr(
        cli.sys,
        "argv",
        [
            "ser",
            "--log-level",
            "INFO",
            "data",
            "download",
            "--dataset",
            "ravdess",
            "--dataset-root",
            str(dataset_root),
            "--manifest-path",
            str(manifest_path),
            "--skip-download",
        ],
    )
    with pytest.raises(SystemExit) as download_exit:
        cli.main()
    assert download_exit.value.code == 0
    assert manifest_path.is_file()

    registry = load_dataset_registry(settings=settings)
    assert "ravdess" in registry
    assert registry["ravdess"].manifest_path == manifest_path

    loaded = data_loader_module.load_utterances()
    assert loaded is not None
    assert [item.label for item in loaded] == ["happy", "sad"]
    assert all(item.corpus == "ravdess" for item in loaded)


def test_cli_help_lists_dataset_commands_and_flags(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Top-level help should advertise dataset workflow commands and key flags."""
    _patch_common_cli_dependencies(monkeypatch)
    monkeypatch.setattr(cli.sys, "argv", ["ser", "--help"])

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 0
    help_text = capsys.readouterr().out
    assert "ser configure --show" in help_text
    assert "ser data download --dataset" in help_text
    assert "--accept-dataset-policy" in help_text
    assert "--accept-dataset-license" in help_text
    assert "--persist" in help_text
    assert "--no-transcript" in help_text
    assert "--labels-csv-path" in help_text
    assert "--audio-base-dir" in help_text
    assert "--skip-download" in help_text
    assert "--accept-license" in help_text


def test_cli_dataset_subcommand_help_includes_configure_flags(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`ser configure --help` should expose dataset consent flags."""
    _patch_common_cli_dependencies(monkeypatch)
    monkeypatch.setattr(cli.sys, "argv", ["ser", "configure", "--help"])

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 0
    help_text = capsys.readouterr().out
    assert "--accept-dataset-policy" in help_text
    assert "--accept-dataset-license" in help_text
    assert "--persist" in help_text
    assert "--show" in help_text


def test_cli_dataset_subcommand_help_includes_download_flags(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`ser data download --help` should expose dataset preparation flags."""
    _patch_common_cli_dependencies(monkeypatch)
    monkeypatch.setattr(cli.sys, "argv", ["ser", "data", "download", "--help"])

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 0
    help_text = capsys.readouterr().out
    assert "--dataset" in help_text
    assert "--dataset-root" in help_text
    assert "--manifest-path" in help_text
    assert "--labels-csv-path" in help_text
    assert "--audio-base-dir" in help_text
    assert "--skip-download" in help_text
    assert "--accept-license" in help_text
