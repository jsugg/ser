"""Behavior tests for CLI argument dispatch and exit semantics."""

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

import ser.__main__ as cli
import ser.config as config_module
import ser.models.emotion_model as emotion_model
import ser.transcript as transcript_module
import ser.utils.timeline_utils as timeline_utils
from ser.domain import EmotionSegment, TimelineEntry, TranscriptWord
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

    monkeypatch.setattr(cli, "configure_logging", _capture_log_level)
    monkeypatch.setattr(cli, "_run_legacy_inference_workflow", lambda _args: None)

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
    """Prediction path should pass language through and save CSV when requested."""
    _patch_common_cli_dependencies(monkeypatch)
    monkeypatch.setattr(
        cli.sys,
        "argv",
        ["ser", "--file", "sample.wav", "--language", "es", "--save_transcript"],
    )

    calls: dict[str, object] = {}
    emotions = [EmotionSegment("happy", 0.0, 1.0)]
    transcript = [TranscriptWord("hola", 0.0, 0.5)]
    timeline = [TimelineEntry(0.0, "happy", "hola")]

    monkeypatch.setattr(emotion_model, "predict_emotions", lambda file: emotions)

    def fake_extract_transcript(
        file_path: str, language: str | None
    ) -> list[TranscriptWord]:
        calls["extract"] = (file_path, language)
        return transcript

    monkeypatch.setattr(
        transcript_module, "extract_transcript", fake_extract_transcript
    )
    monkeypatch.setattr(
        timeline_utils,
        "build_timeline",
        lambda text, emo: timeline if text == transcript and emo == emotions else [],
    )
    monkeypatch.setattr(
        timeline_utils,
        "print_timeline",
        lambda built_timeline: calls.setdefault("printed", built_timeline),
    )

    def fake_save_timeline_to_csv(
        built_timeline: list[TimelineEntry], file_name: str
    ) -> str:
        calls["saved"] = (built_timeline, file_name)
        return "out.csv"

    monkeypatch.setattr(
        timeline_utils, "save_timeline_to_csv", fake_save_timeline_to_csv
    )

    cli.main()

    assert calls["extract"] == ("sample.wav", "es")
    assert calls["printed"] == timeline
    assert calls["saved"] == (timeline, "sample.wav")


def test_cli_prediction_does_not_save_when_flag_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prediction path should skip CSV export unless `--save_transcript` is set."""
    _patch_common_cli_dependencies(monkeypatch)
    monkeypatch.setattr(cli.sys, "argv", ["ser", "--file", "sample.wav"])
    monkeypatch.setattr(
        emotion_model,
        "predict_emotions",
        lambda _file: [EmotionSegment("calm", 0.0, 1.0)],
    )
    monkeypatch.setattr(
        transcript_module,
        "extract_transcript",
        lambda _file, _language: [TranscriptWord("hello", 0.0, 0.5)],
    )
    monkeypatch.setattr(
        timeline_utils,
        "build_timeline",
        lambda _text, _emo: [TimelineEntry(0.0, "calm", "hello")],
    )
    monkeypatch.setattr(timeline_utils, "print_timeline", lambda _timeline: None)

    save_called = {"value": False}

    def fake_save(_timeline: list[TimelineEntry], _file_name: str) -> str:
        save_called["value"] = True
        return "out.csv"

    monkeypatch.setattr(timeline_utils, "save_timeline_to_csv", fake_save)

    cli.main()

    assert save_called["value"] is False


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
