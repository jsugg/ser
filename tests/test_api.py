"""Tests for the stable `ser.api` facade and adjacent owner modules."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

import ser._internal.api.data as api_data_module
import ser._internal.api.diagnostics as api_diagnostics_module
import ser._internal.api.runtime as api_runtime_module
import ser._internal.runtime.commands as runtime_commands_module
import ser._internal.runtime.restricted_backends as restricted_backends_module
import ser.api as api
import ser.config as config_module
from ser.config import AppConfig
from ser.data.application import (
    DatasetRegistrySnapshot,
    DatasetRegistrySnapshotEntry,
    DatasetRegistrySnapshotIssue,
)
from ser.data.dataset_consents import DatasetConsentError
from ser.data.dataset_registry import upsert_dataset_registry_entry
from ser.diagnostics.domain import DiagnosticFinding, DiagnosticReport
from ser.license_check import BackendLicensePolicyError
from ser.runtime import InferenceExecution, InferenceRequest
from ser.runtime.registry import UnsupportedProfileError
from ser.transcript import TranscriptionError


def _settings(tmp_path: Path) -> AppConfig:
    return cast(
        AppConfig,
        SimpleNamespace(
            models=SimpleNamespace(folder=tmp_path / "data" / "models"),
            default_language="en",
            emotions={"03": "happy", "04": "sad"},
            data_loader=SimpleNamespace(max_failed_file_ratio=0.1),
            dataset=SimpleNamespace(subfolder_prefix="Actor_*", extension="*.wav"),
        ),
    )


def test_run_data_command_delegates_to_data_cli(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """API wrapper should delegate `ser data` execution."""
    captured: dict[str, object] = {}
    settings = _settings(tmp_path)

    def _run_data_command(argv: list[str], *, settings: AppConfig) -> int:
        captured["argv"] = argv
        captured["settings"] = settings
        return 7

    monkeypatch.setattr("ser.data.cli.run_data_command", _run_data_command)

    exit_code = api_data_module.run_data_command(
        ["download", "--dataset", "ravdess"],
        settings=settings,
    )

    assert exit_code == 7
    assert captured["argv"] == ["download", "--dataset", "ravdess"]
    assert captured["settings"] is settings


def test_run_doctor_command_delegates_to_diagnostics_command(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """API wrapper should delegate `ser doctor` execution with settings passthrough."""
    captured: dict[str, object] = {}
    settings = _settings(tmp_path)

    def _run_doctor_command(argv: list[str], *, settings: AppConfig | None = None) -> int:
        captured["argv"] = argv
        captured["settings"] = settings
        return 3

    monkeypatch.setattr("ser.diagnostics.command.run_doctor_command", _run_doctor_command)

    exit_code = api_diagnostics_module.run_doctor_command(["--strict"], settings=settings)

    assert exit_code == 3
    assert captured["argv"] == ["--strict"]
    assert captured["settings"] is settings


def test_run_transcription_runtime_calibration_workflow_delegates_to_profiling(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Runtime calibration API wrapper should parse profiles and dispatch profiler."""
    captured: dict[str, object] = {}
    sentinel = object()

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
            sentinel,
        )[1],
    )

    result = runtime_commands_module.run_transcription_runtime_calibration_workflow(
        calibration_file=tmp_path / "sample.wav",
        language="en",
        calibration_iterations=3,
        calibration_profiles="medium,accurate",
    )

    assert result is sentinel
    assert captured["profiles_raw"] == "medium,accurate"
    kwargs = cast(dict[str, object], captured["kwargs"])
    assert kwargs["calibration_file"] == tmp_path / "sample.wav"
    assert kwargs["language"] == "en"
    assert kwargs["iterations_per_profile"] == 3
    assert kwargs["profile_names"] == ("medium", "accurate")


def test_run_transcription_runtime_calibration_command_maps_validation_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Calibration command wrapper should map validation failures to exit code 2."""
    monkeypatch.setattr(
        api_runtime_module,
        "run_transcription_runtime_calibration_cli",
        lambda **_kwargs: (_ for _ in ()).throw(ValueError("missing --file")),
    )

    result, disposition = api_runtime_module.run_transcription_runtime_calibration_command(
        file_path=None,
        language="en",
        calibration_iterations=1,
        calibration_profiles="fast",
    )

    assert result is None
    assert disposition is not None
    assert disposition.exit_code == 2
    assert disposition.message == "missing --file"
    assert disposition.include_traceback is False


def test_run_transcription_runtime_calibration_command_maps_unexpected_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Calibration command wrapper should map unexpected failures to exit code 1."""
    monkeypatch.setattr(
        api_runtime_module,
        "run_transcription_runtime_calibration_cli",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    result, disposition = api_runtime_module.run_transcription_runtime_calibration_command(
        file_path="sample.wav",
        language="en",
        calibration_iterations=1,
        calibration_profiles="fast",
    )

    assert result is None
    assert disposition is not None
    assert disposition.exit_code == 1
    assert disposition.message == "Transcription runtime calibration failed: boom"
    assert disposition.include_traceback is True


def test_run_training_command_maps_training_exceptions_to_disposition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Training command wrapper should map workflow failures to one disposition."""
    monkeypatch.setattr(
        api_runtime_module,
        "run_training_workflow",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("training failed")),
    )

    disposition = api_runtime_module.run_training_command(
        settings=config_module.reload_settings(),
        use_profile_pipeline=True,
    )

    assert disposition is not None
    assert disposition.exit_code == 2
    assert disposition.message == "training failed"
    assert disposition.include_traceback is False


def test_run_training_command_delegates_arguments_on_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Training command wrapper should pass through orchestration arguments."""
    captured: dict[str, object] = {}
    settings = config_module.reload_settings()

    class _StubPipeline:
        def run_training(self) -> None:
            raise AssertionError("unreachable")

        def run_inference(self, request: InferenceRequest) -> InferenceExecution:
            del request
            raise AssertionError("unreachable")

    def _pipeline_builder(_settings: AppConfig) -> _StubPipeline:
        return _StubPipeline()

    def _run_training_workflow(**kwargs: object) -> None:
        captured["kwargs"] = kwargs

    monkeypatch.setattr(api_runtime_module, "run_training_workflow", _run_training_workflow)

    disposition = api_runtime_module.run_training_command(
        settings=settings,
        use_profile_pipeline=False,
        pipeline_builder=_pipeline_builder,
    )

    assert disposition is None
    kwargs = cast(dict[str, object], captured["kwargs"])
    assert kwargs["settings"] is settings
    assert kwargs["use_profile_pipeline"] is False
    assert kwargs["pipeline_builder"] is _pipeline_builder


def test_run_inference_command_validates_missing_file_path() -> None:
    """Inference command wrapper should return exit-1 disposition for missing file."""
    execution, disposition = api_runtime_module.run_inference_command(
        settings=config_module.reload_settings(),
        file_path=None,
        language="en",
        save_transcript=False,
        include_transcript=True,
    )

    assert execution is None
    assert disposition is not None
    assert disposition.exit_code == 1
    assert disposition.message == "No audio file provided for prediction."
    assert disposition.include_traceback is False


def test_run_inference_command_delegates_arguments_on_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inference command wrapper should pass through arguments and return execution."""
    captured: dict[str, object] = {}
    settings = config_module.reload_settings()
    sentinel_execution = cast(InferenceExecution, object())

    class _StubPipeline:
        def run_training(self) -> None:
            raise AssertionError("unreachable")

        def run_inference(self, request: InferenceRequest) -> InferenceExecution:
            del request
            raise AssertionError("unreachable")

    def _pipeline_builder(_settings: AppConfig) -> _StubPipeline:
        return _StubPipeline()

    def _run_inference_workflow(**kwargs: object) -> InferenceExecution:
        captured["kwargs"] = kwargs
        return sentinel_execution

    monkeypatch.setattr(
        api_runtime_module,
        "run_inference_workflow",
        _run_inference_workflow,
    )

    execution, disposition = api_runtime_module.run_inference_command(
        settings=settings,
        file_path="sample.wav",
        language="en",
        save_transcript=True,
        include_transcript=False,
        pipeline_builder=_pipeline_builder,
    )

    assert execution is sentinel_execution
    assert disposition is None
    kwargs = cast(dict[str, object], captured["kwargs"])
    assert kwargs["settings"] is settings
    assert kwargs["file_path"] == "sample.wav"
    assert kwargs["language"] == "en"
    assert kwargs["save_transcript"] is True
    assert kwargs["include_transcript"] is False
    assert kwargs["pipeline_builder"] is _pipeline_builder


def test_run_restricted_backend_cli_gate_short_circuits_without_command_path() -> None:
    """Restricted-backend gate should no-op when no executable CLI path is requested."""
    settings = config_module.reload_settings()

    logs, exit_code = api_runtime_module.run_restricted_backend_cli_gate(
        settings=settings,
        use_profile_pipeline=False,
        train_requested=False,
        file_path=None,
        accept_restricted_backends=False,
        accept_all_restricted_backends=False,
        is_interactive=False,
    )

    assert exit_code is None
    assert logs == ()


def test_run_restricted_backend_cli_gate_short_circuits_opt_in_only_invocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Opt-in-only invocations should persist and return zero without enforcement."""
    settings = config_module.reload_settings()
    monkeypatch.setattr(
        api_runtime_module,
        "_prepare_restricted_backend_opt_in_state",
        lambda **_kwargs: restricted_backends_module.RestrictedBackendOptInState(
            required_backend_ids=("modelscope",),
            persisted_all_count=1,
            persisted_profile_backend_ids=("modelscope",),
            should_exit_zero=True,
        ),
    )
    monkeypatch.setattr(
        api_runtime_module,
        "_enforce_restricted_backends_for_cli",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("should not run")),
    )

    logs, exit_code = api_runtime_module.run_restricted_backend_cli_gate(
        settings=settings,
        use_profile_pipeline=True,
        train_requested=False,
        file_path=None,
        accept_restricted_backends=True,
        accept_all_restricted_backends=False,
        is_interactive=False,
    )

    assert exit_code == 0
    assert len(logs) == 2
    assert logs[0][0] == "info"
    assert "1 backend(s)" in logs[0][1]
    assert logs[1][0] == "info"
    assert "modelscope" in logs[1][1]


def test_run_restricted_backend_cli_gate_maps_policy_errors_to_exit_2(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Restricted-backend gate should map policy errors to CLI exit code 2."""
    settings = config_module.reload_settings()
    monkeypatch.setattr(
        api_runtime_module,
        "_prepare_restricted_backend_opt_in_state",
        lambda **_kwargs: restricted_backends_module.RestrictedBackendOptInState(
            required_backend_ids=("modelscope",),
            persisted_all_count=0,
            persisted_profile_backend_ids=(),
            should_exit_zero=False,
        ),
    )
    monkeypatch.setattr(
        api_runtime_module,
        "_enforce_restricted_backends_for_cli",
        lambda **_kwargs: (_ for _ in ()).throw(BackendLicensePolicyError("blocked")),
    )

    logs, exit_code = api_runtime_module.run_restricted_backend_cli_gate(
        settings=settings,
        use_profile_pipeline=True,
        train_requested=True,
        file_path=None,
        accept_restricted_backends=False,
        accept_all_restricted_backends=False,
        is_interactive=False,
    )

    assert exit_code == 2
    assert logs == (("error", "blocked"),)


def test_prepare_dataset_strict_mode_requires_explicit_consents(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Strict compliance mode should refuse missing restricted consents."""
    settings = _settings(tmp_path)
    manifest_path = tmp_path / "manifests" / "msp-podcast.jsonl"
    monkeypatch.setattr(
        "ser.data.application.run_dataset_prepare_workflow",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("unreachable")),
    )

    with pytest.raises(DatasetConsentError, match="Missing dataset acknowledgements"):
        api.prepare_dataset(
            dataset_id="msp-podcast",
            settings=settings,
            skip_download=True,
            compliance_mode="strict",
            manifest_path=manifest_path,
            labels_csv_path=tmp_path / "labels.csv",
        )


def test_prepare_dataset_accept_license_persists_missing_consents(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """`accept_license` should auto-persist missing restricted consents."""
    settings = _settings(tmp_path)
    manifest_path = tmp_path / "manifests" / "msp-podcast.jsonl"
    captured: dict[str, object] = {}

    def _run_dataset_prepare_workflow(**kwargs: object) -> object:
        captured["kwargs"] = kwargs
        return SimpleNamespace(
            descriptor=SimpleNamespace(dataset_id="msp-podcast"),
            dataset_root=tmp_path / "datasets" / "msp-podcast",
            manifest_path=manifest_path,
            manifest_paths=(manifest_path,),
            downloaded=False,
            source_repo_id=None,
            source_revision=None,
        )

    monkeypatch.setattr(
        "ser.data.application.run_dataset_prepare_workflow",
        _run_dataset_prepare_workflow,
    )

    result = api.prepare_dataset(
        dataset_id="msp-podcast",
        settings=settings,
        skip_download=True,
        accept_license=True,
        compliance_mode="strict",
        manifest_path=manifest_path,
        labels_csv_path=tmp_path / "labels.csv",
    )

    assert result.manifest_paths == (manifest_path,)
    assert result.missing_policy_consents == ()
    assert result.missing_license_consents == ()
    policies, licenses = api.show_dataset_consents(settings=settings)
    assert "academic_only" in policies
    assert "msp-academic-license" in licenses
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["dataset_id"] == "msp-podcast"


def test_prepare_msp_podcast_mirror_delegates_to_data_layer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """API wrapper should delegate MSP mirror preparation with passthrough args."""
    captured: dict[str, object] = {}
    dataset_root = tmp_path / "datasets" / "msp-podcast"
    sentinel = object()

    def _prepare_msp_podcast_from_hf_mirror(**kwargs: object) -> object:
        captured["kwargs"] = kwargs
        return sentinel

    monkeypatch.setattr(
        "ser.data.msp_podcast_mirror.prepare_msp_podcast_from_hf_mirror",
        _prepare_msp_podcast_from_hf_mirror,
    )

    result = api_data_module.prepare_msp_podcast_mirror(
        dataset_root=dataset_root,
        repo_id="org/dataset",
        revision="abc123",
        max_workers=3,
        batch_size=11,
        token="token-value",
    )

    assert result is sentinel
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs == {
        "dataset_root": dataset_root,
        "repo_id": "org/dataset",
        "revision": "abc123",
        "max_workers": 3,
        "batch_size": 11,
        "token": "token-value",
    }


def test_prepare_dataset_passes_source_overrides_to_download(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """prepare_dataset should pass source overrides to dataset acquisition."""
    settings = _settings(tmp_path)
    manifest_path = tmp_path / "manifests" / "msp-podcast.jsonl"
    captured: dict[str, object] = {}

    def _capture_workflow_kwargs(**kwargs: object) -> object:
        captured["workflow_kwargs"] = kwargs
        return SimpleNamespace(
            descriptor=SimpleNamespace(dataset_id="msp-podcast"),
            dataset_root=tmp_path / "datasets" / "msp-podcast",
            manifest_path=manifest_path,
            manifest_paths=(manifest_path,),
            downloaded=True,
            source_repo_id="org/repo",
            source_revision="rev-1",
        )

    monkeypatch.setattr(
        "ser.data.application.run_dataset_prepare_workflow",
        _capture_workflow_kwargs,
    )

    result = api.prepare_dataset(
        dataset_id="msp-podcast",
        settings=settings,
        skip_download=False,
        source_repo_id="org/repo",
        source_revision="rev-1",
        manifest_path=manifest_path,
        labels_csv_path=tmp_path / "labels.csv",
    )

    assert result.downloaded is True
    workflow_kwargs = captured["workflow_kwargs"]
    assert isinstance(workflow_kwargs, dict)
    assert workflow_kwargs["source_repo_id"] == "org/repo"
    assert workflow_kwargs["source_revision"] == "rev-1"


def test_prepare_dataset_rejects_source_overrides_when_skip_download(
    tmp_path: Path,
) -> None:
    """prepare_dataset should reject source overrides when acquisition is disabled."""
    settings = _settings(tmp_path)

    with pytest.raises(ValueError, match="skip_download=True"):
        api.prepare_dataset(
            dataset_id="msp-podcast",
            settings=settings,
            skip_download=True,
            source_repo_id="org/repo",
            labels_csv_path=tmp_path / "labels.csv",
        )


def test_list_registered_datasets_returns_source_provenance(
    tmp_path: Path,
) -> None:
    """Registry API should surface persisted source pin metadata."""
    settings = _settings(tmp_path)
    dataset_root = tmp_path / "datasets" / "msp-podcast"
    manifest_path = tmp_path / "manifests" / "msp-podcast.jsonl"
    upsert_dataset_registry_entry(
        settings=settings,
        dataset_id="msp-podcast",
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        options={
            "labels_csv_path": str(dataset_root / "labels.csv"),
            "audio_base_dir": str(dataset_root / "audio"),
            "source_repo_id": "org/repo",
            "source_revision": "rev-1",
            "source_commit_sha": "abcdef1234567890",
        },
    )

    records = api.list_registered_datasets(settings=settings)

    assert len(records) == 1
    record = records[0]
    assert record.dataset_id == "msp-podcast"
    assert record.dataset_root == dataset_root
    assert record.manifest_path == manifest_path
    assert record.source_repo_id == "org/repo"
    assert record.source_revision == "rev-1"
    assert record.source_commit_sha == "abcdef1234567890"


def test_list_dataset_registry_health_issues_exposes_issue_records(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Health issue API should surface typed registry issue records."""
    settings = _settings(tmp_path)
    monkeypatch.setattr(
        "ser.data.application.collect_dataset_registry_snapshot",
        lambda **kwargs: DatasetRegistrySnapshot(
            entries=(
                DatasetRegistrySnapshotEntry(
                    dataset_id="msp-podcast",
                    dataset_root=tmp_path / "datasets" / "msp-podcast",
                    manifest_path=tmp_path / "manifests" / "msp-podcast.jsonl",
                    options={},
                    source_repo_id=None,
                    source_revision=None,
                ),
            ),
            issues=(
                DatasetRegistrySnapshotIssue(
                    dataset_id="msp-podcast",
                    code="source_provenance_mismatch",
                    message="Mismatch.",
                ),
            ),
        ),
    )

    issues = api.list_dataset_registry_health_issues(settings=settings)

    assert len(issues) == 1
    issue = issues[0]
    assert issue.dataset_id == "msp-podcast"
    assert issue.code == "source_provenance_mismatch"
    assert issue.message == "Mismatch."


def test_train_uses_pipeline_builder_when_enabled(
    tmp_path: Path,
) -> None:
    """Stable training API should use injected pipeline builder when enabled."""
    settings = _settings(tmp_path)
    calls: dict[str, bool] = {"training": False}

    class _FakePipeline:
        def run_training(self) -> None:
            calls["training"] = True

        def run_inference(self, request: InferenceRequest) -> InferenceExecution:
            del request
            raise AssertionError("unreachable")

    api.train(
        settings=settings,
        use_profile_pipeline=True,
        pipeline_builder=lambda _settings: _FakePipeline(),
    )

    assert calls["training"] is True


def test_train_uses_pipeline_builder_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stable training API should keep routing through the pipeline when disabled."""
    settings = config_module.reload_settings()
    captured: dict[str, object] = {"training": False}

    class _FakePipeline:
        def run_training(self) -> None:
            captured["training"] = True

        def run_inference(self, request: InferenceRequest) -> InferenceExecution:
            del request
            raise AssertionError("unreachable")

    def _pipeline_builder(received_settings: AppConfig) -> _FakePipeline:
        captured["settings"] = received_settings
        return _FakePipeline()

    monkeypatch.setattr(
        "ser.models.emotion_model.train_model",
        lambda: (_ for _ in ()).throw(
            AssertionError("Legacy training branch should remain unreachable.")
        ),
    )

    api.train(
        settings=settings,
        use_profile_pipeline=False,
        pipeline_builder=_pipeline_builder,
    )

    assert captured["settings"] is settings
    assert captured["training"] is True


def test_train_passes_scoped_settings_to_pipeline_builder() -> None:
    """Stable training API should pass explicit profile-scoped settings to the builder."""
    base_settings = config_module.reload_settings()
    scoped_settings = replace(base_settings, default_language="pt-BR")
    captured: dict[str, object] = {"training": False}

    class _FakePipeline:
        def run_training(self) -> None:
            captured["training"] = True

        def run_inference(self, request: InferenceRequest) -> InferenceExecution:
            del request
            raise AssertionError("unreachable")

    def _pipeline_builder(received_settings: AppConfig) -> _FakePipeline:
        captured["settings"] = received_settings
        return _FakePipeline()

    api.train(
        settings=scoped_settings,
        use_profile_pipeline=False,
        pipeline_builder=_pipeline_builder,
    )

    assert captured["settings"] is scoped_settings
    assert captured["training"] is True


def test_infer_builds_inference_request(
    tmp_path: Path,
) -> None:
    """Stable inference API should build and pass one runtime inference request."""
    settings = _settings(tmp_path)
    captured: dict[str, object] = {}
    sentinel = object()

    class _FakePipeline:
        def run_training(self) -> None:
            raise AssertionError("unreachable")

        def run_inference(self, request: InferenceRequest) -> InferenceExecution:
            captured["request"] = request
            return cast(InferenceExecution, sentinel)

    result = api.infer(
        tmp_path / "sample.wav",
        settings=settings,
        language="en",
        save_transcript=True,
        include_transcript=False,
        pipeline_builder=lambda _settings: _FakePipeline(),
    )

    assert result is sentinel
    request = cast(InferenceRequest, captured["request"])
    assert request.file_path.endswith("sample.wav")
    assert request.language == "en"
    assert request.save_transcript is True
    assert request.include_transcript is False


def test_classify_inference_exception_for_unsupported_profile() -> None:
    """Unsupported profile errors should map to user-facing exit code 2."""
    disposition = runtime_commands_module.classify_inference_exception(
        UnsupportedProfileError("unsupported profile")
    )
    assert disposition.exit_code == 2
    assert disposition.include_traceback is False


def test_classify_inference_exception_for_transcription_error() -> None:
    """Transcription errors should map to dedicated exit code and traceback."""
    disposition = runtime_commands_module.classify_inference_exception(
        TranscriptionError("transcription failed")
    )
    assert disposition.exit_code == 3
    assert disposition.include_traceback is True


def test_list_profiles_contains_all_runtime_profiles() -> None:
    """Public profile listing should expose all canonical runtime profiles."""
    profiles = api.list_profiles()
    assert profiles == ("fast", "medium", "accurate", "accurate-research")


def test_load_profile_validates_fast_profile_dependencies() -> None:
    """Profile loader should validate capability for supported profile settings."""
    api.load_profile("fast", settings=config_module.reload_settings())


def test_required_restricted_backends_for_profile_returns_research_backend() -> None:
    """Accurate-research profile should require the restricted backend."""
    base = config_module.reload_settings()
    scoped = replace(
        base,
        runtime_flags=replace(
            base.runtime_flags,
            profile_pipeline=True,
            medium_profile=False,
            accurate_profile=False,
            accurate_research_profile=True,
        ),
    )

    required = restricted_backends_module.required_restricted_backends_for_current_profile(
        scoped,
        use_profile_pipeline=True,
    )

    assert required == ("emotion2vec",)


def test_apply_cli_timeout_override_sets_all_profile_timeouts_to_zero() -> None:
    """API timeout override should force all runtime profile timeouts to zero."""
    settings = config_module.reload_settings()

    resolved = api_runtime_module.apply_cli_timeout_override(
        settings,
        disable_timeouts=True,
    )

    assert resolved.fast_runtime.timeout_seconds == 0.0
    assert resolved.medium_runtime.timeout_seconds == 0.0
    assert resolved.accurate_runtime.timeout_seconds == 0.0
    assert resolved.accurate_research_runtime.timeout_seconds == 0.0


def test_preflight_predicates_cover_execution_and_transcription_paths() -> None:
    """Preflight helper predicates should encode expected execution semantics."""
    assert (
        api_diagnostics_module.preflight_command_requested(
            train=False,
            file_path=None,
            calibrate_transcription_runtime=False,
        )
        is False
    )
    assert (
        api_diagnostics_module.preflight_command_requested(
            train=True,
            file_path=None,
            calibrate_transcription_runtime=False,
        )
        is True
    )
    assert (
        api_diagnostics_module.preflight_includes_transcription_checks(
            file_path="sample.wav",
            no_transcript=False,
            calibrate_transcription_runtime=False,
        )
        is True
    )
    assert (
        api_diagnostics_module.preflight_includes_transcription_checks(
            file_path="sample.wav",
            no_transcript=True,
            calibrate_transcription_runtime=False,
        )
        is False
    )
    assert (
        api_diagnostics_module.preflight_includes_transcription_checks(
            file_path=None,
            no_transcript=False,
            calibrate_transcription_runtime=True,
        )
        is True
    )


def test_runtime_profile_helpers_cover_pipeline_and_resolution_paths() -> None:
    """Runtime helper functions should preserve CLI profile resolution semantics."""
    settings = config_module.reload_settings()
    pipeline_settings = replace(
        settings,
        runtime_flags=replace(settings.runtime_flags, profile_pipeline=True),
    )
    assert api_runtime_module.profile_pipeline_enabled(pipeline_settings) is True
    assert (
        api_runtime_module.profile_resolution_requested(
            use_profile_pipeline=False,
            file_path=None,
        )
        is False
    )
    assert (
        api_runtime_module.profile_resolution_requested(
            use_profile_pipeline=False,
            file_path="sample.wav",
        )
        is True
    )
    assert api_runtime_module.resolve_cli_workflow_profile(settings) == "fast"
    resolved_settings = api_runtime_module.apply_cli_profile_override(
        settings,
        "medium",
    )
    assert api_runtime_module.resolve_cli_workflow_profile(resolved_settings) == "medium"


def test_resolve_doctor_command_formats_optional_profile_hint() -> None:
    """Doctor command helper should include profile hint only when provided."""
    assert api_diagnostics_module.resolve_doctor_command(profile=None) == "ser doctor"
    assert (
        api_diagnostics_module.resolve_doctor_command(profile="accurate")
        == "ser doctor --profile accurate"
    )


def test_suppress_preflight_transcription_operational_relogs_marks_emitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Suppress helper should mark known transcription operational issues as emitted."""
    settings = config_module.reload_settings()
    report = DiagnosticReport(
        findings=(
            DiagnosticFinding(
                code="transcription_operational_torio_ffmpeg_abi_mismatch",
                severity="info",
                message="non-blocking",
            ),
        )
    )
    captured: dict[str, object] = {}

    monkeypatch.setattr("ser._internal.api.diagnostics.resolve_profile_name", lambda _s: "fast")
    monkeypatch.setattr(
        "ser._internal.api.diagnostics.resolve_profile_transcription_config",
        lambda _profile: ("faster_whisper", "distil-large-v3", False, True),
    )
    monkeypatch.setattr(
        "ser.transcript.transcript_extractor.mark_compatibility_issues_as_emitted",
        lambda **kwargs: captured.update(kwargs),
    )

    api_diagnostics_module.suppress_preflight_transcription_operational_relogs(
        settings=settings,
        report=report,
    )

    assert captured["backend_id"] == "faster_whisper"
    assert captured["issue_kind"] == "operational"
    assert captured["issue_codes"] == ("torio_ffmpeg_abi_mismatch",)


def test_run_startup_preflight_cli_gate_suppresses_operational_relogs_on_warn_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Warn-mode CLI gate should emit summary logs and suppress duplicate re-logs."""
    settings = config_module.reload_settings()
    report = DiagnosticReport(
        findings=(
            DiagnosticFinding(
                code="transcription_operational_torio_ffmpeg_abi_mismatch",
                severity="warning",
                message="non-blocking",
            ),
        )
    )
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "ser._internal.api.diagnostics.run_startup_preflight",
        lambda **_kwargs: report,
    )
    monkeypatch.setattr(
        "ser._internal.api.diagnostics.format_startup_preflight_one_liner",
        lambda _report, *, doctor_command: f"Startup preflight advisory via `{doctor_command}`.",
    )
    monkeypatch.setattr(
        "ser._internal.api.diagnostics.should_fail_preflight",
        lambda **_kwargs: False,
    )
    monkeypatch.setattr(
        "ser._internal.api.diagnostics.suppress_preflight_transcription_operational_relogs",
        lambda **kwargs: captured.update(kwargs),
    )

    logs, exit_code = api_diagnostics_module.run_startup_preflight_cli_gate(
        settings=settings,
        mode="warn",
        profile=None,
        train_requested=False,
        file_path="sample.wav",
        no_transcript=False,
        calibrate_transcription_runtime=False,
    )

    assert exit_code is None
    assert logs == (("info", "Startup preflight advisory via `ser doctor`."),)
    assert captured["settings"] is settings
    assert captured["report"] is report


def test_api_facade_excludes_removed_cli_compatibility_exports() -> None:
    """Stable facade should not expose the retired CLI compatibility surface."""
    removed_symbols = {
        "WorkflowErrorDisposition",
        "RestrictedBackendOptInState",
        "RestrictedBackendPrompt",
        "apply_cli_profile_override",
        "apply_cli_timeout_override",
        "build_runtime_pipeline",
        "classify_inference_exception",
        "classify_training_exception",
        "collect_missing_restricted_backend_consents",
        "enforce_restricted_backends_for_cli",
        "ensure_restricted_backends_ready_for_command",
        "format_startup_preflight_one_liner",
        "parse_preflight_mode",
        "persist_all_restricted_backend_consents",
        "persist_required_restricted_backends",
        "preflight_command_requested",
        "preflight_includes_transcription_checks",
        "prepare_msp_podcast_mirror",
        "prepare_restricted_backend_opt_in_state",
        "profile_pipeline_enabled",
        "profile_resolution_requested",
        "resolve_cli_workflow_profile",
        "resolve_doctor_command",
        "required_restricted_backends_for_current_profile",
        "run_configure_command",
        "run_data_command",
        "run_doctor_command",
        "run_inference_command",
        "run_inference_workflow",
        "run_restricted_backend_cli_gate",
        "run_startup_preflight_cli_gate",
        "run_training_command",
        "run_training_workflow",
        "run_transcription_runtime_calibration_cli",
        "run_transcription_runtime_calibration_command",
        "run_transcription_runtime_calibration_workflow",
        "should_fail_preflight",
        "suppress_preflight_transcription_operational_relogs",
    }

    assert removed_symbols.isdisjoint(api.__all__)
    for symbol_name in removed_symbols:
        assert not hasattr(api, symbol_name)
        with pytest.raises(AttributeError):
            getattr(api, symbol_name)


def test_internal_runtime_api_surface_is_orchestration_focused() -> None:
    """Internal runtime API surface should only expose orchestration helpers."""
    assert api_runtime_module.__all__ == [
        "apply_cli_profile_override",
        "apply_cli_timeout_override",
        "build_runtime_pipeline",
        "infer",
        "list_profiles",
        "load_profile",
        "profile_pipeline_enabled",
        "profile_resolution_requested",
        "resolve_cli_workflow_profile",
        "run_inference_command",
        "run_inference_workflow",
        "run_restricted_backend_cli_gate",
        "run_training_command",
        "run_transcription_runtime_calibration_command",
        "run_transcription_runtime_calibration_cli",
        "run_training_workflow",
        "train",
    ]


def test_api_public_surface_includes_user_oriented_entrypoints() -> None:
    """Stable API contract should expose user-oriented library entrypoints only."""
    exported = set(api.__all__)
    required = {
        "ComplianceMode",
        "DatasetPrepareResult",
        "DatasetRegistryHealthIssueRecord",
        "DatasetRegistryRecord",
        "configure_dataset_consents",
        "infer",
        "train",
        "list_datasets",
        "list_profiles",
        "load_profile",
        "prepare_dataset",
        "run_startup_preflight",
        "list_registered_datasets",
        "list_dataset_registry_health_issues",
        "show_dataset_consents",
    }
    assert required.issubset(exported)


def test_api_public_surface_snapshot_matches_expected_contract() -> None:
    """Public API export snapshot should only change via explicit contract updates."""
    assert api.__all__ == [
        "ComplianceMode",
        "DatasetPrepareResult",
        "DatasetRegistryHealthIssueRecord",
        "DatasetRegistryRecord",
        "configure_dataset_consents",
        "infer",
        "list_dataset_registry_health_issues",
        "list_datasets",
        "list_profiles",
        "list_registered_datasets",
        "load_profile",
        "prepare_dataset",
        "run_startup_preflight",
        "show_dataset_consents",
        "train",
    ]
