"""Unit tests for in-process transcription orchestration helpers."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import cast

import pytest

from ser._internal.transcription import in_process_orchestration as ipo
from ser.config import AppConfig
from ser.transcript.backends.base import BackendRuntimeRequest

pytestmark = pytest.mark.unit


class _Profile:
    """Minimal transcription profile stub."""

    backend_id = "faster_whisper"


class _Adapter:
    """Deterministic adapter stub used by orchestration tests."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, object, object]] = []
        self.model = object()

    def setup_required(
        self,
        *,
        runtime_request: object,
        settings: object,
    ) -> bool:
        self.calls.append(("setup_required", runtime_request, settings))
        return True

    def prepare_assets(
        self,
        *,
        runtime_request: object,
        settings: object,
    ) -> None:
        self.calls.append(("prepare_assets", runtime_request, settings))

    def load_model(
        self,
        *,
        runtime_request: object,
        settings: object,
    ) -> object:
        self.calls.append(("load_model", runtime_request, settings))
        return self.model


def test_transcription_setup_required_uses_compatibility_checked_runtime_request() -> None:
    """Setup helper should run compatibility checks before asking the adapter."""
    settings = cast(AppConfig, SimpleNamespace())
    active_profile = _Profile()
    runtime_request = BackendRuntimeRequest(
        model_name="tiny",
        use_demucs=False,
        use_vad=False,
    )
    adapter = _Adapter()
    compatibility_calls: list[tuple[object, object, object]] = []

    resolved = ipo.transcription_setup_required(
        active_profile=active_profile,
        settings=settings,
        runtime_request_resolver=lambda profile, resolved_settings: runtime_request,
        compatibility_checker=lambda *, active_profile, settings, runtime_request: compatibility_calls.append(
            (active_profile, settings, runtime_request)
        ),
        adapter_resolver=lambda backend_id: adapter,
    )

    assert resolved is True
    assert compatibility_calls == [(active_profile, settings, runtime_request)]
    assert adapter.calls == [("setup_required", runtime_request, settings)]


def test_prepare_transcription_assets_delegates_to_adapter() -> None:
    """Preparation helper should reuse the same runtime request and adapter seam."""
    settings = cast(AppConfig, SimpleNamespace())
    active_profile = _Profile()
    runtime_request = BackendRuntimeRequest(
        model_name="tiny",
        use_demucs=False,
        use_vad=False,
    )
    adapter = _Adapter()

    ipo.prepare_transcription_assets(
        active_profile=active_profile,
        settings=settings,
        runtime_request_resolver=lambda profile, resolved_settings: runtime_request,
        compatibility_checker=lambda **_kwargs: None,
        adapter_resolver=lambda backend_id: adapter,
    )

    assert adapter.calls == [("prepare_assets", runtime_request, settings)]


def test_load_whisper_model_returns_adapter_model_after_logging_runtime_resolution() -> None:
    """Model loader should resolve settings, runtime request, and adapter model in order."""
    settings = cast(AppConfig, SimpleNamespace())
    active_profile = _Profile()
    runtime_request = BackendRuntimeRequest(
        model_name="tiny",
        use_demucs=False,
        use_vad=False,
        device_spec="cpu",
        precision_candidates=("fp32",),
        memory_tier="low",
    )
    adapter = _Adapter()

    resolved = ipo.load_whisper_model(
        profile=None,
        settings_resolver=lambda: settings,
        profile_resolver=lambda profile: active_profile,
        runtime_request_resolver=lambda profile, resolved_settings: runtime_request,
        compatibility_checker=lambda **_kwargs: None,
        adapter_resolver=lambda backend_id: adapter,
        logger=logging.getLogger("ser.tests.in_process_orchestration"),
        error_factory=RuntimeError,
    )

    assert resolved is adapter.model
    assert adapter.calls == [("load_model", runtime_request, settings)]


def test_load_whisper_model_wraps_runtime_failures_with_domain_error() -> None:
    """Model loader should translate unexpected failures into the transcription domain error."""
    active_profile = _Profile()
    runtime_request = BackendRuntimeRequest(
        model_name="tiny",
        use_demucs=False,
        use_vad=False,
        device_spec="cpu",
        precision_candidates=("fp32",),
        memory_tier="low",
    )
    settings = cast(AppConfig, SimpleNamespace())

    with pytest.raises(RuntimeError, match="Failed to load transcription model") as exc_info:
        ipo.load_whisper_model(
            profile=None,
            settings_resolver=lambda: settings,
            profile_resolver=lambda profile: active_profile,
            runtime_request_resolver=lambda profile, resolved_settings: runtime_request,
            compatibility_checker=lambda **_kwargs: (_ for _ in ()).throw(ValueError("boom")),
            adapter_resolver=lambda backend_id: _Adapter(),
            logger=logging.getLogger("ser.tests.in_process_orchestration"),
            error_factory=RuntimeError,
        )

    assert isinstance(exc_info.value.__cause__, ValueError)


def test_extract_transcript_in_process_runs_setup_and_releases_model_on_success() -> None:
    """In-process transcription should complete all phases and always release model memory."""
    phase_events: list[tuple[str, str]] = []
    release_calls: list[object | None] = []
    settings = cast(AppConfig, SimpleNamespace())
    active_profile = _Profile()
    model = object()

    resolved = ipo.extract_transcript_in_process(
        file_path="sample.wav",
        language="en",
        profile=active_profile,
        settings_resolver=lambda: settings,
        setup_required_checker=lambda *, active_profile, settings: True,
        prepare_assets_runner=lambda *, active_profile, settings: phase_events.append(
            ("prepare_assets", "setup")
        ),
        load_model_fn=lambda profile: model,
        transcribe_with_profile_fn=lambda model, language, file_path, active_profile: ["word"],
        release_memory_fn=lambda *, model: release_calls.append(model),
        phase_started_fn=lambda logger, *, phase_name: phase_events.append(("start", phase_name))
        or 1.0,
        phase_completed_fn=lambda logger, *, phase_name, started_at: phase_events.append(
            ("complete", phase_name)
        )
        or None,
        phase_failed_fn=lambda logger, *, phase_name, started_at: phase_events.append(
            ("failed", phase_name)
        )
        or None,
        logger=logging.getLogger("ser.tests.in_process_orchestration"),
    )

    assert resolved == ["word"]
    assert phase_events == [
        ("start", "transcription_setup"),
        ("prepare_assets", "setup"),
        ("complete", "transcription_setup"),
        ("start", "transcription_model_load"),
        ("complete", "transcription_model_load"),
        ("start", "transcription"),
        ("complete", "transcription"),
    ]
    assert release_calls == [model]


def test_extract_transcript_in_process_reports_transcription_failure_and_releases_model() -> None:
    """Transcription failures should mark the phase as failed and still release model memory."""
    phase_events: list[tuple[str, str]] = []
    release_calls: list[object | None] = []
    active_profile = _Profile()
    model = object()
    settings = cast(AppConfig, SimpleNamespace())

    with pytest.raises(ValueError, match="transcribe failed"):
        ipo.extract_transcript_in_process(
            file_path="sample.wav",
            language="en",
            profile=active_profile,
            settings_resolver=lambda: settings,
            setup_required_checker=lambda *, active_profile, settings: False,
            prepare_assets_runner=lambda *, active_profile, settings: None,
            load_model_fn=lambda profile: model,
            transcribe_with_profile_fn=lambda model, language, file_path, active_profile: (
                (_ for _ in ()).throw(ValueError("transcribe failed"))
            ),
            release_memory_fn=lambda *, model: release_calls.append(model),
            phase_started_fn=lambda logger, *, phase_name: phase_events.append(
                ("start", phase_name)
            )
            or 1.0,
            phase_completed_fn=lambda logger, *, phase_name, started_at: phase_events.append(
                ("complete", phase_name)
            )
            or None,
            phase_failed_fn=lambda logger, *, phase_name, started_at: phase_events.append(
                ("failed", phase_name)
            )
            or None,
            logger=logging.getLogger("ser.tests.in_process_orchestration"),
        )

    assert phase_events == [
        ("start", "transcription_model_load"),
        ("complete", "transcription_model_load"),
        ("start", "transcription"),
        ("failed", "transcription"),
    ]
    assert release_calls == [model]
