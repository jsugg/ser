"""Integration coverage for spawned runtime and transcription process paths."""

from __future__ import annotations

import logging
import multiprocessing as mp
import pickle
from collections.abc import Callable
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest
from tests.utils.helpers import process_spawn_support

from ser._internal.runtime import process_timeout as runtime_process_timeout
from ser._internal.runtime import worker_lifecycle as runtime_worker_lifecycle
from ser._internal.transcription import process_isolation as transcription_process_isolation
from ser._internal.transcription import public_boundary_support as transcription_boundary_support
from ser.config import AppConfig, reload_settings
from ser.domain import TranscriptWord
from ser.transcript.transcript_extractor import TranscriptionProfile

if TYPE_CHECKING:
    from ser.transcript.backends.base import BackendRuntimeRequest
else:
    BackendRuntimeRequest = object

pytestmark = [
    pytest.mark.integration,
    pytest.mark.process_isolation,
    pytest.mark.usefixtures("reset_ambient_settings"),
]


class _RuntimeTimeoutError(TimeoutError):
    """Timeout-domain error used by spawned runtime process integration tests."""


def _log_phase_started(
    _logger: object,
    *,
    phase_name: str,
    profile: str,
) -> float:
    del phase_name, profile
    return 0.0


def _log_phase_completed(
    _logger: object,
    *,
    phase_name: str,
    started_at: float | None = None,
    profile: str | None = None,
) -> None:
    del phase_name, started_at, profile


def _log_phase_failed(
    _logger: object,
    *,
    phase_name: str,
    started_at: float | None = None,
    profile: str | None = None,
) -> None:
    del phase_name, started_at, profile


def _run_runtime_process_timeout(
    payload: process_spawn_support.RuntimeWorkerPayload,
    *,
    timeout_seconds: float,
) -> str:
    logger = logging.getLogger("ser.tests.runtime_process_spawn")

    def _recv_worker_message(connection: Connection, *, stage: str) -> tuple[object, ...]:
        return runtime_worker_lifecycle.recv_worker_message(
            connection=connection,
            stage=stage,
            worker_label="Synthetic runtime",
            error_factory=RuntimeError,
        )

    def _is_setup_complete_message(message: tuple[object, ...]) -> bool:
        return runtime_worker_lifecycle.is_setup_complete_message(
            message=message,
            worker_label="Synthetic runtime",
            error_factory=RuntimeError,
        )

    def _terminate_worker_process(process: object) -> None:
        runtime_worker_lifecycle.terminate_worker_process(
            process=cast(BaseProcess, process),
            terminate_grace_seconds=0.1,
            kill_grace_seconds=0.1,
        )

    def _raise_worker_error(error_type: str, message: str) -> None:
        runtime_worker_lifecycle.raise_worker_error(
            error_type=error_type,
            message=message,
            known_error_factories={"ValueError": ValueError},
            unknown_error_factory=RuntimeError,
            worker_label="Synthetic runtime",
        )

    def _parse_worker_completion_message(message: tuple[object, ...]) -> str:
        return runtime_worker_lifecycle.parse_worker_completion_message(
            worker_message=message,
            worker_label="Synthetic runtime",
            error_factory=RuntimeError,
            raise_worker_error=_raise_worker_error,
            result_type=str,
        )

    return runtime_process_timeout.run_with_process_timeout(
        payload=payload,
        resolve_profile=lambda _payload: "synthetic",
        timeout_seconds=timeout_seconds,
        get_context=mp.get_context,
        logger=logger,
        setup_phase_name="setup",
        inference_phase_name="compute",
        log_phase_started=_log_phase_started,
        log_phase_completed=_log_phase_completed,
        log_phase_failed=_log_phase_failed,
        run_process_setup_compute_handshake=runtime_worker_lifecycle.run_process_setup_compute_handshake,
        worker_target=process_spawn_support.runtime_worker_entry,
        recv_worker_message=_recv_worker_message,
        is_setup_complete_message=_is_setup_complete_message,
        terminate_worker_process=_terminate_worker_process,
        timeout_error_factory=_RuntimeTimeoutError,
        execution_error_factory=RuntimeError,
        worker_label="Synthetic runtime",
        process_join_grace_seconds=0.1,
        parse_worker_completion_message=_parse_worker_completion_message,
    )


def _transcription_runtime_request() -> BackendRuntimeRequest:
    return cast(
        BackendRuntimeRequest,
        SimpleNamespace(
            model_name="tiny",
            use_demucs=False,
            use_vad=True,
            device_spec="cpu",
            device_type="cpu",
            precision_candidates=("float32",),
            memory_tier="not_applicable",
        ),
    )


def test_runtime_process_timeout_executes_real_spawn_worker() -> None:
    """Runtime helper should complete the real spawned-worker happy path."""
    result = _run_runtime_process_timeout(
        process_spawn_support.RuntimeWorkerPayload(result="done"),
        timeout_seconds=1.0,
    )

    assert result == "done"


def test_runtime_process_timeout_maps_worker_errors_from_real_spawn_worker() -> None:
    """Runtime helper should rehydrate worker errors from a real spawned child."""
    with pytest.raises(ValueError, match="bad input"):
        _run_runtime_process_timeout(
            process_spawn_support.RuntimeWorkerPayload(
                error_type="ValueError",
                error_message="bad input",
            ),
            timeout_seconds=1.0,
        )


def test_runtime_process_timeout_enforces_timeout_for_real_spawn_worker() -> None:
    """Runtime helper should terminate a real spawned worker on timeout."""
    with pytest.raises(_RuntimeTimeoutError, match="Synthetic runtime exceeded timeout"):
        _run_runtime_process_timeout(
            process_spawn_support.RuntimeWorkerPayload(compute_delay_seconds=0.2),
            timeout_seconds=0.01,
        )


def test_transcription_process_isolation_executes_real_spawn_worker(
    tmp_path: Path,
) -> None:
    """Transcription helper should complete one real spawned-worker success path."""
    settings = process_spawn_support.FakeSettings(
        models=process_spawn_support.FakeModelsConfig(whisper_download_root=tmp_path),
    )
    profile = process_spawn_support.FakeTranscriptionProfile()

    result = transcription_process_isolation.run_faster_whisper_process_isolated(
        file_path="sample.wav",
        language="en",
        profile=profile,
        settings_resolver=lambda: cast(AppConfig, settings),
        runtime_request_resolver=lambda _profile, _settings: _transcription_runtime_request(),
        payload_factory=cast(
            Callable[..., object],
            process_spawn_support.build_transcription_payload,
        ),
        get_spawn_context=lambda: mp.get_context("spawn"),
        worker_entry=cast(
            Callable[[object, object], None],
            process_spawn_support.transcription_success_worker,
        ),
        recv_worker_message_fn=lambda connection, *, stage: (
            transcription_process_isolation.recv_worker_message(
                connection,
                stage=stage,
                error_factory=RuntimeError,
            )
        ),
        raise_worker_error_fn=lambda message: transcription_process_isolation.raise_worker_error(
            message,
            error_factory=RuntimeError,
        ),
        terminate_worker_process_fn=lambda process: (
            transcription_process_isolation.terminate_worker_process(
                process,
                terminate_grace_seconds=0.1,
                kill_grace_seconds=0.1,
            )
        ),
        transcript_word_factory=TranscriptWord,
        logger=logging.getLogger("ser.tests.transcription_process_spawn"),
        error_factory=RuntimeError,
        terminate_grace_seconds=0.1,
    )

    assert result == [TranscriptWord("hello", 0.0, 0.5), TranscriptWord("world", 0.5, 1.0)]


def test_transcription_process_isolation_maps_real_spawn_worker_errors(
    tmp_path: Path,
) -> None:
    """Transcription helper should surface errors from a real spawned worker."""
    settings = process_spawn_support.FakeSettings(
        models=process_spawn_support.FakeModelsConfig(whisper_download_root=tmp_path),
    )
    profile = process_spawn_support.FakeTranscriptionProfile()

    with pytest.raises(
        RuntimeError,
        match="Transcription worker model_load failed with RuntimeError: boom",
    ):
        transcription_process_isolation.run_faster_whisper_process_isolated(
            file_path="sample.wav",
            language="en",
            profile=profile,
            settings_resolver=lambda: cast(AppConfig, settings),
            runtime_request_resolver=lambda _profile, _settings: _transcription_runtime_request(),
            payload_factory=cast(
                Callable[..., object],
                process_spawn_support.build_transcription_payload,
            ),
            get_spawn_context=lambda: mp.get_context("spawn"),
            worker_entry=cast(
                Callable[[object, object], None],
                process_spawn_support.transcription_error_worker,
            ),
            recv_worker_message_fn=lambda connection, *, stage: (
                transcription_process_isolation.recv_worker_message(
                    connection,
                    stage=stage,
                    error_factory=RuntimeError,
                )
            ),
            raise_worker_error_fn=lambda message: transcription_process_isolation.raise_worker_error(
                message,
                error_factory=RuntimeError,
            ),
            terminate_worker_process_fn=lambda process: (
                transcription_process_isolation.terminate_worker_process(
                    process,
                    terminate_grace_seconds=0.1,
                    kill_grace_seconds=0.1,
                )
            ),
            transcript_word_factory=TranscriptWord,
            logger=logging.getLogger("ser.tests.transcription_process_spawn"),
            error_factory=RuntimeError,
            terminate_grace_seconds=0.1,
        )


def test_transcription_public_boundary_worker_target_is_spawn_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Public faster-whisper boundary should pass a top-level picklable worker target."""
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        transcription_boundary_support,
        "_run_faster_whisper_process_isolated_boundary_impl",
        lambda **kwargs: (captured.update(kwargs), [])[1],
    )

    transcription_boundary_support.run_faster_whisper_process_isolated(
        file_path="sample.wav",
        language="en",
        profile=TranscriptionProfile(
            backend_id="faster_whisper",
            model_name="small",
            use_demucs=False,
            use_vad=True,
        ),
        settings=reload_settings(),
        logger=logging.getLogger("ser.tests.transcription_boundary"),
        error_factory=RuntimeError,
    )

    worker_entry = cast(Callable[..., None], captured["worker_entry"])
    assert callable(worker_entry)
    assert "<locals>" not in worker_entry.__qualname__
    assert pickle.dumps(worker_entry)
