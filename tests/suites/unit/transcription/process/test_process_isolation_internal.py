"""Unit tests for process-isolated transcription orchestration helpers."""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest

from ser._internal.transcription import process_isolation as isolation
from ser._internal.transcription.process_worker import (
    TranscriptionProcessPayload,
    TranscriptionWorkerModelsConfig,
    TranscriptionWorkerSettings,
)
from ser.config import AppConfig
from ser.domain import TranscriptWord
from ser.profiles import TranscriptionBackendId

if TYPE_CHECKING:
    from ser.transcript.backends.base import BackendRuntimeRequest
else:
    BackendRuntimeRequest = object

pytestmark = pytest.mark.unit


@dataclass(frozen=True)
class _Profile:
    """Minimal process-isolated transcription profile stub."""

    backend_id: TranscriptionBackendId = "faster_whisper"
    model_name: str = "tiny"
    use_demucs: bool = False
    use_vad: bool = True


class _Connection:
    """In-memory connection stub for worker message tests."""

    def __init__(self, received: list[object] | None = None) -> None:
        self._received = list(received or [])
        self.sent: list[object] = []
        self.closed = False

    def recv(self) -> object:
        if not self._received:
            raise EOFError()
        return self._received.pop(0)

    def send(self, obj: object) -> None:
        self.sent.append(obj)

    def close(self) -> None:
        self.closed = True


class _Process:
    """Worker process stub that tracks lifecycle method calls."""

    def __init__(self, *, alive_after_join: bool = False) -> None:
        self.args: tuple[object, ...] | None = None
        self.target: object | None = None
        self.daemon: bool | None = None
        self.started = False
        self.join_calls: list[float | None] = []
        self.terminated = False
        self.killed = False
        self.closed = False
        self._alive_after_join = alive_after_join

    def start(self) -> None:
        self.started = True

    def join(self, timeout: float | None = None) -> None:
        self.join_calls.append(timeout)

    def is_alive(self) -> bool:
        return self._alive_after_join and not self.killed

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True

    def close(self) -> None:
        self.closed = True


class _SpawnContext:
    """Spawn-context stub returning predetermined pipe and process objects."""

    def __init__(
        self,
        *,
        parent_connection: _Connection,
        child_connection: _Connection,
        process: _Process,
    ) -> None:
        self.parent_connection = parent_connection
        self.child_connection = child_connection
        self.process = process
        self.pipe_duplex: bool | None = None

    def Pipe(self, duplex: bool = False) -> tuple[_Connection, _Connection]:
        self.pipe_duplex = duplex
        return self.parent_connection, self.child_connection

    def Process(
        self,
        *,
        target: object,
        args: tuple[object, ...],
        daemon: bool,
    ) -> _Process:
        self.process.target = target
        self.process.args = args
        self.process.daemon = daemon
        return self.process


class _Adapter:
    """Backend adapter stub for worker-entry tests."""

    def __init__(self, *, transcript_words: list[object] | None = None) -> None:
        self.transcript_words = transcript_words or [TranscriptWord("hello", 0.0, 0.5)]
        self.calls: list[str] = []

    def setup_required(self, *, runtime_request: object, settings: object) -> bool:
        del runtime_request, settings
        self.calls.append("setup_required")
        return True

    def prepare_assets(self, *, runtime_request: object, settings: object) -> None:
        del runtime_request, settings
        self.calls.append("prepare_assets")

    def load_model(self, *, runtime_request: object, settings: object) -> object:
        del runtime_request, settings
        self.calls.append("load_model")
        return object()

    def transcribe(
        self,
        *,
        model: object,
        runtime_request: object,
        file_path: str,
        language: str,
        settings: object,
    ) -> list[object]:
        del model, runtime_request, file_path, language, settings
        self.calls.append("transcribe")
        return self.transcript_words


def _runtime_request() -> BackendRuntimeRequest:
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


def _worker_payload(*, profile: _Profile | None = None) -> TranscriptionProcessPayload:
    active_profile = profile or _Profile()
    return TranscriptionProcessPayload(
        file_path="sample.wav",
        language="en",
        profile=active_profile,
        runtime_request=_runtime_request(),
        settings=TranscriptionWorkerSettings(
            models=TranscriptionWorkerModelsConfig(
                whisper_download_root=Path("/tmp/cache"),
            )
        ),
    )


def test_should_use_process_isolated_path_only_for_faster_whisper() -> None:
    """Only faster-whisper profiles should use the spawned-worker path."""
    assert isolation.should_use_process_isolated_path(_Profile()) is True
    assert (
        isolation.should_use_process_isolated_path(_Profile(backend_id="stable_whisper")) is False
    )


def test_runtime_request_for_isolated_faster_whisper_uses_cuda_precision_candidates() -> None:
    """CUDA runtime request should preserve device and use legal precision fallbacks."""
    settings = cast(
        AppConfig,
        SimpleNamespace(torch_runtime=SimpleNamespace(device="cuda:1", dtype="auto")),
    )

    resolved = isolation.runtime_request_for_isolated_faster_whisper(
        profile=_Profile(),
        settings=settings,
        error_factory=RuntimeError,
        logger=logging.getLogger("ser.tests.process_isolation"),
    )

    assert resolved.device_spec == "cuda:1"
    assert resolved.device_type == "cuda"
    assert resolved.precision_candidates == ("float16", "float32")


def test_runtime_request_for_isolated_faster_whisper_falls_back_to_cpu_for_unsupported_device(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Unsupported devices should fall back to cpu/float32 with an info log."""
    settings = cast(
        AppConfig,
        SimpleNamespace(torch_runtime=SimpleNamespace(device="mps", dtype="bfloat16")),
    )

    with caplog.at_level(logging.INFO):
        resolved = isolation.runtime_request_for_isolated_faster_whisper(
            profile=_Profile(),
            settings=settings,
            error_factory=RuntimeError,
            logger=logging.getLogger("ser.tests.process_isolation"),
        )

    assert resolved.device_spec == "cpu"
    assert resolved.device_type == "cpu"
    assert resolved.precision_candidates == ("float32",)
    assert "is unsupported; using cpu/float32" in caplog.text


def test_recv_worker_message_wraps_eof_errors() -> None:
    """EOF during worker recv should become a domain-specific runtime error."""
    with pytest.raises(RuntimeError, match="exited before sending setup payload"):
        isolation.recv_worker_message(
            _Connection(),
            stage="setup",
            error_factory=RuntimeError,
        )


def test_recv_worker_message_rejects_malformed_payload() -> None:
    """Non-tuple worker payloads should be rejected immediately."""
    with pytest.raises(RuntimeError, match="malformed payload"):
        isolation.recv_worker_message(
            _Connection(received=["bad-payload"]),
            stage="setup",
            error_factory=RuntimeError,
        )


def test_raise_worker_error_formats_known_worker_error_payload() -> None:
    """Error payloads should be rethrown with stage and type context."""
    with pytest.raises(
        RuntimeError,
        match="Transcription worker transcription failed with ValueError: bad input",
    ):
        isolation.raise_worker_error(
            ("err", "transcription", "ValueError", "bad input"),
            error_factory=RuntimeError,
        )


def test_raise_worker_error_rejects_unexpected_payload_shape() -> None:
    """Unexpected worker payloads should surface a deterministic boundary error."""
    with pytest.raises(RuntimeError, match="unexpected payload shape"):
        isolation.raise_worker_error(
            ("phase", "setup_complete"),
            error_factory=RuntimeError,
        )


def test_terminate_worker_process_kills_after_grace_timeout() -> None:
    """Termination helper should escalate to kill when the worker stays alive."""
    process = _Process(alive_after_join=True)

    isolation.terminate_worker_process(
        process,
        terminate_grace_seconds=0.1,
        kill_grace_seconds=0.2,
    )

    assert process.terminated is True
    assert process.killed is True
    assert process.join_calls == [0.1, 0.2]


def test_transcription_worker_entry_reports_success_and_serializes_words(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Worker entry should emit phase notifications and a serialized success payload."""
    adapter = _Adapter()
    connection = _Connection()
    payload = _worker_payload()
    monkeypatch.setitem(sys.modules, "torch", ModuleType("torch"))

    isolation.transcription_worker_entry(
        payload,
        connection,
        settings_resolver=lambda: payload.settings,
        adapter_resolver=lambda backend_id: adapter,
    )

    assert adapter.calls == ["setup_required", "prepare_assets", "load_model", "transcribe"]
    assert connection.sent == [
        ("phase", "setup_complete"),
        ("phase", "model_loaded"),
        ("ok", [("hello", 0.0, 0.5)]),
    ]
    assert connection.closed is True


def test_transcription_worker_entry_maps_non_serializable_words_to_error_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Worker entry should translate invalid transcript-word payloads into worker errors."""
    adapter = _Adapter(
        transcript_words=[SimpleNamespace(word="hello", start_seconds="bad", end_seconds=0.5)]
    )
    connection = _Connection()
    payload = _worker_payload()
    monkeypatch.setitem(sys.modules, "torch", ModuleType("torch"))

    isolation.transcription_worker_entry(
        payload,
        connection,
        settings_resolver=lambda: payload.settings,
        adapter_resolver=lambda backend_id: adapter,
    )

    assert connection.sent[-1] == (
        "err",
        "transcription",
        "TypeError",
        "Transcription worker produced non-numeric timestamps at index 0.",
    )
    assert connection.closed is True


def test_run_faster_whisper_process_isolated_returns_validated_transcript_words(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Parent orchestration should deserialize worker success payloads into TranscriptWord rows."""
    phase_events: list[tuple[str, str]] = []

    def _log_started(_logger: object, *, phase_name: str) -> float:
        phase_events.append(("start", phase_name))
        return 1.0

    def _log_completed(
        _logger: object,
        *,
        phase_name: str,
        started_at: float,
    ) -> None:
        del started_at
        phase_events.append(("complete", phase_name))

    def _log_failed(
        _logger: object,
        *,
        phase_name: str,
        started_at: float,
    ) -> None:
        del started_at
        phase_events.append(("failed", phase_name))

    monkeypatch.setattr(
        isolation,
        "log_phase_started",
        _log_started,
    )
    monkeypatch.setattr(
        isolation,
        "log_phase_completed",
        _log_completed,
    )
    monkeypatch.setattr(
        isolation,
        "log_phase_failed",
        _log_failed,
    )
    parent_connection = _Connection(
        received=[
            ("phase", "setup_complete"),
            ("phase", "model_loaded"),
            ("ok", [("hello", 0.0, 0.5), ("world", 0.5, 1.0)]),
        ]
    )
    child_connection = _Connection()
    process = _Process()
    context = _SpawnContext(
        parent_connection=parent_connection,
        child_connection=child_connection,
        process=process,
    )
    settings = cast(
        AppConfig, SimpleNamespace(torch_runtime=SimpleNamespace(device="cpu", dtype="auto"))
    )
    profile = _Profile()

    resolved = isolation.run_faster_whisper_process_isolated(
        file_path="sample.wav",
        language="en",
        profile=profile,
        settings_resolver=lambda: settings,
        runtime_request_resolver=lambda _profile, _settings: _runtime_request(),
        payload_factory=lambda **kwargs: kwargs,
        get_spawn_context=lambda: context,
        worker_entry=lambda _payload, _connection: None,
        recv_worker_message_fn=lambda connection, *, stage: isolation.recv_worker_message(
            connection,
            stage=stage,
            error_factory=RuntimeError,
        ),
        raise_worker_error_fn=lambda message: isolation.raise_worker_error(
            message,
            error_factory=RuntimeError,
        ),
        terminate_worker_process_fn=lambda _process: None,
        transcript_word_factory=TranscriptWord,
        logger=logging.getLogger("ser.tests.process_isolation"),
        error_factory=RuntimeError,
        terminate_grace_seconds=0.1,
    )

    assert resolved == [
        TranscriptWord("hello", 0.0, 0.5),
        TranscriptWord("world", 0.5, 1.0),
    ]
    assert context.pipe_duplex is False
    assert process.started is True
    assert child_connection.closed is True
    assert parent_connection.closed is True
    assert process.closed is True
    assert phase_events == [
        ("start", "transcription_setup"),
        ("complete", "transcription_setup"),
        ("start", "transcription_model_load"),
        ("complete", "transcription_model_load"),
        ("start", "transcription"),
        ("complete", "transcription"),
    ]


def test_run_faster_whisper_process_isolated_rejects_malformed_transcript_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Parent orchestration should fail cleanly on malformed worker success payloads."""
    phase_events: list[tuple[str, str]] = []

    def _log_started(_logger: object, *, phase_name: str) -> float:
        phase_events.append(("start", phase_name))
        return 1.0

    def _log_completed(
        _logger: object,
        *,
        phase_name: str,
        started_at: float,
    ) -> None:
        del started_at
        phase_events.append(("complete", phase_name))

    def _log_failed(
        _logger: object,
        *,
        phase_name: str,
        started_at: float,
    ) -> None:
        del started_at
        phase_events.append(("failed", phase_name))

    monkeypatch.setattr(
        isolation,
        "log_phase_started",
        _log_started,
    )
    monkeypatch.setattr(
        isolation,
        "log_phase_completed",
        _log_completed,
    )
    monkeypatch.setattr(
        isolation,
        "log_phase_failed",
        _log_failed,
    )
    context = _SpawnContext(
        parent_connection=_Connection(
            received=[
                ("phase", "setup_complete"),
                ("phase", "model_loaded"),
                ("ok", [("hello", "bad", 0.5)]),
            ]
        ),
        child_connection=_Connection(),
        process=_Process(),
    )
    settings = cast(
        AppConfig, SimpleNamespace(torch_runtime=SimpleNamespace(device="cpu", dtype="auto"))
    )

    with pytest.raises(
        RuntimeError,
        match="non-numeric transcript timestamps at index 0",
    ):
        isolation.run_faster_whisper_process_isolated(
            file_path="sample.wav",
            language="en",
            profile=_Profile(),
            settings_resolver=lambda: settings,
            runtime_request_resolver=lambda _profile, _settings: _runtime_request(),
            payload_factory=lambda **kwargs: kwargs,
            get_spawn_context=lambda: context,
            worker_entry=lambda _payload, _connection: None,
            recv_worker_message_fn=lambda connection, *, stage: isolation.recv_worker_message(
                connection,
                stage=stage,
                error_factory=RuntimeError,
            ),
            raise_worker_error_fn=lambda message: isolation.raise_worker_error(
                message,
                error_factory=RuntimeError,
            ),
            terminate_worker_process_fn=lambda _process: None,
            transcript_word_factory=TranscriptWord,
            logger=logging.getLogger("ser.tests.process_isolation"),
            error_factory=RuntimeError,
            terminate_grace_seconds=0.1,
        )

    assert ("failed", "transcription") in phase_events
