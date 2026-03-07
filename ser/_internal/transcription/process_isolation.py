"""Process-isolated transcription orchestration helpers."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Literal, Never, Protocol, TypeVar, cast

from ser.config import AppConfig
from ser.profiles import TranscriptionBackendId
from ser.runtime.phase_contract import (
    PHASE_TRANSCRIPTION,
    PHASE_TRANSCRIPTION_MODEL_LOAD,
    PHASE_TRANSCRIPTION_SETUP,
)
from ser.runtime.phase_timing import (
    log_phase_completed,
    log_phase_failed,
    log_phase_started,
)
from ser.transcript.backends import BackendRuntimeRequest

type _WorkerPhase = Literal["setup_complete", "model_loaded"]
type WorkerPhaseMessage = tuple[Literal["phase"], _WorkerPhase]
type WorkerSuccessMessage = tuple[Literal["ok"], list[tuple[str, float, float]]]
type WorkerErrorMessage = tuple[Literal["err"], str, str, str]
type WorkerMessage = WorkerPhaseMessage | WorkerSuccessMessage | WorkerErrorMessage


class _ProcessIsolatedProfile(Protocol):
    """Minimal profile contract for process-isolated transcription orchestration."""

    @property
    def backend_id(self) -> TranscriptionBackendId:
        """Returns backend identifier used for process-isolated routing."""
        ...

    @property
    def model_name(self) -> str:
        """Returns backend model identifier."""
        ...

    @property
    def use_demucs(self) -> bool:
        """Returns whether Demucs preprocessing is enabled."""
        ...

    @property
    def use_vad(self) -> bool:
        """Returns whether VAD preprocessing is enabled."""
        ...


class _ProcessPayload(Protocol):
    """Minimal worker payload contract for process-isolated execution."""

    @property
    def file_path(self) -> str:
        """Returns input file path."""
        ...

    @property
    def language(self) -> str:
        """Returns transcription language code."""
        ...

    @property
    def profile(self) -> _ProcessIsolatedProfile:
        """Returns profile metadata for worker backend resolution."""
        ...

    @property
    def runtime_request(self) -> BackendRuntimeRequest:
        """Returns pre-resolved runtime request for worker execution."""
        ...

    @property
    def settings(self) -> "_WorkerSettings":
        """Returns serializable worker settings required by backend adapters."""
        ...


class _WorkerModelsConfig(Protocol):
    """Minimal model settings required inside process-isolated workers."""

    @property
    def whisper_download_root(self) -> Path:
        """Returns download/cache root for Whisper assets."""
        ...


class _WorkerSettings(Protocol):
    """Minimal settings snapshot required inside process-isolated workers."""

    @property
    def models(self) -> _WorkerModelsConfig:
        """Returns model-specific settings required by worker adapters."""
        ...


class _ParentConnection(Protocol):
    """Parent-process connection contract used by worker message loop."""

    def recv(self) -> object:
        """Receives one worker message payload."""
        ...

    def close(self) -> None:
        """Closes parent pipe endpoint."""
        ...


class _ChildConnection(Protocol):
    """Child-process connection contract for worker process entry."""

    def send(self, obj: object) -> None:
        """Sends one worker message payload."""
        ...

    def close(self) -> None:
        """Closes child pipe endpoint."""
        ...


class _WorkerProcess(Protocol):
    """Minimal worker process contract used for lifecycle management."""

    def start(self) -> None:
        """Starts the worker process."""
        ...

    def join(self, timeout: float | None = None) -> None:
        """Waits for process termination up to timeout."""
        ...

    def is_alive(self) -> bool:
        """Returns whether worker is still running."""
        ...

    def terminate(self) -> None:
        """Requests graceful worker termination."""
        ...

    def kill(self) -> None:
        """Force-kills worker process."""
        ...

    def close(self) -> None:
        """Releases process resources."""
        ...


class _SpawnContext(Protocol):
    """Spawn context contract used to create worker pipes and processes."""

    def Pipe(self, duplex: bool = False) -> tuple[_ParentConnection, _ChildConnection]:
        """Creates one unidirectional parent/child connection pair."""
        ...

    def Process(
        self,
        *,
        target: Callable[..., None],
        args: tuple[object, ...],
        daemon: bool,
    ) -> _WorkerProcess:
        """Creates one worker process with the given target and args."""
        ...


type _ErrorFactory = Callable[[str], Exception]
type _SpawnContextResolver = Callable[[], object]
type _PayloadFactory = Callable[..., object]
type _WorkerEntry = Callable[[object, object], None]
type _RuntimeSettingsResolver = Callable[[], AppConfig]
type _WorkerSettingsResolver = Callable[[], _WorkerSettings]
type _TerminateWorkerProcess = Callable[[_WorkerProcess], None]

_TProfile = TypeVar("_TProfile", bound=_ProcessIsolatedProfile)
_TTranscriptWord = TypeVar("_TTranscriptWord")


class _TranscriptWordLike(Protocol):
    """Minimal transcript-word contract used for worker serialization."""

    @property
    def word(self) -> str:
        """Returns transcript token text."""
        ...

    @property
    def start_seconds(self) -> float:
        """Returns token start timestamp in seconds."""
        ...

    @property
    def end_seconds(self) -> float:
        """Returns token end timestamp in seconds."""
        ...


class _TranscriptionAdapter(Protocol):
    """Minimal backend adapter contract required by worker execution."""

    def setup_required(
        self,
        *,
        runtime_request: BackendRuntimeRequest,
        settings: _WorkerSettings,
    ) -> bool:
        """Returns whether setup is needed before model load."""
        ...

    def prepare_assets(
        self,
        *,
        runtime_request: BackendRuntimeRequest,
        settings: _WorkerSettings,
    ) -> None:
        """Prepares backend assets before model load."""
        ...

    def load_model(
        self,
        *,
        runtime_request: BackendRuntimeRequest,
        settings: _WorkerSettings,
    ) -> object:
        """Loads and returns backend model handle."""
        ...

    def transcribe(
        self,
        *,
        model: object,
        runtime_request: BackendRuntimeRequest,
        file_path: str,
        language: str,
        settings: _WorkerSettings,
    ) -> list[object]:
        """Runs backend transcription and returns transcript words."""
        ...


type _AdapterResolver = Callable[[TranscriptionBackendId], object]


def should_use_process_isolated_path(profile: _ProcessIsolatedProfile) -> bool:
    """Returns whether one transcription profile should use worker-process isolation."""
    return profile.backend_id == "faster_whisper"


def runtime_request_for_isolated_faster_whisper(
    *,
    profile: _ProcessIsolatedProfile,
    settings: AppConfig,
    error_factory: _ErrorFactory,
    logger: logging.Logger,
) -> BackendRuntimeRequest:
    """Builds one faster-whisper runtime request without importing torch in worker."""
    if profile.backend_id != "faster_whisper":
        raise error_factory(
            "Process-isolated runtime request only supports faster-whisper backend."
        )
    torch_runtime = getattr(settings, "torch_runtime", None)
    requested_device = getattr(torch_runtime, "device", "cpu")
    requested_dtype = getattr(torch_runtime, "dtype", "auto")
    device_text = (
        requested_device.strip().lower() if isinstance(requested_device, str) else "cpu"
    )
    dtype_text = (
        requested_dtype.strip().lower() if isinstance(requested_dtype, str) else "auto"
    )
    if device_text.startswith("cuda"):
        device_spec = requested_device.strip() or "cuda"
        if dtype_text not in {"auto", "float16", "float32"} and dtype_text:
            logger.info(
                "Process-isolated faster-whisper runtime fallback: requested dtype '%s' "
                "is unsupported; using float16/float32 candidates.",
                dtype_text,
            )
        precision_candidates = (
            (dtype_text,)
            if dtype_text in {"float16", "float32"}
            else ("float16", "float32")
        )
        return BackendRuntimeRequest(
            model_name=profile.model_name,
            use_demucs=profile.use_demucs,
            use_vad=profile.use_vad,
            device_spec=device_spec,
            device_type="cuda",
            precision_candidates=precision_candidates,
            memory_tier="not_applicable",
        )
    if device_text not in {"", "auto", "cpu"}:
        logger.info(
            "Process-isolated faster-whisper runtime fallback: requested device '%s' "
            "is unsupported; using cpu/float32.",
            device_text,
        )
    return BackendRuntimeRequest(
        model_name=profile.model_name,
        use_demucs=profile.use_demucs,
        use_vad=profile.use_vad,
        device_spec="cpu",
        device_type="cpu",
        precision_candidates=("float32",),
        memory_tier="not_applicable",
    )


def recv_worker_message(
    connection: _ParentConnection,
    *,
    stage: str,
    error_factory: _ErrorFactory,
) -> WorkerMessage:
    """Receives one worker message and validates tuple envelope shape."""
    try:
        raw_message = connection.recv()
    except EOFError as err:
        raise error_factory(
            f"Transcription worker exited before sending {stage} payload."
        ) from err
    if not isinstance(raw_message, tuple) or not raw_message:
        raise error_factory("Transcription worker returned malformed payload.")
    return cast(WorkerMessage, raw_message)


def raise_worker_error(
    message: WorkerMessage, *, error_factory: _ErrorFactory
) -> Never:
    """Raises one transcription-domain error from a worker payload."""
    if (
        isinstance(message, tuple)
        and len(message) == 4
        and message[0] == "err"
        and isinstance(message[1], str)
        and isinstance(message[2], str)
        and isinstance(message[3], str)
    ):
        stage, error_type, error_message = message[1], message[2], message[3]
        raise error_factory(
            f"Transcription worker {stage} failed with {error_type}: {error_message}"
        )
    raise error_factory("Transcription worker returned unexpected payload shape.")


def terminate_worker_process(
    process: _WorkerProcess,
    *,
    terminate_grace_seconds: float,
    kill_grace_seconds: float,
) -> None:
    """Terminates one worker process with kill fallback."""
    process.terminate()
    process.join(timeout=terminate_grace_seconds)
    if process.is_alive():
        process.kill()
        process.join(timeout=kill_grace_seconds)


def transcription_worker_entry(
    payload: _ProcessPayload,
    connection: _ChildConnection,
    *,
    settings_resolver: _WorkerSettingsResolver,
    adapter_resolver: _AdapterResolver,
) -> None:
    """Executes faster-whisper transcription inside one isolated worker process."""
    stage = "setup"
    try:
        if payload.profile.backend_id != "faster_whisper":
            raise RuntimeError(
                f"Unsupported process-isolated backend: {payload.profile.backend_id!r}."
            )
        import sys

        # Prevent ctranslate2 from importing torch/functorch in this worker.
        sys.modules["torch"] = cast(ModuleType, None)
        settings = settings_resolver()
        runtime_request = payload.runtime_request
        adapter = cast(
            _TranscriptionAdapter, adapter_resolver(payload.profile.backend_id)
        )
        if adapter.setup_required(runtime_request=runtime_request, settings=settings):
            adapter.prepare_assets(runtime_request=runtime_request, settings=settings)
        connection.send(("phase", "setup_complete"))
        stage = "model_load"
        model = adapter.load_model(runtime_request=runtime_request, settings=settings)
        connection.send(("phase", "model_loaded"))
        stage = "transcription"
        transcript_words = adapter.transcribe(
            model=model,
            runtime_request=runtime_request,
            file_path=payload.file_path,
            language=payload.language,
            settings=settings,
        )
        serialized_words = [
            (
                cast(_TranscriptWordLike, word).word,
                cast(_TranscriptWordLike, word).start_seconds,
                cast(_TranscriptWordLike, word).end_seconds,
            )
            for word in transcript_words
        ]
        connection.send(("ok", serialized_words))
    except BaseException as err:
        connection.send(("err", stage, type(err).__name__, str(err)))
    finally:
        connection.close()


def run_faster_whisper_process_isolated(
    *,
    file_path: str,
    language: str,
    profile: _TProfile,
    settings_resolver: _RuntimeSettingsResolver,
    runtime_request_resolver: Callable[[_TProfile, AppConfig], BackendRuntimeRequest],
    payload_factory: _PayloadFactory,
    get_spawn_context: _SpawnContextResolver,
    worker_entry: _WorkerEntry,
    recv_worker_message_fn: Callable[..., WorkerMessage],
    raise_worker_error_fn: Callable[..., Never],
    terminate_worker_process_fn: _TerminateWorkerProcess,
    transcript_word_factory: Callable[[str, float, float], _TTranscriptWord],
    logger: logging.Logger,
    error_factory: _ErrorFactory,
    terminate_grace_seconds: float,
) -> list[_TTranscriptWord]:
    """Runs faster-whisper setup/load/transcribe inside one spawned worker process."""
    if profile.backend_id != "faster_whisper":
        raise error_factory(
            "Process-isolated transcription only supports faster-whisper backend."
        )
    settings = settings_resolver()
    runtime_request = runtime_request_resolver(profile, settings)
    logger.debug(
        "Process-isolated faster-whisper runtime request resolved (device=%s, precision=%s).",
        runtime_request.device_spec,
        ",".join(runtime_request.precision_candidates),
    )
    payload = payload_factory(
        file_path=file_path,
        language=language,
        profile=profile,
        runtime_request=runtime_request,
        settings=settings,
    )
    context = cast(_SpawnContext, get_spawn_context())
    parent_conn, child_conn = context.Pipe(duplex=False)
    process = context.Process(
        target=worker_entry,
        args=(payload, child_conn),
        daemon=False,
    )
    setup_started_at = log_phase_started(logger, phase_name=PHASE_TRANSCRIPTION_SETUP)
    model_load_started_at: float | None = None
    transcription_started_at: float | None = None
    process.start()
    child_conn.close()
    try:
        setup_message = recv_worker_message_fn(parent_conn, stage="setup")
        if setup_message == ("phase", "setup_complete"):
            log_phase_completed(
                logger,
                phase_name=PHASE_TRANSCRIPTION_SETUP,
                started_at=setup_started_at,
            )
            model_load_started_at = log_phase_started(
                logger,
                phase_name=PHASE_TRANSCRIPTION_MODEL_LOAD,
            )
        else:
            raise_worker_error_fn(setup_message)

        model_load_message = recv_worker_message_fn(parent_conn, stage="model_load")
        if model_load_message == ("phase", "model_loaded"):
            if model_load_started_at is None:
                raise error_factory(
                    "Transcription worker completed model load before phase timer start."
                )
            log_phase_completed(
                logger,
                phase_name=PHASE_TRANSCRIPTION_MODEL_LOAD,
                started_at=model_load_started_at,
            )
            transcription_started_at = log_phase_started(
                logger,
                phase_name=PHASE_TRANSCRIPTION,
            )
        else:
            raise_worker_error_fn(model_load_message)

        completion_message = recv_worker_message_fn(parent_conn, stage="transcription")
        if (
            isinstance(completion_message, tuple)
            and len(completion_message) == 2
            and completion_message[0] == "ok"
            and isinstance(completion_message[1], list)
        ):
            serialized_words = completion_message[1]
            if transcription_started_at is None:
                raise error_factory(
                    "Transcription worker completed transcription before phase timer start."
                )
            log_phase_completed(
                logger,
                phase_name=PHASE_TRANSCRIPTION,
                started_at=transcription_started_at,
            )
            return [
                transcript_word_factory(word, start_seconds, end_seconds)
                for word, start_seconds, end_seconds in serialized_words
            ]
        raise_worker_error_fn(completion_message)
    except Exception:
        if transcription_started_at is not None:
            log_phase_failed(
                logger,
                phase_name=PHASE_TRANSCRIPTION,
                started_at=transcription_started_at,
            )
        elif model_load_started_at is not None:
            log_phase_failed(
                logger,
                phase_name=PHASE_TRANSCRIPTION_MODEL_LOAD,
                started_at=model_load_started_at,
            )
        else:
            log_phase_failed(
                logger,
                phase_name=PHASE_TRANSCRIPTION_SETUP,
                started_at=setup_started_at,
            )
        raise
    finally:
        parent_conn.close()
        process.join(timeout=terminate_grace_seconds)
        if process.is_alive():
            terminate_worker_process_fn(process)
        process.close()


__all__ = [
    "raise_worker_error",
    "recv_worker_message",
    "run_faster_whisper_process_isolated",
    "runtime_request_for_isolated_faster_whisper",
    "should_use_process_isolated_path",
    "terminate_worker_process",
    "transcription_worker_entry",
]
