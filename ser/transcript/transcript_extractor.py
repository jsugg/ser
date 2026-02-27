"""Backend-routed transcript extraction with word-level timestamps."""

from __future__ import annotations

import gc
import logging
import multiprocessing as mp
import sys
from dataclasses import dataclass
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from types import ModuleType
from typing import TYPE_CHECKING, Literal, Never, Protocol, cast

from ser.config import AppConfig, get_settings
from ser.domain import TranscriptWord
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
from ser.transcript.backends import (
    BackendRuntimeRequest,
    CompatibilityReport,
    resolve_transcription_backend_adapter,
)
from ser.transcript.runtime_policy import (
    DEFAULT_MPS_LOW_MEMORY_THRESHOLD_GB,
    resolve_transcription_runtime_policy,
)
from ser.utils.logger import get_logger

if TYPE_CHECKING:
    from stable_whisper.result import WhisperResult

logger: logging.Logger = get_logger(__name__)
_TERMINATE_GRACE_SECONDS = 5.0
_KILL_GRACE_SECONDS = 2.0


class TranscriptionError(RuntimeError):
    """Raised when transcript extraction fails for operational reasons."""


class WhisperWord(Protocol):
    """Protocol for stable-whisper word-level transcript entries."""

    word: str
    start: float | None
    end: float | None


@dataclass(frozen=True)
class TranscriptionProfile:
    """Runtime profile settings used by transcription backends."""

    backend_id: TranscriptionBackendId = "stable_whisper"
    model_name: str = "large-v2"
    use_demucs: bool = True
    use_vad: bool = True


@dataclass(frozen=True)
class _TranscriptionProcessPayload:
    """Serializable payload for one process-isolated transcription attempt."""

    file_path: str
    language: str
    profile: TranscriptionProfile


type _WorkerPhase = Literal["setup_complete", "model_loaded"]
type _WorkerPhaseMessage = tuple[Literal["phase"], _WorkerPhase]
type _WorkerSuccessMessage = tuple[Literal["ok"], list[tuple[str, float, float]]]
type _WorkerErrorMessage = tuple[Literal["err"], str, str, str]
type _WorkerMessage = _WorkerPhaseMessage | _WorkerSuccessMessage | _WorkerErrorMessage


def _resolve_backend_id(raw_backend_id: object) -> TranscriptionBackendId:
    """Normalizes one backend id and validates supported values."""
    if raw_backend_id in {"stable_whisper", "faster_whisper"}:
        return cast(TranscriptionBackendId, raw_backend_id)
    raise TranscriptionError(
        "Unsupported transcription backend id configured. "
        "Expected 'stable_whisper' or 'faster_whisper'."
    )


def resolve_transcription_profile(
    profile: TranscriptionProfile | None = None,
) -> TranscriptionProfile:
    """Resolves profile overrides or falls back to configured defaults."""
    if profile is not None:
        return profile
    settings: AppConfig = get_settings()
    return TranscriptionProfile(
        backend_id=_resolve_backend_id(settings.transcription.backend_id),
        model_name=settings.models.whisper_model.name,
        use_demucs=settings.transcription.use_demucs,
        use_vad=settings.transcription.use_vad,
    )


def _runtime_request_from_profile(
    active_profile: TranscriptionProfile,
    settings: AppConfig,
) -> BackendRuntimeRequest:
    """Builds one backend runtime request from transcription profile settings."""
    torch_runtime = getattr(settings, "torch_runtime", None)
    transcription_settings = getattr(settings, "transcription", None)
    requested_device = getattr(torch_runtime, "device", "cpu")
    requested_dtype = getattr(torch_runtime, "dtype", "auto")
    requested_mps_low_memory_threshold = getattr(
        transcription_settings,
        "mps_low_memory_threshold_gb",
        DEFAULT_MPS_LOW_MEMORY_THRESHOLD_GB,
    )
    runtime_policy = resolve_transcription_runtime_policy(
        backend_id=active_profile.backend_id,
        requested_device=(
            requested_device if isinstance(requested_device, str) else "cpu"
        ),
        requested_dtype=requested_dtype if isinstance(requested_dtype, str) else "auto",
        mps_low_memory_threshold_gb=(
            requested_mps_low_memory_threshold
            if (
                isinstance(requested_mps_low_memory_threshold, int | float)
                and not isinstance(requested_mps_low_memory_threshold, bool)
            )
            else DEFAULT_MPS_LOW_MEMORY_THRESHOLD_GB
        ),
    )
    return BackendRuntimeRequest(
        model_name=active_profile.model_name,
        use_demucs=active_profile.use_demucs,
        use_vad=active_profile.use_vad,
        device_spec=runtime_policy.device_spec,
        device_type=runtime_policy.device_type,
        precision_candidates=runtime_policy.precision_candidates,
        memory_tier=runtime_policy.memory_tier,
    )


def _check_adapter_compatibility(
    *,
    active_profile: TranscriptionProfile,
    settings: AppConfig,
    runtime_request: BackendRuntimeRequest | None = None,
) -> CompatibilityReport:
    """Validates backend compatibility and logs non-blocking compatibility issues."""
    adapter = resolve_transcription_backend_adapter(active_profile.backend_id)
    resolved_runtime_request = (
        _runtime_request_from_profile(active_profile, settings)
        if runtime_request is None
        else runtime_request
    )
    report = adapter.check_compatibility(
        runtime_request=resolved_runtime_request,
        settings=settings,
    )
    if report.noise_issues:
        for issue in report.noise_issues:
            logger.debug(
                "Transcription backend '%s' noise issue [%s]: %s",
                active_profile.backend_id,
                issue.code,
                issue.message,
            )
    if report.operational_issues:
        for issue in report.operational_issues:
            logger.warning(
                "Transcription backend '%s' operational issue [%s]: %s",
                active_profile.backend_id,
                issue.code,
                issue.message,
            )
    if report.has_blocking_issues:
        details = (
            "; ".join(issue.message for issue in report.functional_issues)
            or "backend compatibility validation failed"
        )
        raise TranscriptionError(details)
    return report


def _transcription_setup_required(
    *,
    active_profile: TranscriptionProfile,
    settings: AppConfig,
) -> bool:
    """Returns whether a setup/download phase is needed before model load."""
    runtime_request = _runtime_request_from_profile(active_profile, settings)
    _check_adapter_compatibility(
        active_profile=active_profile,
        settings=settings,
        runtime_request=runtime_request,
    )
    adapter = resolve_transcription_backend_adapter(active_profile.backend_id)
    return adapter.setup_required(
        runtime_request=runtime_request,
        settings=settings,
    )


def _prepare_transcription_assets(
    *,
    active_profile: TranscriptionProfile,
    settings: AppConfig,
) -> None:
    """Ensures required stable-whisper model assets are present locally."""
    runtime_request = _runtime_request_from_profile(active_profile, settings)
    _check_adapter_compatibility(
        active_profile=active_profile,
        settings=settings,
        runtime_request=runtime_request,
    )
    adapter = resolve_transcription_backend_adapter(active_profile.backend_id)
    adapter.prepare_assets(
        runtime_request=runtime_request,
        settings=settings,
    )


def load_whisper_model(profile: TranscriptionProfile | None = None) -> object:
    """Loads the configured transcription model for resolved runtime settings.

    Returns:
        The loaded Whisper model instance.
    """
    settings: AppConfig = get_settings()
    active_profile: TranscriptionProfile = resolve_transcription_profile(profile)
    runtime_request = _runtime_request_from_profile(active_profile, settings)
    try:
        _check_adapter_compatibility(
            active_profile=active_profile,
            settings=settings,
            runtime_request=runtime_request,
        )
        logger.info(
            "Transcription runtime resolved (backend=%s, device=%s, precision=%s, memory_tier=%s).",
            active_profile.backend_id,
            runtime_request.device_spec,
            ",".join(runtime_request.precision_candidates),
            runtime_request.memory_tier,
        )
        adapter = resolve_transcription_backend_adapter(active_profile.backend_id)
        return adapter.load_model(
            runtime_request=runtime_request,
            settings=settings,
        )
    except Exception as err:
        logger.error(msg=f"Failed to load transcription model: {err}", exc_info=True)
        raise TranscriptionError("Failed to load transcription model.") from err


def _should_use_process_isolated_path(profile: TranscriptionProfile) -> bool:
    """Returns whether one transcription profile should execute in a worker process."""
    return profile.backend_id == "faster_whisper"


def _runtime_request_for_isolated_faster_whisper(
    *,
    profile: TranscriptionProfile,
    settings: AppConfig,
) -> BackendRuntimeRequest:
    """Builds one faster-whisper runtime request without importing torch in worker."""
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
    return BackendRuntimeRequest(
        model_name=profile.model_name,
        use_demucs=profile.use_demucs,
        use_vad=profile.use_vad,
        device_spec="cpu",
        device_type="cpu",
        precision_candidates=("float32",),
        memory_tier="not_applicable",
    )


def _run_faster_whisper_process_isolated(
    *,
    file_path: str,
    language: str,
    profile: TranscriptionProfile,
) -> list[TranscriptWord]:
    """Runs faster-whisper setup/load/transcribe inside one spawned worker process."""
    payload = _TranscriptionProcessPayload(
        file_path=file_path,
        language=language,
        profile=profile,
    )
    context = mp.get_context("spawn")
    parent_conn, child_conn = context.Pipe(duplex=False)
    process = context.Process(
        target=_transcription_worker_entry,
        args=(payload, child_conn),
        daemon=False,
    )
    setup_started_at = log_phase_started(
        logger,
        phase_name=PHASE_TRANSCRIPTION_SETUP,
    )
    model_load_started_at: float | None = None
    transcription_started_at: float | None = None
    process.start()
    child_conn.close()
    try:
        setup_message = _recv_worker_message(parent_conn, stage="setup")
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
            _raise_worker_error(setup_message)

        model_load_message = _recv_worker_message(parent_conn, stage="model_load")
        if model_load_message == ("phase", "model_loaded"):
            if model_load_started_at is None:
                raise TranscriptionError(
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
            _raise_worker_error(model_load_message)

        completion_message = _recv_worker_message(parent_conn, stage="transcription")
        if (
            isinstance(completion_message, tuple)
            and len(completion_message) == 2
            and completion_message[0] == "ok"
            and isinstance(completion_message[1], list)
        ):
            serialized_words = completion_message[1]
            if transcription_started_at is None:
                raise TranscriptionError(
                    "Transcription worker completed transcription before phase timer start."
                )
            log_phase_completed(
                logger,
                phase_name=PHASE_TRANSCRIPTION,
                started_at=transcription_started_at,
            )
            return [
                TranscriptWord(word, start_seconds, end_seconds)
                for word, start_seconds, end_seconds in serialized_words
            ]
        _raise_worker_error(completion_message)
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
        process.join(timeout=_TERMINATE_GRACE_SECONDS)
        if process.is_alive():
            _terminate_worker_process(process)
        process.close()


def _recv_worker_message(connection: Connection, *, stage: str) -> _WorkerMessage:
    """Receives one worker message and validates tuple envelope shape."""
    try:
        raw_message = connection.recv()
    except EOFError as err:
        raise TranscriptionError(
            f"Transcription worker exited before sending {stage} payload."
        ) from err
    if not isinstance(raw_message, tuple) or not raw_message:
        raise TranscriptionError("Transcription worker returned malformed payload.")
    return cast(_WorkerMessage, raw_message)


def _raise_worker_error(message: _WorkerMessage) -> Never:
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
        raise TranscriptionError(
            f"Transcription worker {stage} failed with {error_type}: {error_message}"
        )
    raise TranscriptionError("Transcription worker returned unexpected payload shape.")


def _terminate_worker_process(process: BaseProcess) -> None:
    """Terminates a worker process with kill fallback."""
    process.terminate()
    process.join(timeout=_TERMINATE_GRACE_SECONDS)
    if process.is_alive():
        process.kill()
        process.join(timeout=_KILL_GRACE_SECONDS)


def _transcription_worker_entry(
    payload: _TranscriptionProcessPayload,
    connection: Connection,
) -> None:
    """Executes faster-whisper transcription inside one isolated worker process."""
    stage = "setup"
    try:
        if payload.profile.backend_id != "faster_whisper":
            raise RuntimeError(
                f"Unsupported process-isolated backend: {payload.profile.backend_id!r}."
            )
        # Prevent ctranslate2 from importing torch/functorch in this worker.
        sys.modules["torch"] = cast(ModuleType, None)
        settings = get_settings()
        runtime_request = _runtime_request_for_isolated_faster_whisper(
            profile=payload.profile,
            settings=settings,
        )
        adapter = resolve_transcription_backend_adapter(payload.profile.backend_id)
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
            (word.word, word.start_seconds, word.end_seconds)
            for word in transcript_words
        ]
        connection.send(("ok", serialized_words))
    except BaseException as err:
        connection.send(("err", stage, type(err).__name__, str(err)))
    finally:
        connection.close()


def extract_transcript(
    file_path: str,
    language: str | None = None,
    profile: TranscriptionProfile | None = None,
) -> list[TranscriptWord]:
    """Extracts a transcript with per-word timing for an input audio file.

    Args:
        file_path: Path to the audio file.
        language: Language code used by Whisper during transcription.
        profile: Optional runtime overrides for model and preprocessing toggles.

    Returns:
        A list of transcript word entries with timing metadata.
    """
    active_language: str = language or get_settings().default_language
    return _extract_transcript(file_path, active_language, profile)


def _extract_transcript(
    file_path: str,
    language: str,
    profile: TranscriptionProfile | None = None,
) -> list[TranscriptWord]:
    """Internal transcript workflow with backend-specific execution strategy."""
    active_profile = resolve_transcription_profile(profile)
    if _should_use_process_isolated_path(active_profile):
        return _run_faster_whisper_process_isolated(
            file_path=file_path,
            language=language,
            profile=active_profile,
        )
    return _extract_transcript_in_process(
        file_path=file_path,
        language=language,
        profile=active_profile,
    )


def _extract_transcript_in_process(
    *,
    file_path: str,
    language: str,
    profile: TranscriptionProfile,
) -> list[TranscriptWord]:
    """Runs one in-process transcript workflow with phase-aware logging."""
    settings: AppConfig = get_settings()
    active_profile = profile
    model: object | None = None

    if _transcription_setup_required(
        active_profile=active_profile,
        settings=settings,
    ):
        setup_started_at = log_phase_started(
            logger,
            phase_name=PHASE_TRANSCRIPTION_SETUP,
        )
        try:
            _prepare_transcription_assets(
                active_profile=active_profile,
                settings=settings,
            )
        except Exception:
            log_phase_failed(
                logger,
                phase_name=PHASE_TRANSCRIPTION_SETUP,
                started_at=setup_started_at,
            )
            raise
        log_phase_completed(
            logger,
            phase_name=PHASE_TRANSCRIPTION_SETUP,
            started_at=setup_started_at,
        )

    model_load_started_at = log_phase_started(
        logger,
        phase_name=PHASE_TRANSCRIPTION_MODEL_LOAD,
    )
    try:
        model = load_whisper_model(active_profile)
    except Exception:
        log_phase_failed(
            logger,
            phase_name=PHASE_TRANSCRIPTION_MODEL_LOAD,
            started_at=model_load_started_at,
        )
        raise
    log_phase_completed(
        logger,
        phase_name=PHASE_TRANSCRIPTION_MODEL_LOAD,
        started_at=model_load_started_at,
    )

    transcription_started_at = log_phase_started(
        logger,
        phase_name=PHASE_TRANSCRIPTION,
    )
    try:
        try:
            transcript_words = _transcribe_file_with_profile(
                model,
                language,
                file_path,
                active_profile,
            )
        except Exception:
            log_phase_failed(
                logger,
                phase_name=PHASE_TRANSCRIPTION,
                started_at=transcription_started_at,
            )
            raise
        log_phase_completed(
            logger,
            phase_name=PHASE_TRANSCRIPTION,
            started_at=transcription_started_at,
        )
        if not transcript_words:
            logger.info(msg="Transcript extraction succeeded but returned no words.")
        logger.debug(msg="Transcript output formatted successfully.")
        return transcript_words
    finally:
        _release_transcription_runtime_memory(model=model)


def _release_transcription_runtime_memory(*, model: object | None) -> None:
    """Releases best-effort Torch runtime memory after one in-process transcript run."""
    del model
    gc.collect()
    torch_module = sys.modules.get("torch")
    if not isinstance(torch_module, ModuleType):
        return
    mps_module = getattr(torch_module, "mps", None)
    if isinstance(mps_module, ModuleType):
        is_available = getattr(mps_module, "is_available", None)
        empty_cache = getattr(mps_module, "empty_cache", None)
        try:
            if callable(is_available) and is_available() and callable(empty_cache):
                empty_cache()
        except Exception:
            logger.debug(
                "Ignored failure while emptying torch MPS cache after transcription.",
                exc_info=True,
            )
    cuda_module = getattr(torch_module, "cuda", None)
    if isinstance(cuda_module, ModuleType):
        is_available = getattr(cuda_module, "is_available", None)
        empty_cache = getattr(cuda_module, "empty_cache", None)
        try:
            if callable(is_available) and is_available() and callable(empty_cache):
                empty_cache()
        except Exception:
            logger.debug(
                "Ignored failure while emptying torch CUDA cache after transcription.",
                exc_info=True,
            )


def __transcribe_file(
    model: object, language: str, file_path: str
) -> list[TranscriptWord]:
    """Runs a Whisper transcription call and normalizes return types."""
    return _transcribe_file_with_profile(model, language, file_path, profile=None)


def _transcribe_file_with_profile(
    model: object,
    language: str,
    file_path: str,
    profile: TranscriptionProfile | None,
) -> list[TranscriptWord]:
    """Runs a Whisper transcription call using an explicit runtime profile."""
    settings: AppConfig = get_settings()
    active_profile: TranscriptionProfile = resolve_transcription_profile(profile)
    runtime_request = _runtime_request_from_profile(active_profile, settings)
    _check_adapter_compatibility(
        active_profile=active_profile,
        settings=settings,
        runtime_request=runtime_request,
    )
    adapter = resolve_transcription_backend_adapter(active_profile.backend_id)
    try:
        return adapter.transcribe(
            model=model,
            runtime_request=runtime_request,
            file_path=file_path,
            language=language,
            settings=settings,
        )
    except TranscriptionError:
        raise
    except Exception as err:
        logger.error(msg=f"Error processing speech extraction: {err}", exc_info=True)
        raise TranscriptionError("Failed to transcribe audio.") from err


def transcribe_with_model(
    model: object,
    file_path: str,
    language: str,
    profile: TranscriptionProfile | None = None,
) -> list[TranscriptWord]:
    """Transcribes one file with a pre-loaded model for profiling workloads."""
    return _transcribe_file_with_profile(model, language, file_path, profile=profile)


def format_transcript(result: WhisperResult) -> list[TranscriptWord]:
    """Formats a Whisper result object into a word-level timestamp list.

    Args:
        result: Whisper transcription result.

    Returns:
        A list of transcript word entries with timing metadata.
    """
    try:
        words = cast(list[WhisperWord], result.all_words())
    except AttributeError as err:
        logger.error(msg=f"Error extracting words from result: {err}", exc_info=True)
        raise TranscriptionError("Invalid Whisper result object.") from err

    text_with_timestamps: list[TranscriptWord] = [
        TranscriptWord(
            word=str(word.word),
            start_seconds=float(word.start),
            end_seconds=float(word.end),
        )
        for word in words
        if word.start is not None and word.end is not None
    ]
    return text_with_timestamps
