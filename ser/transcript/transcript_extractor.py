"""Backend-routed transcript extraction with word-level timestamps."""

from __future__ import annotations

import importlib
import logging
import os
import warnings
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast
from urllib.parse import urlparse

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
from ser.utils.logger import (
    DependencyLogPolicy,
    get_logger,
    scoped_dependency_log_policy,
)

if TYPE_CHECKING:
    from stable_whisper.result import WhisperResult
    from whisper.model import Whisper

logger: logging.Logger = get_logger(__name__)
_NOISY_FASTER_WHISPER_POLICY = DependencyLogPolicy(
    logger_prefixes=frozenset({"faster_whisper"})
)


class TranscriptionError(RuntimeError):
    """Raised when transcript extraction fails for operational reasons."""


@contextmanager
def _demote_faster_whisper_info_logs() -> Iterator[None]:
    """Demotes faster-whisper INFO records to DEBUG for one transcription call."""
    with scoped_dependency_log_policy(
        policy=_NOISY_FASTER_WHISPER_POLICY,
        keep_demoted=True,
    ):
        yield


class WhisperWord(Protocol):
    """Protocol for stable-whisper word-level transcript entries."""

    word: str
    start: float | None
    end: float | None


class FasterWhisperWord(Protocol):
    """Protocol for faster-whisper word-level transcript entries."""

    word: str
    start: float | None
    end: float | None


class FasterWhisperSegment(Protocol):
    """Protocol for faster-whisper segment objects with word payloads."""

    words: list[FasterWhisperWord] | tuple[FasterWhisperWord, ...] | None


@dataclass(frozen=True)
class TranscriptionProfile:
    """Runtime profile settings used by transcription backends."""

    backend_id: TranscriptionBackendId = "stable_whisper"
    model_name: str = "large-v2"
    use_demucs: bool = True
    use_vad: bool = True


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


def _stable_whisper_download_target(
    *,
    active_profile: TranscriptionProfile,
    settings: AppConfig,
) -> Path | None:
    """Returns stable-whisper checkpoint path for registry-backed model ids."""
    if active_profile.backend_id != "stable_whisper":
        return None
    model_name = active_profile.model_name.strip()
    if not model_name:
        raise TranscriptionError("Transcription model name must be a non-empty string.")
    if Path(model_name).is_file():
        return None
    try:
        whisper_module = importlib.import_module("whisper")
    except ModuleNotFoundError:
        return None
    model_registry = getattr(whisper_module, "_MODELS", None)
    if not isinstance(model_registry, dict):
        return None
    model_url = model_registry.get(model_name)
    if not isinstance(model_url, str):
        return None
    filename = Path(urlparse(model_url).path).name
    if not filename:
        return None
    return settings.models.whisper_download_root / filename


def _faster_whisper_setup_required(
    *,
    active_profile: TranscriptionProfile,
    settings: AppConfig,
) -> bool:
    """Returns whether faster-whisper assets are missing from local cache."""
    if active_profile.backend_id != "faster_whisper":
        return False
    model_name = active_profile.model_name.strip()
    if not model_name:
        raise TranscriptionError("Transcription model name must be a non-empty string.")
    if Path(model_name).is_dir():
        return False
    try:
        fw_utils = importlib.import_module("faster_whisper.utils")
    except ModuleNotFoundError:
        return False
    download_model = getattr(fw_utils, "download_model", None)
    if not callable(download_model):
        return False
    try:
        model_path = download_model(
            model_name,
            local_files_only=True,
            cache_dir=str(settings.models.whisper_download_root),
        )
    except Exception:
        return True
    if isinstance(model_path, str):
        return not Path(model_path).is_dir()
    path_getter = getattr(model_path, "__fspath__", None)
    if not callable(path_getter):
        return False
    resolved_path = path_getter()
    if not isinstance(resolved_path, str):
        return False
    return not Path(resolved_path).is_dir()


def _transcription_setup_required(
    *,
    active_profile: TranscriptionProfile,
    settings: AppConfig,
) -> bool:
    """Returns whether a setup/download phase is needed before model load."""
    stable_target_path = _stable_whisper_download_target(
        active_profile=active_profile,
        settings=settings,
    )
    if stable_target_path is not None:
        return not stable_target_path.is_file()
    return _faster_whisper_setup_required(
        active_profile=active_profile,
        settings=settings,
    )


def _prepare_transcription_assets(
    *,
    active_profile: TranscriptionProfile,
    settings: AppConfig,
) -> None:
    """Ensures required stable-whisper model assets are present locally."""
    target_path = _stable_whisper_download_target(
        active_profile=active_profile,
        settings=settings,
    )
    if target_path is None or target_path.is_file():
        if active_profile.backend_id != "faster_whisper":
            return
        model_name = active_profile.model_name.strip()
        if not model_name or Path(model_name).is_dir():
            return
        try:
            fw_utils = importlib.import_module("faster_whisper.utils")
        except ModuleNotFoundError:
            return
        download_model = getattr(fw_utils, "download_model", None)
        if not callable(download_model):
            return
        os.makedirs(settings.models.whisper_download_root, exist_ok=True)
        download_model(
            model_name,
            local_files_only=False,
            cache_dir=str(settings.models.whisper_download_root),
        )
        return
    try:
        whisper_module = importlib.import_module("whisper")
    except ModuleNotFoundError:
        return
    model_registry = getattr(whisper_module, "_MODELS", None)
    download_fn = getattr(whisper_module, "_download", None)
    if not isinstance(model_registry, dict) or not callable(download_fn):
        return
    model_url = model_registry.get(active_profile.model_name)
    if not isinstance(model_url, str):
        return
    os.makedirs(settings.models.whisper_download_root, exist_ok=True)
    download_fn(
        model_url,
        str(settings.models.whisper_download_root),
        False,
    )


def _load_stable_whisper_model(
    *,
    active_profile: TranscriptionProfile,
    settings: AppConfig,
) -> Whisper:
    """Loads one stable-whisper model for CPU inference."""
    try:
        import stable_whisper
    except ModuleNotFoundError as err:
        raise TranscriptionError(
            "Missing stable-whisper dependencies. Ensure project dependencies are installed."
        ) from err

    download_root: Path = settings.models.whisper_download_root
    torch_cache_root: Path = settings.models.torch_cache_root
    os.makedirs(download_root, exist_ok=True)
    os.makedirs(torch_cache_root, exist_ok=True)
    os.environ["TORCH_HOME"] = str(torch_cache_root)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", module="stable_whisper")
        model: Whisper = stable_whisper.load_model(
            name=active_profile.model_name,
            device="cpu",
            dq=False,
            download_root=str(download_root),
            in_memory=True,
        )
    return model


def _load_faster_whisper_model(
    *,
    active_profile: TranscriptionProfile,
    settings: AppConfig,
) -> object:
    """Loads one faster-whisper model for CPU inference."""
    try:
        faster_whisper_module = importlib.import_module("faster_whisper")
    except ModuleNotFoundError as err:
        raise TranscriptionError(
            "Missing faster-whisper dependencies for transcription backend "
            "'faster_whisper'. Install faster-whisper (for example, "
            "`uv sync --extra full`) or switch to `stable_whisper`."
        ) from err
    whisper_model = getattr(faster_whisper_module, "WhisperModel", None)
    if whisper_model is None:
        raise TranscriptionError("faster-whisper package does not expose WhisperModel.")

    download_root: Path = settings.models.whisper_download_root
    os.makedirs(download_root, exist_ok=True)
    return whisper_model(
        active_profile.model_name,
        device="cpu",
        compute_type="int8",
        download_root=str(download_root),
    )


def load_whisper_model(profile: TranscriptionProfile | None = None) -> object:
    """Loads the configured transcription model for CPU inference.

    Returns:
        The loaded Whisper model instance.
    """
    settings: AppConfig = get_settings()
    active_profile: TranscriptionProfile = resolve_transcription_profile(profile)
    try:
        if active_profile.backend_id == "stable_whisper":
            return _load_stable_whisper_model(
                active_profile=active_profile,
                settings=settings,
            )
        return _load_faster_whisper_model(
            active_profile=active_profile,
            settings=settings,
        )
    except Exception as err:
        logger.error(msg=f"Failed to load transcription model: {err}", exc_info=True)
        raise TranscriptionError("Failed to load transcription model.") from err


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
    """Internal transcript workflow with model loading and formatting."""
    settings: AppConfig = get_settings()
    active_profile: TranscriptionProfile = resolve_transcription_profile(profile)

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
        transcript_words: list[TranscriptWord] = (
            __transcribe_file(model, language, file_path)
            if profile is None
            else _transcribe_file_with_profile(
                model,
                language,
                file_path,
                active_profile,
            )
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


def __transcribe_file(
    model: object, language: str, file_path: str
) -> list[TranscriptWord]:
    """Runs a Whisper transcription call and normalizes return types."""
    return _transcribe_file_with_profile(model, language, file_path, profile=None)


def _normalize_stable_whisper_result(raw_transcript: object) -> WhisperResult:
    """Normalizes raw stable-whisper outputs to a WhisperResult instance."""
    from stable_whisper.result import WhisperResult

    if isinstance(raw_transcript, WhisperResult):
        return raw_transcript
    if isinstance(raw_transcript, dict | list | str):
        return WhisperResult(raw_transcript)
    logger.error(
        "Unexpected transcription result type from stable-whisper: %s",
        type(raw_transcript).__name__,
    )
    raise TranscriptionError(
        "Unexpected transcription result type from stable-whisper."
    )


def _transcribe_with_stable_whisper(
    model: object,
    *,
    language: str,
    file_path: str,
    active_profile: TranscriptionProfile,
) -> list[TranscriptWord]:
    """Executes one transcription call through stable-whisper."""
    transcribe = getattr(model, "transcribe", None)
    if not callable(transcribe):
        raise TranscriptionError(
            "Loaded stable-whisper model does not expose a callable transcribe()."
        )
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            raw_transcript = transcribe(
                audio=file_path,
                language=language,
                verbose=False,
                word_timestamps=True,
                no_speech_threshold=None,
                demucs=active_profile.use_demucs,
                vad=active_profile.use_vad,
            )
    except Exception as err:
        logger.error(msg=f"Error processing speech extraction: {err}", exc_info=True)
        raise TranscriptionError("Failed to transcribe audio.") from err
    return format_transcript(_normalize_stable_whisper_result(raw_transcript))


def _transcribe_with_faster_whisper(
    model: object,
    *,
    language: str,
    file_path: str,
    active_profile: TranscriptionProfile,
) -> list[TranscriptWord]:
    """Executes one transcription call through faster-whisper."""
    transcribe = getattr(model, "transcribe", None)
    if not callable(transcribe):
        raise TranscriptionError(
            "Loaded faster-whisper model does not expose a callable transcribe()."
        )
    if active_profile.use_demucs:
        logger.warning(
            "faster-whisper backend does not support demucs preprocessing; "
            "demucs flag is ignored."
        )
    try:
        with _demote_faster_whisper_info_logs():
            raw_transcribe_result = transcribe(
                file_path,
                language=language,
                word_timestamps=True,
                vad_filter=active_profile.use_vad,
                beam_size=5,
            )
    except Exception as err:
        logger.error(msg=f"Error processing speech extraction: {err}", exc_info=True)
        raise TranscriptionError("Failed to transcribe audio.") from err
    if not isinstance(raw_transcribe_result, tuple) or len(raw_transcribe_result) != 2:
        raise TranscriptionError(
            "Unexpected result envelope returned by faster-whisper transcribe()."
        )
    segments = raw_transcribe_result[0]

    if not isinstance(segments, Iterable):
        raise TranscriptionError(
            "Unexpected segment stream type returned by faster-whisper."
        )

    transcript_words: list[TranscriptWord] = []
    for segment in cast(Iterable[object], segments):
        words = cast(FasterWhisperSegment, segment).words
        if not isinstance(words, list | tuple):
            continue
        for word in words:
            if word.start is None or word.end is None:
                continue
            transcript_words.append(
                TranscriptWord(
                    word=str(word.word),
                    start_seconds=float(word.start),
                    end_seconds=float(word.end),
                )
            )
    return transcript_words


def _transcribe_file_with_profile(
    model: object,
    language: str,
    file_path: str,
    profile: TranscriptionProfile | None,
) -> list[TranscriptWord]:
    """Runs a Whisper transcription call using an explicit runtime profile."""
    active_profile: TranscriptionProfile = resolve_transcription_profile(profile)
    if active_profile.backend_id == "stable_whisper":
        return _transcribe_with_stable_whisper(
            model,
            language=language,
            file_path=file_path,
            active_profile=active_profile,
        )
    if active_profile.backend_id == "faster_whisper":
        return _transcribe_with_faster_whisper(
            model,
            language=language,
            file_path=file_path,
            active_profile=active_profile,
        )
    raise TranscriptionError(
        f"Unsupported transcription backend id: {active_profile.backend_id}."
    )


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
