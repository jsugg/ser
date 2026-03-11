"""Internal entrypoints for public transcript extraction wrappers."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Protocol, TypeVar, cast

from ser.config import AppConfig
from ser.profiles import TranscriptionBackendId

_TProfile = TypeVar("_TProfile", bound="_BackendProfile")
_TTranscriptWord = TypeVar("_TTranscriptWord")
_TTranscriptionError = TypeVar("_TTranscriptionError", bound=BaseException)


class _BackendProfile(Protocol):
    """Minimal profile contract required by transcript entrypoints."""

    @property
    def backend_id(self) -> TranscriptionBackendId:
        """Returns backend identifier used for adapter resolution."""
        ...


class _TranscriptionAdapter(Protocol):
    """Minimal adapter contract for transcript execution entrypoints."""

    def transcribe(
        self,
        *,
        model: object,
        runtime_request: object,
        file_path: str,
        language: str,
        settings: AppConfig,
    ) -> list[object]:
        """Runs backend transcription and returns transcript words."""
        ...


class _RawTranscriptWord(Protocol):
    """Minimal raw transcript-word contract returned by backend results."""

    @property
    def word(self) -> str:
        """Returns transcript token text."""
        ...

    @property
    def start(self) -> float | None:
        """Returns raw token start timestamp in seconds."""
        ...

    @property
    def end(self) -> float | None:
        """Returns raw token end timestamp in seconds."""
        ...


class _WhisperResultLike(Protocol):
    """Minimal stable-whisper style result contract."""

    def all_words(self) -> list[_RawTranscriptWord]:
        """Returns all transcript words exposed by the result object."""
        ...


type _ResolveProfileFn[_TProfile] = Callable[..., _TProfile]
type _ProcessIsolationSelector[_TProfile] = Callable[[_TProfile], bool]
type _TranscriptRunner[_TProfile, _TTranscriptWord] = Callable[
    ...,
    list[_TTranscriptWord],
]
type _RuntimeRequestResolver[_TProfile] = Callable[[_TProfile, AppConfig], object]
type _CompatibilityChecker[_TProfile] = Callable[..., object]
type _AdapterResolver = Callable[[TranscriptionBackendId], object]
type _TranscriptWordFactory[_TTranscriptWord] = Callable[
    [str, float, float],
    _TTranscriptWord,
]
type _ErrorFactory = Callable[[str], Exception]


def extract_transcript(
    file_path: str,
    language: str,
    profile: _TProfile | None,
    *,
    settings: AppConfig,
    resolve_profile_fn: _ResolveProfileFn[_TProfile],
    should_use_process_isolated_path_fn: _ProcessIsolationSelector[_TProfile],
    run_process_isolated_fn: _TranscriptRunner[_TProfile, _TTranscriptWord],
    run_in_process_fn: _TranscriptRunner[_TProfile, _TTranscriptWord],
) -> list[_TTranscriptWord]:
    """Routes transcript execution to isolated or in-process orchestration."""
    active_profile = resolve_profile_fn(profile, settings=settings)
    if should_use_process_isolated_path_fn(active_profile):
        return run_process_isolated_fn(
            file_path=file_path,
            language=language,
            profile=active_profile,
            settings=settings,
        )
    return run_in_process_fn(
        file_path=file_path,
        language=language,
        profile=active_profile,
        settings=settings,
    )


def transcribe_with_profile(
    model: object,
    language: str,
    file_path: str,
    profile: _TProfile | None,
    *,
    settings: AppConfig,
    resolve_profile_fn: _ResolveProfileFn[_TProfile],
    runtime_request_resolver: _RuntimeRequestResolver[_TProfile],
    compatibility_checker: _CompatibilityChecker[_TProfile],
    adapter_resolver: _AdapterResolver,
    passthrough_error_cls: type[_TTranscriptionError],
    logger: logging.Logger,
    error_factory: _ErrorFactory,
) -> list[object]:
    """Runs backend transcription with one resolved runtime profile."""
    active_profile = resolve_profile_fn(profile, settings=settings)
    runtime_request = runtime_request_resolver(active_profile, settings)
    compatibility_checker(
        active_profile=active_profile,
        settings=settings,
        runtime_request=runtime_request,
    )
    adapter = cast(
        _TranscriptionAdapter,
        adapter_resolver(active_profile.backend_id),
    )
    try:
        return adapter.transcribe(
            model=model,
            runtime_request=runtime_request,
            file_path=file_path,
            language=language,
            settings=settings,
        )
    except passthrough_error_cls:
        raise
    except Exception as err:
        logger.error(msg=f"Error processing speech extraction: {err}", exc_info=True)
        raise error_factory("Failed to transcribe audio.") from err


def format_transcript(
    result: object,
    *,
    transcript_word_factory: _TranscriptWordFactory[_TTranscriptWord],
    logger: logging.Logger,
    error_factory: _ErrorFactory,
) -> list[_TTranscriptWord]:
    """Normalizes backend word output into transcript domain objects."""
    try:
        words = cast(_WhisperResultLike, result).all_words()
    except AttributeError as err:
        logger.error(msg=f"Error extracting words from result: {err}", exc_info=True)
        raise error_factory("Invalid Whisper result object.") from err

    return [
        transcript_word_factory(
            str(word.word),
            float(word.start),
            float(word.end),
        )
        for word in words
        if word.start is not None and word.end is not None
    ]


__all__ = [
    "extract_transcript",
    "format_transcript",
    "transcribe_with_profile",
]
