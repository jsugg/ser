"""Execution helpers for stable-whisper transcribe retry/fallback behavior."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from ser.domain import TranscriptWord
from ser.transcript.backends.base import BackendRuntimeRequest
from ser.transcript.runtime_failures import (
    FailureDisposition,
    TranscriptionFailureClassification,
)

if TYPE_CHECKING:
    from ser.config import AppConfig

type BuildTranscribeKwargs = Callable[[BackendRuntimeRequest, str], dict[str, object]]
type InvokeRuntimeTranscribe = Callable[[dict[str, object], str], object]
type ClassifyFailure = Callable[
    [Exception, str, AppConfig],
    TranscriptionFailureClassification,
]
type ReleaseRuntimeMemory = Callable[[], None]
type SummarizeRuntimeError = Callable[[Exception], str]
type MoveModelToCpuRuntime = Callable[[object], bool]
type SetMpsCompatibilityEnabled = Callable[[object], None]
type SetRuntimeDevice = Callable[[object], None]
type NormalizeResult = Callable[[object], object]
type FormatTranscript = Callable[[object], list[TranscriptWord]]


def run_stable_whisper_transcribe_with_retry(
    *,
    model: object,
    runtime_request: BackendRuntimeRequest,
    settings: AppConfig,
    runtime_device_type: str,
    precision_candidates: tuple[str, ...],
    typed_transcribe: Callable[..., object],
    build_transcribe_kwargs: BuildTranscribeKwargs,
    invoke_runtime_transcribe: InvokeRuntimeTranscribe,
    classify_failure: ClassifyFailure,
    release_runtime_memory_for_retry: ReleaseRuntimeMemory,
    summarize_runtime_error: SummarizeRuntimeError,
    move_model_to_cpu_runtime: MoveModelToCpuRuntime,
    set_mps_compatibility_disabled: SetMpsCompatibilityEnabled,
    set_runtime_device_cpu: SetRuntimeDevice,
    normalize_result: NormalizeResult,
    format_transcript: FormatTranscript,
    logger: logging.Logger,
) -> list[TranscriptWord]:
    """Runs transcribe with precision retries and terminal CPU fallback."""
    last_error: Exception | None = None
    for index, precision in enumerate(precision_candidates):
        transcribe_kwargs = build_transcribe_kwargs(runtime_request, precision)
        raw_transcript: object
        try:
            raw_transcript = invoke_runtime_transcribe(
                transcribe_kwargs,
                runtime_device_type,
            )
            return format_transcript(normalize_result(raw_transcript))
        except Exception as err:
            last_error = err
            failure_classification = classify_failure(err, precision, settings)
            if failure_classification.is_retryable:
                release_runtime_memory_for_retry()
            should_force_cpu_now = (
                failure_classification.disposition == FailureDisposition.FAILOVER_CPU_NOW
                and runtime_device_type in {"mps", "cuda"}
            )
            is_final_candidate = index >= len(precision_candidates) - 1
            is_terminal_candidate = is_final_candidate or should_force_cpu_now
            if (
                is_terminal_candidate
                and failure_classification.is_retryable
                and runtime_device_type in {"mps", "cuda"}
            ):
                error_summary = summarize_runtime_error(err)
                if should_force_cpu_now:
                    logger.info(
                        "Transcription hard MPS OOM on %s; switching directly "
                        "to cpu (reason=%s, error=%s).",
                        precision,
                        failure_classification.reason_code,
                        error_summary,
                    )
                else:
                    logger.warning(
                        "Transcription retrying on cpu runtime after %s " "failure (%s).",
                        runtime_device_type,
                        error_summary,
                    )
                if move_model_to_cpu_runtime(model):
                    set_mps_compatibility_disabled(model)
                    set_runtime_device_cpu(model)
                    runtime_device_type = "cpu"
                    cpu_kwargs = build_transcribe_kwargs(
                        _cpu_runtime_request(runtime_request),
                        "float32",
                    )
                    try:
                        raw_transcript = typed_transcribe(**cpu_kwargs)
                        return format_transcript(normalize_result(raw_transcript))
                    except Exception as cpu_err:
                        raise RuntimeError("Failed to transcribe audio.") from cpu_err
            if (
                is_terminal_candidate
                or failure_classification.disposition == FailureDisposition.FAIL_FAST
            ):
                raise RuntimeError("Failed to transcribe audio.") from err
            logger.warning(
                "Transcription retrying with fallback precision after "
                "failure using %s on %s: %s",
                precision,
                runtime_device_type,
                err,
            )
    if last_error is not None:
        raise RuntimeError("Failed to transcribe audio.") from last_error
    raise RuntimeError("Failed to transcribe audio.")


def _cpu_runtime_request(
    runtime_request: BackendRuntimeRequest,
) -> BackendRuntimeRequest:
    """Builds a CPU-only request for terminal transcribe retry fallback."""
    return BackendRuntimeRequest(
        model_name=runtime_request.model_name,
        use_demucs=runtime_request.use_demucs,
        use_vad=runtime_request.use_vad,
        device_spec="cpu",
        device_type="cpu",
        precision_candidates=("float32",),
        memory_tier="not_applicable",
    )


__all__ = ["run_stable_whisper_transcribe_with_retry"]
