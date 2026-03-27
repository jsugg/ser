"""Tests for stable-whisper transcribe execution helper."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import cast

import pytest

from ser.config import AppConfig
from ser.transcript.backends.base import BackendRuntimeRequest
from ser.transcript.backends.stable_whisper_transcribe_execution import (
    run_stable_whisper_transcribe_with_retry,
)
from ser.transcript.runtime_failures import (
    FailureDisposition,
    TranscriptionFailureClassification,
)


def _runtime_request(*, device_type: str, precision: tuple[str, ...]) -> BackendRuntimeRequest:
    return BackendRuntimeRequest(
        model_name="large-v3",
        use_demucs=False,
        use_vad=True,
        device_spec=device_type,
        device_type=device_type,
        precision_candidates=precision,
        memory_tier="low",
    )


def test_run_transcribe_retries_next_precision_then_succeeds() -> None:
    """Retry-next-precision should continue and return formatted transcript."""
    runtime_request = _runtime_request(device_type="cpu", precision=("float16", "float32"))
    release_calls = 0

    def _build_kwargs(request: BackendRuntimeRequest, precision: str) -> dict[str, object]:
        return {"precision": precision, "device_type": request.device_type}

    def _invoke_runtime(kwargs: dict[str, object], _runtime_device: str) -> object:
        if kwargs["precision"] == "float16":
            raise RuntimeError("fp16 failed")
        return {"segments": []}

    def _classify(
        _err: Exception,
        _precision: str,
        _settings: AppConfig,
    ) -> TranscriptionFailureClassification:
        return TranscriptionFailureClassification(
            disposition=FailureDisposition.RETRY_NEXT_PRECISION,
            reason_code="retryable_precision_error",
            is_retryable=True,
        )

    def _release() -> None:
        nonlocal release_calls
        release_calls += 1

    transcript = run_stable_whisper_transcribe_with_retry(
        model=object(),
        runtime_request=runtime_request,
        settings=cast(AppConfig, SimpleNamespace()),
        runtime_device_type="cpu",
        precision_candidates=("float16", "float32"),
        typed_transcribe=lambda **_kwargs: {"segments": []},
        build_transcribe_kwargs=_build_kwargs,
        invoke_runtime_transcribe=_invoke_runtime,
        classify_failure=_classify,
        release_runtime_memory_for_retry=_release,
        summarize_runtime_error=lambda err: str(err),
        move_model_to_cpu_runtime=lambda _model: True,
        set_mps_compatibility_disabled=lambda _model: None,
        set_runtime_device_cpu=lambda _model: None,
        normalize_result=lambda raw: raw,
        format_transcript=lambda _raw: [],
        logger=logging.getLogger(__name__),
    )

    assert transcript == []
    assert release_calls == 1


def test_run_transcribe_terminal_retryable_failure_falls_back_to_cpu() -> None:
    """Terminal retryable failure on accelerator should execute one CPU retry."""
    runtime_request = _runtime_request(device_type="mps", precision=("float16",))
    calls: list[tuple[str, object]] = []
    cpu_runtime_args: list[dict[str, object]] = []
    compatibility_disabled = False
    cpu_runtime_selected = False

    class _Model:
        runtime_device = "mps"

        def to(self, *, device: str) -> _Model:
            self.runtime_device = device
            return self

    model = _Model()

    def _build_kwargs(request: BackendRuntimeRequest, precision: str) -> dict[str, object]:
        kwargs: dict[str, object] = {
            "precision": precision,
            "device_type": request.device_type,
        }
        if request.device_type == "cpu":
            cpu_runtime_args.append(kwargs)
        return kwargs

    def _invoke_runtime(kwargs: dict[str, object], runtime_device: str) -> object:
        calls.append((runtime_device, kwargs["precision"]))
        raise RuntimeError("MPS backend out of memory")

    def _typed_transcribe(**kwargs: object) -> object:
        cpu_runtime_args.append(dict(kwargs))
        return {"segments": []}

    def _classify(
        _err: Exception,
        _precision: str,
        _settings: AppConfig,
    ) -> TranscriptionFailureClassification:
        return TranscriptionFailureClassification(
            disposition=FailureDisposition.FAILOVER_CPU_NOW,
            reason_code="mps_hard_oom",
            is_retryable=True,
        )

    def _set_compatibility_disabled(_model: object) -> None:
        nonlocal compatibility_disabled
        compatibility_disabled = True

    def _set_runtime_device_cpu(_model: object) -> None:
        nonlocal cpu_runtime_selected
        cpu_runtime_selected = True

    def _move_to_cpu(runtime_model: object) -> bool:
        typed_model = cast(_Model, runtime_model)
        typed_model.to(device="cpu")
        return True

    transcript = run_stable_whisper_transcribe_with_retry(
        model=model,
        runtime_request=runtime_request,
        settings=cast(AppConfig, SimpleNamespace()),
        runtime_device_type="mps",
        precision_candidates=("float16",),
        typed_transcribe=_typed_transcribe,
        build_transcribe_kwargs=_build_kwargs,
        invoke_runtime_transcribe=_invoke_runtime,
        classify_failure=_classify,
        release_runtime_memory_for_retry=lambda: None,
        summarize_runtime_error=lambda err: str(err),
        move_model_to_cpu_runtime=_move_to_cpu,
        set_mps_compatibility_disabled=_set_compatibility_disabled,
        set_runtime_device_cpu=_set_runtime_device_cpu,
        normalize_result=lambda raw: raw,
        format_transcript=lambda _raw: [],
        logger=logging.getLogger(__name__),
    )

    assert transcript == []
    assert calls == [("mps", "float16")]
    assert compatibility_disabled is True
    assert cpu_runtime_selected is True
    assert any(
        kwargs.get("device_type") == "cpu" and kwargs.get("precision") == "float32"
        for kwargs in cpu_runtime_args
    )


def test_run_transcribe_fail_fast_raises_runtime_error() -> None:
    """Fail-fast classification should raise stable runtime error contract."""
    runtime_request = _runtime_request(device_type="cpu", precision=("float32",))

    def _classify(
        _err: Exception,
        _precision: str,
        _settings: AppConfig,
    ) -> TranscriptionFailureClassification:
        return TranscriptionFailureClassification(
            disposition=FailureDisposition.FAIL_FAST,
            reason_code="fatal_error",
            is_retryable=False,
        )

    with pytest.raises(RuntimeError, match="Failed to transcribe audio.") as exc_info:
        run_stable_whisper_transcribe_with_retry(
            model=object(),
            runtime_request=runtime_request,
            settings=cast(AppConfig, SimpleNamespace()),
            runtime_device_type="cpu",
            precision_candidates=("float32",),
            typed_transcribe=lambda **_kwargs: {"segments": []},
            build_transcribe_kwargs=lambda _request, _precision: {},
            invoke_runtime_transcribe=lambda _kwargs, _device: (_ for _ in ()).throw(
                ValueError("fatal")
            ),
            classify_failure=_classify,
            release_runtime_memory_for_retry=lambda: None,
            summarize_runtime_error=lambda err: str(err),
            move_model_to_cpu_runtime=lambda _model: False,
            set_mps_compatibility_disabled=lambda _model: None,
            set_runtime_device_cpu=lambda _model: None,
            normalize_result=lambda raw: raw,
            format_transcript=lambda _raw: [],
            logger=logging.getLogger(__name__),
        )

    assert isinstance(exc_info.value.__cause__, ValueError)
