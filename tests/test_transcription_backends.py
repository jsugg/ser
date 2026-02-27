"""Unit tests for transcription backend adapter compatibility behavior."""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from ser.config import AppConfig
from ser.transcript.backends.base import BackendRuntimeRequest
from ser.transcript.backends.faster_whisper import FasterWhisperAdapter
from ser.transcript.backends.stable_whisper import StableWhisperAdapter
from ser.utils.transcription_compat import FASTER_WHISPER_OPENMP_CONFLICT_ISSUE_CODE


@pytest.fixture(autouse=True)
def _disable_openmp_conflict_detection(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keeps faster-whisper compatibility assertions deterministic in unit tests."""
    monkeypatch.setattr(
        "ser.transcript.backends.faster_whisper."
        "has_known_faster_whisper_openmp_runtime_conflict",
        lambda: False,
    )


def test_stable_whisper_compatibility_report_is_noise_aware(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stable adapter should expose noise policy metadata without blocking."""
    adapter = StableWhisperAdapter()
    monkeypatch.setattr(adapter, "_is_module_available", lambda _name: True)
    report = adapter.check_compatibility(
        runtime_request=BackendRuntimeRequest(
            model_name="large-v2",
            use_demucs=True,
            use_vad=True,
        ),
        settings=cast(AppConfig, SimpleNamespace()),
    )

    assert report.has_blocking_issues is False
    assert report.policy_ids == (
        "stable_whisper.invalid_escape_sequence",
        "stable_whisper.fp16_cpu_fallback_warning",
        "stable_whisper.demucs_deprecated_warning",
    )
    assert report.noise_issues


def test_stable_whisper_compatibility_blocks_on_missing_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stable adapter should block when stable_whisper dependency is unavailable."""
    adapter = StableWhisperAdapter()
    monkeypatch.setattr(adapter, "_is_module_available", lambda _name: False)
    report = adapter.check_compatibility(
        runtime_request=BackendRuntimeRequest(
            model_name="large-v2",
            use_demucs=True,
            use_vad=True,
        ),
        settings=cast(AppConfig, SimpleNamespace()),
    )

    assert report.has_blocking_issues is True
    assert any(
        issue.code == "missing_dependency_stable_whisper"
        for issue in report.functional_issues
    )


def test_faster_whisper_demucs_issue_is_operational_not_blocking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Faster adapter should report demucs limitation without blocking execution."""
    adapter = FasterWhisperAdapter()
    monkeypatch.setattr(adapter, "_is_module_available", lambda _name: True)
    report = adapter.check_compatibility(
        runtime_request=BackendRuntimeRequest(
            model_name="distil-large-v3",
            use_demucs=True,
            use_vad=True,
        ),
        settings=cast(AppConfig, SimpleNamespace()),
    )

    assert report.has_blocking_issues is False
    assert any(
        issue.code == "faster_whisper_demucs_unsupported"
        for issue in report.operational_issues
    )
    assert report.policy_ids == ("faster_whisper.info_demotion",)


def test_faster_whisper_mps_runtime_is_operational_not_blocking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Faster adapter should downgrade unsupported MPS runtime to operational issue."""
    adapter = FasterWhisperAdapter()
    monkeypatch.setattr(adapter, "_is_module_available", lambda _name: True)
    report = adapter.check_compatibility(
        runtime_request=BackendRuntimeRequest(
            model_name="distil-large-v3",
            use_demucs=False,
            use_vad=True,
            device_spec="mps",
            device_type="mps",
            precision_candidates=("float16", "float32"),
            memory_tier="low",
        ),
        settings=cast(AppConfig, SimpleNamespace()),
    )

    assert report.has_blocking_issues is False
    assert any(
        issue.code == "faster_whisper_mps_unsupported"
        for issue in report.operational_issues
    )


def test_faster_whisper_openmp_conflict_is_blocking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Known OpenMP runtime collisions should block faster-whisper execution."""
    monkeypatch.setattr(
        "ser.transcript.backends.faster_whisper."
        "has_known_faster_whisper_openmp_runtime_conflict",
        lambda: True,
    )
    adapter = FasterWhisperAdapter()
    monkeypatch.setattr(adapter, "_is_module_available", lambda _name: True)
    report = adapter.check_compatibility(
        runtime_request=BackendRuntimeRequest(
            model_name="distil-large-v3",
            use_demucs=False,
            use_vad=True,
        ),
        settings=cast(AppConfig, SimpleNamespace()),
    )

    assert report.has_blocking_issues is True
    assert any(
        issue.code == FASTER_WHISPER_OPENMP_CONFLICT_ISSUE_CODE
        for issue in report.functional_issues
    )


def test_stable_whisper_transcribe_prefers_denoiser_and_fp16_control(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stable adapter should pass modern denoiser API and disable fp16 on CPU."""
    adapter = StableWhisperAdapter()
    monkeypatch.setattr(adapter, "_normalize_result", lambda raw: raw)
    monkeypatch.setattr(adapter, "_format_transcript", lambda _result: [])
    captured: dict[str, object] = {}

    class _FakeModel:
        def transcribe(
            self,
            *,
            audio: str,
            language: str,
            verbose: bool,
            word_timestamps: bool,
            no_speech_threshold: object,
            vad: bool,
            fp16: bool,
            denoiser: str,
        ) -> dict[str, object]:
            captured["audio"] = audio
            captured["language"] = language
            captured["verbose"] = verbose
            captured["word_timestamps"] = word_timestamps
            captured["no_speech_threshold"] = no_speech_threshold
            captured["vad"] = vad
            captured["fp16"] = fp16
            captured["denoiser"] = denoiser
            return {"segments": []}

    transcript = adapter.transcribe(
        model=_FakeModel(),
        runtime_request=BackendRuntimeRequest(
            model_name="large-v2",
            use_demucs=True,
            use_vad=True,
        ),
        file_path="sample.wav",
        language="en",
        settings=cast(AppConfig, SimpleNamespace()),
    )

    assert transcript == []
    assert captured["audio"] == "sample.wav"
    assert captured["language"] == "en"
    assert captured["verbose"] is False
    assert captured["word_timestamps"] is True
    assert captured["no_speech_threshold"] is None
    assert captured["vad"] is True
    assert captured["fp16"] is False
    assert captured["denoiser"] == "demucs"


def test_stable_whisper_transcribe_uses_legacy_demucs_when_denoiser_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stable adapter should fall back to legacy demucs argument for old signatures."""
    adapter = StableWhisperAdapter()
    monkeypatch.setattr(adapter, "_normalize_result", lambda raw: raw)
    monkeypatch.setattr(adapter, "_format_transcript", lambda _result: [])
    captured: dict[str, object] = {}

    class _LegacyModel:
        def transcribe(
            self,
            *,
            audio: str,
            language: str,
            verbose: bool,
            word_timestamps: bool,
            no_speech_threshold: object,
            vad: bool,
            demucs: bool,
        ) -> dict[str, object]:
            captured["audio"] = audio
            captured["language"] = language
            captured["verbose"] = verbose
            captured["word_timestamps"] = word_timestamps
            captured["no_speech_threshold"] = no_speech_threshold
            captured["vad"] = vad
            captured["demucs"] = demucs
            return {"segments": []}

    transcript = adapter.transcribe(
        model=_LegacyModel(),
        runtime_request=BackendRuntimeRequest(
            model_name="large-v2",
            use_demucs=True,
            use_vad=False,
        ),
        file_path="legacy.wav",
        language="en",
        settings=cast(AppConfig, SimpleNamespace()),
    )

    assert transcript == []
    assert captured["audio"] == "legacy.wav"
    assert captured["language"] == "en"
    assert captured["verbose"] is False
    assert captured["word_timestamps"] is True
    assert captured["no_speech_threshold"] is None
    assert captured["vad"] is False
    assert captured["demucs"] is True


def test_stable_whisper_transcribe_retries_with_fallback_precision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stable adapter should retry with next precision on retryable failure."""
    adapter = StableWhisperAdapter()
    monkeypatch.setattr(adapter, "_normalize_result", lambda raw: raw)
    monkeypatch.setattr(adapter, "_format_transcript", lambda _result: [])
    fp16_flags: list[bool] = []

    class _FakeModel:
        def transcribe(
            self,
            *,
            fp16: bool,
            **_kwargs: object,
        ) -> dict[str, object]:
            fp16_flags.append(fp16)
            if fp16:
                raise NotImplementedError(
                    "Could not run 'aten::empty.memory_format' with arguments "
                    "from the 'SparseMPS' backend."
                )
            return {"segments": []}

    transcript = adapter.transcribe(
        model=_FakeModel(),
        runtime_request=BackendRuntimeRequest(
            model_name="large-v3",
            use_demucs=False,
            use_vad=True,
            device_spec="mps",
            device_type="mps",
            precision_candidates=("float16", "float32"),
            memory_tier="low",
        ),
        file_path="sample.wav",
        language="en",
        settings=cast(AppConfig, SimpleNamespace()),
    )

    assert transcript == []
    assert fp16_flags == [True, False]


def test_stable_whisper_transcribe_hard_mps_oom_shortcuts_to_cpu(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Pure MPS OOM on fp16 should skip MPS float32 retry and fail over to CPU."""
    adapter = StableWhisperAdapter()
    monkeypatch.setattr(adapter, "_normalize_result", lambda raw: raw)
    monkeypatch.setattr(adapter, "_format_transcript", lambda _result: [])
    calls: list[tuple[str, bool]] = []

    class _FakeModel:
        runtime_device = "mps"

        def transcribe(self, *, fp16: bool, **_kwargs: object) -> dict[str, object]:
            calls.append((self.runtime_device, fp16))
            if self.runtime_device == "mps":
                raise RuntimeError("MPS backend out of memory")
            return {"segments": []}

        def to(self, *, device: str) -> _FakeModel:
            self.runtime_device = device
            return self

    caplog.set_level(logging.INFO)
    transcript = adapter.transcribe(
        model=_FakeModel(),
        runtime_request=BackendRuntimeRequest(
            model_name="large-v3",
            use_demucs=False,
            use_vad=True,
            device_spec="mps",
            device_type="mps",
            precision_candidates=("float16", "float32"),
            memory_tier="low",
        ),
        file_path="sample.wav",
        language="en",
        settings=cast(
            AppConfig,
            SimpleNamespace(
                transcription=SimpleNamespace(
                    mps_admission_control_enabled=False,
                    mps_hard_oom_shortcut_enabled=True,
                )
            ),
        ),
    )

    assert transcript == []
    assert calls == [("mps", True), ("cpu", False)]
    assert "switching directly to cpu" in caplog.text


def test_stable_whisper_transcribe_hard_mps_oom_shortcut_can_be_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Disabling hard-OOM shortcut should preserve MPS precision fallback behavior."""
    adapter = StableWhisperAdapter()
    monkeypatch.setattr(adapter, "_normalize_result", lambda raw: raw)
    monkeypatch.setattr(adapter, "_format_transcript", lambda _result: [])
    calls: list[tuple[str, bool]] = []

    class _FakeModel:
        runtime_device = "mps"

        def transcribe(self, *, fp16: bool, **_kwargs: object) -> dict[str, object]:
            calls.append((self.runtime_device, fp16))
            if self.runtime_device == "mps":
                raise RuntimeError("MPS backend out of memory")
            return {"segments": []}

        def to(self, *, device: str) -> _FakeModel:
            self.runtime_device = device
            return self

    transcript = adapter.transcribe(
        model=_FakeModel(),
        runtime_request=BackendRuntimeRequest(
            model_name="large-v3",
            use_demucs=False,
            use_vad=True,
            device_spec="mps",
            device_type="mps",
            precision_candidates=("float16", "float32"),
            memory_tier="low",
        ),
        file_path="sample.wav",
        language="en",
        settings=cast(
            AppConfig,
            SimpleNamespace(
                transcription=SimpleNamespace(
                    mps_admission_control_enabled=False,
                    mps_hard_oom_shortcut_enabled=False,
                )
            ),
        ),
    )

    assert transcript == []
    assert calls == [("mps", True), ("mps", False), ("cpu", False)]


@pytest.mark.parametrize(
    ("failure_type", "failure_message"),
    [
        (RuntimeError, "MPS backend out of memory"),
        (
            NotImplementedError,
            "Could not run 'aten::empty.memory_format' with arguments from the 'SparseMPS' backend.",
        ),
    ],
)
def test_stable_whisper_load_model_retries_on_cpu_after_retryable_mps_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure_type: type[Exception],
    failure_message: str,
) -> None:
    """Stable adapter should keep CPU model when MPS compatibility move fails."""
    adapter = StableWhisperAdapter()
    download_root = tmp_path / "model-cache" / "OpenAI" / "whisper"
    torch_cache_root = tmp_path / "model-cache" / "torch"
    settings = cast(
        AppConfig,
        SimpleNamespace(
            models=SimpleNamespace(
                whisper_download_root=download_root,
                torch_cache_root=torch_cache_root,
            )
        ),
    )
    loaded_model = SimpleNamespace()
    load_devices: list[str] = []

    def _fake_load_model(
        *,
        name: str,
        device: str,
        dq: bool,
        download_root: str,
        in_memory: bool,
    ) -> object:
        del name
        del dq
        del download_root
        del in_memory
        load_devices.append(device)
        return loaded_model

    monkeypatch.setitem(
        sys.modules,
        "stable_whisper",
        SimpleNamespace(load_model=_fake_load_model),
    )
    monkeypatch.setattr(
        "ser.transcript.backends.stable_whisper."
        "enable_stable_whisper_mps_compatibility",
        lambda _model: (_ for _ in ()).throw(failure_type(failure_message)),
    )

    resolved_model = adapter.load_model(
        runtime_request=BackendRuntimeRequest(
            model_name="large-v3",
            use_demucs=False,
            use_vad=True,
            device_spec="mps",
            device_type="mps",
            precision_candidates=("float16", "float32"),
            memory_tier="low",
        ),
        settings=settings,
    )

    assert resolved_model is loaded_model
    assert load_devices == ["cpu"]


def test_stable_whisper_load_model_enables_mps_compatibility_on_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Stable adapter should enable MPS compatibility flow for MPS runtime requests."""
    adapter = StableWhisperAdapter()
    download_root = tmp_path / "model-cache" / "OpenAI" / "whisper"
    torch_cache_root = tmp_path / "model-cache" / "torch"
    settings = cast(
        AppConfig,
        SimpleNamespace(
            models=SimpleNamespace(
                whisper_download_root=download_root,
                torch_cache_root=torch_cache_root,
            ),
            transcription=SimpleNamespace(
                mps_admission_control_enabled=False,
                mps_hard_oom_shortcut_enabled=True,
            ),
        ),
    )
    loaded_model = SimpleNamespace()
    moved_model = object()
    load_devices: list[str] = []

    def _fake_load_model(
        *,
        name: str,
        device: str,
        dq: bool,
        download_root: str,
        in_memory: bool,
    ) -> object:
        del name
        del dq
        del download_root
        del in_memory
        load_devices.append(device)
        return loaded_model

    monkeypatch.setitem(
        sys.modules,
        "stable_whisper",
        SimpleNamespace(load_model=_fake_load_model),
    )
    monkeypatch.setattr(
        "ser.transcript.backends.stable_whisper."
        "enable_stable_whisper_mps_compatibility",
        lambda _model: moved_model,
    )
    monkeypatch.setattr(
        "ser.transcript.backends.stable_whisper."
        "has_known_stable_whisper_sparse_mps_incompatibility",
        lambda: True,
    )

    resolved_model = adapter.load_model(
        runtime_request=BackendRuntimeRequest(
            model_name="large-v3",
            use_demucs=False,
            use_vad=True,
            device_spec="mps",
            device_type="mps",
            precision_candidates=("float16", "float32"),
            memory_tier="low",
        ),
        settings=settings,
    )

    assert resolved_model is moved_model
    assert load_devices == ["cpu"]


def test_stable_whisper_load_model_mps_admission_control_prefers_cpu(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """MPS admission control should keep model on CPU when headroom is insufficient."""
    adapter = StableWhisperAdapter()
    download_root = tmp_path / "model-cache" / "OpenAI" / "whisper"
    torch_cache_root = tmp_path / "model-cache" / "torch"
    settings = cast(
        AppConfig,
        SimpleNamespace(
            models=SimpleNamespace(
                whisper_download_root=download_root,
                torch_cache_root=torch_cache_root,
            ),
            transcription=SimpleNamespace(
                mps_admission_control_enabled=True,
                mps_hard_oom_shortcut_enabled=True,
                mps_admission_min_headroom_mb=64.0,
                mps_admission_safety_margin_mb=64.0,
            ),
        ),
    )
    loaded_model = SimpleNamespace()
    load_devices: list[str] = []
    enable_calls: list[str] = []

    def _fake_load_model(
        *,
        name: str,
        device: str,
        dq: bool,
        download_root: str,
        in_memory: bool,
    ) -> object:
        del name
        del dq
        del download_root
        del in_memory
        load_devices.append(device)
        return loaded_model

    monkeypatch.setitem(
        sys.modules,
        "stable_whisper",
        SimpleNamespace(load_model=_fake_load_model),
    )
    monkeypatch.setattr(
        "ser.transcript.backends.stable_whisper.decide_mps_admission_for_transcription",
        lambda **_kwargs: SimpleNamespace(
            allow_mps=False,
            reason_code="mps_headroom_below_required_budget",
            required_bytes=256 * 1024**2,
            available_bytes=32 * 1024**2,
            required_metric="headroom_budget",
            available_metric="headroom_estimate",
            confidence="high",
        ),
    )

    def _record_enable_call(model: object) -> object:
        enable_calls.append("called")
        return model

    monkeypatch.setattr(
        "ser.transcript.backends.stable_whisper."
        "enable_stable_whisper_mps_compatibility",
        _record_enable_call,
    )

    caplog.set_level(logging.INFO)
    resolved_model = adapter.load_model(
        runtime_request=BackendRuntimeRequest(
            model_name="large-v3",
            use_demucs=False,
            use_vad=True,
            device_spec="mps",
            device_type="mps",
            precision_candidates=("float16", "float32"),
            memory_tier="low",
        ),
        settings=settings,
    )

    assert resolved_model is loaded_model
    assert load_devices == ["cpu"]
    assert enable_calls == []
    assert "MPS admission control switched model_load to cpu" in caplog.text


def test_stable_whisper_transcribe_mps_admission_control_prefers_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Transcribe admission control should switch to CPU before first MPS attempt."""
    adapter = StableWhisperAdapter()
    monkeypatch.setattr(adapter, "_normalize_result", lambda raw: raw)
    monkeypatch.setattr(adapter, "_format_transcript", lambda _result: [])
    calls: list[tuple[str, bool]] = []

    class _FakeModel:
        runtime_device = "mps"

        def transcribe(self, *, fp16: bool, **_kwargs: object) -> dict[str, object]:
            calls.append((self.runtime_device, fp16))
            return {"segments": []}

        def to(self, *, device: str) -> _FakeModel:
            self.runtime_device = device
            return self

    monkeypatch.setattr(
        "ser.transcript.backends.stable_whisper.decide_mps_admission_for_transcription",
        lambda **_kwargs: SimpleNamespace(
            allow_mps=False,
            reason_code="mps_headroom_below_required_budget",
            required_bytes=256 * 1024**2,
            available_bytes=32 * 1024**2,
            required_metric="headroom_budget",
            available_metric="headroom_estimate",
            confidence="high",
        ),
    )

    transcript = adapter.transcribe(
        model=_FakeModel(),
        runtime_request=BackendRuntimeRequest(
            model_name="large-v3",
            use_demucs=False,
            use_vad=True,
            device_spec="mps",
            device_type="mps",
            precision_candidates=("float16", "float32"),
            memory_tier="low",
        ),
        file_path="sample.wav",
        language="en",
        settings=cast(
            AppConfig,
            SimpleNamespace(
                transcription=SimpleNamespace(
                    mps_admission_control_enabled=True,
                    mps_hard_oom_shortcut_enabled=True,
                    mps_admission_min_headroom_mb=64.0,
                    mps_admission_safety_margin_mb=64.0,
                )
            ),
        ),
    )

    assert transcript == []
    assert calls == [("cpu", False)]


def test_stable_whisper_transcribe_low_confidence_admission_denial_is_advisory(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Low-confidence admission denial should allow one MPS attempt before fallback."""
    adapter = StableWhisperAdapter()
    monkeypatch.setattr(adapter, "_normalize_result", lambda raw: raw)
    monkeypatch.setattr(adapter, "_format_transcript", lambda _result: [])
    calls: list[tuple[str, bool]] = []

    class _FakeModel:
        runtime_device = "mps"

        def transcribe(self, *, fp16: bool, **_kwargs: object) -> dict[str, object]:
            calls.append((self.runtime_device, fp16))
            if self.runtime_device == "mps":
                raise RuntimeError("MPS backend out of memory")
            return {"segments": []}

        def to(self, *, device: str) -> _FakeModel:
            self.runtime_device = device
            return self

    monkeypatch.setattr(
        "ser.transcript.backends.stable_whisper.decide_mps_admission_for_transcription",
        lambda **_kwargs: SimpleNamespace(
            allow_mps=False,
            reason_code="mps_headroom_estimate_below_required_budget",
            required_bytes=256 * 1024**2,
            available_bytes=32 * 1024**2,
            required_metric="headroom_budget",
            available_metric="headroom_estimate",
            confidence="low",
        ),
    )

    caplog.set_level(logging.INFO)
    transcript = adapter.transcribe(
        model=_FakeModel(),
        runtime_request=BackendRuntimeRequest(
            model_name="turbo",
            use_demucs=False,
            use_vad=True,
            device_spec="mps",
            device_type="mps",
            precision_candidates=("float16", "float32"),
            memory_tier="low",
        ),
        file_path="sample.wav",
        language="en",
        settings=cast(
            AppConfig,
            SimpleNamespace(
                transcription=SimpleNamespace(
                    mps_admission_control_enabled=True,
                    mps_hard_oom_shortcut_enabled=True,
                )
            ),
        ),
    )

    assert transcript == []
    assert calls == [("mps", True), ("cpu", False)]
    assert "allowing one MPS attempt" in caplog.text


def test_stable_whisper_transcribe_uses_mps_timing_compatibility_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stable adapter should run transcribe in MPS timing compatibility context."""
    adapter = StableWhisperAdapter()
    monkeypatch.setattr(adapter, "_normalize_result", lambda raw: raw)
    monkeypatch.setattr(adapter, "_format_transcript", lambda _result: [])
    context_events: list[str] = []

    @contextmanager
    def _fake_context() -> Iterator[None]:
        context_events.append("enter")
        try:
            yield
        finally:
            context_events.append("exit")

    monkeypatch.setattr(
        "ser.transcript.backends.stable_whisper."
        "is_stable_whisper_mps_compatibility_enabled",
        lambda _model: True,
    )
    monkeypatch.setattr(
        "ser.transcript.backends.stable_whisper."
        "stable_whisper_mps_timing_compatibility_context",
        _fake_context,
    )

    class _FakeModel:
        def transcribe(self, **_kwargs: object) -> dict[str, object]:
            context_events.append("call")
            return {"segments": []}

    transcript = adapter.transcribe(
        model=_FakeModel(),
        runtime_request=BackendRuntimeRequest(
            model_name="large-v3",
            use_demucs=False,
            use_vad=True,
            device_spec="mps",
            device_type="mps",
            precision_candidates=("float32",),
            memory_tier="low",
        ),
        file_path="sample.wav",
        language="en",
        settings=cast(AppConfig, SimpleNamespace()),
    )

    assert transcript == []
    assert context_events == ["enter", "call", "exit"]


def test_stable_whisper_runtime_error_summary_is_single_line() -> None:
    """Stable adapter retry logs should keep runtime errors concise and single-line."""
    adapter = StableWhisperAdapter()
    summary = adapter._summarize_runtime_error(
        RuntimeError("line1\nline2 " + ("x" * 500))
    )

    assert "\n" not in summary
    assert len(summary) <= 180


def test_stable_whisper_accurate_mps_move_failure_uses_cpu_transcribe_directly(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """MPS compatibility-move failure should transcribe directly on CPU in accurate path."""
    adapter = StableWhisperAdapter()
    download_root = tmp_path / "model-cache" / "OpenAI" / "whisper"
    torch_cache_root = tmp_path / "model-cache" / "torch"
    settings = cast(
        AppConfig,
        SimpleNamespace(
            models=SimpleNamespace(
                whisper_download_root=download_root,
                torch_cache_root=torch_cache_root,
            )
        ),
    )
    fp16_flags: list[bool] = []

    class _FakeModel:
        def transcribe(self, *, fp16: bool, **_kwargs: object) -> dict[str, object]:
            fp16_flags.append(fp16)
            return {"segments": []}

        def to(self, *, device: str) -> _FakeModel:
            del device
            return self

    fake_model = _FakeModel()

    def _fake_load_model(
        *,
        name: str,
        device: str,
        dq: bool,
        download_root: str,
        in_memory: bool,
    ) -> object:
        del name
        del device
        del dq
        del download_root
        del in_memory
        return fake_model

    monkeypatch.setitem(
        sys.modules,
        "stable_whisper",
        SimpleNamespace(load_model=_fake_load_model),
    )
    monkeypatch.setattr(
        "ser.transcript.backends.stable_whisper."
        "enable_stable_whisper_mps_compatibility",
        lambda _model: (_ for _ in ()).throw(RuntimeError("MPS backend out of memory")),
    )
    monkeypatch.setattr(adapter, "_normalize_result", lambda raw: raw)
    monkeypatch.setattr(adapter, "_format_transcript", lambda _result: [])

    caplog.set_level(logging.WARNING)
    loaded_model = adapter.load_model(
        runtime_request=BackendRuntimeRequest(
            model_name="large-v3",
            use_demucs=False,
            use_vad=True,
            device_spec="mps",
            device_type="mps",
            precision_candidates=("float16", "float32"),
            memory_tier="low",
        ),
        settings=settings,
    )

    transcript = adapter.transcribe(
        model=loaded_model,
        runtime_request=BackendRuntimeRequest(
            model_name="large-v3",
            use_demucs=False,
            use_vad=True,
            device_spec="mps",
            device_type="mps",
            precision_candidates=("float16", "float32"),
            memory_tier="low",
        ),
        file_path="sample.wav",
        language="en",
        settings=settings,
    )

    assert transcript == []
    assert fp16_flags == [False]
    assert "retrying with fallback precision" not in caplog.text
    assert "retrying on cpu runtime after mps failure" not in caplog.text
