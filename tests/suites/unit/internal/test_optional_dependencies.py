"""Optional features fail at their entry points without loading models."""

import importlib
import importlib.util
import re
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType

import pytest

from ser import api
from ser._internal.api.runtime import run_transcription_runtime_calibration_command
from ser._internal.data.msp_podcast_mirror import _load_hf_clients
from ser._internal.diagnostics.service import format_startup_preflight_one_liner
from ser._internal.runtime.commands import classify_inference_exception, classify_training_exception
from ser._internal.runtime.registry import UnsupportedProfileError
from ser._internal.transcript import profiling
from ser._internal.transcript.transcript_extractor import TranscriptionError
from ser.profiles import ProfileName, TranscriptionBackendId


@pytest.fixture(autouse=True)
def core_dependency_availability(monkeypatch: pytest.MonkeyPatch) -> None:
    find_spec = importlib.util.find_spec
    missing = {
        "ctranslate2",
        "demucs",
        "faster_whisper",
        "funasr",
        "modelscope",
        "onnxruntime",
        "stable_whisper",
        "torch",
        "torchaudio",
        "transformers",
        "whisper",
    }

    def find_core_spec(name: str, package: str | None = None) -> ModuleSpec | None:
        return None if name in missing else find_spec(name, package)

    monkeypatch.setattr(importlib.util, "find_spec", find_core_spec)


@pytest.mark.parametrize("backend", ["faster_whisper", "stable_whisper"])
def test_missing_transcript_extra_precedes_model_loading(
    backend: TranscriptionBackendId, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("WHISPER_BACKEND", backend)
    with pytest.raises(TranscriptionError, match=r"ser\[transcript\]") as raised:
        api.infer(tmp_path / "not-accessed.wav", profile="fast")
    assert backend in str(raised.value)
    assert not classify_inference_exception(raised.value).include_traceback


@pytest.mark.parametrize(
    ("profile", "extra"),
    [("medium", "medium"), ("accurate", "medium"), ("accurate-research", "full")],
)
def test_missing_profile_extra_names_the_install_requirement(
    profile: ProfileName, extra: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SER_ENABLE_RESTRICTED_BACKENDS", "1")
    monkeypatch.setenv("SER_ALLOWED_RESTRICTED_BACKENDS", "emotion2vec")
    with pytest.raises(UnsupportedProfileError, match=re.escape(f"ser[{extra}]")):
        api.load_profile(profile)


def test_missing_training_profile_extra_is_a_usage_error() -> None:
    with pytest.raises(UnsupportedProfileError, match=r"ser\[medium\]") as raised:
        api.train(profile="medium")
    disposition = classify_training_exception(raised.value)
    assert disposition.exit_code == 2
    assert not disposition.include_traceback


@pytest.mark.parametrize("missing", ["huggingface_hub", "pyarrow.parquet"])
def test_missing_mirror_dependencies_name_full(
    missing: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import_module = importlib.import_module
    fake_hub = ModuleType("huggingface_hub")
    fake_hub.__dict__.update(HfApi=object, snapshot_download=lambda: None)

    def import_without_optional_dependency(name: str, package: str | None = None) -> ModuleType:
        if name == missing:
            raise ModuleNotFoundError(name, name=name)
        if name == "huggingface_hub":
            return fake_hub
        return import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", import_without_optional_dependency)
    with pytest.raises(RuntimeError, match=r"ser\[full\]"):
        _load_hf_clients()


@pytest.mark.parametrize("backend", ["faster_whisper", "stable_whisper"])
def test_transcription_startup_failure_names_the_extra(
    backend: TranscriptionBackendId, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WHISPER_BACKEND", backend)
    report = api.run_startup_preflight(include_transcription_checks=True)
    assert "ser[transcript]" in format_startup_preflight_one_liner(
        report, doctor_command="ser doctor"
    )


@pytest.mark.parametrize("backend", ["faster_whisper", "stable_whisper"])
def test_calibration_requires_the_transcript_extra(
    backend: TranscriptionBackendId, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def unexpected_calibration(**_: object) -> None:
        pytest.fail("Calibration started without transcription dependencies")

    monkeypatch.setenv("WHISPER_BACKEND", backend)
    monkeypatch.setattr(profiling, "run_transcription_runtime_calibration", unexpected_calibration)
    audio = tmp_path / "sample.wav"
    audio.touch()

    result, error = run_transcription_runtime_calibration_command(
        file_path=str(audio),
        language="en",
        calibration_iterations=1,
        calibration_profiles="fast",
    )

    assert result is None
    assert error is not None
    assert (error.exit_code, error.include_traceback) == (2, False)
    assert backend in error.message
    assert "ser[transcript]" in error.message


def test_fast_runner_does_not_hide_a_broken_core_install(monkeypatch: pytest.MonkeyPatch) -> None:
    import_module = importlib.import_module

    def broken_runner(name: str, package: str | None = None) -> object:
        if name == "ser._internal.runtime.fast_inference":
            raise ModuleNotFoundError("missing core library", name="scipy")
        return import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", broken_runner)
    with pytest.raises(ModuleNotFoundError, match="missing core library"):
        api.load_profile("fast")
