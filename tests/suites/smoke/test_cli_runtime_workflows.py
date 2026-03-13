"""Cheap smoke coverage for user-visible CLI training and inference workflows."""

from __future__ import annotations

import pytest

import ser.__main__ as cli
import ser.config as config_module
import ser.models.emotion_model as emotion_model
import ser.runtime.fast_inference as fast_inference
import ser.utils.timeline_utils as timeline_utils
from ser.runtime.schema import OUTPUT_SCHEMA_VERSION, InferenceResult, SegmentPrediction

pytestmark = [pytest.mark.smoke, pytest.mark.usefixtures("reset_ambient_settings")]


def _patch_cli_runtime_prerequisites(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keeps CLI runtime smokes deterministic without bypassing pipeline building."""
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    monkeypatch.setattr(cli, "configure_logging", lambda _level=None: None)
    monkeypatch.setattr(
        cli,
        "run_restricted_backend_cli_gate",
        lambda **_kwargs: ((), None),
    )
    monkeypatch.setattr(
        cli,
        "run_startup_preflight_cli_gate",
        lambda **_kwargs: ((), None),
    )


def test_cli_fast_inference_smoke_uses_real_pipeline_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fast CLI inference should succeed without patching away the pipeline builder."""
    _patch_cli_runtime_prerequisites(monkeypatch)
    monkeypatch.setattr(
        cli.sys,
        "argv",
        ["ser", "--file", "sample.wav", "--profile", "fast", "--no-transcript"],
    )
    calls = {"legacy": 0, "detailed": 0, "timeline": 0}

    def _fake_run_fast_inference(
        _request: object,
        _settings: object,
        **_kwargs: object,
    ) -> InferenceResult:
        calls["detailed"] += 1
        return InferenceResult(
            schema_version=OUTPUT_SCHEMA_VERSION,
            segments=[
                SegmentPrediction(
                    emotion="calm",
                    start_seconds=0.0,
                    end_seconds=1.0,
                    confidence=1.0,
                )
            ],
            frames=[],
        )

    def _fake_build_timeline(_transcript: object, _emotions: object) -> list[object]:
        calls["timeline"] += 1
        return []

    monkeypatch.setattr(
        fast_inference,
        "run_fast_inference",
        _fake_run_fast_inference,
    )
    monkeypatch.setattr(
        "ser.transcript.extract_transcript",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("Transcription should not run for --no-transcript.")
        ),
    )
    monkeypatch.setattr(
        timeline_utils,
        "build_timeline",
        _fake_build_timeline,
    )
    monkeypatch.setattr(timeline_utils, "print_timeline", lambda _timeline: None)

    cli.main()

    assert calls["timeline"] == 1
    assert calls["legacy"] == 0
    assert calls["detailed"] == 1


def test_cli_fast_training_smoke_uses_real_pipeline_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fast CLI training should succeed without patching away the pipeline builder."""
    _patch_cli_runtime_prerequisites(monkeypatch)
    monkeypatch.setattr(cli.sys, "argv", ["ser", "--train", "--profile", "fast"])
    captured: dict[str, object] = {}

    def _fake_train_model(*, settings: config_module.AppConfig | None = None) -> None:
        captured["settings"] = settings

    monkeypatch.setattr(emotion_model, "train_model", _fake_train_model)

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 0
    assert captured["settings"] is not None
