"""Unit tests for internal runtime-calibration orchestration."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest

from ser._internal.transcription import runtime_calibration_workflow as workflow_helpers


@dataclass(frozen=True)
class _Settings:
    report_folder: Path
    label: str


@dataclass(frozen=True)
class _Candidate:
    source_profile: str
    backend_id: str
    model_name: str


@dataclass(frozen=True)
class _Metrics:
    backend_id: str
    score: float


@dataclass(frozen=True)
class _Recommendation:
    profile: _Candidate
    recommendation: str
    confidence: str
    reason: str
    metrics: _Metrics


@contextmanager
def _null_override() -> Iterator[None]:
    """Provide a no-op settings override context manager for validation tests."""

    yield None


def test_execute_runtime_calibration_profiles_candidates_and_persists_report(
    tmp_path: Path,
) -> None:
    """Workflow helper should execute candidates inside the override context."""

    captured: dict[str, object] = {"events": []}
    active_settings = _Settings(report_folder=tmp_path, label="base")
    calibration_file = tmp_path / "sample.wav"
    calibration_file.write_bytes(b"audio")
    calibration_settings = _Settings(report_folder=tmp_path, label="calibration")
    candidates = (
        _Candidate("accurate", "stable_whisper", "large"),
        _Candidate("fast", "faster_whisper", "distil-large-v3"),
    )
    metrics_by_profile = {
        "accurate": _Metrics(backend_id="stable_whisper", score=0.8),
        "fast": _Metrics(backend_id="faster_whisper", score=0.6),
    }

    @contextmanager
    def _settings_override(settings: _Settings) -> Iterator[None]:
        events = captured["events"]
        assert isinstance(events, list)
        events.append(("enter", settings.label))
        try:
            yield None
        finally:
            events.append(("exit", settings.label))

    def _persist_profile_report(output_path: Path, payload: dict[str, object]) -> Path:
        captured["persist"] = (output_path, payload)
        return output_path

    execution = workflow_helpers.execute_runtime_calibration(
        active_settings=active_settings,
        calibration_file=calibration_file,
        language="en",
        iterations_per_profile=3,
        profile_names=("accurate", "fast"),
        report_path=None,
        build_runtime_calibration_settings=lambda settings: calibration_settings,
        settings_override=_settings_override,
        runtime_calibration_candidates=lambda profile_names: (
            candidates if profile_names == ("accurate", "fast") else ()
        ),
        calibrate_candidate=lambda candidate, file_path, language, iterations: (
            metrics_by_profile[candidate.source_profile]
            if file_path == calibration_file and language == "en" and iterations == 3
            else _Metrics(backend_id="unexpected", score=0.0)
        ),
        derive_runtime_recommendation=lambda metrics: (
            "prefer_mps" if metrics.backend_id == "stable_whisper" else "prefer_cpu",
            "high",
            f"score={metrics.score}",
        ),
        recommendation_factory=lambda candidate, recommendation, confidence, reason, metrics: _Recommendation(
            profile=candidate,
            recommendation=recommendation,
            confidence=confidence,
            reason=reason,
            metrics=metrics,
        ),
        runtime_calibration_report_path=lambda settings: settings.report_folder
        / "runtime-calibration-report.json",
        persist_profile_report=_persist_profile_report,
        serialize_recommendation=lambda recommendation: {
            "profile": recommendation.profile.source_profile,
            "backend_id": recommendation.profile.backend_id,
            "model_name": recommendation.profile.model_name,
            "recommendation": recommendation.recommendation,
            "confidence": recommendation.confidence,
            "reason": recommendation.reason,
            "metrics": asdict(recommendation.metrics),
        },
        now_utc=lambda: datetime(2026, 3, 9, tzinfo=UTC),
    )

    assert execution.recommendations == (
        _Recommendation(
            profile=candidates[0],
            recommendation="prefer_mps",
            confidence="high",
            reason="score=0.8",
            metrics=metrics_by_profile["accurate"],
        ),
        _Recommendation(
            profile=candidates[1],
            recommendation="prefer_cpu",
            confidence="high",
            reason="score=0.6",
            metrics=metrics_by_profile["fast"],
        ),
    )
    assert execution.report_path == tmp_path / "runtime-calibration-report.json"
    assert captured["events"] == [("enter", "calibration"), ("exit", "calibration")]
    assert captured["persist"] == (
        tmp_path / "runtime-calibration-report.json",
        {
            "created_at_utc": "2026-03-09T00:00:00+00:00",
            "calibration_file": str(calibration_file),
            "iterations_per_profile": 3,
            "profiles": [
                {
                    "profile": "accurate",
                    "backend_id": "stable_whisper",
                    "model_name": "large",
                    "recommendation": "prefer_mps",
                    "confidence": "high",
                    "reason": "score=0.8",
                    "metrics": {
                        "backend_id": "stable_whisper",
                        "score": 0.8,
                    },
                },
                {
                    "profile": "fast",
                    "backend_id": "faster_whisper",
                    "model_name": "distil-large-v3",
                    "recommendation": "prefer_cpu",
                    "confidence": "high",
                    "reason": "score=0.6",
                    "metrics": {
                        "backend_id": "faster_whisper",
                        "score": 0.6,
                    },
                },
            ],
        },
    )


def test_execute_runtime_calibration_requires_existing_calibration_file(
    tmp_path: Path,
) -> None:
    """Workflow helper should fail fast when the calibration sample is missing."""

    with pytest.raises(FileNotFoundError, match="Calibration audio file not found"):
        workflow_helpers.execute_runtime_calibration(
            active_settings=_Settings(report_folder=tmp_path, label="base"),
            calibration_file=tmp_path / "missing.wav",
            language="en",
            iterations_per_profile=2,
            profile_names=("accurate",),
            report_path=None,
            build_runtime_calibration_settings=lambda settings: settings,
            settings_override=lambda _settings: _null_override(),
            runtime_calibration_candidates=lambda _profile_names: (),
            calibrate_candidate=lambda _candidate, _file_path, _language, _iterations: _Metrics(
                backend_id="stable_whisper",
                score=1.0,
            ),
            derive_runtime_recommendation=lambda _metrics: (
                "prefer_mps",
                "high",
                "unused",
            ),
            recommendation_factory=lambda candidate, recommendation, confidence, reason, metrics: _Recommendation(
                profile=candidate,
                recommendation=recommendation,
                confidence=confidence,
                reason=reason,
                metrics=metrics,
            ),
            runtime_calibration_report_path=lambda settings: settings.report_folder
            / "runtime-calibration-report.json",
            persist_profile_report=lambda output_path, _payload: output_path,
            serialize_recommendation=lambda recommendation: {
                "profile": recommendation.profile.source_profile,
            },
        )


def test_execute_runtime_calibration_validates_iteration_count(
    tmp_path: Path,
) -> None:
    """Workflow helper should reject non-positive iteration counts before work."""

    calibration_file = tmp_path / "sample.wav"
    calibration_file.write_bytes(b"audio")

    with pytest.raises(ValueError, match="iterations_per_profile"):
        workflow_helpers.execute_runtime_calibration(
            active_settings=_Settings(report_folder=tmp_path, label="base"),
            calibration_file=calibration_file,
            language="en",
            iterations_per_profile=0,
            profile_names=("accurate",),
            report_path=None,
            build_runtime_calibration_settings=lambda settings: settings,
            settings_override=lambda _settings: _null_override(),
            runtime_calibration_candidates=lambda _profile_names: (),
            calibrate_candidate=lambda _candidate, _file_path, _language, _iterations: _Metrics(
                backend_id="stable_whisper",
                score=1.0,
            ),
            derive_runtime_recommendation=lambda _metrics: (
                "prefer_mps",
                "high",
                "unused",
            ),
            recommendation_factory=lambda candidate, recommendation, confidence, reason, metrics: _Recommendation(
                profile=candidate,
                recommendation=recommendation,
                confidence=confidence,
                reason=reason,
                metrics=metrics,
            ),
            runtime_calibration_report_path=lambda settings: settings.report_folder
            / "runtime-calibration-report.json",
            persist_profile_report=lambda output_path, _payload: output_path,
            serialize_recommendation=lambda recommendation: {
                "profile": recommendation.profile.source_profile,
            },
        )
