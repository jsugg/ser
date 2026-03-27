"""Tests for calibration-driven MPS admission overrides."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import cast

from ser.config import AppConfig
from ser.transcript.backends.base import BackendRuntimeRequest
from ser.transcript.mps_admission import MpsAdmissionDecision
from ser.transcript.mps_admission_overrides import (
    apply_calibrated_mps_admission_override,
)


def _settings(
    *,
    report_path: Path,
    min_confidence: str = "high",
    max_age_hours: float = 168.0,
    enabled: bool = True,
) -> AppConfig:
    """Builds minimal settings object for override-resolution tests."""
    return cast(
        AppConfig,
        SimpleNamespace(
            transcription=SimpleNamespace(
                backend_id="stable_whisper",
                mps_admission_calibration_overrides_enabled=enabled,
                mps_admission_calibration_min_confidence=min_confidence,
                mps_admission_calibration_report_max_age_hours=max_age_hours,
                mps_admission_calibration_report_path=report_path,
            ),
            models=SimpleNamespace(folder=report_path.parent),
        ),
    )


def _write_report(
    *,
    path: Path,
    created_at: datetime,
    recommendation: str,
    confidence: str,
    model_name: str,
) -> None:
    """Writes one runtime-calibration report fixture."""
    payload = {
        "created_at_utc": created_at.isoformat(),
        "profiles": [
            {
                "profile": "accurate",
                "backend_id": "stable_whisper",
                "model_name": model_name,
                "recommendation": recommendation,
                "confidence": confidence,
                "reason": "test fixture",
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _runtime_request(model_name: str) -> BackendRuntimeRequest:
    """Builds one stable-whisper runtime request fixture."""
    return BackendRuntimeRequest(
        model_name=model_name,
        use_demucs=False,
        use_vad=True,
        device_spec="mps",
        device_type="mps",
        precision_candidates=("float16", "float32"),
        memory_tier="low",
    )


def test_calibrated_override_prefer_cpu_forces_admission_denial(tmp_path: Path) -> None:
    """High-confidence prefer_cpu recommendation should deny MPS admission."""
    report_path = tmp_path / "calibration.json"
    _write_report(
        path=report_path,
        created_at=datetime.now(tz=UTC),
        recommendation="prefer_cpu",
        confidence="high",
        model_name="large",
    )
    settings = _settings(report_path=report_path)
    heuristic = MpsAdmissionDecision(
        allow_mps=True,
        reason_code="mps_headroom_sufficient",
        required_bytes=1,
        available_bytes=2,
        required_metric="headroom_budget",
        available_metric="headroom",
        confidence="high",
    )

    resolved = apply_calibrated_mps_admission_override(
        settings=settings,
        runtime_request=_runtime_request("large"),
        phase="model_load",
        heuristic_decision=heuristic,
    )

    assert resolved.allow_mps is False
    assert resolved.reason_code == "calibration_prefer_cpu_model_load"


def test_calibrated_override_prefer_mps_relaxes_low_confidence_deny(
    tmp_path: Path,
) -> None:
    """prefer_mps should allow MPS when heuristic denial is low-confidence only."""
    report_path = tmp_path / "calibration.json"
    _write_report(
        path=report_path,
        created_at=datetime.now(tz=UTC),
        recommendation="prefer_mps",
        confidence="high",
        model_name="turbo",
    )
    settings = _settings(report_path=report_path)
    heuristic = MpsAdmissionDecision(
        allow_mps=False,
        reason_code="mps_headroom_estimate_below_required_budget",
        required_bytes=100,
        available_bytes=50,
        required_metric="headroom_budget",
        available_metric="headroom_estimate",
        confidence="low",
    )

    resolved = apply_calibrated_mps_admission_override(
        settings=settings,
        runtime_request=_runtime_request("turbo"),
        phase="transcribe",
        heuristic_decision=heuristic,
    )

    assert resolved.allow_mps is True
    assert resolved.reason_code == "calibration_prefer_mps_transcribe"


def test_calibrated_override_does_not_relax_high_confidence_deny(
    tmp_path: Path,
) -> None:
    """High-confidence heuristic denials should not be overridden to allow MPS."""
    report_path = tmp_path / "calibration.json"
    _write_report(
        path=report_path,
        created_at=datetime.now(tz=UTC),
        recommendation="prefer_mps",
        confidence="high",
        model_name="large",
    )
    settings = _settings(report_path=report_path)
    heuristic = MpsAdmissionDecision(
        allow_mps=False,
        reason_code="mps_headroom_below_required_budget",
        required_bytes=100,
        available_bytes=10,
        required_metric="headroom_budget",
        available_metric="headroom",
        confidence="high",
    )

    resolved = apply_calibrated_mps_admission_override(
        settings=settings,
        runtime_request=_runtime_request("large"),
        phase="transcribe",
        heuristic_decision=heuristic,
    )

    assert resolved == heuristic


def test_calibrated_override_ignores_stale_report(tmp_path: Path) -> None:
    """Stale calibration reports should be ignored safely."""
    report_path = tmp_path / "calibration.json"
    _write_report(
        path=report_path,
        created_at=datetime.now(tz=UTC) - timedelta(days=30),
        recommendation="prefer_cpu",
        confidence="high",
        model_name="large",
    )
    settings = _settings(
        report_path=report_path,
        max_age_hours=24.0,
    )
    heuristic = MpsAdmissionDecision(
        allow_mps=True,
        reason_code="mps_headroom_sufficient",
        required_bytes=1,
        available_bytes=2,
        required_metric="headroom_budget",
        available_metric="headroom",
        confidence="high",
    )

    resolved = apply_calibrated_mps_admission_override(
        settings=settings,
        runtime_request=_runtime_request("large"),
        phase="model_load",
        heuristic_decision=heuristic,
    )

    assert resolved == heuristic


def test_calibrated_override_respects_minimum_confidence(tmp_path: Path) -> None:
    """Recommendations below configured confidence threshold should be ignored."""
    report_path = tmp_path / "calibration.json"
    _write_report(
        path=report_path,
        created_at=datetime.now(tz=UTC),
        recommendation="prefer_cpu",
        confidence="medium",
        model_name="large",
    )
    settings = _settings(
        report_path=report_path,
        min_confidence="high",
    )
    heuristic = MpsAdmissionDecision(
        allow_mps=True,
        reason_code="mps_headroom_sufficient",
        required_bytes=1,
        available_bytes=2,
        required_metric="headroom_budget",
        available_metric="headroom",
        confidence="high",
    )

    resolved = apply_calibrated_mps_admission_override(
        settings=settings,
        runtime_request=_runtime_request("large"),
        phase="model_load",
        heuristic_decision=heuristic,
    )

    assert resolved == heuristic
