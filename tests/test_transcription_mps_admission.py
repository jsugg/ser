"""Tests for dynamic MPS admission control decisions."""

from __future__ import annotations

import pytest

from ser.transcript import mps_admission


def test_mps_admission_allows_when_pressure_snapshot_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unavailable pressure telemetry should keep MPS enabled conservatively."""
    monkeypatch.setattr(
        mps_admission,
        "capture_mps_pressure_snapshot",
        lambda: mps_admission.MpsPressureSnapshot(
            is_available=False,
            current_allocated_bytes=None,
            driver_allocated_bytes=None,
            recommended_max_bytes=None,
            headroom_bytes=None,
        ),
    )

    decision = mps_admission.decide_mps_admission_for_transcription(
        model_name="turbo",
        phase="transcribe",
    )

    assert decision.allow_mps is True
    assert decision.reason_code == "mps_pressure_unavailable"


def test_mps_admission_denies_when_headroom_is_below_required_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Insufficient MPS headroom should force CPU fallback for transcription."""
    monkeypatch.setattr(
        mps_admission,
        "capture_mps_pressure_snapshot",
        lambda: mps_admission.MpsPressureSnapshot(
            is_available=True,
            current_allocated_bytes=0,
            driver_allocated_bytes=0,
            recommended_max_bytes=0,
            headroom_bytes=32 * 1024**2,
        ),
    )

    decision = mps_admission.decide_mps_admission_for_transcription(
        model_name="large",
        phase="model_load",
        min_headroom_mb=64.0,
        safety_margin_mb=64.0,
    )

    assert decision.allow_mps is False
    assert decision.reason_code == "mps_headroom_below_required_budget"
    assert decision.available_bytes == 32 * 1024**2
    assert decision.required_bytes is not None
    assert decision.required_bytes > decision.available_bytes


def test_mps_admission_allows_when_headroom_is_sufficient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sufficient MPS headroom should keep MPS runtime admitted."""
    monkeypatch.setattr(
        mps_admission,
        "capture_mps_pressure_snapshot",
        lambda: mps_admission.MpsPressureSnapshot(
            is_available=True,
            current_allocated_bytes=0,
            driver_allocated_bytes=0,
            recommended_max_bytes=0,
            headroom_bytes=512 * 1024**2,
        ),
    )

    decision = mps_admission.decide_mps_admission_for_transcription(
        model_name="turbo",
        phase="transcribe",
        min_headroom_mb=32.0,
        safety_margin_mb=16.0,
    )

    assert decision.allow_mps is True
    assert decision.reason_code == "mps_headroom_sufficient"


def test_mps_admission_uses_driver_guard_when_headroom_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Driver-allocation guard should deny MPS when nearing known pressure limits."""
    monkeypatch.setattr(
        mps_admission,
        "capture_mps_pressure_snapshot",
        lambda: mps_admission.MpsPressureSnapshot(
            is_available=True,
            current_allocated_bytes=None,
            driver_allocated_bytes=int(3.30 * float(1024**3)),
            recommended_max_bytes=None,
            headroom_bytes=None,
        ),
    )

    decision = mps_admission.decide_mps_admission_for_transcription(
        model_name="turbo",
        phase="model_load",
    )

    assert decision.allow_mps is False
    assert decision.reason_code == "mps_headroom_estimate_below_required_budget"


def test_mps_admission_denies_large_model_load_when_headroom_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Large model load should fail over to CPU when MPS headroom cannot be measured."""
    monkeypatch.setattr(
        mps_admission,
        "capture_mps_pressure_snapshot",
        lambda: mps_admission.MpsPressureSnapshot(
            is_available=True,
            current_allocated_bytes=None,
            driver_allocated_bytes=None,
            recommended_max_bytes=None,
            headroom_bytes=None,
        ),
    )

    decision = mps_admission.decide_mps_admission_for_transcription(
        model_name="large",
        phase="model_load",
    )

    assert decision.allow_mps is False
    assert decision.reason_code == "mps_headroom_unknown_large_model"


def test_mps_admission_model_load_budget_tracks_model_footprint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Large model-load admission should require near-footprint headroom budget."""
    monkeypatch.setattr(
        mps_admission,
        "capture_mps_pressure_snapshot",
        lambda: mps_admission.MpsPressureSnapshot(
            is_available=True,
            current_allocated_bytes=0,
            driver_allocated_bytes=0,
            recommended_max_bytes=int(4.00 * float(1024**3)),
            headroom_bytes=int(4.00 * float(1024**3)),
        ),
    )

    decision = mps_admission.decide_mps_admission_for_transcription(
        model_name="large",
        phase="model_load",
        min_headroom_mb=64.0,
        safety_margin_mb=64.0,
    )

    assert decision.required_bytes is not None
    assert decision.required_bytes >= int(3.15 * float(1024**3))
