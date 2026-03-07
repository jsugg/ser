"""Dynamic MPS admission control for transcription runtime decisions."""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass
from types import ModuleType
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from ser.config import AppConfig
    from ser.transcript.backends.base import BackendRuntimeRequest

DEFAULT_MPS_ADMISSION_MIN_HEADROOM_MB: float = 64.0
DEFAULT_MPS_ADMISSION_SAFETY_MARGIN_MB: float = 64.0

type TranscriptionPhase = Literal["model_load", "transcribe"]
type MpsAdmissionHeuristicResolver = Callable[
    ...,
    MpsAdmissionDecision,
]


@dataclass(frozen=True, slots=True)
class MpsPressureSnapshot:
    """Best-effort snapshot of current MPS allocator pressure."""

    is_available: bool
    current_allocated_bytes: int | None
    driver_allocated_bytes: int | None
    recommended_max_bytes: int | None
    headroom_bytes: int | None


@dataclass(frozen=True, slots=True)
class MpsAdmissionDecision:
    """Decision for whether MPS should be used for a transcription phase."""

    allow_mps: bool
    reason_code: str
    required_bytes: int | None
    available_bytes: int | None
    required_metric: str
    available_metric: str
    confidence: Literal["high", "medium", "low"]


def decide_mps_admission_for_transcription(
    *,
    model_name: str,
    phase: TranscriptionPhase,
    min_headroom_mb: float = DEFAULT_MPS_ADMISSION_MIN_HEADROOM_MB,
    safety_margin_mb: float = DEFAULT_MPS_ADMISSION_SAFETY_MARGIN_MB,
) -> MpsAdmissionDecision:
    """Decides whether current MPS headroom can safely run one transcription phase."""
    snapshot = capture_mps_pressure_snapshot()
    model_required_bytes = _estimate_required_headroom_bytes(
        model_name=model_name,
        phase=phase,
    )
    min_required_bytes = _mb_to_bytes(min_headroom_mb)
    safety_margin_bytes = _mb_to_bytes(safety_margin_mb)
    required_headroom_bytes = (
        max(min_required_bytes, model_required_bytes) + safety_margin_bytes
    )
    if not snapshot.is_available:
        return MpsAdmissionDecision(
            allow_mps=True,
            reason_code="mps_pressure_unavailable",
            required_bytes=None,
            available_bytes=None,
            required_metric="headroom_budget",
            available_metric="headroom_estimate",
            confidence="low",
        )
    if snapshot.headroom_bytes is None:
        if phase == "model_load" and _is_large_whisper_model(model_name):
            estimated_model_footprint_bytes = _estimate_model_footprint_bytes(
                model_name
            )
            return MpsAdmissionDecision(
                allow_mps=False,
                reason_code="mps_headroom_unknown_large_model",
                required_bytes=estimated_model_footprint_bytes,
                available_bytes=snapshot.driver_allocated_bytes,
                required_metric="model_footprint_estimate",
                available_metric="driver_allocated",
                confidence="medium",
            )
        estimated_headroom_bytes = _estimate_headroom_without_recommended_max(
            snapshot=snapshot,
            model_name=model_name,
            phase=phase,
        )
        if estimated_headroom_bytes is None:
            return MpsAdmissionDecision(
                allow_mps=True,
                reason_code="mps_headroom_unknown",
                required_bytes=None,
                available_bytes=None,
                required_metric="headroom_budget",
                available_metric="headroom_estimate",
                confidence="low",
            )
        allow_mps = estimated_headroom_bytes >= required_headroom_bytes
        return MpsAdmissionDecision(
            allow_mps=allow_mps,
            reason_code=(
                "mps_headroom_estimate_sufficient"
                if allow_mps
                else "mps_headroom_estimate_below_required_budget"
            ),
            required_bytes=required_headroom_bytes,
            available_bytes=estimated_headroom_bytes,
            required_metric="headroom_budget",
            available_metric="headroom_estimate",
            confidence="low",
        )
    allow_mps = snapshot.headroom_bytes >= required_headroom_bytes
    return MpsAdmissionDecision(
        allow_mps=allow_mps,
        reason_code=(
            "mps_headroom_sufficient"
            if allow_mps
            else "mps_headroom_below_required_budget"
        ),
        required_bytes=required_headroom_bytes,
        available_bytes=snapshot.headroom_bytes,
        required_metric="headroom_budget",
        available_metric="headroom",
        confidence="high",
    )


def resolve_mps_admission_decision(
    *,
    settings: AppConfig,
    runtime_request: BackendRuntimeRequest,
    phase: TranscriptionPhase,
    logger: logging.Logger,
    heuristic_resolver: MpsAdmissionHeuristicResolver | None = None,
) -> MpsAdmissionDecision | None:
    """Resolves calibrated MPS admission decision when control is enabled."""
    if not mps_admission_control_enabled(settings):
        return None
    transcription_settings = getattr(settings, "transcription", None)
    configured_min_headroom_mb = getattr(
        transcription_settings,
        "mps_admission_min_headroom_mb",
        DEFAULT_MPS_ADMISSION_MIN_HEADROOM_MB,
    )
    configured_safety_margin_mb = getattr(
        transcription_settings,
        "mps_admission_safety_margin_mb",
        DEFAULT_MPS_ADMISSION_SAFETY_MARGIN_MB,
    )
    min_headroom_mb = (
        configured_min_headroom_mb
        if isinstance(configured_min_headroom_mb, int | float)
        and not isinstance(configured_min_headroom_mb, bool)
        else DEFAULT_MPS_ADMISSION_MIN_HEADROOM_MB
    )
    safety_margin_mb = (
        configured_safety_margin_mb
        if isinstance(configured_safety_margin_mb, int | float)
        and not isinstance(configured_safety_margin_mb, bool)
        else DEFAULT_MPS_ADMISSION_SAFETY_MARGIN_MB
    )
    resolve_heuristic = (
        decide_mps_admission_for_transcription
        if heuristic_resolver is None
        else heuristic_resolver
    )
    heuristic_decision = resolve_heuristic(
        model_name=runtime_request.model_name,
        phase=phase,
        min_headroom_mb=float(min_headroom_mb),
        safety_margin_mb=float(safety_margin_mb),
    )
    # Local import avoids cyclic dependency with mps_admission_overrides.
    from ser.transcript.mps_admission_overrides import (
        apply_calibrated_mps_admission_override,
    )

    resolved_decision = apply_calibrated_mps_admission_override(
        settings=settings,
        runtime_request=runtime_request,
        phase=phase,
        heuristic_decision=heuristic_decision,
    )
    if resolved_decision.reason_code != heuristic_decision.reason_code:
        logger.info(
            "MPS admission calibrated override applied for %s "
            "(model=%s, reason=%s, confidence=%s).",
            phase,
            runtime_request.model_name,
            resolved_decision.reason_code,
            resolved_decision.confidence,
        )
    return resolved_decision


def mps_admission_control_enabled(settings: AppConfig) -> bool:
    """Returns whether dynamic MPS admission control is enabled by config."""
    transcription_settings = getattr(settings, "transcription", None)
    enabled = getattr(transcription_settings, "mps_admission_control_enabled", True)
    return bool(enabled)


def mps_hard_oom_shortcut_enabled(settings: AppConfig) -> bool:
    """Returns whether hard MPS OOM shortcut is enabled by config."""
    transcription_settings = getattr(settings, "transcription", None)
    enabled = getattr(transcription_settings, "mps_hard_oom_shortcut_enabled", True)
    return bool(enabled)


def should_enforce_transcribe_admission(decision: MpsAdmissionDecision) -> bool:
    """Returns whether one pre-transcribe admission denial should be enforced."""
    if decision.confidence == "high":
        return True
    return decision.reason_code == "mps_headroom_unknown_large_model"


def log_mps_admission_control_fallback(
    *,
    phase: str,
    decision: MpsAdmissionDecision,
    logger: logging.Logger,
) -> None:
    """Logs one concise MPS admission-control fallback message."""
    logger.info(
        "MPS admission control switched %s to cpu "
        "(reason=%s, confidence=%s, required_%s=%s, available_%s=%s).",
        phase,
        decision.reason_code,
        decision.confidence,
        decision.required_metric,
        format_gib_short(decision.required_bytes),
        decision.available_metric,
        format_gib_short(decision.available_bytes),
    )


def capture_mps_pressure_snapshot() -> MpsPressureSnapshot:
    """Captures one best-effort MPS memory pressure snapshot."""
    torch_module = sys.modules.get("torch")
    if not isinstance(torch_module, ModuleType):
        return MpsPressureSnapshot(
            is_available=False,
            current_allocated_bytes=None,
            driver_allocated_bytes=None,
            recommended_max_bytes=None,
            headroom_bytes=None,
        )
    backends = getattr(torch_module, "backends", None)
    mps_backend = getattr(backends, "mps", None)
    is_available = getattr(mps_backend, "is_available", None)
    is_built = getattr(mps_backend, "is_built", None)
    if not callable(is_available) or not callable(is_built):
        return MpsPressureSnapshot(
            is_available=False,
            current_allocated_bytes=None,
            driver_allocated_bytes=None,
            recommended_max_bytes=None,
            headroom_bytes=None,
        )
    try:
        if not is_available() or not is_built():
            return MpsPressureSnapshot(
                is_available=False,
                current_allocated_bytes=None,
                driver_allocated_bytes=None,
                recommended_max_bytes=None,
                headroom_bytes=None,
            )
    except Exception:
        return MpsPressureSnapshot(
            is_available=False,
            current_allocated_bytes=None,
            driver_allocated_bytes=None,
            recommended_max_bytes=None,
            headroom_bytes=None,
        )
    mps_module = getattr(torch_module, "mps", None)
    if not isinstance(mps_module, ModuleType):
        return MpsPressureSnapshot(
            is_available=False,
            current_allocated_bytes=None,
            driver_allocated_bytes=None,
            recommended_max_bytes=None,
            headroom_bytes=None,
        )

    current_allocated_bytes = _safe_read_mps_bytes(
        module=mps_module,
        attribute_name="current_allocated_memory",
    )
    driver_allocated_bytes = _safe_read_mps_bytes(
        module=mps_module,
        attribute_name="driver_allocated_memory",
    )
    recommended_max_bytes = _safe_read_mps_bytes(
        module=mps_module,
        attribute_name="recommended_max_memory",
    )
    used_candidates = tuple(
        value
        for value in (current_allocated_bytes, driver_allocated_bytes)
        if value is not None
    )
    effective_used_bytes = max(used_candidates) if used_candidates else None
    headroom_bytes = None
    if recommended_max_bytes is not None and effective_used_bytes is not None:
        headroom_bytes = max(recommended_max_bytes - effective_used_bytes, 0)
    return MpsPressureSnapshot(
        is_available=True,
        current_allocated_bytes=current_allocated_bytes,
        driver_allocated_bytes=driver_allocated_bytes,
        recommended_max_bytes=recommended_max_bytes,
        headroom_bytes=headroom_bytes,
    )


def format_gib_short(value_bytes: int | None) -> str:
    """Formats one byte value as a short GiB string for concise logs."""
    if value_bytes is None:
        return "unknown"
    gib_value = float(value_bytes) / float(1024**3)
    return f"{gib_value:.2f} GiB"


def _safe_read_mps_bytes(*, module: ModuleType, attribute_name: str) -> int | None:
    """Reads one optional MPS memory counter."""
    reader = getattr(module, attribute_name, None)
    if not callable(reader):
        return None
    try:
        value = reader()
    except Exception:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        parsed = int(value)
        return parsed if parsed >= 0 else None
    return None


def _estimate_required_headroom_bytes(
    *,
    model_name: str,
    phase: TranscriptionPhase,
) -> int:
    """Estimates required MPS headroom for one model and transcription phase."""
    normalized = model_name.strip().lower()
    if phase == "model_load":
        # Model load requires room for weights and allocator overhead.
        model_footprint_bytes = _estimate_model_footprint_bytes(model_name)
        minimum_budget_bytes = _mb_to_bytes(96.0)
        return max(model_footprint_bytes, minimum_budget_bytes)
    if normalized in {"large", "large-v1", "large-v2", "large-v3"}:
        return _mb_to_bytes(128.0)
    if normalized in {"turbo", "large-v3-turbo", "distil-large-v3"}:
        return _mb_to_bytes(96.0)
    if normalized.startswith("medium"):
        return _mb_to_bytes(96.0)
    if normalized.startswith("small"):
        return _mb_to_bytes(64.0)
    if normalized.startswith("base") or normalized.startswith("tiny"):
        return _mb_to_bytes(48.0)
    return _mb_to_bytes(64.0)


def _estimate_driver_allocation_guard_bytes(
    *,
    model_name: str,
    phase: TranscriptionPhase,
) -> int:
    """Estimates one absolute driver-allocation guard when headroom is unavailable."""
    normalized = model_name.strip().lower()
    if phase == "model_load":
        if normalized in {"large", "large-v1", "large-v2", "large-v3"}:
            return int(3.10 * float(1024**3))
        if normalized in {"turbo", "large-v3-turbo", "distil-large-v3"}:
            return int(3.25 * float(1024**3))
        return int(3.20 * float(1024**3))
    if normalized in {"large", "large-v1", "large-v2", "large-v3"}:
        return int(3.20 * float(1024**3))
    if normalized in {"turbo", "large-v3-turbo", "distil-large-v3"}:
        return int(3.25 * float(1024**3))
    return int(3.30 * float(1024**3))


def _estimate_headroom_without_recommended_max(
    *,
    snapshot: MpsPressureSnapshot,
    model_name: str,
    phase: TranscriptionPhase,
) -> int | None:
    """Estimates headroom when torch does not expose recommended_max_memory()."""
    if snapshot.driver_allocated_bytes is None:
        return None
    driver_guard_bytes = _estimate_driver_allocation_guard_bytes(
        model_name=model_name,
        phase=phase,
    )
    guard_headroom_bytes = max(driver_guard_bytes - snapshot.driver_allocated_bytes, 0)
    system_available_bytes = _read_system_available_memory_bytes()
    if system_available_bytes is None:
        return guard_headroom_bytes
    return min(guard_headroom_bytes, system_available_bytes)


def _read_system_available_memory_bytes() -> int | None:
    """Reads host available memory as one soft upper bound for unified-memory pressure."""
    try:
        import psutil
    except ModuleNotFoundError:
        return None
    try:
        available_bytes = int(psutil.virtual_memory().available)
    except (AttributeError, OSError, ValueError):
        return None
    return available_bytes if available_bytes >= 0 else None


def _is_large_whisper_model(model_name: str) -> bool:
    """Returns whether one model belongs to Whisper large family variants."""
    normalized = model_name.strip().lower()
    return normalized in {"large", "large-v1", "large-v2", "large-v3"}


def _estimate_model_footprint_bytes(model_name: str) -> int:
    """Estimates one model footprint for conservative admission diagnostics."""
    normalized = model_name.strip().lower()
    if normalized in {"large", "large-v1", "large-v2", "large-v3"}:
        return int(3.10 * float(1024**3))
    if normalized in {"turbo", "large-v3-turbo", "distil-large-v3"}:
        return int(1.55 * float(1024**3))
    if normalized.startswith("medium"):
        return int(1.20 * float(1024**3))
    if normalized.startswith("small"):
        return int(0.75 * float(1024**3))
    if normalized.startswith("base") or normalized.startswith("tiny"):
        return int(0.45 * float(1024**3))
    return int(1.00 * float(1024**3))


def _mb_to_bytes(value_mb: float) -> int:
    """Converts one MB value to bytes with input normalization."""
    normalized = value_mb if value_mb > 0.0 else 0.0
    return int(normalized * float(1024**2))
