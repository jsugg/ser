"""Calibration-driven overrides for dynamic MPS admission decisions."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

from ser.transcript.backends.base import BackendRuntimeRequest
from ser.transcript.mps_admission import MpsAdmissionDecision

if TYPE_CHECKING:
    from ser.config import AppConfig

type RecommendationConfidence = Literal["high", "medium", "low"]
type RuntimeRecommendation = Literal["prefer_cpu", "prefer_mps", "mps_with_failover"]
type TranscriptionPhase = Literal["model_load", "transcribe"]

_REPORT_FILE_NAME = "transcription_runtime_calibration_report.json"
_CONFIDENCE_RANK: dict[RecommendationConfidence, int] = {
    "low": 1,
    "medium": 2,
    "high": 3,
}
_RECOMMENDATION_PRIORITY: dict[RuntimeRecommendation, int] = {
    "prefer_cpu": 3,
    "mps_with_failover": 2,
    "prefer_mps": 1,
}


@dataclass(frozen=True, slots=True)
class CalibrationRecommendation:
    """One calibration recommendation row keyed by backend/model."""

    backend_id: str
    model_name: str
    recommendation: RuntimeRecommendation
    confidence: RecommendationConfidence
    reason: str
    source_profile: str


def apply_calibrated_mps_admission_override(
    *,
    settings: AppConfig,
    runtime_request: BackendRuntimeRequest,
    phase: TranscriptionPhase,
    heuristic_decision: MpsAdmissionDecision,
) -> MpsAdmissionDecision:
    """Applies one optional calibration-based override over heuristic admission."""
    recommendation = _resolve_recommendation(
        settings=settings,
        runtime_request=runtime_request,
    )
    if recommendation is None:
        return heuristic_decision

    if recommendation.recommendation == "prefer_cpu":
        return MpsAdmissionDecision(
            allow_mps=False,
            reason_code=f"calibration_prefer_cpu_{phase}",
            required_bytes=heuristic_decision.required_bytes,
            available_bytes=heuristic_decision.available_bytes,
            required_metric=heuristic_decision.required_metric,
            available_metric=heuristic_decision.available_metric,
            confidence=recommendation.confidence,
        )

    if recommendation.recommendation == "prefer_mps":
        if heuristic_decision.allow_mps or heuristic_decision.confidence == "high":
            return heuristic_decision
        return MpsAdmissionDecision(
            allow_mps=True,
            reason_code=f"calibration_prefer_mps_{phase}",
            required_bytes=heuristic_decision.required_bytes,
            available_bytes=heuristic_decision.available_bytes,
            required_metric=heuristic_decision.required_metric,
            available_metric=heuristic_decision.available_metric,
            confidence=recommendation.confidence,
        )

    if heuristic_decision.allow_mps or heuristic_decision.confidence == "high":
        return heuristic_decision
    return MpsAdmissionDecision(
        allow_mps=True,
        reason_code=f"calibration_mps_with_failover_{phase}",
        required_bytes=heuristic_decision.required_bytes,
        available_bytes=heuristic_decision.available_bytes,
        required_metric=heuristic_decision.required_metric,
        available_metric=heuristic_decision.available_metric,
        confidence=recommendation.confidence,
    )


def _resolve_recommendation(
    *,
    settings: AppConfig,
    runtime_request: BackendRuntimeRequest,
) -> CalibrationRecommendation | None:
    """Resolves one recommendation matching runtime backend/model selectors."""
    logger = logging.getLogger(__name__)
    transcription_settings = getattr(settings, "transcription", None)
    if transcription_settings is None:
        logger.debug(
            "MPS admission calibrated override skipped: transcription settings unavailable."
        )
        return None
    if not bool(
        getattr(
            transcription_settings,
            "mps_admission_calibration_overrides_enabled",
            True,
        )
    ):
        logger.debug(
            "MPS admission calibrated override skipped for %s: disabled by config.",
            runtime_request.model_name,
        )
        return None
    report_path = _resolve_report_path(settings)
    created_at_utc, recommendations = _read_report(path=report_path)
    if created_at_utc is None:
        logger.debug(
            "MPS admission calibrated override skipped for %s: report unavailable or invalid (%s).",
            runtime_request.model_name,
            report_path,
        )
        return None
    max_age_hours = _resolve_max_age_hours(settings)
    if datetime.now(tz=UTC) - created_at_utc > timedelta(hours=max_age_hours):
        logger.debug(
            "MPS admission calibrated override skipped for %s: report is stale "
            "(created_at=%s, max_age_hours=%.1f).",
            runtime_request.model_name,
            created_at_utc.isoformat(),
            max_age_hours,
        )
        return None
    backend_id = (
        str(getattr(transcription_settings, "backend_id", "stable_whisper"))
        .strip()
        .lower()
    )
    key = (backend_id, runtime_request.model_name)
    recommendation = recommendations.get(key)
    if recommendation is None:
        logger.debug(
            "MPS admission calibrated override skipped for %s: no matching recommendation "
            "(backend=%s).",
            runtime_request.model_name,
            backend_id,
        )
        return None
    minimum_confidence = _resolve_minimum_confidence(settings)
    if not _confidence_at_least(
        recommendation.confidence,
        threshold=minimum_confidence,
    ):
        logger.debug(
            "MPS admission calibrated override skipped for %s: confidence %s below threshold %s.",
            runtime_request.model_name,
            recommendation.confidence,
            minimum_confidence,
        )
        return None
    logger.debug(
        "MPS admission calibrated override candidate loaded for %s/%s "
        "(recommendation=%s, confidence=%s).",
        backend_id,
        runtime_request.model_name,
        recommendation.recommendation,
        recommendation.confidence,
    )
    return recommendation


def _resolve_report_path(settings: AppConfig) -> Path:
    """Resolves report path using explicit config or default models folder path."""
    transcription_settings = getattr(settings, "transcription", None)
    configured = getattr(
        transcription_settings,
        "mps_admission_calibration_report_path",
        None,
    )
    if isinstance(configured, Path):
        return configured
    if isinstance(configured, str) and configured.strip():
        return Path(configured).expanduser()
    models = getattr(settings, "models", None)
    folder = getattr(models, "folder", None)
    if isinstance(folder, Path):
        return folder / _REPORT_FILE_NAME
    if isinstance(folder, str) and folder.strip():
        return Path(folder).expanduser() / _REPORT_FILE_NAME
    return Path(_REPORT_FILE_NAME)


def _resolve_minimum_confidence(settings: AppConfig) -> RecommendationConfidence:
    """Reads minimum confidence threshold for calibrated overrides."""
    transcription_settings = getattr(settings, "transcription", None)
    configured = (
        str(
            getattr(
                transcription_settings,
                "mps_admission_calibration_min_confidence",
                "high",
            )
        )
        .strip()
        .lower()
    )
    if configured in {"high", "medium", "low"}:
        return cast(RecommendationConfidence, configured)
    return "high"


def _resolve_max_age_hours(settings: AppConfig) -> float:
    """Reads maximum accepted calibration report age in hours."""
    transcription_settings = getattr(settings, "transcription", None)
    configured = getattr(
        transcription_settings,
        "mps_admission_calibration_report_max_age_hours",
        168.0,
    )
    if isinstance(configured, int | float) and not isinstance(configured, bool):
        return max(float(configured), 1.0)
    return 168.0


def _confidence_at_least(
    value: RecommendationConfidence,
    *,
    threshold: RecommendationConfidence,
) -> bool:
    """Returns whether recommendation confidence meets configured threshold."""
    return _CONFIDENCE_RANK[value] >= _CONFIDENCE_RANK[threshold]


@lru_cache(maxsize=8)
def _read_report_cached(
    resolved_path: str,
    mtime_ns: int,
) -> tuple[datetime | None, dict[tuple[str, str], CalibrationRecommendation]]:
    """Reads one calibration report from disk with mtime-aware cache key."""
    del mtime_ns
    path = Path(resolved_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logging.getLogger(__name__).debug(
            "Failed to load transcription runtime calibration report from %s.",
            path,
            exc_info=True,
        )
        return None, {}

    created_at_raw = payload.get("created_at_utc")
    if not isinstance(created_at_raw, str) or not created_at_raw.strip():
        return None, {}
    try:
        created_at_utc = datetime.fromisoformat(created_at_raw)
    except ValueError:
        return None, {}
    if created_at_utc.tzinfo is None:
        created_at_utc = created_at_utc.replace(tzinfo=UTC)
    rows = payload.get("profiles")
    if not isinstance(rows, list):
        return created_at_utc, {}
    recommendations: dict[tuple[str, str], CalibrationRecommendation] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        backend_id_raw = row.get("backend_id")
        model_name_raw = row.get("model_name")
        recommendation_raw = row.get("recommendation")
        confidence_raw = row.get("confidence")
        if not (
            isinstance(backend_id_raw, str)
            and isinstance(model_name_raw, str)
            and isinstance(recommendation_raw, str)
            and isinstance(confidence_raw, str)
        ):
            continue
        backend_id = backend_id_raw.strip().lower()
        model_name = model_name_raw.strip()
        recommendation_text = recommendation_raw.strip().lower()
        confidence_text = confidence_raw.strip().lower()
        if recommendation_text not in _RECOMMENDATION_PRIORITY:
            continue
        if confidence_text not in _CONFIDENCE_RANK:
            continue
        recommendation = CalibrationRecommendation(
            backend_id=backend_id,
            model_name=model_name,
            recommendation=cast(RuntimeRecommendation, recommendation_text),
            confidence=cast(RecommendationConfidence, confidence_text),
            reason=str(row.get("reason", "")).strip(),
            source_profile=str(row.get("profile", "")).strip(),
        )
        key = (backend_id, model_name)
        existing = recommendations.get(key)
        if existing is None:
            recommendations[key] = recommendation
            continue
        if (
            _CONFIDENCE_RANK[recommendation.confidence]
            > _CONFIDENCE_RANK[existing.confidence]
        ):
            recommendations[key] = recommendation
            continue
        if (
            _CONFIDENCE_RANK[recommendation.confidence]
            == _CONFIDENCE_RANK[existing.confidence]
            and _RECOMMENDATION_PRIORITY[recommendation.recommendation]
            > _RECOMMENDATION_PRIORITY[existing.recommendation]
        ):
            recommendations[key] = recommendation
    return created_at_utc, recommendations


def _read_report(
    *,
    path: Path,
) -> tuple[datetime | None, dict[tuple[str, str], CalibrationRecommendation]]:
    """Reads one calibration report using cached parser keyed by mtime."""
    try:
        stat_result = path.stat()
    except OSError:
        return None, {}
    return _read_report_cached(str(path), stat_result.st_mtime_ns)
