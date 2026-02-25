"""Deterministic temporal segment canonicalization helpers."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


class SegmentLike(Protocol):
    """Structural segment contract used for canonicalization."""

    @property
    def emotion(self) -> str:
        """Segment emotion label."""
        ...

    @property
    def start_seconds(self) -> float:
        """Segment start timestamp in seconds."""
        ...

    @property
    def end_seconds(self) -> float:
        """Segment end timestamp in seconds."""
        ...


@dataclass(frozen=True)
class CanonicalSegment:
    """Canonical non-overlapping segment record."""

    emotion: str
    start_seconds: float
    end_seconds: float


@dataclass(frozen=True)
class _CandidateSegment:
    """Validated candidate used during canonicalization."""

    emotion: str
    start_seconds: float
    end_seconds: float
    confidence: float | None


@dataclass
class _MutableCanonicalSegment:
    """Mutable segment used for in-place canonical assembly."""

    emotion: str
    start_seconds: float
    end_seconds: float


def _read_optional_confidence(segment: SegmentLike) -> float | None:
    """Returns finite confidence when available on the input segment."""
    confidence_value = getattr(segment, "confidence", None)
    if confidence_value is None:
        return None
    try:
        confidence = float(confidence_value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(confidence):
        return None
    return confidence


def _build_candidate(segment: SegmentLike) -> _CandidateSegment | None:
    """Builds a validated candidate segment or returns ``None`` when invalid."""
    emotion = str(segment.emotion).strip()
    if not emotion:
        return None
    start_seconds = float(segment.start_seconds)
    end_seconds = float(segment.end_seconds)
    if not (math.isfinite(start_seconds) and math.isfinite(end_seconds)):
        return None
    if end_seconds <= start_seconds:
        return None
    return _CandidateSegment(
        emotion=emotion,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        confidence=_read_optional_confidence(segment),
    )


def _winner_for_same_start(
    candidates: list[_CandidateSegment],
) -> _CandidateSegment:
    """Chooses deterministic winner when multiple labels share a start timestamp."""
    by_label: dict[str, _CandidateSegment] = {}
    for candidate in candidates:
        existing = by_label.get(candidate.emotion)
        if existing is None or candidate.end_seconds > existing.end_seconds:
            by_label[candidate.emotion] = candidate

    deduplicated = list(by_label.values())
    return min(
        deduplicated,
        key=lambda candidate: (
            -(
                candidate.confidence
                if candidate.confidence is not None
                else float("-inf")
            ),
            candidate.emotion,
        ),
    )


def _append_candidate(
    canonical: list[_MutableCanonicalSegment],
    candidate: _CandidateSegment,
) -> None:
    """Appends a candidate while preserving non-overlap canonical invariants."""
    incoming = _MutableCanonicalSegment(
        emotion=candidate.emotion,
        start_seconds=candidate.start_seconds,
        end_seconds=candidate.end_seconds,
    )
    if not canonical:
        canonical.append(incoming)
        return

    previous = canonical[-1]
    if incoming.start_seconds < previous.end_seconds:
        if incoming.emotion == previous.emotion:
            previous.end_seconds = max(previous.end_seconds, incoming.end_seconds)
            return
        previous.end_seconds = incoming.start_seconds
        if previous.end_seconds <= previous.start_seconds:
            canonical.pop()
        canonical.append(incoming)
        return

    if (
        incoming.start_seconds == previous.end_seconds
        and incoming.emotion == previous.emotion
    ):
        previous.end_seconds = max(previous.end_seconds, incoming.end_seconds)
        return

    canonical.append(incoming)


def canonicalize_segments(segments: Sequence[SegmentLike]) -> list[CanonicalSegment]:
    """Canonicalizes segments into sorted, non-overlapping, positive-duration output.

    Rules:
    1. Adjacent/overlapping segments with the same label are merged.
    2. Overlaps with different labels are truncated at the newer segment start.
    3. Conflicting labels at the same start are resolved by higher confidence,
       then lexical label order when confidence is unavailable/tied.
    """
    validated = [
        candidate
        for candidate in (_build_candidate(segment) for segment in segments)
        if candidate is not None
    ]
    if not validated:
        return []

    validated.sort(key=lambda candidate: (candidate.start_seconds, candidate.end_seconds))
    selected_by_start: list[_CandidateSegment] = []
    group_start = 0
    while group_start < len(validated):
        group_end = group_start + 1
        group_start_seconds = validated[group_start].start_seconds
        while (
            group_end < len(validated)
            and validated[group_end].start_seconds == group_start_seconds
        ):
            group_end += 1
        selected_by_start.append(
            _winner_for_same_start(validated[group_start:group_end])
        )
        group_start = group_end

    canonical_mutable: list[_MutableCanonicalSegment] = []
    for candidate in selected_by_start:
        _append_candidate(canonical_mutable, candidate)

    return [
        CanonicalSegment(
            emotion=segment.emotion,
            start_seconds=segment.start_seconds,
            end_seconds=segment.end_seconds,
        )
        for segment in canonical_mutable
        if segment.end_seconds > segment.start_seconds
    ]
