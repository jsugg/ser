"""Deterministic medium-segment postprocessing utilities."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import fmean

from ser.runtime.schema import FramePrediction, SegmentPrediction


@dataclass(frozen=True)
class SegmentPostprocessingConfig:
    """Controls smoothing, hysteresis, and short-segment cleanup."""

    smoothing_window_frames: int = 3
    hysteresis_enter_confidence: float = 0.60
    hysteresis_exit_confidence: float = 0.45
    min_segment_duration_seconds: float = 0.40


def postprocess_frame_predictions(
    frame_predictions: Sequence[FramePrediction],
    *,
    config: SegmentPostprocessingConfig,
) -> list[SegmentPrediction]:
    """Converts frame predictions into stable medium segments.

    Args:
        frame_predictions: Ordered frame-level predictions.
        config: Postprocessing controls.

    Returns:
        Segment-level predictions after smoothing, hysteresis, and cleanup.
    """
    if not frame_predictions:
        return []
    _validate_config(config)

    smoothed_labels = _smooth_labels(
        labels=[frame.emotion for frame in frame_predictions],
        window_size=config.smoothing_window_frames,
    )
    stabilized_labels = _apply_confidence_hysteresis(
        labels=smoothed_labels,
        frame_predictions=frame_predictions,
        enter_confidence=config.hysteresis_enter_confidence,
        exit_confidence=config.hysteresis_exit_confidence,
    )
    segments = _build_segments(frame_predictions, stabilized_labels)
    merged_segments = _merge_short_segments(
        segments,
        min_duration_seconds=config.min_segment_duration_seconds,
    )
    return _merge_adjacent_same_label(merged_segments)


def _validate_config(config: SegmentPostprocessingConfig) -> None:
    """Validates postprocessing controls before execution."""
    if config.smoothing_window_frames < 1:
        raise ValueError("smoothing_window_frames must be greater than or equal to 1.")
    if config.hysteresis_enter_confidence < 0.0:
        raise ValueError("hysteresis_enter_confidence cannot be negative.")
    if config.hysteresis_exit_confidence < 0.0:
        raise ValueError("hysteresis_exit_confidence cannot be negative.")
    if config.hysteresis_enter_confidence < config.hysteresis_exit_confidence:
        raise ValueError(
            "hysteresis_enter_confidence must be greater than or equal to "
            "hysteresis_exit_confidence."
        )
    if config.min_segment_duration_seconds < 0.0:
        raise ValueError("min_segment_duration_seconds cannot be negative.")


def _smooth_labels(*, labels: Sequence[str], window_size: int) -> list[str]:
    """Applies deterministic majority-vote smoothing over label sequence."""
    if not labels:
        return []
    if window_size <= 1:
        return [str(label) for label in labels]

    radius = window_size // 2
    smoothed: list[str] = []
    for index, label in enumerate(labels):
        start = max(0, index - radius)
        end = min(len(labels), index + radius + 1)
        window_labels = [str(item) for item in labels[start:end]]
        counts = Counter(window_labels)
        max_count = max(counts.values())
        candidates = [item for item, count in counts.items() if count == max_count]
        if label in candidates:
            smoothed.append(str(label))
            continue
        previous_label = smoothed[-1] if smoothed else str(labels[0])
        if previous_label in candidates:
            smoothed.append(previous_label)
            continue
        smoothed.append(sorted(candidates)[0])
    return smoothed


def _apply_confidence_hysteresis(
    *,
    labels: Sequence[str],
    frame_predictions: Sequence[FramePrediction],
    enter_confidence: float,
    exit_confidence: float,
) -> list[str]:
    """Applies confidence-gated label transitions to reduce jitter."""
    if len(labels) != len(frame_predictions):
        raise ValueError("labels and frame_predictions must have identical length.")
    if not labels:
        return []
    if enter_confidence <= 0.0 and exit_confidence <= 0.0:
        return [str(label) for label in labels]

    stabilized: list[str] = [str(labels[0])]
    current_label = str(labels[0])
    current_confidence = float(frame_predictions[0].confidence)
    for index in range(1, len(labels)):
        candidate_label = str(labels[index])
        candidate_confidence = float(frame_predictions[index].confidence)
        if candidate_label == current_label:
            current_confidence = candidate_confidence
            stabilized.append(current_label)
            continue

        can_switch = candidate_confidence >= enter_confidence and (
            current_confidence <= exit_confidence
            or candidate_confidence >= current_confidence
        )
        if can_switch:
            current_label = candidate_label
            current_confidence = candidate_confidence
        stabilized.append(current_label)
    return stabilized


def _build_segments(
    frame_predictions: Sequence[FramePrediction],
    labels: Sequence[str],
) -> list[SegmentPrediction]:
    """Builds contiguous segments from frame predictions and resolved labels."""
    if not frame_predictions:
        return []
    if len(frame_predictions) != len(labels):
        raise ValueError("frame_predictions and labels must have identical length.")

    segment_ranges: list[tuple[int, int, str]] = []
    start_index = 0
    active_label = str(labels[0])
    for index in range(1, len(labels)):
        label = str(labels[index])
        if label == active_label:
            continue
        segment_ranges.append((start_index, index - 1, active_label))
        start_index = index
        active_label = label
    segment_ranges.append((start_index, len(labels) - 1, active_label))

    segments: list[SegmentPrediction] = []
    for start, end, emotion in segment_ranges:
        frame_slice = frame_predictions[start : end + 1]
        segments.append(
            SegmentPrediction(
                emotion=emotion,
                start_seconds=float(frame_slice[0].start_seconds),
                end_seconds=float(frame_slice[-1].end_seconds),
                confidence=float(fmean(item.confidence for item in frame_slice)),
                probabilities=_aggregate_probabilities(
                    [item.probabilities for item in frame_slice]
                ),
            )
        )
    return segments


def _merge_short_segments(
    segments: Sequence[SegmentPrediction],
    *,
    min_duration_seconds: float,
) -> list[SegmentPrediction]:
    """Merges segments shorter than configured duration into adjacent segments."""
    if not segments:
        return []
    if min_duration_seconds <= 0.0:
        return list(segments)
    if len(segments) == 1:
        return list(segments)

    merged = list(segments)
    index = 0
    while index < len(merged):
        if len(merged) == 1:
            break
        current = merged[index]
        if _segment_duration(current) >= min_duration_seconds:
            index += 1
            continue

        if index == 0:
            target_index = 1
        elif index == len(merged) - 1:
            target_index = index - 1
        else:
            previous = merged[index - 1]
            following = merged[index + 1]
            target_index = (
                index - 1 if previous.confidence >= following.confidence else index + 1
            )

        target = merged[target_index]
        merged_segment = _merge_into_target(target=target, source=current)
        if target_index < index:
            merged[target_index] = merged_segment
            del merged[index]
            index = max(0, target_index)
            continue

        merged[target_index] = merged_segment
        del merged[index]
        index = max(0, target_index - 1)
    return merged


def _merge_adjacent_same_label(
    segments: Sequence[SegmentPrediction],
) -> list[SegmentPrediction]:
    """Collapses adjacent segments with equal labels."""
    if not segments:
        return []
    normalized: list[SegmentPrediction] = [segments[0]]
    for segment in segments[1:]:
        previous = normalized[-1]
        if segment.emotion != previous.emotion:
            normalized.append(segment)
            continue
        normalized[-1] = _merge_into_target(target=previous, source=segment)
    return normalized


def _merge_into_target(
    *,
    target: SegmentPrediction,
    source: SegmentPrediction,
) -> SegmentPrediction:
    """Merges one segment into a target segment while preserving target emotion."""
    target_duration = _segment_duration(target)
    source_duration = _segment_duration(source)
    total_duration = target_duration + source_duration
    if total_duration <= 0.0:
        confidence = float(fmean([target.confidence, source.confidence]))
    else:
        confidence = (
            (target.confidence * target_duration)
            + (source.confidence * source_duration)
        ) / total_duration

    probabilities = _merge_probability_maps(
        target=target.probabilities,
        source=source.probabilities,
        target_weight=max(target_duration, 1e-12),
        source_weight=max(source_duration, 1e-12),
    )
    return SegmentPrediction(
        emotion=target.emotion,
        start_seconds=min(target.start_seconds, source.start_seconds),
        end_seconds=max(target.end_seconds, source.end_seconds),
        confidence=float(confidence),
        probabilities=probabilities,
    )


def _merge_probability_maps(
    *,
    target: dict[str, float] | None,
    source: dict[str, float] | None,
    target_weight: float,
    source_weight: float,
) -> dict[str, float] | None:
    """Computes weighted average for optional segment-level probability mappings."""
    if target is None and source is None:
        return None
    if target is None:
        return {key: float(value) for key, value in source.items()} if source else None
    if source is None:
        return {key: float(value) for key, value in target.items()}

    total = target_weight + source_weight
    labels = sorted(set(target.keys()) | set(source.keys()))
    return {
        label: float(
            (
                (target.get(label, 0.0) * target_weight)
                + (source.get(label, 0.0) * source_weight)
            )
            / total
        )
        for label in labels
    }


def _segment_duration(segment: SegmentPrediction) -> float:
    """Returns safe segment duration in seconds."""
    return max(0.0, float(segment.end_seconds) - float(segment.start_seconds))


def _aggregate_probabilities(
    probabilities: Sequence[dict[str, float] | None],
) -> dict[str, float] | None:
    """Aggregates frame-level probabilities into one segment-level mapping."""
    valid = [item for item in probabilities if item is not None]
    if not valid:
        return None
    labels = sorted({label for item in valid for label in item.keys()})
    return {
        label: float(fmean(float(item.get(label, 0.0)) for item in valid))
        for label in labels
    }
