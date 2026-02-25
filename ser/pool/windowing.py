"""Temporal pooling-window generation utilities for encoded sequences."""

from __future__ import annotations

import numpy as np

from ser.repr import EncodedSequence, PoolingWindow


def temporal_pooling_windows(
    encoded: EncodedSequence,
    *,
    window_size_seconds: float,
    window_stride_seconds: float,
) -> list[PoolingWindow]:
    """Builds deterministic temporal pooling windows over an encoded sequence.

    Args:
        encoded: Frame-level encoded representation with explicit timestamps.
        window_size_seconds: Temporal window size in seconds.
        window_stride_seconds: Window stride in seconds.

    Returns:
        Ordered pooling windows that cover the encoded timeline.

    Raises:
        ValueError: If configuration values are non-positive.
    """
    if window_size_seconds <= 0.0 or not np.isfinite(window_size_seconds):
        raise ValueError("window_size_seconds must be a positive finite float.")
    if window_stride_seconds <= 0.0 or not np.isfinite(window_stride_seconds):
        raise ValueError("window_stride_seconds must be a positive finite float.")

    clip_start = float(encoded.frame_start_seconds[0])
    clip_end = float(encoded.frame_end_seconds[-1])
    clip_duration = clip_end - clip_start
    if clip_duration <= 0.0:
        raise ValueError("Encoded sequence duration must be positive.")

    effective_window = min(window_size_seconds, clip_duration)
    if np.isclose(effective_window, clip_duration):
        return [PoolingWindow(start_seconds=clip_start, end_seconds=clip_end)]

    windows: list[PoolingWindow] = []
    epsilon = 1e-9

    cursor = clip_start
    while cursor + effective_window <= clip_end + epsilon:
        end = min(clip_end, cursor + effective_window)
        windows.append(PoolingWindow(start_seconds=cursor, end_seconds=end))
        cursor += window_stride_seconds

    if not windows:
        return [
            PoolingWindow(
                start_seconds=max(clip_start, clip_end - effective_window),
                end_seconds=clip_end,
            )
        ]

    if windows[-1].end_seconds < clip_end - epsilon:
        tail_start = max(clip_start, clip_end - effective_window)
        tail_window = PoolingWindow(start_seconds=tail_start, end_seconds=clip_end)
        previous = windows[-1]
        if not (
            np.isclose(previous.start_seconds, tail_window.start_seconds)
            and np.isclose(previous.end_seconds, tail_window.end_seconds)
        ):
            windows.append(tail_window)

    return windows
