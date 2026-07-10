"""Medium-profile pooled-feature noise control helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class MediumNoiseControlStats:
    """Window-level filtering statistics for medium training traceability."""

    total_windows: int = 0
    kept_windows: int = 0
    dropped_low_std_windows: int = 0
    dropped_cap_windows: int = 0
    forced_keep_windows: int = 0


def merge_medium_noise_stats(
    base: MediumNoiseControlStats,
    incoming: MediumNoiseControlStats,
) -> MediumNoiseControlStats:
    """Aggregates per-clip medium noise-control counters."""
    return MediumNoiseControlStats(
        total_windows=base.total_windows + incoming.total_windows,
        kept_windows=base.kept_windows + incoming.kept_windows,
        dropped_low_std_windows=(base.dropped_low_std_windows + incoming.dropped_low_std_windows),
        dropped_cap_windows=base.dropped_cap_windows + incoming.dropped_cap_windows,
        forced_keep_windows=base.forced_keep_windows + incoming.forced_keep_windows,
    )


def apply_medium_noise_controls(
    pooled_features: NDArray[np.float64],
    *,
    min_window_std: float,
    max_windows_per_clip: int,
) -> tuple[NDArray[np.float64], MediumNoiseControlStats]:
    """Applies deterministic label-noise controls to pooled medium features."""
    if pooled_features.ndim != 2 or int(pooled_features.shape[1]) <= 0:
        raise RuntimeError("Medium pooled features must be a non-empty 2D matrix.")
    total_windows = int(pooled_features.shape[0])
    if total_windows == 0:
        raise RuntimeError("Medium pooled feature matrix contains zero rows.")
    feature_width = int(pooled_features.shape[1])
    if feature_width % 2 != 0:
        raise RuntimeError("Medium pooled feature width must be even (mean+std concatenation).")

    std_components = pooled_features[:, feature_width // 2 :]
    std_scores = np.linalg.norm(std_components, axis=1) / np.sqrt(feature_width / 2.0)

    keep_mask = np.ones(total_windows, dtype=np.bool_)
    dropped_low_std_windows = 0
    forced_keep_windows = 0
    if min_window_std > 0.0:
        keep_mask = std_scores >= min_window_std
        if not np.any(keep_mask):
            keep_mask[int(np.argmax(std_scores))] = True
            forced_keep_windows = 1
        dropped_low_std_windows = total_windows - int(np.sum(keep_mask))

    filtered = np.asarray(pooled_features[keep_mask], dtype=np.float64)
    dropped_cap_windows = 0
    if max_windows_per_clip > 0 and int(filtered.shape[0]) > max_windows_per_clip:
        selected_indices = np.linspace(
            0,
            int(filtered.shape[0]) - 1,
            num=max_windows_per_clip,
            dtype=np.int64,
        )
        dropped_cap_windows = int(filtered.shape[0]) - max_windows_per_clip
        filtered = np.asarray(filtered[selected_indices], dtype=np.float64)

    return filtered, MediumNoiseControlStats(
        total_windows=total_windows,
        kept_windows=int(filtered.shape[0]),
        dropped_low_std_windows=dropped_low_std_windows,
        dropped_cap_windows=dropped_cap_windows,
        forced_keep_windows=forced_keep_windows,
    )


__all__ = [
    "MediumNoiseControlStats",
    "apply_medium_noise_controls",
    "merge_medium_noise_stats",
]
