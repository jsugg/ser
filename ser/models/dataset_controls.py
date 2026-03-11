"""Dataset-control projection helpers for training report metadata."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol


class UtteranceLike(Protocol):
    """Minimal utterance contract used by dataset-controls projection."""

    @property
    def corpus(self) -> str: ...

    @property
    def language(self) -> str | None: ...


class _DatasetConfigLike(Protocol):
    """Structural contract for settings exposing manifest-path configuration."""

    @property
    def manifest_paths(self) -> tuple[Path, ...]: ...


class _SettingsLike(Protocol):
    """Structural contract for settings exposing dataset configuration."""

    @property
    def dataset(self) -> _DatasetConfigLike: ...


def resolve_registry_manifest_paths(*, settings: object) -> tuple[Path, ...]:
    """Resolves registry manifest paths when optional registry data is available."""
    try:
        from ser.data.dataset_registry import (
            load_dataset_registry,
            registered_manifest_paths,
        )

        settings_obj: Any = settings
        registry = load_dataset_registry(settings=settings_obj)
        if registry:
            return tuple(sorted(set(registered_manifest_paths(settings=settings_obj))))
    except Exception:
        # Optional feature; keep non-registry mode on any lookup/config errors.
        return ()
    return ()


def build_dataset_controls(
    *,
    utterances: Sequence[UtteranceLike],
    manifest_paths: tuple[Path, ...],
    resolve_registry_manifest_paths: Callable[[], Sequence[Path]],
) -> dict[str, object]:
    """Builds deterministic dataset controls payload for training reports."""
    corpus_counts = dict(Counter(utterance.corpus for utterance in utterances))
    language_counts = dict(Counter((utterance.language or "unknown") for utterance in utterances))

    resolved_manifest_paths = [str(path) for path in manifest_paths]
    mode = "manifest" if resolved_manifest_paths else "glob"
    if not resolved_manifest_paths:
        registry_manifest_paths = list(resolve_registry_manifest_paths())
        if registry_manifest_paths:
            mode = "registry"
            resolved_manifest_paths = [str(path) for path in sorted(set(registry_manifest_paths))]

    return {
        "mode": mode,
        "manifest_paths": resolved_manifest_paths,
        "utterance_count": len(utterances),
        "corpus_counts": corpus_counts,
        "language_counts": language_counts,
    }


def build_dataset_controls_for_settings(
    utterances: Sequence[UtteranceLike],
    *,
    read_settings: Callable[[], _SettingsLike],
    resolve_registry_manifest_paths_for_settings: Callable[..., Sequence[Path]],
) -> dict[str, object]:
    """Builds dataset controls using manifest configuration from current settings."""
    settings = read_settings()
    return build_dataset_controls(
        utterances=utterances,
        manifest_paths=tuple(settings.dataset.manifest_paths),
        resolve_registry_manifest_paths=lambda: (
            resolve_registry_manifest_paths_for_settings(settings=settings)
        ),
    )


__all__ = [
    "build_dataset_controls",
    "build_dataset_controls_for_settings",
    "resolve_registry_manifest_paths",
]
