"""Model artifact persistence and report I/O helpers."""

from __future__ import annotations

import importlib
import json
import os
import pickle
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, TypeVar

from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from ser.utils.logger import get_logger

type EmotionClassifier = MLPClassifier | Pipeline
_PersistedArtifactsT = TypeVar("_PersistedArtifactsT")

logger = get_logger(__name__)


class _ModelsConfigLike(Protocol):
    """Structural contract for model artifact destinations."""

    @property
    def model_file(self) -> Path: ...

    @property
    def secure_model_file(self) -> Path: ...


class _SettingsLike(Protocol):
    """Structural contract for settings exposing model artifact destinations."""

    @property
    def models(self) -> _ModelsConfigLike: ...


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Writes binary payload atomically to avoid partial artifacts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    try:
        with tmp_path.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def atomic_write_text(path: Path, payload: str) -> None:
    """Writes UTF-8 text atomically to avoid truncated JSON output."""
    atomic_write_bytes(path, payload.encode("utf-8"))


def persist_pickle_artifact(path: Path, artifact: dict[str, object]) -> None:
    """Serializes and stores the pickle envelope for broad compatibility."""
    serialized = pickle.dumps(artifact, protocol=pickle.HIGHEST_PROTOCOL)
    atomic_write_bytes(path, serialized)


def persist_secure_artifact(path: Path, model: EmotionClassifier) -> bool:
    """Attempts to persist a secure model format via `skops`, if available."""
    try:
        skops_io: Any = importlib.import_module("skops.io")
    except ModuleNotFoundError:
        logger.info(
            "Optional dependency `skops` is not installed; skipping secure model "
            "artifact generation."
        )
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    try:
        skops_io.dump(model, str(tmp_path))
        os.replace(tmp_path, path)
        return True
    except Exception as err:
        logger.warning(
            "Failed to persist secure model artifact at %s: %s. Continuing with "
            "pickle artifact.",
            path,
            err,
        )
        return False
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def persist_training_report(report: dict[str, object], path: Path) -> None:
    """Persists a deterministic JSON training report to disk."""
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    atomic_write_text(path, serialized)


def persist_model_artifacts_for_settings(
    model: EmotionClassifier,
    artifact: dict[str, object],
    *,
    read_settings: Callable[[], _SettingsLike],
    persist_pickle: Callable[[Path, dict[str, object]], None],
    persist_secure: Callable[[Path, EmotionClassifier], bool],
    persisted_artifacts_factory: Callable[[Path, Path | None], _PersistedArtifactsT],
) -> _PersistedArtifactsT:
    """Persists pickle/secure artifacts using the current settings destinations."""
    settings = read_settings()
    pickle_path = settings.models.model_file
    secure_path = settings.models.secure_model_file

    persist_pickle(pickle_path, artifact)
    secure_saved = persist_secure(secure_path, model)
    return persisted_artifacts_factory(
        pickle_path,
        secure_path if secure_saved else None,
    )


def read_training_report_feature_size(path: Path) -> int | None:
    """Reads expected feature size from the training report when available."""
    if not path.exists():
        return None

    try:
        with path.open("r", encoding="utf-8") as report_fh:
            payload = json.load(report_fh)
    except Exception as err:
        logger.warning("Failed to parse training report at %s: %s", path, err)
        return None

    if not isinstance(payload, dict):
        return None

    feature_size = payload.get("feature_vector_size")
    if isinstance(feature_size, int) and feature_size > 0:
        return feature_size

    artifact_metadata = payload.get("artifact_metadata")
    if isinstance(artifact_metadata, dict):
        nested_feature_size = artifact_metadata.get("feature_vector_size")
        if isinstance(nested_feature_size, int) and nested_feature_size > 0:
            return nested_feature_size
    return None
