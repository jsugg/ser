"""Tests for artifact-persistence helper delegation seams."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sklearn.neural_network import MLPClassifier

from ser.models import artifact_persistence


@dataclass(frozen=True)
class _ModelsConfig:
    model_file: Path
    secure_model_file: Path


@dataclass(frozen=True)
class _Settings:
    models: _ModelsConfig


def test_persist_model_artifacts_for_settings_uses_configured_paths() -> None:
    """Settings-aware persistence helper should use current model destinations."""
    calls: dict[str, object] = {}

    def _persist_pickle(path: Path, artifact: dict[str, object]) -> None:
        calls["pickle_path"] = path
        calls["artifact"] = artifact

    def _persist_secure(path: Path, model: object) -> bool:
        calls["secure_path"] = path
        calls["model"] = model
        return True

    persisted = artifact_persistence.persist_model_artifacts_for_settings(
        MLPClassifier(hidden_layer_sizes=(1,), max_iter=1, random_state=0),
        {"metadata": {}},
        read_settings=lambda: _Settings(
            models=_ModelsConfig(
                model_file=Path("models/ser_model.pkl"),
                secure_model_file=Path("models/ser_model.skops"),
            ),
        ),
        persist_pickle=_persist_pickle,
        persist_secure=_persist_secure,
        persisted_artifacts_factory=lambda pickle_path, secure_path: (
            pickle_path,
            secure_path,
        ),
    )

    assert calls["pickle_path"] == Path("models/ser_model.pkl")
    assert calls["secure_path"] == Path("models/ser_model.skops")
    assert persisted == (
        Path("models/ser_model.pkl"),
        Path("models/ser_model.skops"),
    )
