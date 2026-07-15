"""Tests for artifact-persistence helper delegation seams."""

from __future__ import annotations

import errno
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
from sklearn.neural_network import MLPClassifier

from ser._internal.models import artifact_persistence  # noqa: TID251
from ser._internal.models.training_execution import (  # noqa: TID251
    finalize_profile_training_report,
)
from ser._internal.models.training_orchestration import training_operation_scope  # noqa: TID251
from ser._internal.models.training_readiness import (  # noqa: TID251
    OptionalArtifactError,
    TrainingOperation,
)
from ser._internal.models.training_types import (  # noqa: TID251
    PersistedArtifactsLike,
    TrainingEvaluation,
)


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


def test_secure_artifact_continues_only_for_explicit_optional_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = MLPClassifier(hidden_layer_sizes=(1,), max_iter=1, random_state=0)

    def _raise_optional(_model: object, _path: str) -> None:
        raise OptionalArtifactError("advisory format unavailable")

    monkeypatch.setattr(
        artifact_persistence.importlib,
        "import_module",
        lambda _name: SimpleNamespace(dump=_raise_optional),
    )
    with training_operation_scope(TrainingOperation()) as state:
        assert (
            artifact_persistence.persist_secure_artifact(tmp_path / "model.skops", model) is False
        )
        assert state.containment_counts == {
            "optional_artifact:optional_artifact_failed:continue": 1
        }

    def _raise_unknown(_model: object, _path: str) -> None:
        raise RuntimeError("serializer invariant failed")

    monkeypatch.setattr(
        artifact_persistence.importlib,
        "import_module",
        lambda _name: SimpleNamespace(dump=_raise_unknown),
    )
    with pytest.raises(RuntimeError, match="serializer invariant"):
        artifact_persistence.persist_secure_artifact(tmp_path / "model.skops", model)


def test_secure_artifact_contains_only_target_local_write_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "model.skops"
    model = MLPClassifier(hidden_layer_sizes=(1,), max_iter=1, random_state=0)

    def _raise_target_write(_model: object, path: str) -> None:
        raise OSError(errno.ENOSPC, "disk full", path)

    monkeypatch.setattr(
        artifact_persistence.importlib,
        "import_module",
        lambda _name: SimpleNamespace(dump=_raise_target_write),
    )
    with training_operation_scope(TrainingOperation()) as state:
        assert artifact_persistence.persist_secure_artifact(target, model) is False
        assert state.containment_counts == {
            "optional_artifact:optional_artifact_failed:continue": 1
        }

    def _raise_other_path(_model: object, _path: str) -> None:
        raise OSError(errno.ENOSPC, "disk full", str(tmp_path / "required.pkl"))

    monkeypatch.setattr(
        artifact_persistence.importlib,
        "import_module",
        lambda _name: SimpleNamespace(dump=_raise_other_path),
    )
    with pytest.raises(OSError) as error:
        artifact_persistence.persist_secure_artifact(target, model)
    assert error.value.filename == str(tmp_path / "required.pkl")


def test_secure_artifact_serialization_value_error_remains_aborting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise_value_error(_model: object, _path: str) -> None:
        raise ValueError("unsupported estimator graph")

    monkeypatch.setattr(
        artifact_persistence.importlib,
        "import_module",
        lambda _name: SimpleNamespace(dump=_raise_value_error),
    )
    with pytest.raises(ValueError, match="unsupported estimator graph"):
        artifact_persistence.persist_secure_artifact(
            tmp_path / "model.skops",
            MLPClassifier(),
        )

    def _raise_transitive_missing(_name: str) -> object:
        raise ModuleNotFoundError("No module named 'required_runtime'", name="required_runtime")

    monkeypatch.setattr(
        artifact_persistence.importlib,
        "import_module",
        _raise_transitive_missing,
    )
    with pytest.raises(ModuleNotFoundError, match="required_runtime"):
        artifact_persistence.persist_secure_artifact(
            tmp_path / "model.skops",
            MLPClassifier(),
        )


@pytest.mark.parametrize("failure", ["missing_skops", "target_write"])
def test_optional_secure_failure_is_durable_in_final_training_report(
    failure: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if failure == "missing_skops":

        def _missing(_name: str) -> object:
            raise ModuleNotFoundError("No module named 'skops'", name="skops")

        monkeypatch.setattr(artifact_persistence.importlib, "import_module", _missing)
    else:

        def _raise_target_write(_model: object, path: str) -> None:
            raise OSError(errno.ENOSPC, "disk full", path)

        monkeypatch.setattr(
            artifact_persistence.importlib,
            "import_module",
            lambda _name: SimpleNamespace(dump=_raise_target_write),
        )

    persisted = SimpleNamespace(
        pickle_path=tmp_path / "model.pkl",
        secure_path=None,
    )
    with training_operation_scope(TrainingOperation()):
        assert (
            artifact_persistence.persist_secure_artifact(
                tmp_path / "model.skops",
                MLPClassifier(),
            )
            is False
        )
        report = finalize_profile_training_report(
            profile_label="Fixture",
            logger=artifact_persistence.logger,
            evaluation=TrainingEvaluation(1.0, 1.0, 1.0, {}),
            ser_metrics={},
            artifact_metadata={},
            persisted_artifacts=cast(PersistedArtifactsLike, persisted),
            x_train=np.ones((2, 2), dtype=np.float64),
            x_test=np.ones((1, 2), dtype=np.float64),
            y_train=["calm", "happy"],
            y_test=["calm"],
            provenance={"training_robustness": {"statistics": {"containment": {}}}},
            data_controls={},
            build_training_report=lambda **kwargs: kwargs,
            persist_training_report=lambda _report: None,
            report_destination=tmp_path / "report.json",
        )

    provenance = report["provenance"]
    assert isinstance(provenance, dict)
    robustness = provenance["training_robustness"]
    assert isinstance(robustness, dict)
    statistics = robustness["statistics"]
    assert isinstance(statistics, dict)
    assert statistics["containment"] == {"optional_artifact:optional_artifact_failed:continue": 1}
