"""Training plans use the package's version when distribution metadata is absent."""

from importlib.metadata import PackageNotFoundError

import pytest

import ser
from ser._internal.models import training_orchestration


def test_metadata_free_training_version(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(distribution: str) -> str:
        raise PackageNotFoundError(distribution)

    monkeypatch.setattr(ser, "__version__", "7.7.7+fallback")
    monkeypatch.setattr(training_orchestration, "version", missing)
    assert training_orchestration._package_version() == "7.7.7+fallback"
