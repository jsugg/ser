"""Fixtures for the utterance-analysis contract tests (ADR 0001).

Fixtures never call the analyser: a stub that raises inside a fixture would turn an
expected failure into a test error.
"""

from __future__ import annotations

import os
from collections import Counter
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from tests.suites.unit.analysis import _scripted_backend
from tests.suites.unit.analysis._contract import _ANALYZERS, IoGuard, io_guard, read_only_directory
from tests.suites.unit.analysis._stand_ins import (
    CANONICAL_RATE_HZ,
    LABELS,
    FailingClassifier,
    tone,
    write_model,
)
from tests.suites.unit.analysis._synthetic_artifacts import SyntheticSafeFormat

import ser._internal.analysis.artifact as artifact_module
import ser._internal.analysis.backend as backend_module
import ser._internal.analysis.worker as worker_module
from ser._internal.features.feature_extractor import extract_feature_from_signal
from ser._internal.utils import audio_utils
from ser._internal.utils.audio_utils import CanonicalAudio


@pytest.fixture(autouse=True)
def close_analyzers() -> Iterator[None]:
    yield
    _ANALYZERS.close()


@pytest.fixture(scope="session")
def fast_feature_size() -> int:
    """Length of the handcrafted feature vector under default settings."""
    return int(extract_feature_from_signal(tone(3.0), CANONICAL_RATE_HZ).shape[0])


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Directory whose name is a token that must never appear in logs or errors."""
    directory: Path = tmp_path_factory.mktemp("model-dir-token-91c2")
    return directory


@pytest.fixture(scope="session")
def fast_model_path(model_dir: Path, fast_feature_size: int) -> Path:
    """A legacy pickle like the ones ``ser --train`` writes today."""
    return write_model(model_dir / "fast.pkl", feature_size=fast_feature_size)


@pytest.fixture(scope="session")
def other_fast_model_path(model_dir: Path, fast_feature_size: int) -> Path:
    """Same labels as ``fast_model_path``, different weights."""
    return write_model(model_dir / "fast-other.pkl", feature_size=fast_feature_size, seed=1)


@pytest.fixture(scope="session")
def revision_model_path(model_dir: Path, fast_feature_size: int) -> Path:
    return write_model(
        model_dir / "fast-rev.pkl", feature_size=fast_feature_size, model_revision="rev-7"
    )


@pytest.fixture(scope="session")
def failing_model_path(model_dir: Path, fast_feature_size: int) -> Path:
    return write_model(
        model_dir / "fast-failing.pkl",
        feature_size=fast_feature_size,
        classifier_type=FailingClassifier,
    )


@pytest.fixture(scope="session")
def incompatible_model_paths(model_dir: Path, fast_feature_size: int) -> dict[str, Path]:
    """Legacy files that exist but cannot serve the ``fast`` profile."""
    corrupt = model_dir / "corrupt.pkl"
    corrupt.write_bytes(b"\x80not a model artifact")
    return {
        "other-profile": write_model(
            model_dir / "medium.pkl",
            feature_size=fast_feature_size,
            profile="medium",
            backend_id="hf_xlsr",
        ),
        "feature-size": write_model(model_dir / "fast-size.pkl", feature_size=7),
        "corrupt": corrupt,
    }


@pytest.fixture(scope="session")
def seven_label_model_path(model_dir: Path, fast_feature_size: int) -> Path:
    return write_model(
        model_dir / "fast-seven.pkl", feature_size=fast_feature_size, labels=LABELS[:7], seed=2
    )


@pytest.fixture
def scripted(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Registers the synthetic safe format and the scripted backend for fast and medium."""
    formats = (SyntheticSafeFormat(), artifact_module.LegacyPickleFormat())
    loaders = {"fast": _scripted_backend.load_scripted, "medium": _scripted_backend.load_scripted}
    monkeypatch.setattr(artifact_module, "artifact_formats", lambda: formats)
    monkeypatch.setattr(backend_module, "backend_loaders", lambda: loaders)
    _scripted_backend.reset()
    yield
    for held in _scripted_backend.GATES.values():
        held.release.set()


@pytest.fixture
def spies(monkeypatch: pytest.MonkeyPatch) -> Counter[str]:
    """Counts model file reads and worker starts in this process."""
    counts: Counter[str] = Counter()

    def counting[**P, R](name: str, original: Callable[P, R]) -> Callable[P, R]:
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            counts[name] += 1
            return original(*args, **kwargs)

        return wrapper

    monkeypatch.setattr(
        artifact_module,
        "read_model_file",
        counting("read_model_file", artifact_module.read_model_file),
    )
    monkeypatch.setattr(
        worker_module, "start_worker", counting("start_worker", worker_module.start_worker)
    )
    return counts


@pytest.fixture
def canonical_calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[object, ...]]:
    """Records every call of the shared conversion seam and what it returned."""
    calls: list[tuple[object, ...]] = []
    original = audio_utils.to_canonical_audio

    def spy(
        samples: NDArray[np.generic], sample_rate: int, *, channel_axis: int = 1
    ) -> CanonicalAudio:
        result = original(samples, sample_rate, channel_axis=channel_axis)
        calls.append((samples, sample_rate, channel_axis, result))
        return result

    monkeypatch.setattr(audio_utils, "to_canonical_audio", spy)
    return calls


@pytest.fixture
def guard() -> IoGuard:
    return io_guard()


@pytest.fixture
def read_only_cwd_and_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Read-only working and temporary directories, for this process and the worker."""
    if os.geteuid() == 0:
        pytest.skip("root ignores directory permissions")
    work = read_only_directory(tmp_path / "work")
    temp = read_only_directory(tmp_path / "tmp")
    monkeypatch.chdir(work)
    monkeypatch.setenv("TMPDIR", str(temp))
    monkeypatch.setattr("tempfile.tempdir", None)
    monkeypatch.setattr("sys.dont_write_bytecode", True)
    yield work
    work.chmod(0o755)
    temp.chmod(0o755)
