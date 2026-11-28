"""Fixtures for the utterance-analysis contract tests (ADR 0001).

Fixtures never call the analyser: a stub that raises inside a fixture would turn an
expected failure into a test error.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.suites.unit.analysis._stand_ins import (
    CANONICAL_RATE_HZ,
    LABELS,
    FailingClassifier,
    tone,
    write_model,
)

from ser._internal.features.feature_extractor import extract_feature_from_signal


@pytest.fixture(scope="session")
def fast_feature_size() -> int:
    """Length of the handcrafted feature vector under default settings."""
    return int(extract_feature_from_signal(tone(3.0), CANONICAL_RATE_HZ).shape[0])


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Directory whose name is a token that must never appear in logs or errors."""
    return tmp_path_factory.mktemp("model-dir-token-91c2")


@pytest.fixture(scope="session")
def fast_model_path(model_dir: Path, fast_feature_size: int) -> Path:
    return write_model(model_dir / "fast.pkl", feature_size=fast_feature_size)


@pytest.fixture(scope="session")
def other_fast_model_path(model_dir: Path, fast_feature_size: int) -> Path:
    """Same labels as ``fast_model_path``, different weights."""
    return write_model(model_dir / "fast-other.pkl", feature_size=fast_feature_size, seed=1)


@pytest.fixture(scope="session")
def seven_label_model_path(model_dir: Path, fast_feature_size: int) -> Path:
    return write_model(
        model_dir / "fast-seven.pkl",
        feature_size=fast_feature_size,
        labels=LABELS[:7],
        seed=2,
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
    """Model files that exist but cannot serve the ``fast`` profile."""
    corrupt = model_dir / "corrupt.pkl"
    corrupt.write_bytes(b"not a model artifact")
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
