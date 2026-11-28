"""Caller mistakes and unusable models raise one documented exception family."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.suites.unit.analysis._contract import NOT_IMPLEMENTED, load_fast
from tests.suites.unit.analysis._stand_ins import CANONICAL_RATE_HZ, tone

import ser.config as config
from ser._internal.analysis import (
    AnalysisError,
    AnalysisModelLoadError,
    AnalysisModelUnavailableError,
    AnalysisUsageError,
    UtteranceAnalyzer,
)
from ser.profiles import ProfileName
from ser.runtime.contracts import AnalysisConfig


@NOT_IMPLEMENTED
def test_missing_model_file_fails_at_load(tmp_path: Path) -> None:
    missing = tmp_path / "missing-model-token-5d1e" / "ser_model.pkl"

    with pytest.raises(AnalysisModelUnavailableError) as raised:
        load_fast(missing)

    assert isinstance(raised.value, AnalysisError)
    assert "token-5d1e" not in str(raised.value)


@NOT_IMPLEMENTED
@pytest.mark.usefixtures("reset_ambient_settings")
def test_no_configured_model_fails_at_load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    settings = config.reload_settings()

    with pytest.raises(AnalysisModelUnavailableError):
        UtteranceAnalyzer.load(profile="fast", settings=settings)


@NOT_IMPLEMENTED
@pytest.mark.parametrize("kind", ["other-profile", "feature-size", "corrupt"])
def test_incompatible_model_file_fails_at_load(
    incompatible_model_paths: dict[str, Path], kind: str
) -> None:
    with pytest.raises(AnalysisModelLoadError) as raised:
        load_fast(incompatible_model_paths[kind])

    assert isinstance(raised.value, AnalysisError)


@NOT_IMPLEMENTED
@pytest.mark.parametrize("profile", ["accurate", "accurate-research"])
def test_profile_without_an_analysis_backend_is_a_usage_error(
    fast_model_path: Path, profile: ProfileName
) -> None:
    with pytest.raises(AnalysisUsageError):
        UtteranceAnalyzer.load(profile=profile, model_path=fast_model_path)


@NOT_IMPLEMENTED
@pytest.mark.parametrize(
    "bad_config",
    [
        AnalysisConfig(min_duration_s=-0.1),
        AnalysisConfig(min_duration_s=2.0, max_duration_s=1.0),
        AnalysisConfig(timeout_floor_s=-1.0),
        AnalysisConfig(timeout_per_audio_second=0.0),
        AnalysisConfig(silence_rms_dbfs=float("nan")),
        AnalysisConfig(max_threads=0),
        AnalysisConfig(memory_budget_mb=0),
    ],
    ids=[
        "negative-min",
        "min-above-max",
        "negative-timeout-floor",
        "zero-timeout-rate",
        "nan-silence",
        "zero-threads",
        "zero-memory",
    ],
)
def test_invalid_configuration_is_a_usage_error(
    fast_model_path: Path, bad_config: AnalysisConfig
) -> None:
    with pytest.raises(AnalysisUsageError):
        load_fast(fast_model_path, bad_config)


@NOT_IMPLEMENTED
def test_analyze_after_close_is_a_usage_error(fast_model_path: Path) -> None:
    with load_fast(fast_model_path) as analyzer:
        assert analyzer.analyze(tone(1.0), sample_rate_hz=CANONICAL_RATE_HZ)["status"] == "ok"

    analyzer.close()
    with pytest.raises(AnalysisUsageError):
        analyzer.analyze(tone(1.0), sample_rate_hz=CANONICAL_RATE_HZ)
