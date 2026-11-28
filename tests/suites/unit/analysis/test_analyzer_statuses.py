"""Bad audio yields a status, never an exception; only ``ok`` carries labels."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from tests.suites.unit.analysis._contract import NOT_IMPLEMENTED, assert_valid_record, load_fast
from tests.suites.unit.analysis._stand_ins import (
    CANONICAL_RATE_HZ,
    noise,
    silence,
    tone,
    write_wav,
)

from ser.runtime.contracts import AnalysisConfig
from ser.runtime.schema import UtteranceStatus

_MIN_SAMPLES = round(AnalysisConfig().min_duration_s * CANONICAL_RATE_HZ)


@NOT_IMPLEMENTED
@pytest.mark.parametrize(
    ("samples", "expected"),
    [
        pytest.param(tone(0.2), "too_short", id="0.2s"),
        pytest.param(np.zeros(0, dtype=np.float32), "too_short", id="empty"),
        pytest.param(tone(0.0, num_samples=_MIN_SAMPLES - 1), "too_short", id="min-minus-one"),
        pytest.param(tone(0.0, num_samples=_MIN_SAMPLES), "ok", id="exactly-min"),
        pytest.param(tone(120.5), "too_long", id="120.5s"),
        pytest.param(silence(0.1), "too_short", id="short-silence-is-too-short"),
        pytest.param(silence(1.0), "silent", id="digital-silence"),
        pytest.param(noise(1.0, rms_dbfs=-80.0), "silent", id="noise-at-80dbfs"),
        pytest.param(tone(1.0, rms_dbfs=-40.0), "ok", id="tone-at-40dbfs"),
    ],
)
def test_duration_and_level_limits(
    fast_model_path: Path, samples: NDArray[np.float32], expected: UtteranceStatus
) -> None:
    analyzer = load_fast(fast_model_path)

    record = analyzer.analyze(samples, sample_rate_hz=CANONICAL_RATE_HZ)

    assert record["status"] == expected
    assert_valid_record(record, analyzer.model_identity)
    assert record["audio"]["duration_s"] == pytest.approx(
        samples.size / CANONICAL_RATE_HZ, abs=1e-3
    )


@NOT_IMPLEMENTED
def test_limits_come_from_the_configuration(fast_model_path: Path) -> None:
    config = AnalysisConfig(max_duration_s=2.0, silence_rms_dbfs=-30.0)
    analyzer = load_fast(fast_model_path, config)
    max_samples = 2 * CANONICAL_RATE_HZ

    def status(samples: NDArray[np.float32]) -> str:
        return analyzer.analyze(samples, sample_rate_hz=CANONICAL_RATE_HZ)["status"]

    assert status(tone(0.0, num_samples=max_samples)) == "ok"
    assert status(tone(0.0, num_samples=max_samples + 1)) == "too_long"
    assert status(tone(1.0, rms_dbfs=-40.0)) == "silent"


def _missing(directory: Path) -> Path:
    return directory / "missing.wav"


def _empty(directory: Path) -> Path:
    path = directory / "empty.wav"
    path.write_bytes(b"")
    return path


def _truncated_header(directory: Path) -> Path:
    whole = write_wav(directory / "whole.wav", tone(1.0), subtype="PCM_16").read_bytes()
    path = directory / "truncated.wav"
    path.write_bytes(whole[:20])
    return path


def _not_audio(directory: Path) -> Path:
    path = directory / "text.wav"
    path.write_bytes(b"this is not audio\n" * 64)
    return path


def _directory(directory: Path) -> Path:
    path = directory / "folder.wav"
    path.mkdir()
    return path


@NOT_IMPLEMENTED
@pytest.mark.parametrize(
    "make_path",
    [_missing, _empty, _truncated_header, _not_audio, _directory],
    ids=["missing", "empty", "truncated-header", "not-audio", "directory"],
)
def test_undecodable_files_are_unreadable(
    tmp_path: Path, fast_model_path: Path, make_path: Callable[[Path], Path]
) -> None:
    path = make_path(tmp_path)
    analyzer = load_fast(fast_model_path)

    record = analyzer.analyze(path)

    assert record["status"] == "unreadable"
    assert_valid_record(record, analyzer.model_identity)
    assert record["audio"] == {"duration_s": None, "input_sample_rate_hz": None}


@NOT_IMPLEMENTED
@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf], ids=["nan", "inf", "-inf"])
def test_non_finite_samples_are_unreadable(fast_model_path: Path, bad_value: float) -> None:
    samples = tone(2.0)
    samples[100] = bad_value
    analyzer = load_fast(fast_model_path)

    record = analyzer.analyze(samples, sample_rate_hz=CANONICAL_RATE_HZ)

    assert record["status"] == "unreadable"
    assert_valid_record(record, analyzer.model_identity)
    assert record["audio"]["input_sample_rate_hz"] == CANONICAL_RATE_HZ


@NOT_IMPLEMENTED
def test_exceeding_the_deadline_is_a_timeout(fast_model_path: Path) -> None:
    config = AnalysisConfig(timeout_floor_s=0.0, timeout_per_audio_second=1e-9)
    analyzer = load_fast(fast_model_path, config)

    record = analyzer.analyze(tone(2.0), sample_rate_hz=CANONICAL_RATE_HZ)

    assert record["status"] == "timeout"
    assert_valid_record(record, analyzer.model_identity)


@NOT_IMPLEMENTED
def test_a_failing_model_is_a_model_error(failing_model_path: Path) -> None:
    analyzer = load_fast(failing_model_path)

    record = analyzer.analyze(tone(2.0), sample_rate_hz=CANONICAL_RATE_HZ)

    assert record["status"] == "model_error"
    assert_valid_record(record, analyzer.model_identity)
