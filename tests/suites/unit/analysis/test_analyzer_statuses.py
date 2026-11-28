"""Bad audio and a failing model yield a status, never an exception; only ``ok`` carries labels."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np
import pytest
from tests.suites.unit.analysis._contract import (
    INLINE,
    assert_valid_record,
    load_artifact,
    load_baseline,
)
from tests.suites.unit.analysis._scripted_backend import SECRET
from tests.suites.unit.analysis._stand_ins import CANONICAL_RATE_HZ, tone, write_wav
from tests.suites.unit.analysis._synthetic_artifacts import write_artifact

from ser.api import AnalysisConfig


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


def _truncated_payload(directory: Path) -> Path:
    whole = write_wav(directory / "whole.wav", tone(1.0), subtype="PCM_16").read_bytes()
    path = directory / "short-payload.wav"
    path.write_bytes(whole[: len(whole) // 2])
    return path


def _truncated_data_chunk(directory: Path) -> Path:
    whole = write_wav(directory / "whole.wav", tone(1.0), subtype="PCM_16").read_bytes()
    truncated = bytearray(whole[: len(whole) // 2])
    truncated[4:8] = (len(truncated) - 8).to_bytes(4, "little")
    path = directory / "short-data-chunk.wav"
    path.write_bytes(truncated)
    return path


def _not_audio(directory: Path) -> Path:
    path = directory / "text.wav"
    path.write_bytes(b"this is not audio\n" * 64)
    return path


def _directory(directory: Path) -> Path:
    path = directory / "folder.wav"
    path.mkdir()
    return path


@pytest.mark.parametrize(
    "make_path",
    [
        _missing,
        _empty,
        _truncated_header,
        _truncated_payload,
        _truncated_data_chunk,
        _not_audio,
        _directory,
    ],
    ids=[
        "missing",
        "empty",
        "truncated-header",
        "truncated-payload",
        "truncated-data-chunk",
        "not-audio",
        "directory",
    ],
)
def test_undecodable_files_are_unreadable(
    tmp_path: Path, fast_model_path: Path, make_path: Callable[[Path], Path]
) -> None:
    path = make_path(tmp_path)
    analyzer = load_baseline(fast_model_path)

    record = analyzer.analyze(path)

    assert record["status"] == "unreadable"
    assert_valid_record(record, analyzer.model_manifest)
    assert record["audio"] == {"duration_s": None, "input_sample_rate_hz": None}


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf], ids=["nan", "inf", "-inf"])
def test_non_finite_samples_are_unreadable(fast_model_path: Path, bad_value: float) -> None:
    samples = tone(2.0)
    samples[100] = bad_value
    analyzer = load_baseline(fast_model_path)

    record = analyzer.analyze(samples, sample_rate_hz=CANONICAL_RATE_HZ)

    assert record["status"] == "unreadable"
    assert_valid_record(record, analyzer.model_manifest)
    assert record["audio"]["input_sample_rate_hz"] == CANONICAL_RATE_HZ


def test_a_failing_legacy_model_is_a_model_error(failing_model_path: Path) -> None:
    analyzer = load_baseline(failing_model_path)

    record = analyzer.analyze(tone(2.0), sample_rate_hz=CANONICAL_RATE_HZ)

    assert record["status"] == "model_error"
    assert_valid_record(record, analyzer.model_manifest)


@pytest.mark.usefixtures("scripted")
@pytest.mark.parametrize("config", [AnalysisConfig(), INLINE], ids=["worker", "inline"])
def test_a_raising_backend_is_a_model_error_that_leaks_nothing(
    tmp_path: Path, capfd: pytest.CaptureFixture[str], config: AnalysisConfig
) -> None:
    path = write_artifact(tmp_path / "model.ser", script={"raise": "runtime"})
    analyzer = load_artifact(path, config)

    record = analyzer.analyze(tone(1.0), sample_rate_hz=CANONICAL_RATE_HZ)

    assert record["status"] == "model_error"
    assert_valid_record(record, analyzer.model_manifest)
    assert SECRET not in repr(record)
    captured = capfd.readouterr()
    assert (captured.out, captured.err) == ("", "")


_INVALID_SCORES: dict[str, Mapping[str, object]] = {
    "nan": {"labels": {"a": float("nan"), "b": 0.5, "c": 0.5}},
    "negative": {"labels": {"a": -0.1, "b": 0.6, "c": 0.5}},
    "missing-label": {"labels": {"a": 0.5, "b": 0.5}},
    "extra-label": {"labels": {"a": 0.4, "b": 0.3, "c": 0.2, "d": 0.1}},
    "sum-low": {"labels": {"a": 0.3, "b": 0.3, "c": 0.3}},
    "sum-high": {"labels": {"a": 0.5, "b": 0.5, "c": 0.01}},
    "above-one-within-sum-tolerance": {"labels": {"a": 1.0000005, "b": 0.0, "c": 0.0}},
    "frame-gap": {"frames": "gap"},
    "frame-outside": {"frames": "outside"},
}


@pytest.mark.usefixtures("scripted")
@pytest.mark.parametrize("script", list(_INVALID_SCORES.values()), ids=list(_INVALID_SCORES))
def test_invalid_backend_scores_are_a_model_error(
    tmp_path: Path, script: Mapping[str, object]
) -> None:
    analyzer = load_artifact(write_artifact(tmp_path / "model.ser", script=script))

    record = analyzer.analyze(tone(1.0), sample_rate_hz=CANONICAL_RATE_HZ)

    assert record["status"] == "model_error"
    assert_valid_record(record, analyzer.model_manifest)
