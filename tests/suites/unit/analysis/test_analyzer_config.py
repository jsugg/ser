"""Configuration: enforced limits at their boundaries, unenforced targets, process policy."""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import threadpoolctl
from numpy.typing import NDArray
from tests.suites.unit.analysis._contract import (
    INLINE,
    NOT_IMPLEMENTED,
    assert_valid_record,
    load_artifact,
)
from tests.suites.unit.analysis._stand_ins import CANONICAL_RATE_HZ, noise, tone
from tests.suites.unit.analysis._synthetic_artifacts import write_artifact

from ser.api import AnalysisConfig, AnalysisTargets, AnalysisUsageError, UtteranceStatus


def _with(**changes: Any) -> AnalysisConfig:
    return dataclasses.replace(AnalysisConfig(), **changes)


def _targets(**changes: Any) -> AnalysisConfig:
    return _with(targets=dataclasses.replace(AnalysisTargets(), **changes))


_INVALID: dict[str, AnalysisConfig] = {
    "min-negative": _with(min_duration_s=-0.1),
    "min-zero": _with(min_duration_s=0.0),
    "min-nan": _with(min_duration_s=float("nan")),
    "max-below-min": _with(min_duration_s=2.0, max_duration_s=1.0),
    "max-inf": _with(max_duration_s=float("inf")),
    "floor-zero": _with(timeout_floor_s=0.0),
    "floor-nan": _with(timeout_floor_s=float("nan")),
    "rate-zero": _with(timeout_per_audio_second=0.0),
    "rate-inf": _with(timeout_per_audio_second=float("inf")),
    "silence-nan": _with(silence_rms_dbfs=float("nan")),
    "silence-above-full-scale": _with(silence_rms_dbfs=0.1),
    "silence-minus-inf": _with(silence_rms_dbfs=float("-inf")),
    "isolation": _with(isolation="thread"),
    "threads-zero": _with(max_threads=0),
    "threads-bool": _with(max_threads=True),
    "threads-float": _with(max_threads=1.5),
    "nice-negative": _with(worker_nice=-1),
    "nice-above-19": _with(worker_nice=20),
    "start-timeout-zero": _with(worker_start_timeout_s=0.0),
    "p95-zero": _targets(p95_real_time_factor=0.0),
    "cold-start-negative": _targets(cold_start_s=-1.0),
    "memory-zero": _targets(steady_memory_bytes=0),
    "memory-float": _targets(steady_memory_bytes=1e9),
}


@pytest.mark.usefixtures("scripted")
@pytest.mark.parametrize("bad_config", list(_INVALID.values()), ids=list(_INVALID))
def test_invalid_configuration_is_a_usage_error(tmp_path: Path, bad_config: AnalysisConfig) -> None:
    path = write_artifact(tmp_path / "model.ser")

    with pytest.raises(AnalysisUsageError):
        load_artifact(path, bad_config)


def _samples(count: int) -> NDArray[np.float32]:
    return tone(0.0, num_samples=count)


_DEFAULT_MIN = round(0.3 * CANONICAL_RATE_HZ)
_DEFAULT_MAX = 120 * CANONICAL_RATE_HZ
_BOUNDARIES = [
    pytest.param(None, _samples(_DEFAULT_MIN - 1), "too_short", id="default-min-minus-one"),
    pytest.param(None, _samples(_DEFAULT_MIN), "ok", id="default-min"),
    pytest.param(None, _samples(_DEFAULT_MAX), "ok", id="default-max"),
    pytest.param(None, _samples(_DEFAULT_MAX + 1), "too_long", id="default-max-plus-one"),
    pytest.param(None, noise(1.0, rms_dbfs=-59.9), "ok", id="default-silence-above"),
    pytest.param(None, noise(1.0, rms_dbfs=-60.1), "silent", id="default-silence-below"),
    pytest.param(_with(min_duration_s=1.0), _samples(15_999), "too_short", id="min-1s-below"),
    pytest.param(_with(min_duration_s=1.0), _samples(16_000), "ok", id="min-1s"),
    pytest.param(_with(max_duration_s=2.0), _samples(32_000), "ok", id="max-2s"),
    pytest.param(_with(max_duration_s=2.0), _samples(32_001), "too_long", id="max-2s-above"),
    pytest.param(
        _with(silence_rms_dbfs=-30.0), noise(1.0, rms_dbfs=-29.9), "ok", id="silence-30-above"
    ),
    pytest.param(
        _with(silence_rms_dbfs=-30.0), noise(1.0, rms_dbfs=-30.1), "silent", id="silence-30-below"
    ),
]


@NOT_IMPLEMENTED
@pytest.mark.usefixtures("scripted")
@pytest.mark.parametrize(("config", "samples", "expected"), _BOUNDARIES)
def test_limits_apply_exactly_at_their_boundaries(
    tmp_path: Path,
    config: AnalysisConfig | None,
    samples: NDArray[np.float32],
    expected: UtteranceStatus,
) -> None:
    analyzer = load_artifact(write_artifact(tmp_path / "model.ser"), config)

    record = analyzer.analyze(samples, sample_rate_hz=CANONICAL_RATE_HZ)

    assert record["status"] == expected
    assert_valid_record(record, analyzer.model_manifest)


@NOT_IMPLEMENTED
@pytest.mark.usefixtures("scripted")
def test_performance_targets_are_not_enforced(tmp_path: Path) -> None:
    path = write_artifact(tmp_path / "model.ser", script={"extract_sleep_s": 1.0})
    analyzer = load_artifact(path, _targets(p95_real_time_factor=0.1, cold_start_s=0.01))

    record = analyzer.analyze(tone(1.0), sample_rate_hz=CANONICAL_RATE_HZ)

    assert record["status"] == "ok"


@NOT_IMPLEMENTED
@pytest.mark.usefixtures("scripted")
@pytest.mark.parametrize(("nice", "threads"), [(19, 2), (5, 1)], ids=["default", "custom"])
def test_worker_runs_at_low_priority_with_capped_threads_and_no_gpu(
    tmp_path: Path, nice: int, threads: int
) -> None:
    expected_nice = max(nice, os.nice(0))
    probe = {"nice": expected_nice, "max_threads": threads, "gpu_hidden": True}
    path = write_artifact(tmp_path / "model.ser", script={"probe": probe})
    config = None if (nice, threads) == (19, 2) else _with(worker_nice=nice, max_threads=threads)

    record = load_artifact(path, config).analyze(tone(1.0), sample_rate_hz=CANONICAL_RATE_HZ)

    assert record["status"] == "ok"


@NOT_IMPLEMENTED
@pytest.mark.usefixtures("scripted")
def test_inline_mode_leaves_the_caller_process_alone(tmp_path: Path) -> None:
    path = write_artifact(tmp_path / "model.ser", script={"probe": {}})
    before = (os.nice(0), threadpoolctl.threadpool_info(), os.environ.get("CUDA_VISIBLE_DEVICES"))

    record = load_artifact(path, INLINE).analyze(tone(1.0), sample_rate_hz=CANONICAL_RATE_HZ)

    after = (os.nice(0), threadpoolctl.threadpool_info(), os.environ.get("CUDA_VISIBLE_DEVICES"))
    assert record["status"] == "ok"
    assert after == before
