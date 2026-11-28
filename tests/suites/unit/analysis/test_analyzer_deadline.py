"""Hard deadline: ``analyze`` returns ``timeout`` by its deadline, whatever the backend does.

The worker that holds the model is evicted at the deadline and restarted; calls made
while it restarts wait at most their own deadline. Inline isolation checks the
deadline only between stages and is not a hard bound.
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from tests.suites.unit.analysis._contract import (
    NOT_IMPLEMENTED,
    assert_valid_record,
    load_artifact,
)
from tests.suites.unit.analysis._stand_ins import CANONICAL_RATE_HZ, tone
from tests.suites.unit.analysis._synthetic_artifacts import write_artifact

from ser._internal.analysis.pipeline import analysis_deadline_s
from ser.api import AnalysisConfig, UtteranceAnalyzer, UtteranceRecord

_MARGIN_S = 0.5


def _timed(call: Callable[[], UtteranceRecord]) -> tuple[UtteranceRecord, float]:
    start = time.monotonic()
    record = call()
    return record, time.monotonic() - start


def _analyse(analyzer: UtteranceAnalyzer, samples: NDArray[np.float32]) -> UtteranceRecord:
    return analyzer.analyze(samples, sample_rate_hz=CANONICAL_RATE_HZ)


def _until_settled(
    analyzer: UtteranceAnalyzer, samples: NDArray[np.float32], config: AnalysisConfig
) -> list[UtteranceRecord]:
    """Calls until a status other than ``timeout``; each call must keep its deadline."""
    records: list[UtteranceRecord] = []
    give_up = time.monotonic() + config.worker_start_timeout_s + 10.0
    while time.monotonic() < give_up:
        record, elapsed = _timed(lambda: _analyse(analyzer, samples))
        assert elapsed <= config.timeout_floor_s + _MARGIN_S
        records.append(record)
        if record["status"] != "timeout":
            return records
    raise AssertionError("the worker never came back")


@NOT_IMPLEMENTED
@pytest.mark.parametrize(
    ("duration_s", "expected_s"),
    [(None, 10.0), (0.3, 10.0), (5.0, 10.0), (5.5, 11.0), (120.0, 240.0)],
)
def test_deadline_is_the_floor_or_two_seconds_per_audio_second(
    duration_s: float | None, expected_s: float
) -> None:
    assert analysis_deadline_s(AnalysisConfig(), duration_s) == pytest.approx(expected_s)


_BRANCHES = AnalysisConfig(timeout_floor_s=2.0, timeout_per_audio_second=1.0)


@NOT_IMPLEMENTED
@pytest.mark.usefixtures("scripted")
@pytest.mark.parametrize(
    ("seconds", "sleep_s"),
    [(0.5, 1.2), (4.0, 3.0)],
    ids=["floor-covers-short-audio", "per-second-covers-long-audio"],
)
def test_both_deadline_branches_allow_work_within_them(
    tmp_path: Path, seconds: float, sleep_s: float
) -> None:
    path = write_artifact(tmp_path / "model.ser", script={"score_sleep_s": sleep_s})
    analyzer = load_artifact(path, _BRANCHES)

    assert _analyse(analyzer, tone(seconds))["status"] == "ok"


@NOT_IMPLEMENTED
@pytest.mark.usefixtures("scripted")
def test_a_hung_backend_times_out_on_time_and_the_worker_comes_back(
    tmp_path: Path, spies: Counter[str]
) -> None:
    config = AnalysisConfig(timeout_floor_s=1.0, timeout_per_audio_second=0.5)
    path = write_artifact(tmp_path / "model.ser", script={"hang_if_duration_at_least_s": 0.9})
    analyzer = load_artifact(path, config)

    hung, elapsed = _timed(lambda: _analyse(analyzer, tone(1.0)))
    recovered = _until_settled(analyzer, tone(0.5), config)

    assert hung["status"] == "timeout"
    assert elapsed <= 1.0 + _MARGIN_S
    assert_valid_record(hung, analyzer.model_manifest)
    assert recovered[-1]["status"] == "ok"
    assert spies["start_worker"] == 2
    assert spies["read_model_file"] == 1


@NOT_IMPLEMENTED
@pytest.mark.usefixtures("scripted")
def test_a_call_that_times_out_in_the_queue_does_not_evict_the_running_one(
    tmp_path: Path, spies: Counter[str]
) -> None:
    config = AnalysisConfig(timeout_floor_s=1.0, timeout_per_audio_second=2.0)
    path = write_artifact(tmp_path / "model.ser", script={"score_sleep_s": 2.0})
    analyzer = load_artifact(path, config)

    with ThreadPoolExecutor(max_workers=2) as pool:
        running = pool.submit(_analyse, analyzer, tone(1.5))
        time.sleep(0.2)
        queued = pool.submit(_timed, lambda: _analyse(analyzer, tone(0.4)))
        queued_record, queued_elapsed = queued.result()
        running_record = running.result()

    assert queued_record["status"] == "timeout"
    assert queued_elapsed <= 1.0 + _MARGIN_S
    assert running_record["status"] == "ok"
    assert spies["start_worker"] == 1


@NOT_IMPLEMENTED
@pytest.mark.usefixtures("scripted")
def test_a_worker_that_dies_is_a_model_error_and_is_replaced(
    tmp_path: Path, spies: Counter[str]
) -> None:
    config = AnalysisConfig(timeout_floor_s=5.0)
    path = write_artifact(tmp_path / "model.ser", script={"die": True})
    analyzer = load_artifact(path, config)

    died = _analyse(analyzer, tone(1.0))
    again = _until_settled(analyzer, tone(1.0), config)

    assert died["status"] == "model_error"
    assert again[-1]["status"] == "model_error"
    assert_valid_record(died, analyzer.model_manifest)
    assert spies["start_worker"] >= 2
    assert spies["read_model_file"] == 1


@NOT_IMPLEMENTED
@pytest.mark.usefixtures("scripted")
def test_a_failed_restart_is_a_model_error_and_the_next_call_retries(tmp_path: Path) -> None:
    config = AnalysisConfig(timeout_floor_s=1.0, timeout_per_audio_second=0.5)
    marker = tmp_path / "refuse-restart"
    script = {"hang_if_duration_at_least_s": 0.9, "fail_load_if_exists": str(marker)}
    analyzer = load_artifact(write_artifact(tmp_path / "model.ser", script=script), config)
    marker.touch()

    assert _analyse(analyzer, tone(1.0))["status"] == "timeout"
    failed = _until_settled(analyzer, tone(0.5), config)
    marker.unlink()
    recovered = _until_settled(analyzer, tone(0.5), config)

    assert failed[-1]["status"] == "model_error"
    assert recovered[-1]["status"] == "ok"


@NOT_IMPLEMENTED
@pytest.mark.usefixtures("scripted")
def test_inline_isolation_checks_the_deadline_between_stages_only(tmp_path: Path) -> None:
    config = AnalysisConfig(isolation="inline", timeout_floor_s=0.5, timeout_per_audio_second=0.1)
    path = write_artifact(tmp_path / "model.ser", script={"score_sleep_s": 1.5})
    analyzer = load_artifact(path, config)

    record, elapsed = _timed(lambda: _analyse(analyzer, tone(1.0)))

    assert record["status"] == "timeout"
    assert elapsed >= 1.5
