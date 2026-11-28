"""Load once, stay quiet and offline, and be deterministic, thread-safe and closable."""

from __future__ import annotations

import logging
import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import numpy as np
import pytest
from numpy.typing import NDArray
from tests.suites.unit.analysis import _scripted_backend
from tests.suites.unit.analysis._contract import (
    INLINE,
    IoGuard,
    load_artifact,
    load_baseline,
    without_timing,
)
from tests.suites.unit.analysis._stand_ins import CANONICAL_RATE_HZ, noise, tone, write_wav
from tests.suites.unit.analysis._synthetic_artifacts import write_artifact

from ser._internal.analysis.analyzer import _InlineSession
from ser._internal.utils import dsp
from ser.api import (
    AnalysisClosedError,
    AnalysisConfig,
    AnalysisModelLoadError,
    UtteranceAnalyzer,
    UtteranceRecord,
)

_MODES = pytest.mark.parametrize("config", [AnalysisConfig(), INLINE], ids=["worker", "inline"])
_SENTINEL = "caller-path-token-3f7a"


def _clips() -> list[NDArray[np.float32]]:
    return [noise(1.5 + index, seed=index) for index in range(4)]


def _analyse(analyzer: UtteranceAnalyzer, samples: NDArray[np.float32]) -> UtteranceRecord:
    return analyzer.analyze(samples, sample_rate_hz=CANONICAL_RATE_HZ)


@pytest.mark.usefixtures("scripted")
def test_the_model_is_read_and_started_once_for_many_analyses(
    tmp_path: Path, spies: Counter[str]
) -> None:
    script = {"name": "counted-in-child", "probe": {"loads_once": True}}
    analyzer = load_artifact(write_artifact(tmp_path / "model.ser", script=script))

    statuses = {_analyse(analyzer, tone(0.5))["status"] for _ in range(100)}

    assert statuses == {"ok"}
    assert (spies["read_model_file"], spies["start_worker"]) == (1, 1)


@pytest.mark.usefixtures("scripted")
def test_real_child_is_offline_and_leaves_redirected_tmp_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker_tmp = tmp_path / "child-tmp"
    worker_tmp.mkdir()
    monkeypatch.setenv("TMPDIR", str(worker_tmp))
    wav = write_wav(tmp_path / "clip.wav", tone(0.5))
    script = {"audit": True, "probe": {"tmpdir": str(worker_tmp)}}

    with load_artifact(write_artifact(tmp_path / "model.ser", script=script)) as analyzer:
        assert analyzer.analyze(wav)["status"] == "ok"
        assert _analyse(analyzer, tone(0.5))["status"] == "ok"

    assert list(worker_tmp.iterdir()) == []


@pytest.mark.usefixtures("scripted")
def test_parent_and_worker_logs_exclude_path_exception_and_score_content(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    score_sentinel = "label-content-token"
    sentinels = [_SENTINEL, _scripted_backend.SECRET, score_sentinel]
    wav = write_wav(tmp_path / _SENTINEL / "clip.wav", tone(0.5))
    caplog.set_level(logging.DEBUG)
    with pytest.raises(AnalysisModelLoadError) as failed_start:
        load_artifact(write_artifact(wav.parent / "bad.ser", script={"fail_load": True}))
    script = {"audit": True, "raise_language": "es", "probe": {"no_log": sentinels}}
    model = write_artifact(wav.parent / "probe.ser", script=script, labels=(score_sentinel,))

    with load_artifact(model) as analyzer:
        assert analyzer.analyze(wav, language="es")["status"] == "model_error"
        for _ in range(2):
            record = analyzer.analyze(wav)
            assert record["labels"] == {score_sentinel: 1.0}

    assert not any(token in str(failed_start.value) for token in sentinels)
    assert all(token not in message for token in sentinels for message in caplog.messages)


@pytest.mark.usefixtures("scripted")
def test_inline_loads_the_backend_once(tmp_path: Path) -> None:
    analyzer = load_artifact(write_artifact(tmp_path / "m.ser", script={"name": "once"}), INLINE)

    for _ in range(20):
        _analyse(analyzer, tone(0.5))

    assert _scripted_backend.LOADS["once"] == 1


@_MODES
def test_the_model_file_is_not_needed_after_load(
    tmp_path: Path, fast_model_path: Path, config: AnalysisConfig
) -> None:
    private_copy = tmp_path / "ser_model.pkl"
    private_copy.write_bytes(fast_model_path.read_bytes())
    analyzer = load_baseline(private_copy, config)
    private_copy.unlink()

    assert {_analyse(analyzer, tone(s))["status"] for s in (1.0, 2.0, 3.0)} == {"ok"}


@_MODES
def test_load_and_analysis_use_no_network(
    tmp_path: Path, fast_model_path: Path, guard: IoGuard, config: AnalysisConfig
) -> None:
    wav = write_wav(tmp_path / "clip.wav", tone(2.0))

    with guard.watching():
        analyzer = load_baseline(fast_model_path, config)
        analyzer.analyze(wav)
        _analyse(analyzer, tone(2.0))
        analyzer.close()

    assert guard.network == []


@_MODES
def test_nothing_is_printed_or_written(
    tmp_path: Path,
    fast_model_path: Path,
    capfd: pytest.CaptureFixture[str],
    guard: IoGuard,
    read_only_cwd_and_tmp: Path,
    config: AnalysisConfig,
) -> None:
    wav = write_wav(tmp_path / "audio" / "clip.wav", tone(2.0))

    with guard.watching(), load_baseline(fast_model_path, config) as analyzer:
        analyzer.analyze(wav)
        analyzer.analyze(tmp_path / "audio" / "missing.wav")
        _analyse(analyzer, tone(0.1))
        _analyse(analyzer, np.zeros(16_000, dtype=np.float32))

    assert capfd.readouterr() == ("", "")
    assert guard.writes == []
    assert list(read_only_cwd_and_tmp.iterdir()) == []


def _logger_state() -> list[tuple[object, ...]]:
    return [
        (lg.level, list(lg.handlers), lg.propagate, lg.disabled)
        for lg in (logging.getLogger(n) for n in ("", "ser", "ser._internal.analysis"))
    ]


@_MODES
def test_logging_and_warnings_are_left_alone(
    tmp_path: Path,
    fast_model_path: Path,
    caplog: pytest.LogCaptureFixture,
    config: AnalysisConfig,
) -> None:
    audio_dir = tmp_path / _SENTINEL
    wav = write_wav(audio_dir / "clip.wav", tone(2.0))
    caplog.set_level(logging.DEBUG)
    before, filters = _logger_state(), list(warnings.filters)

    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always")
        with load_baseline(fast_model_path, config) as analyzer:
            analyzer.analyze(wav)
            analyzer.analyze(audio_dir / "missing.wav")

    assert _logger_state() == before
    assert list(warnings.filters) == filters
    assert [str(w.message) for w in emitted] == []
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []
    assert all(_SENTINEL not in message for message in caplog.messages)


class _SharedDspContentLeak(AssertionError):
    pass


@pytest.mark.xfail(
    strict=True,
    raises=_SharedDspContentLeak,
    reason="Shared DSP logs dependency exceptions before analysis can sanitize them (ADR D14).",
)
def test_fast_dsp_failure_does_not_log_dependency_content(
    fast_model_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError(_scripted_backend.SECRET)

    monkeypatch.setattr(dsp.librosa, "stft", fail)
    caplog.set_level(logging.DEBUG)
    with load_baseline(fast_model_path, INLINE) as analyzer:
        assert _analyse(analyzer, tone(1.0))["status"] == "model_error"
    if any(_scripted_backend.SECRET in message for message in caplog.messages):
        raise _SharedDspContentLeak("Shared DSP exposed exception text.")


def test_results_are_deterministic(fast_model_path: Path) -> None:
    samples = noise(4.0, seed=7)
    first, second = load_baseline(fast_model_path), load_baseline(fast_model_path)

    runs = [_analyse(first, samples), _analyse(first, samples), _analyse(second, samples)]

    assert runs[0]["status"] == "ok"
    assert without_timing(runs[1]) == without_timing(runs[0])
    assert without_timing(runs[2]) == without_timing(runs[0])


@_MODES
def test_one_analyzer_can_be_shared_by_threads(
    fast_model_path: Path, config: AnalysisConfig
) -> None:
    clips = _clips()
    analyzer = load_baseline(fast_model_path, config)

    def analyse(samples: NDArray[np.float32]) -> dict[str, object]:
        return without_timing(_analyse(analyzer, samples))

    sequential = [analyse(samples) for samples in clips]
    with ThreadPoolExecutor(max_workers=4) as pool:
        concurrent = list(pool.map(analyse, clips * 3))

    assert concurrent == sequential * 3


def test_two_analyzers_in_two_threads_are_independent(
    fast_model_path: Path, other_fast_model_path: Path
) -> None:
    clips = _clips()
    analyzers = [load_baseline(fast_model_path), load_baseline(other_fast_model_path)]

    def run_all(index: int) -> list[dict[str, object]]:
        return [without_timing(_analyse(analyzers[index], samples)) for samples in clips]

    sequential = [run_all(0), run_all(1)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        concurrent = list(pool.map(run_all, [0, 1]))

    assert concurrent == sequential
    assert sequential[0][0]["model_sha256"] != sequential[1][0]["model_sha256"]


@pytest.mark.usefixtures("scripted")
def test_inline_calls_on_one_instance_never_overlap_in_the_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = {"gate": "held", "exclusive": True}
    analyzer = load_artifact(write_artifact(tmp_path / "m.ser", script=script), INLINE)
    held = _scripted_backend.gate("held")
    runner = analyzer._session.runner
    assert isinstance(runner, _InlineSession)
    waiting = Event()
    original_wait = runner._condition.wait

    def observe_wait(timeout: float | None = None) -> bool:
        waiting.set()
        return original_wait(timeout)

    monkeypatch.setattr(runner._condition, "wait", observe_wait)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(_analyse, analyzer, tone(0.5))
        try:
            assert held.entered.wait(timeout=10.0)
            second = pool.submit(_analyse, analyzer, tone(0.5))
            assert waiting.wait(timeout=10.0)
            assert held.entries == 1
        finally:
            held.release.set()
        statuses = (first.result(timeout=10.0)["status"], second.result(timeout=10.0)["status"])

    assert statuses == ("ok", "ok")


@pytest.mark.usefixtures("scripted")
def test_inline_instances_do_not_serialise_each_other(tmp_path: Path) -> None:
    holder = load_artifact(write_artifact(tmp_path / "a.ser", script={"gate": "g"}), INLINE)
    waiter = load_artifact(write_artifact(tmp_path / "b.ser", script={"await_gate": "g"}), INLINE)
    held = _scripted_backend.gate("g")

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(_analyse, holder, tone(0.5))
        try:
            assert held.entered.wait(timeout=10.0)
            second = pool.submit(_analyse, waiter, tone(0.5)).result(timeout=10.0)
        finally:
            held.release.set()
        assert first.result(timeout=10.0)["status"] == "ok"

    assert second["status"] == "ok"


@pytest.mark.usefixtures("scripted")
def test_close_waits_for_the_call_in_flight(tmp_path: Path) -> None:
    analyzer = load_artifact(write_artifact(tmp_path / "m.ser", script={"gate": "c"}), INLINE)
    held = _scripted_backend.gate("c")
    runner = analyzer._session.runner
    assert isinstance(runner, _InlineSession)

    with ThreadPoolExecutor(max_workers=2) as pool:
        call = pool.submit(_analyse, analyzer, tone(0.5))
        try:
            assert held.entered.wait(timeout=10.0)
            closing = pool.submit(analyzer.close)
            with runner._condition:
                assert runner._condition.wait_for(lambda: runner._closing, timeout=10.0)
                assert not closing.done()
        finally:
            held.release.set()
        closing.result(timeout=10.0)
        assert call.result(timeout=10.0)["status"] == "ok"


@_MODES
def test_close_is_idempotent_and_ends_the_analyzer(
    fast_model_path: Path, config: AnalysisConfig
) -> None:
    with load_baseline(fast_model_path, config) as analyzer:
        assert _analyse(analyzer, tone(1.0))["status"] == "ok"
    analyzer.close()

    with pytest.raises(AnalysisClosedError):
        _analyse(analyzer, tone(1.0))


@pytest.mark.usefixtures("scripted")
@pytest.mark.parametrize(
    ("failure", "expected"),
    [("keyboard", KeyboardInterrupt), ("memory", MemoryError)],
)
def test_interrupts_and_memory_errors_propagate_inline(
    tmp_path: Path, failure: str, expected: type[BaseException]
) -> None:
    script = {"raise": failure}
    analyzer = load_artifact(write_artifact(tmp_path / "m.ser", script=script), INLINE)

    with pytest.raises(expected):
        _analyse(analyzer, tone(1.0))


@pytest.mark.usefixtures("scripted")
@_MODES
def test_context_exit_closes_before_an_explicit_close(
    tmp_path: Path, config: AnalysisConfig
) -> None:
    with load_artifact(write_artifact(tmp_path / "model.ser"), config) as analyzer:
        assert _analyse(analyzer, tone(1.0))["status"] == "ok"

    with pytest.raises(AnalysisClosedError):
        _analyse(analyzer, tone(1.0))
    analyzer.close()
