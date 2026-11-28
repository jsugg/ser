"""Load once, stay quiet and offline, and be deterministic and thread-safe."""

from __future__ import annotations

import logging
import socket
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import NoReturn

import numpy as np
import pytest
from numpy.typing import NDArray
from tests.suites.unit.analysis._contract import NOT_IMPLEMENTED, load_fast, without_timing
from tests.suites.unit.analysis._stand_ins import CANONICAL_RATE_HZ, noise, tone, write_wav


def _clips() -> list[NDArray[np.float32]]:
    return [noise(1.5 + index, seed=index) for index in range(4)]


@NOT_IMPLEMENTED
def test_model_is_read_once_at_load(tmp_path: Path, fast_model_path: Path) -> None:
    private_copy = tmp_path / "ser_model.pkl"
    private_copy.write_bytes(fast_model_path.read_bytes())
    analyzer = load_fast(private_copy)
    private_copy.unlink()

    for seconds in (1.0, 2.0, 3.0):
        record = analyzer.analyze(tone(seconds), sample_rate_hz=CANONICAL_RATE_HZ)
        assert record["status"] == "ok"


@NOT_IMPLEMENTED
def test_load_and_analysis_use_no_network(
    tmp_path: Path, fast_model_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(*_args: object, **_kwargs: object) -> NoReturn:
        raise AssertionError("network access during analysis")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    wav = write_wav(tmp_path / "clip.wav", tone(2.0))

    analyzer = load_fast(fast_model_path)

    assert analyzer.analyze(wav)["status"] == "ok"
    assert analyzer.analyze(tone(2.0), sample_rate_hz=CANONICAL_RATE_HZ)["status"] == "ok"


@NOT_IMPLEMENTED
def test_nothing_is_printed_or_written(
    tmp_path: Path,
    fast_model_path: Path,
    capfd: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    monkeypatch.chdir(work_dir)
    wav = write_wav(audio_dir / "clip.wav", tone(2.0))
    model_dir_before = sorted(fast_model_path.parent.iterdir())

    with load_fast(fast_model_path) as analyzer:
        analyzer.analyze(wav)
        analyzer.analyze(audio_dir / "missing.wav")
        analyzer.analyze(tone(0.1), sample_rate_hz=CANONICAL_RATE_HZ)

    assert capfd.readouterr() == ("", "")
    assert list(work_dir.iterdir()) == []
    assert sorted(audio_dir.iterdir()) == [wav]
    assert sorted(fast_model_path.parent.iterdir()) == model_dir_before


@NOT_IMPLEMENTED
def test_root_logging_is_left_alone(fast_model_path: Path) -> None:
    root = logging.getLogger()
    level, handlers = root.level, list(root.handlers)
    formatters = [handler.formatter for handler in handlers]

    analyzer = load_fast(fast_model_path)
    analyzer.analyze(tone(2.0), sample_rate_hz=CANONICAL_RATE_HZ)

    assert root.level == level
    assert root.handlers == handlers
    assert [handler.formatter for handler in root.handlers] == formatters


@NOT_IMPLEMENTED
def test_logs_never_contain_caller_paths(
    tmp_path: Path, fast_model_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    audio_dir = tmp_path / "audio-dir-token-3f7a"
    audio_dir.mkdir()
    wav = write_wav(audio_dir / "clip.wav", tone(2.0))
    caplog.set_level(logging.DEBUG)

    analyzer = load_fast(fast_model_path)
    analyzer.analyze(wav)
    analyzer.analyze(audio_dir / "missing.wav")

    for message in caplog.messages:
        assert "token-3f7a" not in message
        assert "token-91c2" not in message
        assert str(audio_dir) not in message


@NOT_IMPLEMENTED
def test_results_are_deterministic(fast_model_path: Path) -> None:
    samples = noise(4.0, seed=7)
    first = load_fast(fast_model_path)
    second = load_fast(fast_model_path)

    runs = [
        first.analyze(samples, sample_rate_hz=CANONICAL_RATE_HZ),
        first.analyze(samples, sample_rate_hz=CANONICAL_RATE_HZ),
        second.analyze(samples, sample_rate_hz=CANONICAL_RATE_HZ),
    ]

    assert runs[0]["status"] == "ok"
    assert without_timing(runs[1]) == without_timing(runs[0])
    assert without_timing(runs[2]) == without_timing(runs[0])


@NOT_IMPLEMENTED
def test_one_analyzer_can_be_shared_by_threads(fast_model_path: Path) -> None:
    clips = _clips()
    analyzer = load_fast(fast_model_path)

    def analyse(samples: NDArray[np.float32]) -> dict[str, object]:
        return without_timing(analyzer.analyze(samples, sample_rate_hz=CANONICAL_RATE_HZ))

    sequential = [analyse(samples) for samples in clips]
    with ThreadPoolExecutor(max_workers=4) as pool:
        concurrent = list(pool.map(analyse, clips * 3))

    assert concurrent == sequential * 3


@NOT_IMPLEMENTED
def test_two_analyzers_in_two_threads_are_independent(
    fast_model_path: Path, other_fast_model_path: Path
) -> None:
    clips = _clips()
    analyzers = [load_fast(fast_model_path), load_fast(other_fast_model_path)]

    def run_all(index: int) -> list[dict[str, object]]:
        analyzer = analyzers[index]
        return [
            without_timing(analyzer.analyze(samples, sample_rate_hz=CANONICAL_RATE_HZ))
            for samples in clips
        ]

    sequential = [run_all(0), run_all(1)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        concurrent = list(pool.map(run_all, [0, 1]))

    assert concurrent == sequential
    assert sequential[0][0]["model_sha256"] != sequential[1][0]["model_sha256"]
