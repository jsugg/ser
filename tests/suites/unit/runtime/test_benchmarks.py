"""Unit tests for local runtime benchmark helpers."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

from ser.runtime import benchmarks

pytestmark = pytest.mark.unit


def test_benchmark_predict_rejects_invalid_run_count() -> None:
    """Benchmark helper should validate run count before timing work."""
    with pytest.raises(ValueError, match="greater than or equal to 1"):
        benchmarks.benchmark_predict(audio_path="sample.wav", runs=0)


def test_benchmark_predict_returns_deterministic_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Benchmark helper should compute summary statistics from timing samples."""
    perf_samples = iter([0.0, 1.0, 1.0, 3.0, 3.0, 6.0])
    monkeypatch.setattr(benchmarks, "_predict_emotions", lambda _audio_path: ["ok"])
    monkeypatch.setattr(benchmarks.time, "perf_counter", lambda: next(perf_samples))

    summary = benchmarks.benchmark_predict(audio_path="sample.wav", runs=3)

    assert summary == {
        "runs": 3,
        "mean_seconds": 2.0,
        "median_seconds": 2.0,
        "p95_seconds": 3.0,
        "min_seconds": 1.0,
        "max_seconds": 3.0,
    }


def test_parse_args_reads_cli_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    """Argument parser should expose required file and optional output controls."""
    monkeypatch.setattr(
        sys,
        "argv",
        ["ser-benchmark", "--file", "sample.wav", "--runs", "7", "--out", "result.json"],
    )

    args = benchmarks._parse_args()

    assert args == argparse.Namespace(file="sample.wav", runs=7, out="result.json")


def test_main_prints_json_when_no_output_file(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CLI wrapper should print benchmark JSON when no output path is requested."""
    payload = {"runs": 2, "mean_seconds": 1.5}
    monkeypatch.setattr(
        benchmarks,
        "_parse_args",
        lambda: argparse.Namespace(file="sample.wav", runs=2, out=None),
    )
    monkeypatch.setattr(benchmarks, "benchmark_predict", lambda **_kwargs: payload)

    benchmarks.main()

    assert capsys.readouterr().out == json.dumps(payload, indent=2, sort_keys=True) + "\n"


def test_main_writes_json_output_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CLI wrapper should write JSON output when an explicit file path is provided."""
    output_path = tmp_path / "benchmarks" / "summary.json"
    payload = {"runs": 4, "mean_seconds": 2.25}
    monkeypatch.setattr(
        benchmarks,
        "_parse_args",
        lambda: argparse.Namespace(file="sample.wav", runs=4, out=str(output_path)),
    )
    monkeypatch.setattr(benchmarks, "benchmark_predict", lambda **_kwargs: payload)

    benchmarks.main()

    expected_text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    assert output_path.read_text(encoding="utf-8") == expected_text
    assert capsys.readouterr().out == expected_text
