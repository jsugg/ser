"""Batch CLI contract tests using generated audio and a tiny stand-in model."""

from __future__ import annotations

import json
import os
import random
import selectors
import signal
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from tests.suites.unit.analysis._json_schema_subset import validate
from tests.suites.unit.analysis._stand_ins import (
    CANONICAL_RATE_HZ,
    FailingClassifier,
    noise,
    silence,
    tone,
    write_model,
    write_wav,
)

from ser._internal.features.feature_extractor import extract_feature_from_signal
from ser.api import AnalysisConfig, UtteranceAnalyzer, utterance_record_json_schema

_PRIVATE = "caller-content-never-logged"
_PAUSE_BEFORE_RENAME = """
import os
import sys
from ser.__main__ import main

output = sys.argv.pop(1)

def pause_before_publish(event, args):
    if event == "os.rename" and os.fspath(args[1]) == output:
        print("ready to rename", file=sys.stderr, flush=True)
        sys.stdin.buffer.read(1)

sys.addaudithook(pause_before_publish)
main()
"""


@pytest.fixture(scope="module")
def batch_model(tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("batch-model")
    size = int(extract_feature_from_signal(tone(3.0), CANONICAL_RATE_HZ).size)
    return write_model(directory / "synthetic.pkl", feature_size=size)


def _argv(*args: str | Path) -> list[str]:
    return [sys.executable, "-m", "ser", "analyze", *map(str, args)]


def _run(*args: str | Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(_argv(*args), capture_output=True, text=True, timeout=90, check=False)


def _rows(path: Path) -> list[dict[str, Any]]:
    data = path.read_bytes()
    assert not data or data.endswith(b"\n")
    return [json.loads(line) for line in data.splitlines()]


def test_batch_records_failures_and_summarizes_resume(tmp_path: Path, batch_model: Path) -> None:
    audio = write_wav(tmp_path / "audio" / "tone.wav", tone(0.35))
    manifest = tmp_path / f"{_PRIVATE}.jsonl"
    manifest.write_text(
        json.dumps({"id": "opaque\né", "path": "audio/tone.wav", "ignore": _PRIVATE})
        + "\n"
        + json.dumps({"id": "second", "path": f"audio/{_PRIVATE}.wav"})
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "output.jsonl"
    args = (
        "--manifest",
        manifest,
        "--out",
        out,
        "--model",
        batch_model,
        "--allow-legacy-baseline",
        "--threads",
        "1",
        "--summary",
        "--log-level",
        "DEBUG",
    )
    first = _run(*args)
    assert first.returncode == 0, first.stderr
    assert first.stdout == ""
    assert _PRIVATE not in first.stderr
    rows = _rows(out)
    assert [row["id"] for row in rows] == ["opaque\né", "second"]
    for row in rows:
        assert row.keys() == {"id", "record"}
        assert validate(row["record"], utterance_record_json_schema("ser.utterance/v1")) == []
    with UtteranceAnalyzer.load(
        profile="fast",
        model_path=batch_model,
        allow_legacy_baseline=True,
        config=AnalysisConfig(max_threads=1),
    ) as analyzer:
        expected = analyzer.analyze(audio)
    expected["timing_ms"] = rows[0]["record"]["timing_ms"]
    assert rows[0]["record"] == expected
    assert _PRIVATE not in out.read_text()
    counts = {"ok": 1, "unreadable": 1}
    assert [row["record"]["status"] for row in rows] == list(counts)
    assert json.loads(first.stderr.splitlines()[-1]) == {
        "read": 2,
        "analyzed": 2,
        "skipped": 0,
        "statuses": counts,
    }

    before = out.read_bytes()
    again = _run(*args)
    assert again.returncode == 0, again.stderr
    assert again.stdout == ""
    assert out.read_bytes() == before
    assert json.loads(again.stderr.splitlines()[-1]) == {
        "read": 2,
        "analyzed": 0,
        "skipped": 2,
        "statuses": counts,
    }


def test_batch_continues_after_model_error(tmp_path: Path) -> None:
    feature_size = int(extract_feature_from_signal(tone(3.0), CANONICAL_RATE_HZ).size)
    model = write_model(
        tmp_path / "model.pkl", feature_size=feature_size, classifier_type=FailingClassifier
    )
    audio = tmp_path / "audio"
    write_wav(audio / "b-error.wav", tone(0.35))
    write_wav(audio / "c-short.wav", tone(0.1))
    write_wav(audio / "d-silent.wav", silence(0.35))
    manifest = tmp_path / "input.jsonl"
    names = ["a-missing.wav", "b-error.wav", "c-short.wav", "d-silent.wav"]
    manifest.write_text(
        "".join(json.dumps({"id": name, "path": f"audio/{name}"}) + "\n" for name in names)
    )
    out = tmp_path / "out.jsonl"
    result = _run("--manifest", manifest, "--out", out, "--model", model, "--allow-legacy-baseline")
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    rows = _rows(out)
    assert [row["id"] for row in rows] == names
    assert [row["record"]["status"] for row in rows] == [
        "unreadable",
        "model_error",
        "too_short",
        "silent",
    ]


def test_batch_directory_is_sorted_and_quiet(tmp_path: Path, batch_model: Path) -> None:
    audio = tmp_path / "audio"
    for name in ("b.WAV", "a.wav", "nested/c.wav"):
        write_wav(audio / name, tone(0.1))
    (audio / "not-a-wav.txt").touch()
    out = tmp_path / "output.jsonl"
    result = _run("--dir", audio, "--out", out, "--model", batch_model, "--allow-legacy-baseline")
    assert result.returncode == 0, result.stderr
    assert result.stdout == result.stderr == ""
    assert [row["id"] for row in _rows(out)] == ["a.wav", "b.WAV"]


@pytest.mark.parametrize(
    "last_line",
    [b'{"id":"first","path":"another.wav"}', b'{"id":"second"}', b"{not-json", b"\xff"],
)
def test_batch_rejects_bad_manifest_without_exposing_content(
    tmp_path: Path,
    last_line: bytes,
) -> None:
    manifest = tmp_path / f"{_PRIVATE}.jsonl"
    manifest.write_bytes(b'{"id":"first","path":"audio/missing.wav"}\n' + last_line + b"\n")
    out = tmp_path / "output.jsonl"
    result = _run("--manifest", manifest, "--out", out)
    assert result.returncode == 2
    assert "manifest row 2" in result.stderr
    assert _PRIVATE not in result.stderr
    assert result.stdout == ""
    assert not out.exists()


@pytest.mark.parametrize("existing", [b'{"id":"clip"}', b'{"id":"another","record":{}}\n'])
def test_batch_refuses_an_invalid_checkpoint(tmp_path: Path, existing: bytes) -> None:
    manifest = tmp_path / "input.jsonl"
    manifest.write_text('{"id":"clip","path":"audio/x.wav"}\n')
    out = tmp_path / "output.jsonl"
    out.write_bytes(existing)
    result = _run("--manifest", manifest, "--out", out)
    assert result.returncode == 2
    assert "existing output" in result.stderr
    assert out.read_bytes() == existing


def test_batch_never_writes_in_the_audio_directory(tmp_path: Path) -> None:
    audio = tmp_path / "audio"
    wav = write_wav(audio / "input.wav", tone(0.1))
    original = wav.read_bytes()
    result = _run("--dir", audio, "--out", audio / "output.jsonl")
    assert result.returncode == 2
    assert "audio directory" in result.stderr
    assert list(audio.iterdir()) == [wav]
    assert wav.read_bytes() == original


def test_batch_empty_input_needs_no_model(tmp_path: Path) -> None:
    manifest = tmp_path / "input.jsonl"
    manifest.touch()
    out = tmp_path / "output.jsonl"
    result = _run("--manifest", manifest, "--out", out, "--summary")
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    assert out.read_bytes() == b""
    assert json.loads(result.stderr) == {"read": 0, "analyzed": 0, "skipped": 0, "statuses": {}}


def test_batch_rejects_a_model_change_on_partial_resume(tmp_path: Path, batch_model: Path) -> None:
    manifest = tmp_path / "input.jsonl"
    manifest.write_text('{"id":"first","path":"audio/missing.wav"}\n')
    out = tmp_path / "output.jsonl"
    args = ("--manifest", manifest, "--out", out, "--allow-legacy-baseline", "--model")
    first = _run(*args, batch_model)
    assert first.returncode == 0, first.stderr
    checkpoint = out.read_bytes()
    with manifest.open("a") as stream:
        stream.write('{"id":"second","path":"audio/missing.wav"}\n')
    changed = tmp_path / f"{_PRIVATE}.pkl"
    changed.write_bytes(batch_model.read_bytes() + b"\x00")
    resume = _run(*args, changed)
    assert resume.returncode == 1
    assert _PRIVATE not in resume.stderr
    assert out.read_bytes() == checkpoint


def test_batch_requires_legacy_opt_in(tmp_path: Path, batch_model: Path) -> None:
    manifest = tmp_path / "input.jsonl"
    manifest.write_text('{"id":"clip","path":"audio/x.wav"}\n')
    out = tmp_path / "output.jsonl"
    result = _run("--manifest", manifest, "--out", out, "--model", batch_model)
    assert result.returncode == 1
    assert "--allow-legacy-baseline" in result.stderr
    assert batch_model.name not in result.stderr
    assert result.stdout == ""
    assert not out.exists()


def test_batch_does_not_echo_unrecognized_arguments(tmp_path: Path) -> None:
    result = _run("--manifest", tmp_path / _PRIVATE, "--unexpected", _PRIVATE)
    assert result.returncode == 2
    assert "usage: ser analyze" in result.stderr
    assert _PRIVATE not in result.stderr
    assert result.stdout == ""


@pytest.mark.skipif(sys.platform != "linux", reason="SIGKILL process-resume proof")
def test_batch_cleans_staging_after_kill_before_rename(tmp_path: Path, batch_model: Path) -> None:
    manifest = tmp_path / "input.jsonl"
    manifest.write_text('{"id":"clip","path":"audio/missing.wav"}\n')
    out = tmp_path / "output[1].jsonl"
    sidecar = out.with_name(f".{out.name}.lock")
    other_staging = {
        tmp_path / ".output1.jsonl.ser-batch-unrelated",
        tmp_path / f".{out.name}.ser-batch-other.ser-batch-abandoned",
    }
    for staging in other_staging:
        staging.write_bytes(b"other batch")
    args = ("--manifest", manifest, "--out", out, "--model", batch_model, "--allow-legacy-baseline")
    argv = [sys.executable, "-c", _PAUSE_BEFORE_RENAME, str(out), "analyze", *map(str, args)]
    _kill_after_event(argv, b"ready to rename")
    assert not out.exists()
    (abandoned,) = set(tmp_path.iterdir()) - {manifest, sidecar} - other_staging
    result = _run(*args)
    assert result.returncode == 0, result.stderr
    assert [row["id"] for row in _rows(out)] == ["clip"]
    assert not abandoned.exists()
    assert abandoned.name.startswith(f".{out.name}.ser-batch-")
    assert all(staging.read_bytes() == b"other batch" for staging in other_staging)


@pytest.mark.skipif(sys.platform != "linux", reason="POSIX output locking")
def test_batch_refuses_concurrent_writer(tmp_path: Path, batch_model: Path) -> None:
    manifest = tmp_path / "input.jsonl"
    manifest.write_text(
        '{"id":"first","path":"audio/x.wav"}\n{"id":"second","path":"audio/y.wav"}\n'
    )
    out = tmp_path / f"{_PRIVATE}.jsonl"
    args = ("--manifest", manifest, "--out", out, "--model", batch_model, "--allow-legacy-baseline")
    argv = [sys.executable, "-c", _PAUSE_BEFORE_RENAME, str(out), "analyze", *map(str, args)]
    with _process_at_event(argv, b"ready to rename") as writer:
        contender = _run(*args)
        assert contender.returncode == 2, contender.stderr
        assert contender.stdout == ""
        assert contender.stderr == "ser analyze: Another batch owns the output lock.\n"
        assert not out.exists()
        stdout, stderr = writer.communicate(input=b"\n", timeout=30)
        assert writer.returncode == 0, stderr
        assert stdout == b""
    assert [row["id"] for row in _rows(out)] == ["first", "second"]
    assert out.with_name(f".{out.name}.lock").is_file()


@contextmanager
def _process_at_event(argv: list[str], target: bytes) -> Iterator[subprocess.Popen[bytes]]:
    with subprocess.Popen(
        argv,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    ) as process:
        assert process.stderr is not None
        deadline = time.monotonic() + 90
        pending = b""
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(process.stderr, selectors.EVENT_READ)
                while True:
                    assert selector.select(max(0, deadline - time.monotonic())), pending
                    chunk = os.read(process.stderr.fileno(), 8192)
                    assert chunk, (process.poll(), pending)
                    lines = (pending + chunk).split(b"\n")
                    pending = lines.pop()
                    if target in lines:
                        break
            yield process
        finally:
            if process.returncode is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)


def _kill_after_event(argv: list[str], target: bytes) -> None:
    with _process_at_event(argv, target) as process:
        process.kill()
        stdout, _ = process.communicate(timeout=10)
        assert process.returncode == -signal.SIGKILL
        assert stdout == b""


@pytest.mark.skipif(sys.platform != "linux", reason="SIGKILL process-resume proof")
def test_batch_sigkill_resume_on_1000_recordings(tmp_path: Path, batch_model: Path) -> None:
    total = 1000
    audio = tmp_path / "audio"
    ids = [f"{i:04d}.wav" for i in range(total)]
    for i, name in enumerate(ids):
        samples = tone(0.32) if i % 2 == 0 else noise(0.32, seed=i)
        write_wav(audio / name, samples, subtype="PCM_16")
    out = tmp_path / "output.jsonl"
    argv = _argv(
        "--dir",
        audio,
        "--out",
        out,
        "--model",
        batch_model,
        "--allow-legacy-baseline",
        "--threads",
        "1",
        "--progress",
        "--summary",
    )
    rng = random.Random(913)
    saved = b""
    for start in (100, 350, 600):
        cut = rng.randrange(start, start + 75)
        _kill_after_event(argv, f"Analyzed {cut}/{total}".encode("ascii"))
        snapshot = out.read_bytes()
        assert len(snapshot) > len(saved)
        assert snapshot.startswith(saved)
        saved = snapshot
        rows = _rows(out)
        assert len(rows) < total
        assert [row["id"] for row in rows] == ids[: len(rows)]

    final = subprocess.run(argv, capture_output=True, text=True, timeout=90, check=False)
    assert final.returncode == 0, final.stderr
    assert final.stdout == ""
    assert out.read_bytes().startswith(saved)
    assert [row["id"] for row in _rows(out)] == ids
    assert json.loads(final.stderr.splitlines()[-1]) == {
        "read": total,
        "analyzed": total - len(rows),
        "skipped": len(rows),
        "statuses": {"ok": total},
    }
