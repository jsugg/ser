"""In-process checks of batch validation, checkpointing, and exit codes."""

from __future__ import annotations

import json
import os
import stat
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Never

import pytest

from ser._internal.analysis.batch import run_analyze_command
from ser.api import (
    UTTERANCE_SCHEMA_VERSION,
    AnalysisModelNotConformingError,
    AnalysisModelUnavailableError,
    UtteranceAnalyzer,
    UtteranceRecord,
)
from ser.config import get_settings

_ROW = '{"id":"first","path":"audio/x.wav"}\n'


def _run(*args: str | Path) -> int:
    return run_analyze_command(list(map(str, args)), settings=get_settings())


@pytest.fixture
def manifest(tmp_path: Path) -> Path:
    path = tmp_path / "input.jsonl"
    path.write_text(_ROW)
    return path


@pytest.fixture
def no_model(monkeypatch: pytest.MonkeyPatch) -> None:
    def load(**kwargs: object) -> Never:
        pytest.fail("The batch tried to load a model")

    monkeypatch.setattr(UtteranceAnalyzer, "load", load)


@pytest.fixture
def analyzed_paths(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    paths: list[Path] = []

    class StandIn:
        def analyze(self, path: Path) -> UtteranceRecord:
            paths.append(path)
            return {
                "schema_version": UTTERANCE_SCHEMA_VERSION,
                "ser_version": "1.0.0",
                "model_id": "test",
                "model_sha256": "a" * 64,
                "label_set_id": "test",
                "labels": None,
                "top_label": None,
                "confidence": None,
                "frames": None,
                "dimensions": None,
                "audio": {"duration_s": None, "input_sample_rate_hz": None},
                "status": "unreadable",
                "timing_ms": {"features": 0, "predict": 0},
            }

    @contextmanager
    def load(**kwargs: object) -> Iterator[StandIn]:
        yield StandIn()

    monkeypatch.setattr(UtteranceAnalyzer, "load", load)
    return paths


@pytest.mark.usefixtures("no_model")
@pytest.mark.parametrize("args", [("--threads", "0"), ("--dir", "private-directory"), ("--oops",)])
def test_batch_invalid_arguments_raise_exit_2(
    manifest: Path, args: tuple[str, ...], capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as error:
        _run("--manifest", manifest, "--out", manifest.parent / "out.jsonl", *args)
    assert error.value.code == 2
    assert capsys.readouterr().err.endswith("invalid arguments (see ser analyze --help)\n")


@pytest.mark.usefixtures("no_model")
@pytest.mark.parametrize(
    ("flag", "source", "destination"),
    [
        ("--manifest", "missing.jsonl", "out.jsonl"),
        ("--manifest", "directory", "out.jsonl"),
        ("--dir", "missing-directory", "out.jsonl"),
        ("--dir", "input.jsonl", "out.jsonl"),
        ("--manifest", "input.jsonl", "missing-directory/out.jsonl"),
        ("--manifest", "input.jsonl", "input.jsonl/out.jsonl"),
        ("--manifest", "input.jsonl", "directory"),
    ],
)
def test_batch_bad_command_paths_exit_2(
    manifest: Path,
    flag: str,
    source: str,
    destination: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = manifest.parent
    directory = root / "directory"
    directory.mkdir()
    assert _run(flag, root / source, "--out", root / destination) == 2
    stdout, stderr = capsys.readouterr()
    assert stdout == ""
    assert stderr.startswith("ser analyze: ")
    assert str(root / source) not in stderr and str(root / destination) not in stderr
    assert set(root.iterdir()) == {manifest, directory}


@pytest.mark.usefixtures("no_model")
@pytest.mark.skipif(os.name != "posix", reason="POSIX directory permissions")
def test_batch_unwritable_output_directory_fails_before_model_load(
    manifest: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    directory = manifest.parent / "read-only"
    directory.mkdir()
    out = directory / "out.jsonl"
    sidecar = directory / ".out.jsonl.lock"
    sidecar.touch()
    directory.chmod(0o555)
    try:
        if os.access(directory, os.W_OK):
            pytest.skip("process can write to read-only directories")
        assert _run("--manifest", manifest, "--out", out) == 1
        assert capsys.readouterr().err == "ser analyze: batch input or output I/O failed.\n"
        assert list(directory.iterdir()) == [sidecar]
    finally:
        directory.chmod(0o755)


@pytest.mark.usefixtures("no_model")
@pytest.mark.parametrize(
    ("line", "message"),
    [(_ROW, "Duplicate ID in manifest row 2."), ("{\n", "Invalid manifest row 2.")],
)
def test_batch_invalid_manifest(
    manifest: Path, line: str, message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest.write_text(_ROW + line)
    out = manifest.parent / "out.jsonl"
    assert _run("--manifest", manifest, "--out", out) == 2
    assert capsys.readouterr().err == f"ser analyze: {message}\n"
    assert not out.exists()


@pytest.mark.usefixtures("no_model")
def test_batch_empty_manifest_publishes_empty_output(
    manifest: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest.write_bytes(b"")
    out = manifest.parent / "out.jsonl"
    assert _run("--manifest", manifest, "--out", out, "--summary") == 0
    assert out.read_bytes() == b""
    assert json.loads(capsys.readouterr().err) == {
        "read": 0,
        "analyzed": 0,
        "skipped": 0,
        "statuses": {},
    }


@pytest.mark.usefixtures("no_model")
@pytest.mark.skipif(os.name != "posix", reason="POSIX file modes")
@pytest.mark.parametrize("umask", [0o002, 0o400])
def test_batch_new_output_is_private(manifest: Path, umask: int) -> None:
    manifest.write_bytes(b"")
    out = manifest.parent / "out.jsonl"
    old_umask = os.umask(umask)
    try:
        assert _run("--manifest", manifest, "--out", out) == 0
    finally:
        os.umask(old_umask)
    assert stat.S_IMODE(out.stat().st_mode) == 0o600


@pytest.mark.skipif(os.name != "posix", reason="POSIX file modes")
def test_batch_preserves_mode_on_partial_resume(
    manifest: Path, analyzed_paths: list[Path], capsys: pytest.CaptureFixture[str]
) -> None:
    out = manifest.parent / "out.jsonl"
    args = ("--manifest", manifest, "--out", out)
    assert _run(*args) == 0
    out.chmod(0o640)
    checkpoint = out.read_bytes()
    manifest.write_text(_ROW + '{"id":"second","path":"audio/y.wav"}\n')
    assert _run(*args, "--progress", "--summary") == 0
    assert stat.S_IMODE(out.stat().st_mode) == 0o640
    assert out.read_bytes().startswith(checkpoint)
    assert [json.loads(line)["id"] for line in out.read_bytes().splitlines()] == ["first", "second"]
    assert analyzed_paths == [manifest.parent / "audio/x.wav", manifest.parent / "audio/y.wav"]
    progress, summary = capsys.readouterr().err.splitlines()
    assert progress == "Analyzed 2/2"
    assert json.loads(summary) == {
        "read": 2,
        "analyzed": 1,
        "skipped": 1,
        "statuses": {"unreadable": 2},
    }


@pytest.mark.usefixtures("no_model")
@pytest.mark.parametrize("name", [".private.jsonl.lock", "..private.jsonl.ser-batch-abcd_123"])
def test_batch_refuses_reserved_output(
    manifest: Path, name: str, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest.write_bytes(b"")
    assert _run("--manifest", manifest, "--out", manifest.parent / name) == 2
    assert capsys.readouterr() == ("", "ser analyze: The output name is reserved.\n")
    assert set(manifest.parent.iterdir()) == {manifest}


@pytest.mark.usefixtures("no_model")
def test_batch_refuses_symlink_lock(manifest: Path, capsys: pytest.CaptureFixture[str]) -> None:
    pytest.importorskip("fcntl")
    manifest.write_bytes(b"")
    out = manifest.parent / "out.jsonl"
    lock = manifest.parent / ".out.jsonl.lock"
    target = manifest.parent / "absent"
    lock.symlink_to(target)
    assert _run("--manifest", manifest, "--out", out) == 2
    assert capsys.readouterr() == ("", "ser analyze: The output lock is a symbolic link.\n")
    assert lock.readlink() == target
    assert set(manifest.parent.iterdir()) == {manifest, lock}


@pytest.mark.usefixtures("no_model")
def test_batch_held_lock_leaves_staging_untouched(
    manifest: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fcntl = pytest.importorskip("fcntl")
    out = manifest.parent / "out.jsonl"
    stale = manifest.parent / ".out.jsonl.ser-batch-k3j9x_2q"
    stale.write_bytes(b"staging")
    with (manifest.parent / ".out.jsonl.lock").open("wb") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert _run("--manifest", manifest, "--out", out) == 2
    assert capsys.readouterr().err == "ser analyze: Another batch owns the output lock.\n"
    assert stale.read_bytes() == b"staging"
    assert not out.exists()


@pytest.mark.usefixtures("no_model")
def test_batch_removes_stale_staging_on_complete_resume(manifest: Path) -> None:
    pytest.importorskip("fcntl")
    manifest.write_bytes(b"")
    out = manifest.parent / "out.jsonl"
    out.touch()
    stale = manifest.parent / ".out.jsonl.ser-batch-k3j9x_2q"
    stale.touch()
    assert _run("--manifest", manifest, "--out", out) == 0
    assert not stale.exists()


@pytest.mark.usefixtures("no_model")
def test_batch_without_fcntl_keeps_stale_staging(
    manifest: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "fcntl", None)
    manifest.write_bytes(b"")
    out = manifest.parent / "out.jsonl"
    stale = manifest.parent / ".out.jsonl.ser-batch-k3j9x_2q"
    stale.touch()
    assert _run("--manifest", manifest, "--out", out) == 0
    assert stale.exists()
    assert not (manifest.parent / ".out.jsonl.lock").exists()


@pytest.mark.usefixtures("no_model")
@pytest.mark.parametrize(
    ("checkpoint", "diagnostic"),
    [
        (b'{"id":"first"}', "ordered input prefix"),
        (b'{"id":"first","record":{}}\n', "utterance record"),
    ],
)
def test_batch_rejects_invalid_checkpoint(
    manifest: Path, checkpoint: bytes, diagnostic: str, capsys: pytest.CaptureFixture[str]
) -> None:
    out = manifest.parent / "out.jsonl"
    out.write_bytes(checkpoint)
    assert _run("--manifest", manifest, "--out", out) == 2
    assert diagnostic in capsys.readouterr().err
    assert out.read_bytes() == checkpoint


@pytest.mark.parametrize(
    ("error", "diagnostic"),
    [
        (AnalysisModelNotConformingError, "--allow-legacy-baseline"),
        (AnalysisModelUnavailableError, "check the model and settings"),
    ],
)
def test_batch_model_load_error_exits_1(
    manifest: Path,
    error: type[Exception],
    diagnostic: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def load(**kwargs: object) -> Never:
        raise error("private-model-path")

    monkeypatch.setattr(UtteranceAnalyzer, "load", load)
    out = manifest.parent / "out.jsonl"
    assert _run("--manifest", manifest, "--out", out) == 1
    stderr = capsys.readouterr().err
    assert diagnostic in stderr and "private-model-path" not in stderr
    assert not out.exists()
