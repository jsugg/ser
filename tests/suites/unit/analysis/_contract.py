"""Shared helpers for the utterance-analysis contract tests.

Tests reach the interface through ``ser.api`` only. Internal modules appear solely as
monkeypatch targets, which the implementation must call through their module
(``artifact.read_model_file``, ``artifact.artifact_formats``,
``backend.backend_loaders``, ``worker.start_worker``, ``audio_utils.to_canonical_audio``).
"""

from __future__ import annotations

import json
import math
import os
import stat
import sys
import types
import typing
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from tests.suites.unit.analysis._json_schema_subset import validate

from ser.api import (
    AnalysisConfig,
    ModelManifest,
    UtteranceAnalyzer,
    UtteranceRecord,
    utterance_record_json_schema,
)

PROBABILITY_SUM_TOLERANCE = 1e-6
INLINE = AnalysisConfig(isolation="inline")
_ANALYZERS = ExitStack()


def load_baseline(
    model_path: str | os.PathLike[str],
    config: AnalysisConfig | None = None,
) -> UtteranceAnalyzer:
    """Loads a legacy ``fast`` pickle, which only loads as an explicit baseline."""
    return _ANALYZERS.enter_context(
        UtteranceAnalyzer.load(
            profile="fast", model_path=model_path, allow_legacy_baseline=True, config=config
        )
    )


def load_artifact(
    model_path: str | os.PathLike[str],
    config: AnalysisConfig | None = None,
) -> UtteranceAnalyzer:
    """Loads a synthetic safe artifact as a conforming ``fast`` model."""
    return _ANALYZERS.enter_context(
        UtteranceAnalyzer.load(profile="fast", model_path=model_path, config=config)
    )


def assert_valid_record(record: UtteranceRecord, manifest: ModelManifest) -> None:
    """Checks the schema, the Python types, and the clauses JSON Schema cannot express."""
    identity = manifest.identity
    assert validate(record, utterance_record_json_schema()) == []
    assert record_type_errors(record) == []
    assert json.loads(json.dumps(record, allow_nan=False)) == record
    assert record["model_id"] == identity.model_id
    assert record["model_sha256"] == identity.model_sha256
    assert record["label_set_id"] == identity.label_set_id
    if record["status"] != "ok":
        return

    labels = record["labels"]
    frames = record["frames"]
    duration_s = record["audio"]["duration_s"]
    assert labels is not None and frames is not None and duration_s is not None
    assert set(labels) == set(identity.labels)
    assert math.isclose(sum(labels.values()), 1.0, abs_tol=PROBABILITY_SUM_TOLERANCE)
    expected_top = min(labels, key=lambda label: (-labels[label], label))
    assert record["top_label"] == expected_top
    assert record["confidence"] == labels[expected_top]

    assert frames[0]["start_s"] == 0.0
    assert frames[-1]["end_s"] == pytest.approx(duration_s, abs=1e-3)
    covered_to = 0.0
    for frame in frames:
        assert frame["start_s"] <= covered_to + 1e-9, "frames leave a gap"
        assert 0.0 <= frame["start_s"] < frame["end_s"] <= duration_s + 1e-3
        covered_to = max(covered_to, frame["end_s"])
        assert set(frame["labels"]) == set(identity.labels)
        assert math.isclose(sum(frame["labels"].values()), 1.0, abs_tol=PROBABILITY_SUM_TOLERANCE)
    starts = [frame["start_s"] for frame in frames]
    assert starts == sorted(starts)


def record_type_errors(
    value: object, annotation: object = UtteranceRecord, path: str = "$"
) -> list[str]:
    """Python-type check against the TypedDicts; stricter than the JSON Schema.

    The schema accepts ``310.0`` as an integer, as JSON does; a record SER builds must
    still carry a Python ``int`` there, and a ``float`` wherever the type says float.
    """
    origin = typing.get_origin(annotation)
    if annotation is None or annotation is type(None):
        return [] if value is None else [f"{path}: expected None"]
    if origin is typing.Literal:
        return [] if value in typing.get_args(annotation) else [f"{path}: not in literal"]
    if origin in (types.UnionType, typing.Union):
        options = typing.get_args(annotation)
        if any(not record_type_errors(value, option, path) for option in options):
            return []
        return [f"{path}: matches no member of {annotation}"]
    if isinstance(annotation, typing.TypeAliasType):
        return record_type_errors(value, annotation.__value__, path)
    if typing.is_typeddict(annotation):
        if not isinstance(value, dict):
            return [f"{path}: expected object"]
        hints = typing.get_type_hints(annotation)
        mapping = typing.cast(dict[str, object], value)
        found = [] if set(mapping) == set(hints) else [f"{path}: keys differ"]
        for name, hint in hints.items():
            if name in mapping:
                found += record_type_errors(mapping[name], hint, f"{path}.{name}")
        return found
    if origin is dict:
        if not isinstance(value, dict):
            return [f"{path}: expected dict"]
        _, item = typing.get_args(annotation)
        errors: list[str] = []
        for key, child in typing.cast(dict[object, object], value).items():
            if type(key) is not str:
                errors.append(f"{path}: non-string key")
            errors += record_type_errors(child, item, f"{path}[{key!r}]")
        return errors
    if origin is list:
        if not isinstance(value, list):
            return [f"{path}: expected list"]
        (item,) = typing.get_args(annotation)
        return [
            error
            for index, child in enumerate(typing.cast(list[object], value))
            for error in record_type_errors(child, item, f"{path}[{index}]")
        ]
    if annotation in (int, float, str):
        return [] if type(value) is annotation else [f"{path}: expected {annotation}"]
    raise TypeError(f"unhandled annotation {annotation!r}")


def without_timing(record: UtteranceRecord) -> dict[str, object]:
    """The record minus ``timing_ms``, the only field excluded from determinism."""
    return {key: value for key, value in record.items() if key != "timing_ms"}


_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC
_WRITE_EVENTS = frozenset(
    {"os.remove", "os.rename", "os.mkdir", "os.rmdir", "os.truncate", "os.link", "os.symlink"}
)
_NETWORK_EVENTS = frozenset(
    {
        "socket.connect",
        "socket.bind",
        "socket.getaddrinfo",
        "socket.gethostbyname",
        "socket.sendto",
    }
)


@dataclass
class IoGuard:
    """Records file writes and network use in this process while active.

    Audit hooks cannot be removed, so one hook is installed per process and does
    nothing while inactive. The worker process is covered by a read-only ``TMPDIR``
    and working directory instead.
    """

    active: bool = False
    writes: list[str] = field(default_factory=list)
    network: list[str] = field(default_factory=list)

    @contextmanager
    def watching(self) -> Iterator[None]:
        self.writes.clear()
        self.network.clear()
        self.active = True
        try:
            yield
        finally:
            self.active = False

    def hook(self, event: str, args: tuple[object, ...]) -> None:
        if not self.active:
            return
        if event == "open":
            path, mode, flags = args
            if isinstance(path, int):
                try:
                    if not stat.S_ISREG(os.fstat(path).st_mode):
                        return
                except OSError:
                    return
            writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (
                isinstance(flags, int) and flags & _WRITE_FLAGS
            )
            if writing and path != os.devnull:
                self.writes.append(f"open {path!r}")
        elif event in _WRITE_EVENTS:
            self.writes.append(event)
        elif event in _NETWORK_EVENTS:
            self.network.append(event)


_GUARD: IoGuard | None = None


def io_guard() -> IoGuard:
    global _GUARD
    if _GUARD is None:
        _GUARD = IoGuard()
        sys.addaudithook(_GUARD.hook)
    return _GUARD


def read_only_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    path.chmod(0o555)
    return path
