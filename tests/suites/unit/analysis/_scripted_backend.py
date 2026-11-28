"""A backend whose behaviour is the JSON payload of a synthetic artifact.

Registered for the contract tests in place of the real loaders. Process-local state
(``LOADS``, ``GATES``, ``CALLS``) is observable only with ``isolation="inline"``.
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import numpy as np
import threadpoolctl
from numpy.typing import NDArray
from tests.suites.unit.analysis._contract import IoGuard, io_guard

from ser._internal.analysis.backend import BackendFrame, BackendLoadRequest, BackendScores
from ser._internal.utils.audio_utils import CanonicalAudio
from ser.runtime.contracts import LanguageHint

CENTROID_LABELS: tuple[str, ...] = ("high", "low")
LANGUAGE_LABELS: tuple[str, ...] = ("en", "es", "none", "pt")
CANONICAL_RATE_HZ = 16_000
SECRET = "backend-secret-token-77ab"

LOADS: Counter[str] = Counter()


def advance_time(seconds: float) -> None:
    raise AssertionError("Scripted elapsed time requires the virtual clock fixture.")


@dataclass
class Gate:
    """Lets a test hold a backend inside ``score`` and see that it got there."""

    entered: threading.Event = field(default_factory=threading.Event)
    release: threading.Event = field(default_factory=threading.Event)
    entries: int = 0


GATES: dict[str, Gate] = {}


@dataclass(frozen=True)
class ScriptedCall:
    samples: NDArray[np.float32]
    sample_rate: int


CALLS: dict[str, list[ScriptedCall]] = {}


def reset() -> None:
    LOADS.clear()
    GATES.clear()
    CALLS.clear()


def gate(name: str) -> Gate:
    return GATES.setdefault(name, Gate())


def load_scripted(request: BackendLoadRequest) -> ScriptedBackend:
    script = cast(dict[str, object], json.loads(request.payload))
    audit = _ChildAudit() if script.get("audit") else None
    marker = script.get("fail_load_if_exists")
    if script.get("fail_load") or (isinstance(marker, str) and Path(marker).exists()):
        raise RuntimeError(f"scripted load failure {SECRET}")
    LOADS[str(script.get("name", ""))] += 1
    return ScriptedBackend(script, request, audit)


class _ChildAudit(logging.Handler):
    """Observes the real child process without any file or network side channel."""

    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []
        self.io: IoGuard = io_guard()
        self.io.active = True
        root = logging.getLogger()
        root.addHandler(self)
        root.setLevel(logging.DEBUG)
        logging.getLogger(__name__).debug("child audit positive control")

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(self.format(record))


class ScriptedBackend:
    def __init__(
        self, script: dict[str, object], request: BackendLoadRequest, audit: _ChildAudit | None
    ) -> None:
        self._script = script
        self._labels = request.manifest.identity.labels
        self._device = request.device
        self._active = 0
        self._lock = threading.Lock()
        self._audit = audit

    def extract(self, audio: CanonicalAudio, *, deadline_monotonic_s: float) -> object:
        if (
            audio.sample_rate != CANONICAL_RATE_HZ
            or audio.samples.ndim != 1
            or audio.samples.dtype != np.float32
        ):
            raise RuntimeError("scripted backend received non-canonical audio")
        self._elapse("extract_elapsed_s")
        record = self._script.get("record")
        if isinstance(record, str):
            CALLS.setdefault(record, []).append(
                ScriptedCall(audio.samples.copy(), audio.sample_rate)
            )
        return audio

    def score(
        self,
        features: object,
        *,
        language: LanguageHint | None,
        deadline_monotonic_s: float,
    ) -> BackendScores:
        audio = cast(CanonicalAudio, features)
        with self._lock:
            self._active += 1
            reentered = self._active > 1
        try:
            if reentered and self._script.get("exclusive"):
                raise RuntimeError("scripted backend entered twice")
            return self._score(audio, language)
        finally:
            with self._lock:
                self._active -= 1

    def close(self) -> None:
        return None

    def _score(self, audio: CanonicalAudio, language: LanguageHint | None) -> BackendScores:
        script = self._script
        duration_s = audio.samples.size / CANONICAL_RATE_HZ
        gate_name = script.get("gate")
        if isinstance(gate_name, str):
            held = gate(gate_name)
            held.entries += 1
            held.entered.set()
            held.release.wait(timeout=30.0)
        awaited = script.get("await_gate")
        if isinstance(awaited, str):
            gate(awaited).entered.wait(timeout=30.0)
        hang_from = script.get("hang_if_duration_at_least_s")
        if isinstance(hang_from, int | float) and duration_s >= hang_from:
            threading.Event().wait(timeout=15.0)
        self._elapse("score_elapsed_s")
        if language is not None and language == script.get("raise_language"):
            raise RuntimeError(f"scripted failure {SECRET}")
        self._raise_if_scripted()
        self._probe()
        labels = self._labels_for(audio, language)
        return BackendScores(labels=labels, frames=self._frames(labels, duration_s))

    def _raise_if_scripted(self) -> None:
        failure = self._script.get("raise")
        if failure == "runtime":
            raise RuntimeError(f"scripted failure {SECRET}")
        if failure == "keyboard":
            raise KeyboardInterrupt
        if failure == "memory":
            raise MemoryError
        if self._script.get("die"):
            os._exit(70)

    def _probe(self) -> None:
        probe = self._script.get("probe")
        if not isinstance(probe, dict):
            return
        expected = cast(dict[str, object], probe)
        if self._device != "cpu":
            raise RuntimeError("probe: device")
        if "nice" in expected and os.nice(0) != expected["nice"]:
            raise RuntimeError("probe: nice")
        limit = expected.get("max_threads")
        if isinstance(limit, int) and any(
            cast(int, pool["num_threads"]) > limit for pool in threadpoolctl.threadpool_info()
        ):
            raise RuntimeError("probe: threads")
        if expected.get("gpu_hidden") and os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise RuntimeError("probe: gpu")
        if expected.get("loads_once") and LOADS[str(self._script.get("name", ""))] != 1:
            raise RuntimeError("probe: load count")
        if self._audit is not None:
            audit = self._audit
            if audit.io.writes or audit.io.network:
                raise RuntimeError("probe: child I/O")
            directory = expected.get("tmpdir")
            if isinstance(directory, str) and (
                os.environ.get("TMPDIR") != directory or list(Path(directory).iterdir())
            ):
                raise RuntimeError("probe: child temp directory")
            sentinels = expected.get("no_log")
            if isinstance(sentinels, list) and (
                "child audit positive control" not in audit.messages
                or any(str(value) in message for value in sentinels for message in audit.messages)
            ):
                raise RuntimeError("probe: child logs")

    def _labels_for(self, audio: CanonicalAudio, language: LanguageHint | None) -> dict[str, float]:
        mode = self._script.get("mode", "fixed")
        if mode == "centroid":
            spectrum = np.abs(np.fft.rfft(audio.samples.astype(np.float64)))
            freqs = np.fft.rfftfreq(audio.samples.size, d=1.0 / CANONICAL_RATE_HZ)
            power = np.square(spectrum)
            centroid = float(np.sum(freqs * power) / max(float(np.sum(power)), 1e-12))
            high = 1.0 / (1.0 + math.exp(-(centroid - 1000.0) / 500.0))
            return {"high": high, "low": 1.0 - high}
        if mode == "language":
            chosen = language or "none"
            return {label: 1.0 if label == chosen else 0.0 for label in self._labels}
        fixed = self._script.get("labels")
        if isinstance(fixed, dict):
            return dict(cast(Mapping[str, float], fixed))
        return {label: 1.0 / len(self._labels) for label in self._labels}

    def _frames(self, labels: dict[str, float], duration_s: float) -> tuple[BackendFrame, ...]:
        frame_labels = self._script.get("frame_labels")
        if isinstance(frame_labels, dict):
            labels = dict(cast(Mapping[str, float], frame_labels))
        frames = self._script.get("frames")
        if frames == "gap":
            return (
                BackendFrame(0.0, duration_s / 3, labels),
                BackendFrame(duration_s / 2, duration_s, labels),
            )
        if frames == "outside":
            return (BackendFrame(0.0, duration_s + 1.0, labels),)
        return (BackendFrame(0.0, duration_s, labels),)

    def _elapse(self, key: str) -> None:
        value = self._script.get(key)
        if isinstance(value, (int, float)):
            advance_time(value)
