"""Bounded, offline evidence for training-readiness backend smoke checks."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import overload

import numpy as np
import pytest
import soundfile as sf
from numpy.typing import NDArray

from ser._internal.config.schema import (  # noqa: TID251
    AudioReadConfig,
    MediumRuntimeConfig,
    RuntimeFlags,
)
from ser._internal.data.manifest import Utterance  # noqa: TID251
from ser._internal.models import training_orchestration  # noqa: TID251
from ser._internal.models.medium_noise_controls import (  # noqa: TID251
    apply_medium_noise_controls,
)
from ser._internal.models.training_readiness import (  # noqa: TID251
    TrainingOperation,
    select_smoke_samples,
)
from ser._internal.pool import mean_std_pool, temporal_pooling_windows  # noqa: TID251
from ser._internal.repr import EncodedSequence, PoolingWindow  # noqa: TID251
from ser.config import AppConfig


class _CountingSequence(Sequence[Utterance]):
    """Sequence that records full traversals without relying on timing."""

    def __init__(self, rows: list[Utterance]) -> None:
        self._rows = rows
        self.iterations = 0
        self.visits = 0

    def __len__(self) -> int:
        return len(self._rows)

    @overload
    def __getitem__(self, index: int) -> Utterance: ...

    @overload
    def __getitem__(self, index: slice) -> list[Utterance]: ...

    def __getitem__(self, index: int | slice) -> Utterance | list[Utterance]:
        return self._rows[index]

    def __iter__(self) -> Iterator[Utterance]:
        self.iterations += 1
        for row in self._rows:
            self.visits += 1
            yield row


class _StubBackend:
    """Small deterministic sequence backend with observable inference inputs."""

    backend_id = "stub-medium"
    feature_dim = 3

    def __init__(self) -> None:
        self.encode_calls = 0
        self.decoded_inputs: list[NDArray[np.float32]] = []
        self.outputs: list[EncodedSequence] = []

    def encode_sequence(
        self,
        audio: NDArray[np.float32],
        sample_rate: int,
    ) -> EncodedSequence:
        self.encode_calls += 1
        decoded = np.asarray(audio, dtype=np.float32)
        self.decoded_inputs.append(decoded.copy())
        chunks = np.array_split(decoded, 5)
        embeddings = np.asarray(
            [
                (float(chunk.mean()), float(chunk.std()), float(np.max(np.abs(chunk))))
                for chunk in chunks
            ],
            dtype=np.float32,
        )
        frame_duration = (decoded.size / sample_rate) / len(chunks)
        starts = np.arange(len(chunks), dtype=np.float64) * frame_duration
        encoded = EncodedSequence(
            embeddings=embeddings,
            frame_start_seconds=starts,
            frame_end_seconds=starts + frame_duration,
            backend_id=self.backend_id,
        )
        self.outputs.append(encoded)
        return encoded

    def pool(
        self,
        encoded: EncodedSequence,
        windows: Sequence[PoolingWindow],
    ) -> NDArray[np.float64]:
        return mean_std_pool(encoded, windows)


def _write_tiny_audio(path: Path, *, phase: float) -> None:
    samples = np.arange(800, dtype=np.float32)
    signal = 2.0 * np.sin((samples / 31.0) + phase)
    stereo = np.column_stack((signal, signal * 0.25))
    subtype = "FLOAT" if path.suffix == ".wav" else "PCM_16"
    sf.write(path, stereo, 8_000, subtype=subtype)


def _inventory(paths: dict[tuple[str, str], Path], size: int) -> list[Utterance]:
    combinations = (("en", ".wav"), ("en", ".flac"), ("es", ".wav"), ("es", ".flac"))
    return [
        Utterance(
            schema_version=1,
            sample_id=f"sample-{index:05d}",
            corpus="fixture",
            audio_path=paths[combinations[index % len(combinations)]],
            label="calm" if index % 2 == 0 else "happy",
            language=combinations[index % len(combinations)][0],
        )
        for index in range(size)
    ]


def test_readiness_selection_is_single_pass_capped_and_covers_format_language(
    tmp_path: Path,
) -> None:
    paths = {
        (language, suffix): tmp_path / f"{language}-{suffix.removeprefix('.')}" f"{suffix}"
        for language in ("en", "es")
        for suffix in (".wav", ".flac")
    }
    tracked = _CountingSequence(_inventory(paths, 4_096))

    selected = select_smoke_samples(tracked, cap=4)

    assert tracked.iterations == 1
    assert tracked.visits == len(tracked)
    assert len(selected) == 4
    assert {(row.language, row.audio_path.suffix) for row in selected} == {
        ("en", ".wav"),
        ("en", ".flac"),
        ("es", ".wav"),
        ("es", ".flac"),
    }
    assert [row.sample_id for row in selected] == [
        row.sample_id for row in select_smoke_samples(list(reversed(tracked)), cap=4)
    ]


def test_stub_backend_smoke_decodes_normalizes_and_round_trips_isolated_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = {
        (language, suffix): tmp_path / f"{language}-{suffix.removeprefix('.')}{suffix}"
        for language in ("en", "es")
        for suffix in (".wav", ".flac")
    }
    for phase, path in enumerate(paths.values()):
        _write_tiny_audio(path, phase=float(phase))
    samples = tuple(select_smoke_samples(_inventory(paths, 20_000), cap=4))
    backend = _StubBackend()
    settings = AppConfig(
        emotions={"01": "calm", "02": "happy"},
        tmp_folder=tmp_path / "tmp",
        runtime_flags=RuntimeFlags(profile_pipeline=True, medium_profile=True),
        audio_read=AudioReadConfig(max_retries=1, retry_delay_seconds=0.0),
        medium_runtime=MediumRuntimeConfig(
            pool_window_size_seconds=0.04,
            pool_window_stride_seconds=0.02,
        ),
    )
    monkeypatch.setattr(
        training_orchestration,
        "_profile_smoke_runtime",
        lambda _settings: (
            backend.backend_id,
            "stub/model@revision",
            "cpu",
            "float32",
            backend,
        ),
    )
    probe_cache = tmp_path / "isolated-smoke-cache"

    with training_orchestration.training_operation_scope(TrainingOperation()):
        first = training_orchestration.run_selected_backend_smoke(
            settings=settings,
            samples=samples,
            probe_cache_dir=probe_cache,
        )
    with training_orchestration.training_operation_scope(TrainingOperation()):
        second = training_orchestration.run_selected_backend_smoke(
            settings=settings,
            samples=samples,
            probe_cache_dir=probe_cache,
        )

    assert backend.encode_calls == 4
    assert all(row.ndim == 1 and row.dtype == np.float32 for row in backend.decoded_inputs)
    assert all(
        np.all(np.isfinite(row)) and np.max(np.abs(row)) <= 1.0 for row in backend.decoded_inputs
    )
    assert all(output.embeddings.shape == (5, backend.feature_dim) for output in backend.outputs)
    assert all(np.all(np.isfinite(output.embeddings)) for output in backend.outputs)
    assert {output.backend_id for output in backend.outputs} == {backend.backend_id}
    assert first == second
    assert first.attempted == first.succeeded == len(samples) == 4
    assert first.feature_dim == backend.feature_dim * 2
    assert first.cache_round_trip is True
    assert (first.backend_id, first.model_id, first.device, first.dtype) == (
        backend.backend_id,
        "stub/model@revision",
        "cpu",
        "float32",
    )
    cache_paths = list((probe_cache / "embeddings").rglob("*.npz"))
    assert len(cache_paths) == 4
    with np.load(cache_paths[0], allow_pickle=False) as payload:
        assert str(np.asarray(payload["backend_id"]).item()) == backend.backend_id


def test_temporal_pooling_and_noise_controls_preserve_final_feature_dimension() -> None:
    encoded = EncodedSequence(
        embeddings=np.asarray(
            [[0.0, 0.0], [0.0, 0.0], [1.0, -1.0], [2.0, -2.0], [3.0, -3.0]],
            dtype=np.float32,
        ),
        frame_start_seconds=np.arange(5, dtype=np.float64) * 0.1,
        frame_end_seconds=(np.arange(5, dtype=np.float64) + 1.0) * 0.1,
        backend_id="stub-medium",
    )
    windows = temporal_pooling_windows(
        encoded,
        window_size_seconds=0.2,
        window_stride_seconds=0.1,
    )
    pooled = mean_std_pool(encoded, windows)
    controlled, stats = apply_medium_noise_controls(
        pooled,
        min_window_std=0.05,
        max_windows_per_clip=2,
    )

    assert pooled.shape == (4, 4)
    assert controlled.shape == (2, 4)
    assert np.all(np.isfinite(controlled))
    assert stats.total_windows == 4
    assert stats.kept_windows == 2
    assert stats.dropped_low_std_windows >= 1
    assert stats.dropped_cap_windows >= 1
