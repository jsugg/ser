"""Behavior tests for audio normalization and decoding guards."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ser.utils import audio_utils as au

pytestmark = pytest.mark.filterwarnings(
    "ignore:path is deprecated.*:DeprecationWarning"
)


def test_prepare_audio_buffer_converts_stereo_to_mono() -> None:
    """Stereo buffers should be down-mixed before normalization."""
    stereo = np.asarray([[0.2, 0.0], [0.4, 0.0]], dtype=np.float32)

    prepared = au._prepare_audio_buffer(stereo)

    assert prepared.ndim == 1
    assert prepared.tolist() == pytest.approx([0.5, 1.0])


def test_prepare_audio_buffer_rejects_empty_audio() -> None:
    """Empty decoded buffers should fail fast with a clear error."""
    with pytest.raises(OSError, match="contains no samples"):
        au._prepare_audio_buffer(np.asarray([], dtype=np.float32))


def test_read_audio_file_uses_soundfile_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When librosa fails, soundfile should be used on the same retry cycle."""
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"fake-audio")

    monkeypatch.setattr(
        au.librosa,
        "load",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("decode failure")),
    )

    class DummySoundFile:
        """Minimal context manager for soundfile fallback tests."""

        samplerate = 16000

        def __init__(self, _path: str) -> None:
            pass

        def __enter__(self) -> "DummySoundFile":
            return self

        def __exit__(
            self,
            _exc_type: type[BaseException] | None,
            _exc: BaseException | None,
            _tb: object,
        ) -> None:
            return None

        def read(self, dtype: str = "float32") -> np.ndarray:
            assert dtype == "float32"
            return np.asarray([[0.2, 0.0], [0.4, 0.0]], dtype=np.float32)

    monkeypatch.setattr(au.sf, "SoundFile", DummySoundFile)
    monkeypatch.setattr(
        au,
        "get_settings",
        lambda: SimpleNamespace(
            audio_read=SimpleNamespace(max_retries=2, retry_delay_seconds=0.0)
        ),
    )

    audio, sample_rate = au.read_audio_file(str(audio_path))

    assert sample_rate == 16000
    assert audio.tolist() == pytest.approx([0.5, 1.0])


def test_read_audio_file_rejects_invalid_segment_bounds(tmp_path: Path) -> None:
    """Segment arguments should be validated before decode attempts."""
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"fake-audio")

    with pytest.raises(ValueError, match="start_seconds"):
        au.read_audio_file(str(audio_path), start_seconds=-0.1)
    with pytest.raises(ValueError, match="duration_seconds"):
        au.read_audio_file(str(audio_path), duration_seconds=0.0)


def test_read_audio_file_segment_uses_librosa_offset_duration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Segment reads should pass offset/duration to librosa."""
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"fake-audio")
    captured: dict[str, object] = {}

    def _fake_librosa_load(
        path: str, *, sr: None, offset: float, duration: float
    ) -> tuple[np.ndarray, int]:
        captured["path"] = path
        captured["sr"] = sr
        captured["offset"] = offset
        captured["duration"] = duration
        return np.asarray([0.2, 0.4], dtype=np.float32), 22050

    monkeypatch.setattr(au.librosa, "load", _fake_librosa_load)
    monkeypatch.setattr(
        au,
        "get_settings",
        lambda: SimpleNamespace(
            audio_read=SimpleNamespace(max_retries=1, retry_delay_seconds=0.0)
        ),
    )

    audio, sample_rate = au.read_audio_file(
        str(audio_path),
        start_seconds=1.5,
        duration_seconds=0.75,
    )

    assert sample_rate == 22050
    assert audio.tolist() == pytest.approx([0.5, 1.0])
    assert captured["path"] == str(audio_path)
    assert captured["sr"] is None
    assert captured["offset"] == pytest.approx(1.5)
    assert captured["duration"] == pytest.approx(0.75)
