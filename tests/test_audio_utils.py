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
