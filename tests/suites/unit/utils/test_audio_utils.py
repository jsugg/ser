"""Behavior tests for audio normalization and decoding guards."""

from pathlib import Path

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from numpy.typing import NDArray

import ser._internal.utils.audio_utils as au
from ser.config import AudioReadConfig

pytestmark = pytest.mark.filterwarnings("ignore:path is deprecated.*:DeprecationWarning")


@pytest.mark.parametrize("rate", [8_000, 16_000, 44_100, 48_000])
def test_canonical_audio_preserves_band_limited_signal(rate: int) -> None:
    def signal(sample_rate: int) -> NDArray[np.float64]:
        time = np.arange(sample_rate) / sample_rate
        envelope = np.sin(np.pi * time) ** 2
        return np.asarray(
            envelope
            * (0.4 * np.sin(2 * np.pi * 440 * time) + 0.25 * np.sin(2 * np.pi * 3_200 * time)),
            dtype=np.float64,
        )

    canonical = au.to_canonical_audio(signal(rate), rate)

    assert canonical.sample_rate == 16_000
    assert canonical.original_sample_rate == rate
    assert canonical.duration_seconds == 1.0
    assert canonical.samples.shape == (16_000,)
    assert canonical.samples.dtype == np.float32
    np.testing.assert_allclose(canonical.samples, signal(16_000), atol=2e-5, rtol=0)


def test_canonical_audio_preserves_16khz_mono_float32_samples() -> None:
    samples = np.asarray([-0.5, -0.0, 0.25, 0.75], dtype=np.float32)

    canonical = au.to_canonical_audio(samples, 16_000)

    assert canonical.samples.dtype == np.float32
    assert canonical.samples.tobytes() == samples.tobytes()
    assert canonical.duration_seconds == 4 / 16_000


def test_canonical_audio_averages_channels_without_changing_input() -> None:
    stereo = np.asarray([[0.5, 0.0], [1.0, -0.5], [2.0, 0.0]], dtype=np.float64)
    original = stereo.copy()

    canonical = au.to_canonical_audio(stereo, 16_000)

    np.testing.assert_array_equal(canonical.samples, [0.25, 0.25, 0.5])
    np.testing.assert_array_equal(stereo, original)
    assert canonical.duration_seconds == 3 / 16_000


@pytest.mark.parametrize("scale", [2**15, 2**31])
def test_canonical_audio_scales_signed_pcm_to_full_scale(scale: int) -> None:
    dtype = np.int16 if scale == 2**15 else np.int32
    samples = np.asarray([-scale, -scale // 2, 0, scale // 2, scale - 1], dtype=dtype)

    canonical = au.to_canonical_audio(samples, 16_000)

    expected = np.asarray([-1, -0.5, 0, 0.5, (scale - 1) / scale], dtype=np.float32)
    np.testing.assert_array_equal(canonical.samples, expected)


def test_canonical_audio_centers_unsigned_pcm() -> None:
    samples = np.asarray([0, 64, 128, 192, 255], dtype=np.uint8)

    canonical = au.to_canonical_audio(samples, 16_000)

    np.testing.assert_array_equal(canonical.samples, [-1.0, -0.5, 0.0, 0.5, 127 / 128])


@pytest.mark.parametrize(
    ("samples", "rate", "match"),
    [
        (np.empty(0), 16_000, "at least one sample"),
        (np.empty((3, 0)), 16_000, "at least one sample"),
        (np.asarray([np.nan]), 16_000, "finite"),
        (np.asarray([[0.0, np.inf]]), 16_000, "finite"),
        (np.asarray([-np.inf]), 16_000, "finite"),
        (np.ones(1), 0, "sample_rate"),
        (np.ones(1), -1, "sample_rate"),
        (np.ones(1), 16_000.5, "sample_rate"),
        (np.ones((1, 1, 1)), 16_000, "shape"),
        (np.asarray(0.5), 16_000, "shape"),
        (np.asarray([1j]), 16_000, "dtype"),
    ],
)
def test_canonical_audio_rejects_invalid_input(
    samples: NDArray[np.generic], rate: int, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        au.to_canonical_audio(samples, rate)


@given(
    values=st.lists(st.floats(-2, 2, width=32), min_size=1, max_size=100),
    rate=st.integers(1_000, 192_000),
)
def test_canonical_audio_has_deterministic_length_and_range(values: list[float], rate: int) -> None:
    samples = np.asarray(values, dtype=np.float32)

    canonical = au.to_canonical_audio(samples, rate)

    assert canonical.samples.shape == ((len(values) * 16_000 + rate - 1) // rate,)
    assert canonical.duration_seconds == len(values) / rate
    assert np.all(np.abs(canonical.samples) <= 1.0)
    assert canonical.samples.tobytes() == au.to_canonical_audio(samples, rate).samples.tobytes()


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

    audio, sample_rate = au.read_audio_file(
        str(audio_path),
        audio_read_config=AudioReadConfig(max_retries=2, retry_delay_seconds=0.0),
    )

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
    audio, sample_rate = au.read_audio_file(
        str(audio_path),
        start_seconds=1.5,
        duration_seconds=0.75,
        audio_read_config=AudioReadConfig(max_retries=1, retry_delay_seconds=0.0),
    )

    assert sample_rate == 22050
    assert audio.tolist() == pytest.approx([0.5, 1.0])
    assert captured["path"] == str(audio_path)
    assert captured["sr"] is None
    assert captured["offset"] == pytest.approx(1.5)
    assert captured["duration"] == pytest.approx(0.75)


def test_read_audio_file_rejects_lfs_pointer_without_decoder_retries(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """LFS pointers should produce an actionable permanent integrity failure immediately."""
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(
        b"version https://git-lfs.github.com/spec/v1\noid sha256:0123456789abcdef\nsize 1234\n"
    )

    def _unexpected(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("decoder or retry invoked")

    monkeypatch.setattr(au.librosa, "load", _unexpected)
    monkeypatch.setattr(au.sf, "SoundFile", _unexpected)
    monkeypatch.setattr(au.time, "sleep", _unexpected)

    with pytest.raises(au.AudioIntegrityError, match="git lfs pull"):
        au.read_audio_file(
            str(audio_path),
            audio_read_config=AudioReadConfig(max_retries=3, retry_delay_seconds=1.0),
        )


def test_read_audio_file_chains_the_terminal_decoder_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Generic read errors should retain the decoder cause for diagnostics."""
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"not-audio")
    terminal_error = RuntimeError("soundfile decoder failed")
    monkeypatch.setattr(
        au.librosa,
        "load",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("librosa failed")),
    )
    monkeypatch.setattr(
        au.sf,
        "SoundFile",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(terminal_error),
    )

    with pytest.raises(OSError, match="Error reading") as exc_info:
        au.read_audio_file(
            str(audio_path),
            audio_read_config=AudioReadConfig(max_retries=1, retry_delay_seconds=0.0),
        )

    assert exc_info.value.__cause__ is terminal_error
