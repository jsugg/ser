"""Characterize how the medium XLS-R encoder handles native 48 kHz audio."""

import importlib
from importlib.util import find_spec

import numpy as np
import pytest
from numpy.typing import NDArray
from tests.suites.unit.repr.test_hf_xlsr_backend import _FakeFeatureExtractor, _FakeModel

from ser._internal.repr import XLSRBackend


def test_medium_forwards_48khz_audio_to_preprocessor_unchanged() -> None:
    calls: list[tuple[int, NDArray[np.float32]]] = []

    class RecordingFeatureExtractor(_FakeFeatureExtractor):
        def __call__(
            self,
            audio: NDArray[np.float32],
            *,
            sampling_rate: int,
            return_tensors: str,
            padding: bool,
        ) -> dict[str, object]:
            calls.append((sampling_rate, audio.copy()))
            return super().__call__(
                audio, sampling_rate=sampling_rate, return_tensors=return_tensors, padding=padding
            )

    audio = np.linspace(-0.5, 0.5, 48_000, dtype=np.float32)
    model = _FakeModel(hidden_size=2)
    backend = XLSRBackend(feature_extractor=RecordingFeatureExtractor(), model=model, device="cpu")

    encoded = backend.encode_sequence(audio, sample_rate=48_000)

    [(rate, forwarded)] = calls
    assert rate == 48_000
    np.testing.assert_array_equal(forwarded, audio)
    assert model.call_sizes == [48_000]
    assert encoded.frame_end_seconds[-1] == pytest.approx(1.0)


@pytest.mark.skipif(find_spec("transformers") is None, reason="transformers is not installed")
@pytest.mark.xfail(
    strict=True,
    raises=ValueError,
    reason="XLS-R forwards native 48 kHz input to a 16 kHz Wav2Vec2 preprocessor.",
)
def test_medium_encodes_48khz_audio_with_real_preprocessor() -> None:
    transformers = importlib.import_module("transformers")
    extractor = transformers.Wav2Vec2FeatureExtractor(sampling_rate=16_000)
    backend = XLSRBackend(
        feature_extractor=extractor, model=_FakeModel(hidden_size=2), device="cpu"
    )
    audio = np.linspace(-0.5, 0.5, 48_000, dtype=np.float32)

    try:
        backend.encode_sequence(audio, sample_rate=48_000)
    except ValueError as error:
        assert "16000" in str(error) and "48000" in str(error)
        raise
