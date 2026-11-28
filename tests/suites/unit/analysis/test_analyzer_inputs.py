"""Input forms, and the one canonical conversion every input goes through."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from tests.suites.unit.analysis._contract import (
    INLINE,
    NOT_IMPLEMENTED,
    assert_valid_record,
    load_artifact,
    load_baseline,
    without_timing,
)
from tests.suites.unit.analysis._scripted_backend import CALLS, CENTROID_LABELS, LANGUAGE_LABELS
from tests.suites.unit.analysis._stand_ins import CANONICAL_RATE_HZ, sine, tone, write_wav
from tests.suites.unit.analysis._synthetic_artifacts import write_artifact

from ser._internal.utils.audio_utils import CanonicalAudio
from ser.api import AnalysisConfig, AnalysisUsageError, LanguageHint, UtteranceAnalyzer


def _centroid_model(tmp_path: Path, config: AnalysisConfig | None = None) -> UtteranceAnalyzer:
    path = write_artifact(
        tmp_path / "centroid.ser", script={"mode": "centroid"}, labels=CENTROID_LABELS
    )
    return load_artifact(path, config)


@NOT_IMPLEMENTED
def test_wav_path_may_be_str_or_pathlike(tmp_path: Path, fast_model_path: Path) -> None:
    wav = write_wav(tmp_path / "clip.wav", tone(2.0))
    analyzer = load_baseline(fast_model_path)

    from_path = analyzer.analyze(wav)
    from_str = analyzer.analyze(str(wav))

    assert from_path["status"] == "ok"
    assert_valid_record(from_path, analyzer.model_manifest)
    assert without_timing(from_str) == without_timing(from_path)


@NOT_IMPLEMENTED
def test_samples_and_float_wav_of_the_same_audio_give_the_same_record(
    tmp_path: Path, fast_model_path: Path
) -> None:
    samples = tone(2.0)
    wav = write_wav(tmp_path / "clip.wav", samples, subtype="FLOAT")
    analyzer = load_baseline(fast_model_path)

    from_samples = analyzer.analyze(samples, sample_rate_hz=CANONICAL_RATE_HZ)
    from_file = analyzer.analyze(wav)

    assert from_samples["status"] == "ok"
    assert without_timing(from_samples) == without_timing(from_file)


@pytest.mark.usefixtures("scripted")
@pytest.mark.parametrize("rate", [8_000, 44_100, 48_000])
def test_every_input_rate_is_resampled_to_the_same_scores(tmp_path: Path, rate: int) -> None:
    analyzer = _centroid_model(tmp_path)
    reference = analyzer.analyze(sine(900.0, 2.0), sample_rate_hz=CANONICAL_RATE_HZ)
    samples = sine(900.0, 2.0, sample_rate_hz=rate)
    wav = write_wav(tmp_path / "clip.wav", samples, sample_rate_hz=rate)

    for record in (analyzer.analyze(samples, sample_rate_hz=rate), analyzer.analyze(wav)):
        assert record["status"] == "ok"
        assert record["labels"] is not None and reference["labels"] is not None
        for label in CENTROID_LABELS:
            assert record["labels"][label] == pytest.approx(reference["labels"][label], abs=0.05)
        assert record["top_label"] == reference["top_label"]
        assert record["audio"]["input_sample_rate_hz"] == rate
        assert record["audio"]["duration_s"] == pytest.approx(2.0, abs=1e-3)


@pytest.mark.usefixtures("scripted")
def test_stereo_is_mixed_by_the_canonical_conversion_not_by_dropping_a_channel(
    tmp_path: Path,
) -> None:
    left, right = sine(600.0, 2.0), sine(1_800.0, 2.0)
    stereo = write_wav(tmp_path / "stereo.wav", np.stack([left, right], axis=1))
    mixed = write_wav(tmp_path / "mixed.wav", ((left + right) / 2).astype(np.float32))
    left_only = write_wav(tmp_path / "left.wav", left)
    analyzer = _centroid_model(tmp_path)

    from_stereo, from_mix, from_left = (analyzer.analyze(p) for p in (stereo, mixed, left_only))

    assert from_stereo["labels"] is not None and from_mix["labels"] is not None
    assert from_left["labels"] is not None
    assert from_stereo["labels"]["high"] == pytest.approx(from_mix["labels"]["high"], abs=1e-3)
    assert abs(from_stereo["labels"]["high"] - from_left["labels"]["high"]) > 0.1


@pytest.mark.usefixtures("scripted")
def test_native_channels_and_rate_reach_the_shared_seam(
    tmp_path: Path, canonical_calls: list[tuple[object, ...]]
) -> None:
    stereo = np.stack([sine(600.0, 1.0, sample_rate_hz=44_100)] * 2, axis=1)
    wav = write_wav(tmp_path / "stereo.wav", stereo, sample_rate_hz=44_100)
    analyzer = _centroid_model(tmp_path, INLINE)

    assert analyzer.analyze(wav)["status"] == "ok"

    assert len(canonical_calls) == 1
    samples, rate, channel_axis, result = canonical_calls[0]
    assert isinstance(samples, np.ndarray) and samples.shape == (44_100, 2)
    assert (rate, channel_axis) == (44_100, 1)
    assert isinstance(result, CanonicalAudio)
    assert result.sample_rate == CANONICAL_RATE_HZ and result.samples.ndim == 1


_RATE_44K = 44_100
_OUT_OF_RANGE = {
    "empty": (0, "too_short"),
    "below-min": (round(0.3 * _RATE_44K) - 1, "too_short"),
    "above-max": (120 * _RATE_44K + 1, "too_long"),
}


@pytest.mark.usefixtures("scripted")
@pytest.mark.parametrize(
    ("count", "expected"), list(_OUT_OF_RANGE.values()), ids=list(_OUT_OF_RANGE)
)
def test_out_of_range_sources_are_classified_before_conversion(
    tmp_path: Path, canonical_calls: list[tuple[object, ...]], count: int, expected: str
) -> None:
    analyzer = _centroid_model(tmp_path, INLINE)

    record = analyzer.analyze(np.zeros(count, dtype=np.float32), sample_rate_hz=_RATE_44K)

    assert record["status"] == expected
    assert record["audio"]["duration_s"] == pytest.approx(count / _RATE_44K, abs=1e-9)
    assert record["audio"]["input_sample_rate_hz"] == _RATE_44K
    assert canonical_calls == []


@pytest.mark.usefixtures("scripted")
@pytest.mark.parametrize("as_file", [False, True], ids=["array", "pcm16"])
def test_source_and_canonical_samples_reach_their_respective_boundaries(
    tmp_path: Path, canonical_calls: list[tuple[object, ...]], as_file: bool
) -> None:
    rate = 8_000
    pcm = np.tile(np.array([1_024, -2_048, 4_096, -8_192], dtype=np.int16), 1_000)
    samples = pcm.astype(np.float32) / np.float32(32_768)
    path = write_artifact(tmp_path / "model.ser", script={"record": "samples"})
    with load_artifact(path, INLINE) as analyzer:
        if as_file:
            wav = write_wav(tmp_path / "clip.wav", samples, sample_rate_hz=rate, subtype="PCM_16")
            record = analyzer.analyze(wav)
        else:
            record = analyzer.analyze(samples, sample_rate_hz=rate)

    assert record["status"] == "ok"
    [(source, source_rate, axis, canonical)] = canonical_calls
    assert isinstance(source, np.ndarray)
    if source.dtype == np.int16:
        np.testing.assert_array_equal(source.reshape(-1), pcm)
    else:
        np.testing.assert_array_equal(source.reshape(-1), samples)
    assert (source_rate, axis) == (rate, 1)
    assert isinstance(canonical, CanonicalAudio)
    [received] = CALLS["samples"]
    assert received.sample_rate == CANONICAL_RATE_HZ
    np.testing.assert_array_equal(received.samples, canonical.samples)


@pytest.mark.usefixtures("scripted")
def test_opposite_stereo_channels_cancel_to_silence(tmp_path: Path) -> None:
    samples = sine(600.0, 1.0)
    stereo = write_wav(tmp_path / "stereo.wav", np.stack([samples, -samples], axis=1))
    with _centroid_model(tmp_path, INLINE) as analyzer:
        assert analyzer.analyze(stereo)["status"] == "silent"


@NOT_IMPLEMENTED
def test_integer_pcm_wav_is_accepted(tmp_path: Path, fast_model_path: Path) -> None:
    wav = write_wav(tmp_path / "clip.wav", tone(2.0), subtype="PCM_16")
    analyzer = load_baseline(fast_model_path)

    assert analyzer.analyze(wav)["status"] == "ok"


@pytest.mark.usefixtures("scripted")
@pytest.mark.parametrize("language", [None, "en", "es", "pt"])
def test_the_language_hint_reaches_the_backend(
    tmp_path: Path, language: LanguageHint | None
) -> None:
    path = write_artifact(
        tmp_path / "lang.ser", script={"mode": "language"}, labels=LANGUAGE_LABELS
    )
    analyzer = load_artifact(path)

    record = analyzer.analyze(tone(1.0), sample_rate_hz=CANONICAL_RATE_HZ, language=language)

    assert record["top_label"] == (language or "none")


@NOT_IMPLEMENTED
def test_unknown_language_hint_is_a_usage_error(fast_model_path: Path) -> None:
    analyzer: Any = load_baseline(fast_model_path)

    with pytest.raises(AnalysisUsageError):
        analyzer.analyze(tone(2.0), sample_rate_hz=CANONICAL_RATE_HZ, language="fr")


# Typed loosely so that calls the signature rejects can still be made.
type Misuse = Callable[[Any, Path], object]

_MISUSES: dict[str, Misuse] = {
    "two-dimensional": lambda a, _: a.analyze(
        np.stack([tone(1.0), tone(1.0)], axis=1), sample_rate_hz=CANONICAL_RATE_HZ
    ),
    "int16": lambda a, _: a.analyze(
        (tone(1.0) * 32767).astype(np.int16), sample_rate_hz=CANONICAL_RATE_HZ
    ),
    "float64": lambda a, _: a.analyze(
        tone(1.0).astype(np.float64), sample_rate_hz=CANONICAL_RATE_HZ
    ),
    "list": lambda a, _: a.analyze(tone(1.0).tolist(), sample_rate_hz=CANONICAL_RATE_HZ),
    "missing-rate": lambda a, _: a.analyze(tone(1.0)),
    "zero-rate": lambda a, _: a.analyze(tone(1.0), sample_rate_hz=0),
    "negative-rate": lambda a, _: a.analyze(tone(1.0), sample_rate_hz=-16_000),
    "float-rate": lambda a, _: a.analyze(tone(1.0), sample_rate_hz=16_000.0),
    "bool-rate": lambda a, _: a.analyze(tone(1.0), sample_rate_hz=True),
    "path-with-rate": lambda a, wav: a.analyze(wav, sample_rate_hz=16_000),
}


@NOT_IMPLEMENTED
@pytest.mark.parametrize("misuse", list(_MISUSES.values()), ids=list(_MISUSES))
def test_malformed_arguments_are_usage_errors(
    tmp_path: Path, fast_model_path: Path, misuse: Misuse
) -> None:
    wav = write_wav(tmp_path / "clip.wav", tone(1.0))
    analyzer = load_baseline(fast_model_path)

    with pytest.raises(AnalysisUsageError):
        misuse(analyzer, wav)
