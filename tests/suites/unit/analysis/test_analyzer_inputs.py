"""Input forms: WAV path or mono float32 samples plus rate, and an optional language."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from tests.suites.unit.analysis._contract import (
    NOT_IMPLEMENTED,
    assert_valid_record,
    load_fast,
    without_timing,
)
from tests.suites.unit.analysis._stand_ins import CANONICAL_RATE_HZ, tone, write_wav

from ser._internal.analysis import AnalysisUsageError


@NOT_IMPLEMENTED
def test_wav_path_may_be_str_or_pathlike(tmp_path: Path, fast_model_path: Path) -> None:
    wav = write_wav(tmp_path / "clip.wav", tone(2.0))
    analyzer = load_fast(fast_model_path)

    from_path = analyzer.analyze(wav)
    from_str = analyzer.analyze(str(wav))

    assert from_path["status"] == "ok"
    assert_valid_record(from_path, analyzer.model_identity)
    assert without_timing(from_str) == without_timing(from_path)


@NOT_IMPLEMENTED
def test_samples_and_float_wav_of_the_same_audio_give_the_same_record(
    tmp_path: Path, fast_model_path: Path
) -> None:
    samples = tone(2.0)
    wav = write_wav(tmp_path / "clip.wav", samples, subtype="FLOAT")
    analyzer = load_fast(fast_model_path)

    from_samples = analyzer.analyze(samples, sample_rate_hz=CANONICAL_RATE_HZ)
    from_file = analyzer.analyze(wav)

    assert from_samples["status"] == "ok"
    assert without_timing(from_samples) == without_timing(from_file)


@NOT_IMPLEMENTED
@pytest.mark.parametrize("rate", [8_000, 22_050, 44_100, 48_000])
def test_any_input_rate_is_converted_and_reported(
    tmp_path: Path, fast_model_path: Path, rate: int
) -> None:
    samples = tone(2.0, sample_rate_hz=rate)
    wav = write_wav(tmp_path / "clip.wav", samples, sample_rate_hz=rate)
    analyzer = load_fast(fast_model_path)

    for record in (analyzer.analyze(samples, sample_rate_hz=rate), analyzer.analyze(wav)):
        assert record["status"] == "ok"
        assert_valid_record(record, analyzer.model_identity)
        assert record["audio"]["input_sample_rate_hz"] == rate
        assert record["audio"]["duration_s"] == pytest.approx(2.0, abs=1e-3)


@NOT_IMPLEMENTED
def test_multichannel_wav_is_mixed_down(tmp_path: Path, fast_model_path: Path) -> None:
    mono = tone(2.0)
    mono_wav = write_wav(tmp_path / "mono.wav", mono)
    stereo_wav = write_wav(tmp_path / "stereo.wav", np.stack([mono, mono], axis=1))
    analyzer = load_fast(fast_model_path)

    assert without_timing(analyzer.analyze(stereo_wav)) == without_timing(
        analyzer.analyze(mono_wav)
    )


@NOT_IMPLEMENTED
def test_integer_pcm_wav_is_accepted(tmp_path: Path, fast_model_path: Path) -> None:
    wav = write_wav(tmp_path / "clip.wav", tone(2.0), subtype="PCM_16")
    analyzer = load_fast(fast_model_path)

    assert analyzer.analyze(wav)["status"] == "ok"


@NOT_IMPLEMENTED
@pytest.mark.parametrize("language", [None, "en", "es", "pt"])
def test_language_hint_is_optional(fast_model_path: Path, language: Any) -> None:
    analyzer = load_fast(fast_model_path)

    record = analyzer.analyze(tone(2.0), sample_rate_hz=CANONICAL_RATE_HZ, language=language)

    assert record["status"] == "ok"


@NOT_IMPLEMENTED
def test_unknown_language_hint_is_a_usage_error(fast_model_path: Path) -> None:
    analyzer: Any = load_fast(fast_model_path)

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
    "path-with-rate": lambda a, wav: a.analyze(wav, sample_rate_hz=16_000),
}


@NOT_IMPLEMENTED
@pytest.mark.parametrize("misuse", list(_MISUSES.values()), ids=list(_MISUSES))
def test_malformed_arguments_are_usage_errors(
    tmp_path: Path, fast_model_path: Path, misuse: Misuse
) -> None:
    wav = write_wav(tmp_path / "clip.wav", tone(1.0))
    analyzer = load_fast(fast_model_path)

    with pytest.raises(AnalysisUsageError):
        misuse(analyzer, wav)
