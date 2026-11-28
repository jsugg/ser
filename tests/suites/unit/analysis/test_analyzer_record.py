"""The ``ser.utterance/v1`` record of a successful analysis: scores, frames, timing, identity."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from tests.suites.unit.analysis._contract import (
    INLINE,
    assert_valid_record,
    load_artifact,
    load_baseline,
)
from tests.suites.unit.analysis._stand_ins import (
    CANONICAL_RATE_HZ,
    WindowClassifier,
    noise,
    tone,
    write_model,
)
from tests.suites.unit.analysis._synthetic_artifacts import write_artifact

import ser
from ser._internal.analysis.backend import BackendFrame, coverage_weighted_mean, frame_layout
from ser._internal.analysis.pipeline import top_label
from ser.api import UTTERANCE_SCHEMA_VERSION, UtteranceFrame

# The fast backend's window and hop (ser/_internal/models/artifact_envelope.py).
_FAST_FRAME_S = 3.0
_FAST_STRIDE_S = 1.0


def _pooled(frames: list[UtteranceFrame], duration_s: float) -> dict[str, float]:
    """Coverage-weighted mean written out independently of the implementation."""
    edges = sorted(
        {0.0, duration_s, *(f["start_s"] for f in frames), *(f["end_s"] for f in frames)}
    )
    totals: dict[str, float] = dict.fromkeys(frames[0]["labels"], 0.0)
    for left, right in zip(edges, edges[1:], strict=False):
        covering = [f for f in frames if f["start_s"] <= left and right <= f["end_s"]]
        for frame in covering:
            for label, value in frame["labels"].items():
                totals[label] += value * (right - left) / len(covering)
    return {label: total / duration_s for label, total in totals.items()}


@pytest.mark.parametrize("seconds", [0.5, 3.0, 7.3])
def test_ok_record_meets_every_record_clause(fast_model_path: Path, seconds: float) -> None:
    analyzer = load_baseline(fast_model_path)

    record = analyzer.analyze(tone(seconds), sample_rate_hz=CANONICAL_RATE_HZ)

    assert record["status"] == "ok"
    assert_valid_record(record, analyzer.model_manifest)
    assert record["dimensions"] is None
    assert record["audio"] == {
        "duration_s": pytest.approx(seconds, abs=1e-3),
        "input_sample_rate_hz": CANONICAL_RATE_HZ,
    }


@pytest.mark.parametrize(
    ("duration_s", "expected"),
    [
        (0.5, ((0.0, 0.5),)),
        (3.0, ((0.0, 3.0),)),
        (7.3, ((0.0, 3.0), (1.0, 4.0), (2.0, 5.0), (3.0, 6.0), (4.0, 7.0), (5.0, 7.3))),
    ],
)
def test_fast_frame_layout(duration_s: float, expected: tuple[tuple[float, float], ...]) -> None:
    layout = frame_layout(duration_s, frame_size_s=_FAST_FRAME_S, stride_s=_FAST_STRIDE_S)

    assert len(layout) == len(expected)
    for span, wanted in zip(layout, expected, strict=True):
        assert span == pytest.approx(wanted)


def test_fast_records_use_the_fast_frame_layout(fast_model_path: Path) -> None:
    analyzer = load_baseline(fast_model_path)

    record = analyzer.analyze(noise(7.3, rms_dbfs=-25.0), sample_rate_hz=CANONICAL_RATE_HZ)

    assert record["frames"] is not None
    spans = tuple((frame["start_s"], frame["end_s"]) for frame in record["frames"])
    expected = frame_layout(7.3, frame_size_s=_FAST_FRAME_S, stride_s=_FAST_STRIDE_S)
    assert len(spans) == len(expected)
    for span, wanted in zip(spans, expected, strict=True):
        assert span == pytest.approx(wanted, abs=1e-3)


def test_fast_utterance_scores_are_the_coverage_weighted_mean_of_its_frames(
    fast_model_path: Path,
) -> None:
    analyzer = load_baseline(fast_model_path)

    record = analyzer.analyze(noise(7.3, rms_dbfs=-25.0), sample_rate_hz=CANONICAL_RATE_HZ)

    assert record["labels"] is not None and record["frames"] is not None
    duration_s = record["audio"]["duration_s"]
    assert duration_s is not None
    expected = _pooled(record["frames"], duration_s)
    assert record["labels"] == pytest.approx(expected, abs=1e-6)


def test_coverage_weighted_mean_weighs_every_instant_once() -> None:
    frames = (
        BackendFrame(0.0, 3.0, {"a": 1.0, "b": 0.0}),
        BackendFrame(1.0, 3.0, {"a": 0.0, "b": 1.0}),
    )

    assert coverage_weighted_mean(frames, 3.0) == pytest.approx({"a": 2 / 3, "b": 1 / 3})


def test_ties_go_to_the_alphabetically_first_label() -> None:
    assert top_label({"b": 0.4, "a": 0.4, "c": 0.2}) == ("a", 0.4)


@pytest.mark.usefixtures("scripted")
def test_backend_utterance_scores_are_the_authority_for_top_label(tmp_path: Path) -> None:
    script = {
        "labels": {"a": 0.2, "b": 0.5, "c": 0.3},
        "frame_labels": {"a": 1.0, "b": 0.0, "c": 0.0},
    }
    analyzer = load_artifact(write_artifact(tmp_path / "model.ser", script=script), INLINE)

    record = analyzer.analyze(tone(1.0), sample_rate_hz=CANONICAL_RATE_HZ)

    assert record["labels"] == pytest.approx(script["labels"])
    assert record["top_label"] == "b"
    assert record["confidence"] == pytest.approx(0.5)
    assert record["frames"] == [{"start_s": 0.0, "end_s": 1.0, "labels": script["frame_labels"]}]


@pytest.mark.usefixtures("scripted")
def test_public_record_breaks_score_ties_alphabetically(tmp_path: Path) -> None:
    path = write_artifact(tmp_path / "model.ser", labels=("b", "a"))
    with load_artifact(path, INLINE) as analyzer:
        record = analyzer.analyze(tone(1.0), sample_rate_hz=CANONICAL_RATE_HZ)

    assert record["top_label"] == "a"
    assert record["confidence"] == 0.5


@pytest.mark.usefixtures("scripted")
def test_invalid_frame_probability_rejects_valid_utterance_scores(tmp_path: Path) -> None:
    script = {"frame_labels": {"a": 1.0000005, "b": 0.0, "c": 0.0}}
    with load_artifact(write_artifact(tmp_path / "model.ser", script=script), INLINE) as analyzer:
        record = analyzer.analyze(tone(1.0), sample_rate_hz=CANONICAL_RATE_HZ)

    assert record["status"] == "model_error"
    assert_valid_record(record, analyzer.model_manifest)


@pytest.mark.usefixtures("scripted")
def test_valid_scores_are_renormalised_before_they_are_written(tmp_path: Path) -> None:
    script = {"labels": {"a": 0.2000004, "b": 0.5, "c": 0.3}}
    analyzer = load_artifact(write_artifact(tmp_path / "model.ser", script=script), INLINE)
    total = math.fsum(script["labels"].values())

    record = analyzer.analyze(tone(1.0), sample_rate_hz=CANONICAL_RATE_HZ)

    assert record["status"] == "ok"
    assert record["labels"] is not None and record["frames"] is not None
    for scores in [record["labels"], *(frame["labels"] for frame in record["frames"])]:
        assert math.fsum(scores.values()) == pytest.approx(1.0, abs=1e-12)
        for label, value in script["labels"].items():
            assert scores[label] == pytest.approx(value / total, rel=1e-12)


@pytest.mark.usefixtures("virtual_analysis_time")
def test_timing_reports_features_and_predict_separately(tmp_path: Path) -> None:
    script = {"extract_elapsed_s": 0.3, "score_elapsed_s": 0.2}
    analyzer = load_artifact(write_artifact(tmp_path / "model.ser", script=script), INLINE)

    ok = analyzer.analyze(tone(1.0), sample_rate_hz=CANONICAL_RATE_HZ)
    short = analyzer.analyze(tone(0.1), sample_rate_hz=CANONICAL_RATE_HZ)

    assert ok["timing_ms"] == {"features": 300, "predict": 200}
    assert short["status"] == "too_short"
    assert short["timing_ms"]["predict"] == 0


def test_records_of_two_models_on_the_same_audio_each_name_their_model(
    fast_model_path: Path, other_fast_model_path: Path
) -> None:
    first, second = load_baseline(fast_model_path), load_baseline(other_fast_model_path)
    samples = tone(2.0)

    one = first.analyze(samples, sample_rate_hz=CANONICAL_RATE_HZ)
    two = second.analyze(samples, sample_rate_hz=CANONICAL_RATE_HZ)

    assert_valid_record(one, first.model_manifest)
    assert_valid_record(two, second.model_manifest)
    assert one["model_sha256"] != two["model_sha256"]


def test_record_names_the_schema_and_ser_versions(fast_model_path: Path) -> None:
    analyzer = load_baseline(fast_model_path)

    record = analyzer.analyze(tone(2.0), sample_rate_hz=CANONICAL_RATE_HZ)

    assert record["schema_version"] == UTTERANCE_SCHEMA_VERSION
    assert record["ser_version"] == ser.__version__


def test_fast_public_record_pools_asymmetric_windows(
    tmp_path: Path, fast_feature_size: int
) -> None:
    model = write_model(
        tmp_path / "windows.pkl",
        feature_size=fast_feature_size,
        labels=("a", "b"),
        classifier_type=WindowClassifier,
    )
    with load_baseline(model) as analyzer:
        record = analyzer.analyze(tone(4.5), sample_rate_hz=CANONICAL_RATE_HZ)

    assert record["status"] == "ok"
    assert record["labels"] == pytest.approx({"a": 11 / 27, "b": 16 / 27})
    assert record["frames"] == [
        {"start_s": 0.0, "end_s": 3.0, "labels": {"a": 1.0, "b": 0.0}},
        {"start_s": 1.0, "end_s": 4.0, "labels": {"a": 0.0, "b": 1.0}},
        {"start_s": 2.0, "end_s": 4.5, "labels": {"a": 0.0, "b": 1.0}},
    ]
