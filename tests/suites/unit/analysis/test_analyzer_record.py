"""The ``ser.utterance/v1`` record of a successful analysis and its identity fields."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from tests.suites.unit.analysis._contract import NOT_IMPLEMENTED, assert_valid_record, load_fast
from tests.suites.unit.analysis._stand_ins import CANONICAL_RATE_HZ, LABELS, noise, tone

import ser
from ser.runtime.schema import UTTERANCE_SCHEMA_VERSION


@NOT_IMPLEMENTED
@pytest.mark.parametrize("seconds", [0.5, 3.0, 7.3])
def test_ok_record_meets_every_record_clause(fast_model_path: Path, seconds: float) -> None:
    analyzer = load_fast(fast_model_path)

    record = analyzer.analyze(tone(seconds), sample_rate_hz=CANONICAL_RATE_HZ)

    assert record["status"] == "ok"
    assert_valid_record(record, analyzer.model_identity)
    assert record["dimensions"] is None
    assert record["audio"] == {
        "duration_s": pytest.approx(seconds, abs=1e-3),
        "input_sample_rate_hz": CANONICAL_RATE_HZ,
    }


@NOT_IMPLEMENTED
def test_frames_cover_the_whole_utterance(fast_model_path: Path) -> None:
    analyzer = load_fast(fast_model_path)

    record = analyzer.analyze(noise(7.3, rms_dbfs=-25.0), sample_rate_hz=CANONICAL_RATE_HZ)

    frames = record["frames"]
    assert frames is not None and len(frames) >= 2
    assert frames[0]["start_s"] == 0.0
    assert frames[-1]["end_s"] == pytest.approx(7.3, abs=1e-3)


@NOT_IMPLEMENTED
def test_record_names_versions_and_the_exact_model_file(fast_model_path: Path) -> None:
    analyzer = load_fast(fast_model_path)

    record = analyzer.analyze(tone(2.0), sample_rate_hz=CANONICAL_RATE_HZ)

    identity = analyzer.model_identity
    assert record["schema_version"] == UTTERANCE_SCHEMA_VERSION
    assert record["ser_version"] == ser.__version__
    assert identity.profile == "fast"
    assert identity.labels == LABELS
    assert identity.model_sha256 == hashlib.sha256(fast_model_path.read_bytes()).hexdigest()
    assert record["model_sha256"] == identity.model_sha256
    assert record["model_id"] == identity.model_id
    assert record["label_set_id"] == identity.label_set_id


@NOT_IMPLEMENTED
def test_another_model_file_has_another_model_id_and_the_same_label_set(
    fast_model_path: Path, other_fast_model_path: Path
) -> None:
    first = load_fast(fast_model_path).model_identity
    second = load_fast(other_fast_model_path).model_identity

    assert second.model_id != first.model_id
    assert second.model_sha256 != first.model_sha256
    assert second.label_set_id == first.label_set_id


@NOT_IMPLEMENTED
def test_a_changed_label_set_has_another_label_set_id(
    fast_model_path: Path, seven_label_model_path: Path
) -> None:
    eight = load_fast(fast_model_path)
    seven = load_fast(seven_label_model_path)

    record = seven.analyze(tone(2.0), sample_rate_hz=CANONICAL_RATE_HZ)

    assert seven.model_identity.label_set_id != eight.model_identity.label_set_id
    assert_valid_record(record, seven.model_identity)
    assert record["labels"] is not None and set(record["labels"]) == set(LABELS[:7])
