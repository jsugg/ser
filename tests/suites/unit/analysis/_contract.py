"""Shared helpers for the utterance-analysis contract tests."""

from __future__ import annotations

import json
import math
import os

import pytest
from tests.suites.unit.analysis._json_schema_subset import validate

from ser._internal.analysis import UtteranceAnalyzer, utterance_record_json_schema
from ser.runtime.contracts import AnalysisConfig, ModelIdentity
from ser.runtime.schema import UtteranceRecord

PROBABILITY_SUM_TOLERANCE = 1e-6

NOT_IMPLEMENTED = pytest.mark.xfail(
    strict=True,
    raises=NotImplementedError,
    reason="UtteranceAnalyzer is a stub until brief A1-2; remove this marker then.",
)


def load_fast(
    model_path: str | os.PathLike[str],
    config: AnalysisConfig | None = None,
) -> UtteranceAnalyzer:
    """Loads the ``fast`` analyser from an explicit model file."""
    return UtteranceAnalyzer.load(profile="fast", model_path=model_path, config=config)


def assert_valid_record(record: UtteranceRecord, identity: ModelIdentity) -> None:
    """Checks the schema plus the clauses JSON Schema cannot express."""
    assert validate(record, utterance_record_json_schema()) == []
    assert json.loads(json.dumps(record, allow_nan=False)) == record
    assert record["model_id"] == identity.model_id
    assert record["model_sha256"] == identity.model_sha256
    assert record["label_set_id"] == identity.label_set_id
    if record["status"] != "ok":
        return

    labels = record["labels"]
    frames = record["frames"]
    duration_s = record["audio"]["duration_s"]
    assert labels is not None and frames is not None and duration_s is not None
    assert set(labels) == set(identity.labels)
    assert math.isclose(sum(labels.values()), 1.0, abs_tol=PROBABILITY_SUM_TOLERANCE)
    expected_top = min(labels, key=lambda label: (-labels[label], label))
    assert record["top_label"] == expected_top
    assert record["confidence"] == labels[expected_top]

    starts = [frame["start_s"] for frame in frames]
    assert starts == sorted(starts)
    for frame in frames:
        assert 0.0 <= frame["start_s"] < frame["end_s"] <= duration_s + 1e-3
        assert set(frame["labels"]) == set(identity.labels)
        assert math.isclose(sum(frame["labels"].values()), 1.0, abs_tol=PROBABILITY_SUM_TOLERANCE)


def without_timing(record: UtteranceRecord) -> dict[str, object]:
    """The record minus ``timing_ms``, the only field excluded from determinism."""
    return {key: value for key, value in record.items() if key != "timing_ms"}
