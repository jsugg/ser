"""The ``ser.utterance/v1`` JSON Schema: example records, rejections, and type parity."""

from __future__ import annotations

import typing
from collections.abc import Callable

import pytest
from tests.suites.unit.analysis._json_schema_subset import unsupported_keywords, validate

from ser._internal.analysis import utterance_record_json_schema
from ser.runtime.schema import (
    UTTERANCE_SCHEMA_VERSION,
    UtteranceAudio,
    UtteranceFrame,
    UtteranceRecord,
    UtteranceStatus,
    UtteranceTiming,
)

_LABELS = {
    "angry": 0.07,
    "calm": 0.11,
    "disgust": 0.02,
    "fearful": 0.05,
    "happy": 0.09,
    "neutral": 0.41,
    "sad": 0.21,
    "surprised": 0.04,
}


def _ok_record() -> dict[str, object]:
    """The example of the integration contract, with placeholders filled in."""
    return {
        "schema_version": "ser.utterance/v1",
        "ser_version": "1.1.0",
        "model_id": "medium-xlsr300m-head-bench1",
        "model_sha256": "0" * 64,
        "label_set_id": "ravdess8/v1",
        "labels": dict(_LABELS),
        "top_label": "neutral",
        "confidence": 0.41,
        "frames": [{"start_s": 0.0, "end_s": 3.0, "labels": dict(_LABELS)}],
        "dimensions": None,
        "audio": {"duration_s": 6.2, "input_sample_rate_hz": 16000},
        "status": "ok",
        "timing_ms": {"features": 310, "predict": 4},
    }


def _too_short_record() -> dict[str, object]:
    record = _ok_record()
    record.update(
        labels=None,
        top_label=None,
        confidence=None,
        frames=None,
        status="too_short",
        audio={"duration_s": 0.1, "input_sample_rate_hz": 16000},
        timing_ms={"features": 0, "predict": 0},
    )
    return record


def _unreadable_record() -> dict[str, object]:
    record = _too_short_record()
    record.update(status="unreadable", audio={"duration_s": None, "input_sample_rate_hz": None})
    return record


@pytest.mark.parametrize("build", [_ok_record, _too_short_record, _unreadable_record])
def test_example_records_validate(build: Callable[[], dict[str, object]]) -> None:
    assert validate(build(), utterance_record_json_schema()) == []


def _set(path: tuple[str, ...], value: object) -> Callable[[dict[str, object]], None]:
    def mutate(record: dict[str, object]) -> None:
        target: dict[str, object] = record
        for key in path[:-1]:
            target = typing.cast(dict[str, object], target[key])
        target[path[-1]] = value

    return mutate


def _delete(key: str) -> Callable[[dict[str, object]], None]:
    def mutate(record: dict[str, object]) -> None:
        del record[key]

    return mutate


@pytest.mark.parametrize(
    ("build", "mutate"),
    [
        pytest.param(_ok_record, _set(("schema_version",), "ser.utterance/v2"), id="version"),
        pytest.param(_ok_record, _set(("status",), "error"), id="unknown-status"),
        pytest.param(_ok_record, _set(("turn_id",), "t-1"), id="extra-field"),
        pytest.param(_ok_record, _delete("label_set_id"), id="missing-identity"),
        pytest.param(_ok_record, _set(("model_sha256",), "ABC"), id="bad-sha"),
        pytest.param(_ok_record, _set(("labels",), None), id="ok-without-labels"),
        pytest.param(_ok_record, _set(("frames",), []), id="ok-without-frames"),
        pytest.param(_ok_record, _set(("confidence",), 1.5), id="confidence-range"),
        pytest.param(_ok_record, _set(("labels", "sad"), -0.1), id="negative-probability"),
        pytest.param(_ok_record, _set(("dimensions",), {"valence": 0.1}), id="dimensions"),
        pytest.param(_ok_record, _set(("audio", "duration_s"), None), id="ok-no-duration"),
        pytest.param(_ok_record, _set(("timing_ms", "predict"), 4.5), id="timing-float"),
        pytest.param(_ok_record, _set(("timing_ms", "total"), 9), id="timing-extra"),
        pytest.param(_too_short_record, _set(("labels",), dict(_LABELS)), id="labels-on-error"),
        pytest.param(_too_short_record, _set(("top_label",), "sad"), id="top-label-on-error"),
        pytest.param(_too_short_record, _set(("frames",), []), id="frames-on-error"),
    ],
)
def test_schema_rejects_contract_violations(
    build: Callable[[], dict[str, object]],
    mutate: Callable[[dict[str, object]], None],
) -> None:
    record = build()
    mutate(record)
    assert validate(record, utterance_record_json_schema()) != []


def test_schema_uses_only_checked_keywords() -> None:
    assert unsupported_keywords(utterance_record_json_schema()) == set()


def _properties(schema: object, *path: str) -> dict[str, object]:
    node = typing.cast(dict[str, object], schema)
    for key in path:
        node = typing.cast(
            dict[str, object], typing.cast(dict[str, object], node["properties"])[key]
        )
    return node


def test_schema_matches_record_types() -> None:
    schema = utterance_record_json_schema()
    ok_branch = typing.cast(dict[str, object], schema["then"])
    frame_schema = typing.cast(dict[str, object], _properties(ok_branch, "frames")["items"])
    pairs: list[tuple[type, dict[str, object]]] = [
        (UtteranceRecord, schema),
        (UtteranceAudio, _properties(schema, "audio")),
        (UtteranceTiming, _properties(schema, "timing_ms")),
        (UtteranceFrame, frame_schema),
    ]
    for record_type, node in pairs:
        keys = set(typing.get_type_hints(record_type))
        assert set(typing.cast(list[str], node["required"])) == keys
        assert set(typing.cast(dict[str, object], node["properties"])) == keys
        assert node["additionalProperties"] is False

    statuses = typing.get_args(UtteranceStatus.__value__)
    assert _properties(schema, "status")["enum"] == list(statuses)
    assert _properties(schema, "schema_version")["const"] == UTTERANCE_SCHEMA_VERSION
