"""The ``ser.utterance/v1`` JSON Schema: example records, rejections, and type parity."""

from __future__ import annotations

import json
import math
import types
import typing
from collections.abc import Callable
from pathlib import Path

import pytest
from tests.suites.unit.analysis._contract import record_type_errors
from tests.suites.unit.analysis._json_schema_subset import unsupported_keywords, validate

from ser.api import (
    UTTERANCE_SCHEMA_VERSION,
    UtteranceAudio,
    UtteranceFrame,
    UtteranceRecord,
    UtteranceStatus,
    UtteranceTiming,
    utterance_record_json_schema,
)

_V1_FIXTURE = Path(__file__).parent / "fixtures" / "utterance_v1_ok.json"
_FAILURE_STATUSES = [s for s in typing.get_args(UtteranceStatus.__value__) if s != "ok"]
_IDENTITY_FIELDS = ["schema_version", "ser_version", "model_id", "model_sha256", "label_set_id"]
_LABEL_FIELDS: dict[str, object] = {
    "labels": {"neutral": 1.0},
    "top_label": "neutral",
    "confidence": 0.41,
    "frames": [{"start_s": 0.0, "end_s": 1.0, "labels": {"neutral": 1.0}}],
}


def _ok_record() -> dict[str, object]:
    """The committed v1 example: once published, it must stay valid."""
    return typing.cast(dict[str, object], json.loads(_V1_FIXTURE.read_text()))


def _failure_record(status: str) -> dict[str, object]:
    record = _ok_record()
    record.update(
        labels=None,
        top_label=None,
        confidence=None,
        frames=None,
        status=status,
        timing_ms={"features": 0, "predict": 0},
    )
    if status == "unreadable":
        record["audio"] = {"duration_s": None, "input_sample_rate_hz": None}
    return record


def test_published_v1_record_validates_against_v1() -> None:
    schema = utterance_record_json_schema("ser.utterance/v1")
    assert validate(_ok_record(), schema) == []
    assert record_type_errors(_ok_record()) == []


def test_unknown_schema_version_is_refused() -> None:
    with pytest.raises(ValueError):
        utterance_record_json_schema(typing.cast(typing.Any, "ser.utterance/v0"))


@pytest.mark.parametrize("status", _FAILURE_STATUSES)
def test_every_failure_status_validates_with_null_labels(status: str) -> None:
    assert validate(_failure_record(status), utterance_record_json_schema()) == []


@pytest.mark.parametrize("status", _FAILURE_STATUSES)
@pytest.mark.parametrize("field", list(_LABEL_FIELDS))
def test_failure_statuses_reject_every_label_field(status: str, field: str) -> None:
    record = _failure_record(status)
    record[field] = _LABEL_FIELDS[field]
    assert validate(record, utterance_record_json_schema()) != []


@pytest.mark.parametrize("field", _IDENTITY_FIELDS)
def test_every_identity_field_is_required(field: str) -> None:
    record = _ok_record()
    del record[field]
    assert validate(record, utterance_record_json_schema()) != []


def _set(path: tuple[str, ...], value: object) -> Callable[[dict[str, object]], None]:
    def mutate(record: dict[str, object]) -> None:
        target: dict[str, object] = record
        for key in path[:-1]:
            target = typing.cast(dict[str, object], target[key])
        target[path[-1]] = value

    return mutate


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(_set(("schema_version",), "ser.utterance/v2"), id="version"),
        pytest.param(_set(("status",), "error"), id="unknown-status"),
        pytest.param(_set(("turn_id",), "t-1"), id="extra-field"),
        pytest.param(_set(("model_sha256",), "ABC"), id="bad-sha"),
        pytest.param(_set(("model_id",), "has space"), id="model-id-space"),
        pytest.param(_set(("labels",), None), id="ok-without-labels"),
        pytest.param(_set(("frames",), []), id="ok-without-frames"),
        pytest.param(_set(("confidence",), 1.5), id="confidence-range"),
        pytest.param(_set(("labels", "sad"), -0.1), id="negative-probability"),
        pytest.param(_set(("dimensions",), {"valence": 0.1}), id="dimensions"),
        pytest.param(_set(("audio", "duration_s"), None), id="ok-no-duration"),
        pytest.param(_set(("timing_ms", "predict"), 4.5), id="timing-fraction"),
        pytest.param(_set(("timing_ms", "predict"), True), id="timing-bool"),
        pytest.param(_set(("timing_ms", "total"), 9), id="timing-extra"),
    ],
)
def test_schema_rejects_contract_violations(mutate: Callable[[dict[str, object]], None]) -> None:
    record = _ok_record()
    mutate(record)
    assert validate(record, utterance_record_json_schema()) != []


def test_integral_floats_are_json_integers_but_not_record_ints() -> None:
    record = _ok_record()
    record["timing_ms"] = {"features": 310.0, "predict": 4}

    assert validate(record, utterance_record_json_schema()) == []
    assert record_type_errors(record) != []


@pytest.mark.parametrize(
    ("instance", "valid"),
    [(310, True), (310.0, True), (10**400, True), (310.5, False), (True, False), (math.inf, False)],
    ids=["int", "integral-float", "huge-int", "fraction", "bool", "infinity"],
)
def test_the_integer_rule_follows_draft_2020_12(instance: object, valid: bool) -> None:
    assert (validate(instance, {"type": "integer"}) == []) is valid


def test_schema_uses_only_checked_keywords() -> None:
    assert unsupported_keywords(utterance_record_json_schema()) == set()


def _node(schema: object, *path: str) -> dict[str, object]:
    node = typing.cast(dict[str, object], schema)
    for key in path:
        node = typing.cast(
            dict[str, object], typing.cast(dict[str, object], node["properties"])[key]
        )
    return node


def _annotation_types(annotation: object) -> set[str]:
    if annotation is None or annotation is type(None):
        return {"null"}
    if isinstance(annotation, typing.TypeAliasType):
        return _annotation_types(annotation.__value__)
    origin = typing.get_origin(annotation)
    if origin in (types.UnionType, typing.Union):
        return set().union(*(_annotation_types(arg) for arg in typing.get_args(annotation)))
    if origin is typing.Literal:
        return {"string"}
    if origin is dict or typing.is_typeddict(annotation):
        return {"object"}
    if origin is list:
        return {"array"}
    return {{int: "integer", float: "number", str: "string"}[typing.cast(type, annotation)]}


def _schema_types(node: dict[str, object]) -> set[str]:
    if "type" in node:
        declared = node["type"]
        return {declared} if isinstance(declared, str) else set(typing.cast(list[str], declared))
    values = [node["const"]] if "const" in node else typing.cast(list[object], node["enum"])
    return {"string" for value in values if isinstance(value, str)}


def test_schema_matches_record_types() -> None:
    schema = utterance_record_json_schema()
    frame_schema = typing.cast(
        dict[str, object], _node(typing.cast(dict[str, object], schema["then"]), "frames")["items"]
    )
    pairs: list[tuple[type, dict[str, object]]] = [
        (UtteranceRecord, schema),
        (UtteranceAudio, _node(schema, "audio")),
        (UtteranceTiming, _node(schema, "timing_ms")),
        (UtteranceFrame, frame_schema),
    ]
    for record_type, node in pairs:
        hints = typing.get_type_hints(record_type)
        assert set(typing.cast(list[str], node["required"])) == set(hints)
        assert set(typing.cast(dict[str, object], node["properties"])) == set(hints)
        assert node["additionalProperties"] is False
        for key, hint in hints.items():
            assert _annotation_types(hint) == _schema_types(_node(node, key)), key

    assert _node(schema, "status")["enum"] == list(typing.get_args(UtteranceStatus.__value__))
    assert _node(schema, "schema_version")["const"] == UTTERANCE_SCHEMA_VERSION
