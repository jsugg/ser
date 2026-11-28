"""JSON Schema (draft 2020-12) of the ``ser.utterance/v1`` record."""

from __future__ import annotations

import copy

from ser.runtime.schema import UTTERANCE_SCHEMA_VERSION

_PROBABILITY: dict[str, object] = {"type": "number", "minimum": 0.0, "maximum": 1.0}
_LABELS: dict[str, object] = {
    "type": "object",
    "minProperties": 1,
    "propertyNames": {"minLength": 1},
    "additionalProperties": _PROBABILITY,
}
_IDENTIFIER: dict[str, object] = {"type": "string", "pattern": r"^\S+$"}

_UTTERANCE_RECORD_SCHEMA: dict[str, object] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "urn:ser:schema:ser.utterance:v1",
    "title": "ser.utterance/v1",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "schema_version",
        "ser_version",
        "model_id",
        "model_sha256",
        "label_set_id",
        "labels",
        "top_label",
        "confidence",
        "frames",
        "dimensions",
        "audio",
        "status",
        "timing_ms",
    ],
    "properties": {
        "schema_version": {"const": UTTERANCE_SCHEMA_VERSION},
        "ser_version": {"type": "string", "minLength": 1},
        "model_id": _IDENTIFIER,
        "model_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "label_set_id": _IDENTIFIER,
        "labels": {"type": ["object", "null"]},
        "top_label": {"type": ["string", "null"]},
        "confidence": {"type": ["number", "null"]},
        "frames": {"type": ["array", "null"]},
        "dimensions": {"type": "null"},
        "audio": {
            "type": "object",
            "additionalProperties": False,
            "required": ["duration_s", "input_sample_rate_hz"],
            "properties": {
                "duration_s": {"type": ["number", "null"], "minimum": 0.0},
                "input_sample_rate_hz": {"type": ["integer", "null"], "minimum": 1},
            },
        },
        "status": {
            "enum": [
                "ok",
                "too_short",
                "too_long",
                "silent",
                "unreadable",
                "timeout",
                "model_error",
            ]
        },
        "timing_ms": {
            "type": "object",
            "additionalProperties": False,
            "required": ["features", "predict"],
            "properties": {
                "features": {"type": "integer", "minimum": 0},
                "predict": {"type": "integer", "minimum": 0},
            },
        },
    },
    "if": {"properties": {"status": {"const": "ok"}}},
    "then": {
        "properties": {
            "labels": _LABELS,
            "top_label": {"type": "string", "minLength": 1},
            "confidence": _PROBABILITY,
            "frames": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["start_s", "end_s", "labels"],
                    "properties": {
                        "start_s": {"type": "number", "minimum": 0.0},
                        "end_s": {"type": "number", "minimum": 0.0},
                        "labels": _LABELS,
                    },
                },
            },
            "audio": {
                "properties": {
                    "duration_s": {"type": "number"},
                    "input_sample_rate_hz": {"type": "integer"},
                }
            },
        }
    },
    "else": {
        "properties": {
            "labels": {"type": "null"},
            "top_label": {"type": "null"},
            "confidence": {"type": "null"},
            "frames": {"type": "null"},
        }
    },
}


def utterance_record_json_schema() -> dict[str, object]:
    """Returns a fresh copy of the ``ser.utterance/v1`` JSON Schema.

    Constraints JSON Schema cannot express (probabilities summing to 1, ``top_label``
    being the most probable label, frames lying inside the audio) are stated in
    docs/adr/0001-utterance-analysis.md.
    """
    return copy.deepcopy(_UTTERANCE_RECORD_SCHEMA)


__all__ = ["utterance_record_json_schema"]
