"""Validator for the JSON Schema keywords the ``ser.utterance/v1`` schema uses.

The project has no JSON Schema library; this covers exactly the keywords in
``SUPPORTED_KEYWORDS`` and refuses schemas that use any other.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import cast

SUPPORTED_KEYWORDS = frozenset(
    {
        "$schema",
        "$id",
        "title",
        "type",
        "const",
        "enum",
        "required",
        "properties",
        "additionalProperties",
        "minProperties",
        "propertyNames",
        "minLength",
        "pattern",
        "minimum",
        "maximum",
        "minItems",
        "items",
        "if",
        "then",
        "else",
    }
)

type Schema = Mapping[str, object]


def unsupported_keywords(schema: object) -> set[str]:
    """Returns every keyword in ``schema`` (recursively) this validator cannot check."""
    found: set[str] = set()
    if not isinstance(schema, Mapping):
        return found
    node = cast(Schema, schema)
    found |= set(node) - SUPPORTED_KEYWORDS
    for key in ("propertyNames", "items", "if", "then", "else"):
        found |= unsupported_keywords(node.get(key))
    additional = node.get("additionalProperties")
    if isinstance(additional, Mapping):
        found |= unsupported_keywords(additional)
    properties = node.get("properties")
    if isinstance(properties, Mapping):
        for child in cast(Schema, properties).values():
            found |= unsupported_keywords(child)
    return found


def validate(instance: object, schema: Schema, path: str = "$") -> list[str]:
    """Returns one message per violation; an empty list means valid."""
    errors: list[str] = []
    if "type" in schema and not _matches_type(instance, schema["type"]):
        return [f"{path}: expected type {schema['type']}, got {type(instance).__name__}"]
    if "const" in schema and not _json_equal(instance, schema["const"]):
        errors.append(f"{path}: expected {schema['const']!r}")
    if "enum" in schema and not any(
        _json_equal(instance, option) for option in cast(Sequence[object], schema["enum"])
    ):
        errors.append(f"{path}: {instance!r} not in enum")
    if isinstance(instance, str):
        if len(instance) < cast(int, schema.get("minLength", 0)):
            errors.append(f"{path}: shorter than minLength")
        pattern = schema.get("pattern")
        if isinstance(pattern, str) and re.search(pattern, instance) is None:
            errors.append(f"{path}: does not match {pattern!r}")
    if _is_number(instance):
        number = cast(float, instance)
        if "minimum" in schema and number < cast(float, schema["minimum"]):
            errors.append(f"{path}: below minimum")
        if "maximum" in schema and number > cast(float, schema["maximum"]):
            errors.append(f"{path}: above maximum")
    if isinstance(instance, list):
        items = cast(list[object], instance)
        if len(items) < cast(int, schema.get("minItems", 0)):
            errors.append(f"{path}: fewer than minItems")
        item_schema = schema.get("items")
        if isinstance(item_schema, Mapping):
            for index, item in enumerate(items):
                errors += validate(item, cast(Schema, item_schema), f"{path}[{index}]")
    if isinstance(instance, dict):
        errors += _validate_object(cast(dict[str, object], instance), schema, path)
    condition = schema.get("if")
    if isinstance(condition, Mapping):
        branch = schema.get("else" if validate(instance, cast(Schema, condition)) else "then")
        if isinstance(branch, Mapping):
            errors += validate(instance, cast(Schema, branch), path)
    return errors


def _validate_object(instance: dict[str, object], schema: Schema, path: str) -> list[str]:
    errors: list[str] = []
    for key in cast(Sequence[str], schema.get("required", ())):
        if key not in instance:
            errors.append(f"{path}: missing {key!r}")
    if len(instance) < cast(int, schema.get("minProperties", 0)):
        errors.append(f"{path}: fewer than minProperties")
    names = schema.get("propertyNames")
    properties = cast(Schema, schema.get("properties", {}))
    additional = schema.get("additionalProperties", True)
    for key, value in instance.items():
        if isinstance(names, Mapping):
            errors += validate(key, cast(Schema, names), f"{path}.<name {key!r}>")
        child_path = f"{path}.{key}"
        if key in properties:
            errors += validate(value, cast(Schema, properties[key]), child_path)
        elif additional is False:
            errors.append(f"{path}: unexpected property {key!r}")
        elif isinstance(additional, Mapping):
            errors += validate(value, cast(Schema, additional), child_path)
    return errors


def _matches_type(instance: object, expected: object) -> bool:
    names = [expected] if isinstance(expected, str) else cast(Sequence[str], expected)
    return any(_is_type(instance, name) for name in names)


def _is_type(instance: object, name: str) -> bool:
    if name == "null":
        return instance is None
    if name == "boolean":
        return isinstance(instance, bool)
    if name == "integer":
        return isinstance(instance, int) and not isinstance(instance, bool)
    if name == "number":
        return _is_number(instance)
    if name == "string":
        return isinstance(instance, str)
    if name == "array":
        return isinstance(instance, list)
    if name == "object":
        return isinstance(instance, dict)
    raise ValueError(f"unknown JSON type {name!r}")


def _is_number(instance: object) -> bool:
    return isinstance(instance, int | float) and not isinstance(instance, bool)


def _json_equal(left: object, right: object) -> bool:
    return type(left) is type(right) and left == right
