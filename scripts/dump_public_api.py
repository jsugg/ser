#!/usr/bin/env python3
"""Dump the reviewed tier-1 public API surface as stable JSON."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

import griffe

SCHEMA_VERSION = 1
SNAPSHOT_PATH = Path("tests/suites/integration/architecture/public_api_snapshot.json")
TIER_ONE_MODULES = (
    "ser",
    "ser.api",
    "ser.config",
    "ser.domain",
    "ser.profiles",
    "ser.utils",
)

type JsonValue = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
type JsonObject = dict[str, JsonValue]


def _text(value: object) -> str | None:
    """Returns a stable string representation for Griffe expression values."""
    if value is None:
        return None
    return str(value)


def _kind(member: Any) -> str:
    """Returns Griffe's stable kind value."""
    return str(member.kind.value)


def _member_name(member: Any) -> str:
    """Returns a member name without resolving aliases."""
    return str(member.name)


def _parse_dunder_all(module: Any, module_name: str) -> tuple[str, ...] | None:
    """Reads a literal module `__all__` declaration when present."""
    all_member = module.members.get("__all__")
    if all_member is None:
        return None
    raw_value = _text(all_member.value)
    if raw_value is None:
        raise ValueError(f"{module_name}.__all__ must be a literal sequence of strings.")
    parsed = ast.literal_eval(raw_value)
    if not isinstance(parsed, list | tuple) or not all(isinstance(item, str) for item in parsed):
        raise ValueError(f"{module_name}.__all__ must be a literal sequence of strings.")
    return tuple(parsed)


def _exported_names(module: Any, module_name: str) -> tuple[str, ...]:
    """Returns reviewed export names for one tier-1 module."""
    explicit_exports = _parse_dunder_all(module, module_name)
    if explicit_exports is not None:
        return tuple(sorted(explicit_exports))

    return tuple(
        sorted(
            name
            for name, member in module.members.items()
            if not name.startswith("_") and not bool(getattr(member, "is_alias", False))
        )
    )


def _parameter_snapshot(parameter: Any) -> JsonObject:
    """Returns a stable snapshot for one callable parameter."""
    snapshot: JsonObject = {
        "annotation": _text(parameter.annotation),
        "default": _text(parameter.default),
        "kind": str(parameter.kind.value),
        "name": str(parameter.name),
    }
    return snapshot


def _signature(member: Any) -> JsonObject:
    """Returns a stable callable signature snapshot."""
    parameters = [_parameter_snapshot(parameter) for parameter in member.parameters]
    return {
        "parameters": parameters,
        "returns": _text(member.returns),
        "signature": _format_signature(parameters, _text(member.returns)),
    }


def _format_signature(parameters: list[JsonObject], returns: str | None) -> str:
    """Formats parameters into a compact, review-friendly signature string."""
    parts: list[str] = []
    inserted_keyword_separator = False
    for parameter in parameters:
        kind = parameter["kind"]
        if kind == "keyword-only" and not inserted_keyword_separator:
            parts.append("*")
            inserted_keyword_separator = True

        name = str(parameter["name"])
        if kind == "variadic positional":
            name = f"*{name}"
            inserted_keyword_separator = True
        elif kind == "variadic keyword":
            name = f"**{name}"

        annotation = parameter["annotation"]
        if isinstance(annotation, str):
            name = f"{name}: {annotation}"

        default = parameter["default"]
        if isinstance(default, str):
            name = f"{name} = {default}"
        parts.append(name)

    return_annotation = returns if returns is not None else "None"
    return f"({', '.join(parts)}) -> {return_annotation}"


def _attribute_snapshot(member: Any) -> JsonObject:
    """Returns a stable attribute or type-alias snapshot."""
    return {
        "annotation": _text(getattr(member, "annotation", None)),
        "kind": _kind(member),
        "value": _text(getattr(member, "value", None)),
    }


def _class_snapshot(member: Any) -> JsonObject:
    """Returns a stable class snapshot, including reviewed public members."""
    snapshot: JsonObject = {
        "bases": [_text(base) for base in member.bases],
        "kind": _kind(member),
        "members": {
            name: _member_snapshot(child)
            for name, child in sorted(member.members.items())
            if not name.startswith("_")
        },
    }
    return snapshot


def _alias_snapshot(member: Any) -> JsonObject:
    """Returns a stable alias snapshot without forcing target resolution."""
    target_path = getattr(member, "target_path", None)
    return {
        "kind": _kind(member),
        "target_path": str(target_path) if target_path is not None else None,
    }


def _member_snapshot(member: Any) -> JsonObject:
    """Returns the stable public-contract snapshot for one exported member."""
    if bool(getattr(member, "is_alias", False)):
        return _alias_snapshot(member)
    kind = _kind(member)
    if kind in {"function", "method"}:
        snapshot = _signature(member)
        snapshot["kind"] = kind
        return snapshot
    if kind == "class":
        return _class_snapshot(member)
    if kind in {"attribute", "type alias"}:
        return _attribute_snapshot(member)
    return {"kind": kind}


def dump_public_api(repo_root: Path) -> JsonObject:
    """Builds the stable tier-1 public API snapshot."""
    modules: dict[str, JsonValue] = {}
    search_paths = [str(repo_root)]
    for module_name in TIER_ONE_MODULES:
        module = griffe.load(module_name, search_paths=search_paths)
        exports: dict[str, JsonValue] = {}
        for export_name in _exported_names(module, module_name):
            member = module.members.get(export_name)
            if member is None:
                raise ValueError(f"{module_name}.__all__ exports missing member {export_name!r}.")
            exports[export_name] = _member_snapshot(member)
        modules[module_name] = {
            "exports": exports,
            "source": "explicit __all__" if "__all__" in module.members else "public definitions",
        }

    return {
        "modules": modules,
        "schema_version": SCHEMA_VERSION,
        "tier_one_modules": list(TIER_ONE_MODULES),
    }


def _json_text(snapshot: JsonObject) -> str:
    """Serializes snapshot data with deterministic formatting."""
    return json.dumps(snapshot, indent=2, sort_keys=True) + "\n"


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        action="store_true",
        help=f"write {SNAPSHOT_PATH} instead of printing to stdout",
    )
    args = parser.parse_args()

    repo_root = Path.cwd()
    snapshot_text = _json_text(dump_public_api(repo_root))
    if args.write:
        output_path = repo_root / SNAPSHOT_PATH
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(snapshot_text, encoding="utf-8")
    else:
        print(snapshot_text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
