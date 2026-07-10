"""Contracts for the self-sufficient `ser.api` type vocabulary."""

from __future__ import annotations

import inspect
import typing

import ser.api as api


def _type_hints(value: object, path: str) -> dict[str, object]:
    """Resolves annotations or raises an actionable public-contract failure."""
    try:
        return dict(typing.get_type_hints(value))
    except (NameError, TypeError) as error:
        raise AssertionError(f"Unable to resolve public annotation at {path}: {error}") from error


def _walk_annotation(
    value: object,
    *,
    path: str,
    visited_classes: set[int],
    missing: dict[tuple[str, str], set[str]],
) -> None:
    """Records first-party types reachable through one public annotation value."""
    if isinstance(value, (tuple, list, set, frozenset)):
        for item in value:
            _walk_annotation(
                item,
                path=path,
                visited_classes=visited_classes,
                missing=missing,
            )
        return

    origin = typing.get_origin(value)
    if origin is not None:
        for argument in typing.get_args(value):
            _walk_annotation(
                argument,
                path=path,
                visited_classes=visited_classes,
                missing=missing,
            )
        return

    if type(value).__name__ == "TypeAliasType":
        alias_value = getattr(value, "__value__", None)
        if alias_value is not None:
            _walk_annotation(
                alias_value,
                path=path,
                visited_classes=visited_classes,
                missing=missing,
            )
        return

    if inspect.isclass(value):
        module_name = getattr(value, "__module__", "")
        type_name = getattr(value, "__name__", "")
        if (
            isinstance(module_name, str)
            and module_name.startswith("ser")
            and isinstance(type_name, str)
        ):
            if getattr(api, type_name, None) is not value:
                missing.setdefault((module_name, type_name), set()).add(path)

        if id(value) in visited_classes:
            return
        visited_classes.add(id(value))
        for annotation_name, annotation in _type_hints(value, path).items():
            _walk_annotation(
                annotation,
                path=f"{path}.{annotation_name}",
                visited_classes=visited_classes,
                missing=missing,
            )
        for member_name, member in value.__dict__.items():
            if member_name.startswith("_"):
                continue
            if isinstance(member, (classmethod, staticmethod)):
                member = member.__func__
            if isinstance(member, property):
                if member.fget is not None:
                    _walk_annotation(
                        member.fget,
                        path=f"{path}.{member_name}",
                        visited_classes=visited_classes,
                        missing=missing,
                    )
            elif inspect.isfunction(member):
                _walk_annotation(
                    member,
                    path=f"{path}.{member_name}",
                    visited_classes=visited_classes,
                    missing=missing,
                )
        return

    if inspect.isfunction(value):
        for annotation_name, annotation in _type_hints(value, path).items():
            _walk_annotation(
                annotation,
                path=f"{path}.{annotation_name}",
                visited_classes=visited_classes,
                missing=missing,
            )


def test_public_api_reexports_every_reachable_first_party_annotation_type() -> None:
    """Consumers should never need an implementation-side import for API annotations."""
    missing: dict[tuple[str, str], set[str]] = {}
    visited_classes: set[int] = set()

    for export_name in api.__all__:
        _walk_annotation(
            getattr(api, export_name),
            path=f"ser.api.{export_name}",
            visited_classes=visited_classes,
            missing=missing,
        )

    assert not missing, "Missing direct ser.api type exports:\n" + "\n".join(
        f"- {module_name}.{type_name}: {', '.join(sorted(paths))}"
        for (module_name, type_name), paths in sorted(missing.items())
    )
