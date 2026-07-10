#!/usr/bin/env python3
"""Run pyright verifytypes and enforce the configured completeness ratchet."""

from __future__ import annotations

import json
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

PYRIGHT_VERIFYTYPES_COMMAND = (
    "pyright",
    "--verifytypes",
    "ser",
    "--ignoreexternal",
    "--outputjson",
)


def _load_threshold(repo_root: Path) -> float:
    """Loads the configured type-completeness threshold from pyproject."""
    pyproject = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))
    threshold = pyproject["tool"]["ser"]["type_completeness"]["threshold"]
    if not isinstance(threshold, int | float):
        raise TypeError("[tool.ser.type_completeness].threshold must be a number.")
    return float(threshold)


def _run_pyright(repo_root: Path) -> dict[str, Any]:
    """Runs pyright verifytypes and returns its JSON payload."""
    completed = subprocess.run(
        PYRIGHT_VERIFYTYPES_COMMAND,
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as err:
        sys.stderr.write(completed.stdout)
        sys.stderr.write(completed.stderr)
        raise RuntimeError("pyright --verifytypes did not emit valid JSON.") from err
    if not isinstance(payload, dict):
        raise RuntimeError("pyright --verifytypes JSON payload must be an object.")
    return payload


def _read_score(payload: dict[str, Any]) -> float:
    """Reads the verifytypes completeness score with shape validation."""
    completeness = payload.get("typeCompleteness")
    if not isinstance(completeness, dict):
        raise RuntimeError("pyright JSON missing typeCompleteness object.")
    score = completeness.get("completenessScore")
    if not isinstance(score, int | float):
        raise RuntimeError("pyright JSON missing numeric completenessScore.")
    return float(score)


def _read_error_count(payload: dict[str, Any]) -> int:
    """Reads the pyright summary error count."""
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        raise RuntimeError("pyright JSON missing summary object.")
    error_count = summary.get("errorCount")
    if not isinstance(error_count, int):
        raise RuntimeError("pyright JSON missing integer summary.errorCount.")
    return error_count


def main() -> int:
    """CLI entry point."""
    repo_root = Path.cwd()
    threshold = _load_threshold(repo_root)
    payload = _run_pyright(repo_root)
    error_count = _read_error_count(payload)
    score = _read_score(payload)
    print(f"pyright verifytypes completeness: {score:.10f} (threshold {threshold:.10f})")
    if error_count:
        print(f"pyright verifytypes reported {error_count} errors.", file=sys.stderr)
        return 1
    if score < threshold:
        print(
            f"pyright verifytypes completeness {score:.10f} is below threshold {threshold:.10f}.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
