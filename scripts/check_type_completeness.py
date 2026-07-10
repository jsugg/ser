#!/usr/bin/env python3
"""Run pyright verifytypes and enforce the configured completeness ratchet."""

from __future__ import annotations

import json
import subprocess
import sys
import tomllib
from math import isfinite
from pathlib import Path
from typing import cast

MINIMUM_COMPLETENESS_THRESHOLD = 0.95
REPO_ROOT = Path(__file__).resolve().parent.parent

PYRIGHT_VERIFYTYPES_COMMAND = (
    "pyright",
    "--verifytypes",
    "ser",
    "--ignoreexternal",
    "--outputjson",
)


def _mapping(value: object, description: str) -> dict[str, object]:
    """Validates one JSON or TOML mapping at an external boundary."""
    if not isinstance(value, dict):
        raise RuntimeError(f"{description} must be a mapping.")
    return cast(dict[str, object], value)


def _load_threshold(repo_root: Path) -> float:
    """Loads the configured type-completeness threshold from pyproject."""
    raw_pyproject: object = tomllib.loads(
        (repo_root / "pyproject.toml").read_text(encoding="utf-8")
    )
    pyproject = _mapping(raw_pyproject, "pyproject.toml")
    tool = _mapping(pyproject.get("tool"), "[tool]")
    ser = _mapping(tool.get("ser"), "[tool.ser]")
    completeness = _mapping(ser.get("type_completeness"), "[tool.ser.type_completeness]")
    threshold = completeness.get("threshold")
    if isinstance(threshold, bool) or not isinstance(threshold, int | float):
        raise TypeError("[tool.ser.type_completeness].threshold must be a number.")
    parsed_threshold = float(threshold)
    if (
        not isfinite(parsed_threshold)
        or not MINIMUM_COMPLETENESS_THRESHOLD <= parsed_threshold <= 1.0
    ):
        raise ValueError(
            "[tool.ser.type_completeness].threshold must be finite and between "
            f"{MINIMUM_COMPLETENESS_THRESHOLD:.2f} and 1.00."
        )
    return parsed_threshold


def _run_pyright(repo_root: Path) -> dict[str, object]:
    """Runs pyright verifytypes and returns its JSON payload."""
    completed = subprocess.run(
        PYRIGHT_VERIFYTYPES_COMMAND,
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        diagnostics = completed.stderr.strip()
        diagnostic_suffix = f"\n{diagnostics}" if diagnostics else ""
        raise RuntimeError(
            "pyright --verifytypes exited with status "
            f"{completed.returncode}.{diagnostic_suffix}"
        )
    try:
        raw_payload: object = json.loads(completed.stdout)
    except json.JSONDecodeError as err:
        sys.stderr.write(completed.stdout)
        sys.stderr.write(completed.stderr)
        raise RuntimeError("pyright --verifytypes did not emit valid JSON.") from err
    if not isinstance(raw_payload, dict):
        raise RuntimeError("pyright --verifytypes JSON payload must be an object.")
    return cast(dict[str, object], raw_payload)


def _read_score(payload: dict[str, object]) -> float:
    """Reads the verifytypes completeness score with shape validation."""
    completeness = payload.get("typeCompleteness")
    if not isinstance(completeness, dict):
        raise RuntimeError("pyright JSON missing typeCompleteness object.")
    score = completeness.get("completenessScore")
    if isinstance(score, bool) or not isinstance(score, int | float):
        raise RuntimeError("pyright JSON missing numeric completenessScore.")
    parsed_score = float(score)
    if not isfinite(parsed_score) or not 0.0 <= parsed_score <= 1.0:
        raise RuntimeError("pyright JSON completenessScore must be finite and between 0.0 and 1.0.")
    return parsed_score


def _read_error_count(payload: dict[str, object]) -> int:
    """Reads the pyright summary error count."""
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        raise RuntimeError("pyright JSON missing summary object.")
    error_count = summary.get("errorCount")
    if isinstance(error_count, bool) or not isinstance(error_count, int) or error_count < 0:
        raise RuntimeError("pyright JSON missing integer summary.errorCount.")
    return error_count


def main() -> int:
    """CLI entry point."""
    repo_root = REPO_ROOT
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
