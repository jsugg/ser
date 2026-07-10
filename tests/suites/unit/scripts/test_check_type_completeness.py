"""Regression contracts for the type-completeness verification script."""

from __future__ import annotations

import os
import stat
import subprocess
import sys
from pathlib import Path


def test_checker_rejects_nonzero_pyright_with_valid_json(repo_root: Path, tmp_path: Path) -> None:
    """A valid-looking payload must not hide a failed pyright process."""
    fake_pyright = tmp_path / "pyright"
    fake_pyright.write_text(
        "\n".join(
            (
                f"#!{sys.executable}",
                "import json",
                "raise SystemExit((print(json.dumps({'typeCompleteness': {'completenessScore': 1.0}, 'summary': {'errorCount': 0}})) or 7))",
                "",
            )
        ),
        encoding="utf-8",
    )
    fake_pyright.chmod(fake_pyright.stat().st_mode | stat.S_IXUSR)

    environment = os.environ.copy()
    environment["PATH"] = f"{tmp_path}{os.pathsep}{environment['PATH']}"
    completed = subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "check_type_completeness.py")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode != 0
    assert "pyright --verifytypes exited with status 7" in completed.stderr


def test_checker_rejects_nonfinite_completeness_score(repo_root: Path, tmp_path: Path) -> None:
    """A NaN score must fail closed instead of bypassing numeric comparison."""
    fake_pyright = tmp_path / "pyright"
    fake_pyright.write_text(
        "\n".join(
            (
                f"#!{sys.executable}",
                'print(\'{"typeCompleteness": {"completenessScore": NaN}, "summary": {"errorCount": 0}}\')',
                "",
            )
        ),
        encoding="utf-8",
    )
    fake_pyright.chmod(fake_pyright.stat().st_mode | stat.S_IXUSR)

    environment = os.environ.copy()
    environment["PATH"] = f"{tmp_path}{os.pathsep}{environment['PATH']}"
    completed = subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "check_type_completeness.py")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode != 0
    assert "completenessScore must be finite" in completed.stderr


def test_checker_rejects_malformed_pyright_json(repo_root: Path, tmp_path: Path) -> None:
    """Malformed output must not become a passing completeness result."""
    fake_pyright = tmp_path / "pyright"
    fake_pyright.write_text(
        "\n".join((f"#!{sys.executable}", "print('not JSON')", "")),
        encoding="utf-8",
    )
    fake_pyright.chmod(fake_pyright.stat().st_mode | stat.S_IXUSR)

    environment = os.environ.copy()
    environment["PATH"] = f"{tmp_path}{os.pathsep}{environment['PATH']}"
    completed = subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "check_type_completeness.py")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode != 0
    assert "did not emit valid JSON" in completed.stderr


def test_checker_runs_pyright_from_repository_root(repo_root: Path, tmp_path: Path) -> None:
    """The checker must not validate a package resolved from its caller's directory."""
    fake_pyright = tmp_path / "pyright"
    fake_pyright.write_text(
        "\n".join(
            (
                f"#!{sys.executable}",
                "import json",
                "import os",
                "from pathlib import Path",
                "Path(os.environ['PYRIGHT_CWD_RECORD']).write_text(str(Path.cwd()), encoding='utf-8')",
                "print(json.dumps({'typeCompleteness': {'completenessScore': 1.0}, 'summary': {'errorCount': 0}}))",
                "",
            )
        ),
        encoding="utf-8",
    )
    fake_pyright.chmod(fake_pyright.stat().st_mode | stat.S_IXUSR)

    record_path = tmp_path / "pyright-cwd.txt"
    environment = os.environ.copy()
    environment["PATH"] = f"{tmp_path}{os.pathsep}{environment['PATH']}"
    environment["PYRIGHT_CWD_RECORD"] = str(record_path)
    completed = subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "check_type_completeness.py")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr
    assert record_path.read_text(encoding="utf-8") == str(repo_root)
