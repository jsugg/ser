"""Contract tests for executable README Python examples."""

from __future__ import annotations

import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytestmark = pytest.mark.topology_contract

_README_PYTHON_BLOCK_RE = re.compile(r"```python\n(.*?)\n```", re.DOTALL)


def _readme_python_blocks(readme_path: Path) -> tuple[str, ...]:
    """Extracts fenced Python examples from README."""
    return tuple(
        match.group(1).strip()
        for match in _README_PYTHON_BLOCK_RE.finditer(readme_path.read_text(encoding="utf-8"))
    )


def _example_runner_source(example_source: str, example_index: int) -> str:
    """Builds an isolated runner with a lightweight inference stub."""
    return textwrap.dedent(f"""
        from __future__ import annotations

        from types import SimpleNamespace

        import ser.api


        class _Emotion:
            def __init__(self, emotion: str, start_seconds: float, end_seconds: float) -> None:
                self.emotion = emotion
                self.start_seconds = start_seconds
                self.end_seconds = end_seconds


        def _fake_infer(*_args: object, **_kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(emotions=(_Emotion("neutral", 0.0, 1.0),))


        ser.api.infer = _fake_infer
        _example_source = {example_source!r}
        exec(
            compile(_example_source, "README.md python block {example_index}", "exec"),
            {{"__name__": "__readme_example__"}},
        )
        """)


def test_readme_python_examples_execute(repo_root: Path) -> None:
    """README Python blocks should stay executable with documented public APIs."""
    examples = _readme_python_blocks(repo_root / "README.md")
    assert examples, "README.md should contain at least one fenced Python example."

    failures: list[str] = []
    for example_index, example_source in enumerate(examples, start=1):
        completed = subprocess.run(
            [sys.executable, "-c", _example_runner_source(example_source, example_index)],
            cwd=repo_root,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            failures.append(
                "\n".join(
                    (
                        f"README Python example {example_index} failed.",
                        f"stdout:\n{completed.stdout}",
                        f"stderr:\n{completed.stderr}",
                    )
                )
            )

    assert failures == []
