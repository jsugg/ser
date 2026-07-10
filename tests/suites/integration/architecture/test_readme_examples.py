"""Contract tests for executable README Python examples."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytestmark = pytest.mark.topology_contract

_FENCE_OPEN_PREFIX = "```"
_PYTHON_FENCE_LANGUAGES = frozenset({"python", "python3"})
_NOEXEC_DIRECTIVE = "noexec"


def _readme_python_blocks(readme_path: Path) -> tuple[str, ...]:
    """Extracts executable fenced Python examples and rejects malformed fences."""
    examples: list[str] = []
    open_fence: tuple[int, bool, list[str]] | None = None

    for line_number, line in enumerate(
        readme_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        stripped_line = line.strip()
        if open_fence is not None:
            opened_at, excluded, source_lines = open_fence
            if stripped_line == _FENCE_OPEN_PREFIX:
                source = "\n".join(source_lines).strip()
                if not excluded:
                    if not source:
                        raise AssertionError(
                            f"README Python fence opened on line {opened_at} is empty."
                        )
                    examples.append(source)
                open_fence = None
            else:
                source_lines.append(line)
            continue

        if not stripped_line.startswith(_FENCE_OPEN_PREFIX):
            continue
        info = stripped_line.removeprefix(_FENCE_OPEN_PREFIX).strip()
        if not info:
            continue
        language, *directives = info.split()
        if language.lower() not in _PYTHON_FENCE_LANGUAGES:
            continue
        if directives not in ([], [_NOEXEC_DIRECTIVE]):
            raise AssertionError(
                f"README Python fence on line {line_number} has unsupported directives: "
                f"{', '.join(directives)}. Use `{_NOEXEC_DIRECTIVE}` to exclude an illustrative block."
            )
        open_fence = (line_number, directives == [_NOEXEC_DIRECTIVE], [])

    if open_fence is not None:
        opened_at, _, _ = open_fence
        raise AssertionError(
            f"README Python fence opened on line {opened_at} has no closing fence."
        )

    return tuple(examples)


def _example_runner_source(example_source: str, example_index: int) -> str:
    """Builds an isolated runner that stubs only the inference owner seam."""
    return textwrap.dedent(f"""
        from __future__ import annotations

        import ser.api
        import ser.domain


        def _fake_runtime_infer(
            file_path: object,
            *,
            profile: object = None,
            language: object = None,
            save_transcript: object = False,
            include_transcript: object = True,
            subtitle_output_path: object = None,
            subtitle_format: object = None,
            settings: object = None,
            pipeline_builder: object = None,
        ) -> ser.api.InferenceExecution:
            if file_path != "sample.wav" or profile != "fast":
                raise AssertionError("README example did not call ser.api.infer as documented.")
            return ser.api.InferenceExecution(
                profile="fast",
                output_schema_version="README-test",
                backend_id="README-test",
                emotions=[ser.domain.EmotionSegment("neutral", 0.0, 1.0)],
                transcript=[],
                timeline=[],
            )


        _example_source = {example_source!r}
        original_runtime_infer = ser.api._runtime_api.infer
        try:
            ser.api._runtime_api.infer = _fake_runtime_infer
            exec(
                compile(_example_source, "README.md python block {example_index}", "exec"),
                {{"__name__": "__readme_example__"}},
            )
        finally:
            ser.api._runtime_api.infer = original_runtime_infer
        """)


def _run_example(
    repo_root: Path,
    example_source: str,
    example_index: int = 1,
) -> subprocess.CompletedProcess[str]:
    """Runs one generated README example runner in a fresh interpreter."""
    return subprocess.run(
        [sys.executable, "-c", _example_runner_source(example_source, example_index)],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )


def test_readme_python_examples_execute(repo_root: Path) -> None:
    """README Python blocks should stay executable with documented public APIs."""
    examples = _readme_python_blocks(repo_root / "README.md")
    assert examples, "README.md should contain at least one fenced Python example."

    failures: list[str] = []
    for example_index, example_source in enumerate(examples, start=1):
        completed = _run_example(repo_root, example_source, example_index)
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


def test_readme_python_blocks_reject_unclosed_or_unsupported_fences(tmp_path: Path) -> None:
    """Executable Python fences should not be silently skipped."""
    readme_path = tmp_path / "README.md"

    readme_path.write_text("```python3\nprint('missing close')\n", encoding="utf-8")
    with pytest.raises(AssertionError, match="no closing fence"):
        _readme_python_blocks(readme_path)

    readme_path.write_text("```python skip\nprint('unsupported')\n```\n", encoding="utf-8")
    with pytest.raises(AssertionError, match="unsupported directives"):
        _readme_python_blocks(readme_path)


def test_readme_python_blocks_allow_explicit_non_executable_examples(tmp_path: Path) -> None:
    """Illustrative Python blocks require the explicit `noexec` directive."""
    readme_path = tmp_path / "README.md"
    readme_path.write_text(
        "```python noexec\nnot valid Python\n```\n\n```python\nprint('checked')\n```\n",
        encoding="utf-8",
    )

    assert _readme_python_blocks(readme_path) == ("print('checked')",)


def test_readme_runner_keeps_the_real_public_infer_contract(repo_root: Path) -> None:
    """The fixture must not mask missing facade functions or invalid API arguments."""
    identity_check = _run_example(
        repo_root,
        "import ser.api\nassert ser.api.infer.__module__ == 'ser.api'",
    )
    assert identity_check.returncode == 0, identity_check.stderr or identity_check.stdout

    invalid_arguments = _run_example(
        repo_root,
        "import ser.api\nser.api.infer('sample.wav', unsupported_keyword=True)",
    )
    assert invalid_arguments.returncode != 0
    assert "unsupported_keyword" in invalid_arguments.stderr
