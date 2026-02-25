"""Pipeline-level compatibility contracts for refactor safety."""

from pathlib import Path
from types import SimpleNamespace

import pytest

import ser.__main__ as cli
import ser.models.emotion_model as emotion_model
from ser import domain


def _patch_cli_prerequisites(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patches shared CLI dependencies for deterministic contract tests."""
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    monkeypatch.setattr(
        cli,
        "reload_settings",
        lambda: SimpleNamespace(default_language="en"),
    )


def test_domain_contract_shape() -> None:
    """Domain tuple field names and order remain stable for consumers."""
    assert domain.TranscriptWord._fields == ("word", "start_seconds", "end_seconds")
    assert domain.EmotionSegment._fields == ("emotion", "start_seconds", "end_seconds")
    assert domain.TimelineEntry._fields == ("timestamp_seconds", "emotion", "speech")


def test_cli_missing_file_exit_code_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CLI exits with code 1 when prediction file argument is omitted."""
    _patch_cli_prerequisites(monkeypatch)
    monkeypatch.setattr(cli.sys, "argv", ["ser"])

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 1


def test_cli_train_exit_code_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI exits with code 0 after successful `--train` dispatch."""
    _patch_cli_prerequisites(monkeypatch)
    monkeypatch.setattr(cli.sys, "argv", ["ser", "--train"])

    called = {"train": False}

    def fake_train_model() -> None:
        called["train"] = True

    monkeypatch.setattr(emotion_model, "train_model", fake_train_model)

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 0
    assert called["train"] is True


def test_model_load_candidate_order_and_uniqueness(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Model load candidate order remains secure-first with deduplicated entries."""
    primary_dir = tmp_path / "primary"
    monkeypatch.setattr(
        emotion_model,
        "get_settings",
        lambda: SimpleNamespace(
            models=SimpleNamespace(
                folder=primary_dir,
                secure_model_file=primary_dir / "ser_model.skops",
                model_file=primary_dir / "ser_model.pkl",
                secure_model_file_name="ser_model.skops",
                model_file_name="ser_model.pkl",
            )
        ),
    )

    candidates = emotion_model._model_load_candidates()

    assert [candidate.artifact_format for candidate in candidates] == [
        "skops",
        "pickle",
    ]
    keys = {(str(candidate.path), candidate.artifact_format) for candidate in candidates}
    assert len(keys) == len(candidates)
