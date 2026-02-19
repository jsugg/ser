"""Behavior tests for resilient dataset loading."""

from types import SimpleNamespace

import numpy as np
import pytest

from ser.data import data_loader as dl


def _build_settings(max_failed_file_ratio: float = 0.5) -> SimpleNamespace:
    """Returns a minimal settings object for data-loader tests."""
    return SimpleNamespace(
        emotions={"03": "happy", "04": "sad"},
        dataset=SimpleNamespace(glob_pattern="unused"),
        models=SimpleNamespace(num_cores=1),
        data_loader=SimpleNamespace(
            max_workers=1,
            max_failed_file_ratio=max_failed_file_ratio,
        ),
        training=SimpleNamespace(
            test_size=0.5,
            random_state=42,
            stratify_split=True,
        ),
    )


def test_process_file_reports_unexpected_name_format() -> None:
    """Files without expected RAVDESS tokens should be skipped explicitly."""
    result = dl.process_file("bad.wav", {"happy"}, {"03": "happy"})

    assert result.sample is None
    assert result.error is not None
    assert "unexpected name format" in result.error


def test_load_data_aborts_when_failure_ratio_exceeds_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A high extraction failure ratio should stop training early."""
    monkeypatch.setattr(
        dl, "get_settings", lambda: _build_settings(max_failed_file_ratio=0.4)
    )
    monkeypatch.setattr(
        dl.glob, "glob", lambda _pattern: ["a.wav", "b.wav", "c.wav", "d.wav"]
    )

    def fake_process_file(
        file: str, observed_emotions: set[str], emotion_map: dict[str, str]
    ) -> dl.ProcessFileResult:
        assert observed_emotions
        assert emotion_map
        if file == "a.wav":
            return dl.ProcessFileResult(
                sample=(np.asarray([1.0, 2.0], dtype=np.float64), "happy"),
                error=None,
            )
        return dl.ProcessFileResult(sample=None, error=f"failure for {file}")

    monkeypatch.setattr(dl, "process_file", fake_process_file)

    with pytest.raises(RuntimeError, match="exceeded configured limit"):
        dl.load_data(test_size=0.5)


def test_load_data_returns_split_for_valid_samples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Valid extracted samples should be split into train/test partitions."""
    monkeypatch.setattr(
        dl, "get_settings", lambda: _build_settings(max_failed_file_ratio=1.0)
    )
    monkeypatch.setattr(
        dl.glob, "glob", lambda _pattern: ["a.wav", "b.wav", "c.wav", "d.wav"]
    )

    sample_map: dict[str, tuple[np.ndarray, str]] = {
        "a.wav": (np.asarray([1.0, 2.0], dtype=np.float64), "happy"),
        "b.wav": (np.asarray([1.1, 2.1], dtype=np.float64), "happy"),
        "c.wav": (np.asarray([3.0, 4.0], dtype=np.float64), "sad"),
        "d.wav": (np.asarray([3.1, 4.1], dtype=np.float64), "sad"),
    }

    def fake_process_file(
        file: str, observed_emotions: set[str], emotion_map: dict[str, str]
    ) -> dl.ProcessFileResult:
        assert observed_emotions
        assert emotion_map
        return dl.ProcessFileResult(sample=sample_map[file], error=None)

    monkeypatch.setattr(dl, "process_file", fake_process_file)

    split = dl.load_data(test_size=0.5)

    assert split is not None
    x_train, x_test, y_train, y_test = split
    assert x_train.shape[1] == 2
    assert x_test.shape[1] == 2
    assert len(y_train) + len(y_test) == 4
