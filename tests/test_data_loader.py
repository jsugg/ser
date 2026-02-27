"""Behavior tests for resilient dataset loading."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ser.data import data_loader as dl
from ser.data.manifest import MANIFEST_SCHEMA_VERSION


def _build_settings(max_failed_file_ratio: float = 0.5) -> SimpleNamespace:
    """Returns a minimal settings object for data-loader tests."""
    return SimpleNamespace(
        emotions={"03": "happy", "04": "sad"},
        dataset=SimpleNamespace(
            glob_pattern="unused",
            folder=Path("unused"),
            manifest_paths=(),
        ),
        models=SimpleNamespace(
            num_cores=1,
            folder=Path("unused/models"),
        ),
        data_loader=SimpleNamespace(
            max_workers=1,
            max_failed_file_ratio=max_failed_file_ratio,
        ),
        training=SimpleNamespace(
            test_size=0.5,
            random_state=42,
            stratify_split=True,
        ),
        default_language="en",
    )


def test_process_file_reports_unexpected_name_format() -> None:
    """Files without expected RAVDESS tokens should be skipped explicitly."""
    result = dl.process_file("bad.wav", {"happy"}, {"03": "happy"})

    assert result.sample is None
    assert result.error is not None
    assert "unexpected name format" in result.error


def test_extract_ravdess_speaker_id_from_path() -> None:
    """RAVDESS actor IDs should be parsed from full file paths."""
    path = "ser/dataset/ravdess/Actor_24/03-01-05-01-02-01-24.wav"
    assert dl.extract_ravdess_speaker_id_from_path(path) == "24"
    assert dl.extract_ravdess_speaker_id_from_path("invalid.wav") is None


def test_load_labeled_audio_paths_returns_expected_pairs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Labeled path loader should keep only supported emotions with valid names."""
    monkeypatch.setattr(
        dl, "get_settings", lambda: _build_settings(max_failed_file_ratio=1.0)
    )
    monkeypatch.setattr(
        dl.glob,
        "glob",
        lambda _pattern: [
            "Actor_01/03-01-03-01-01-01-01.wav",
            "Actor_02/03-01-04-01-01-01-02.wav",
            "Actor_03/03-01-08-01-01-01-03.wav",
            "invalid.wav",
        ],
    )

    samples = dl.load_labeled_audio_paths()

    assert samples == [
        ("Actor_01/03-01-03-01-01-01-01.wav", "happy"),
        ("Actor_02/03-01-04-01-01-01-02.wav", "sad"),
    ]


def test_load_labeled_audio_paths_aborts_on_high_parse_failure_ratio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Label-loader should stop early when parse failures exceed threshold."""
    monkeypatch.setattr(
        dl, "get_settings", lambda: _build_settings(max_failed_file_ratio=0.3)
    )
    monkeypatch.setattr(
        dl.glob,
        "glob",
        lambda _pattern: [
            "bad-a.wav",
            "bad-b.wav",
            "Actor_01/03-01-03-01-01-01-01.wav",
        ],
    )

    with pytest.raises(RuntimeError, match="exceeded configured limit"):
        dl.load_labeled_audio_paths()


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


def test_load_utterances_prefers_manifest_when_configured(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Manifest mode should load utterances and preserve canonical labels."""
    manifest_path = tmp_path / "dataset.jsonl"
    manifest_path.write_text(
        "\n".join(
            [
                (
                    '{"schema_version": 1, "sample_id": "ravdess:a.wav", '
                    '"corpus": "ravdess", "audio_path": "a.wav", '
                    '"label": "happy", "speaker_id": "ravdess:1", "split": "train"}'
                ),
                (
                    '{"schema_version": 1, "sample_id": "ravdess:b.wav", '
                    '"corpus": "ravdess", "audio_path": "b.wav", '
                    '"label": "sad", "speaker_id": "ravdess:2", "split": "test"}'
                ),
            ]
        ),
        encoding="utf-8",
    )
    settings = _build_settings(max_failed_file_ratio=1.0)
    settings.dataset.manifest_paths = (manifest_path,)
    settings.dataset.folder = tmp_path
    monkeypatch.setattr(dl, "get_settings", lambda: settings)

    utterances = dl.load_utterances()

    assert utterances is not None
    assert [item.sample_id for item in utterances] == ["ravdess:a.wav", "ravdess:b.wav"]
    assert all(item.schema_version == MANIFEST_SCHEMA_VERSION for item in utterances)
    assert all(item.audio_path.is_absolute() for item in utterances)


def test_load_utterances_rejects_duplicate_sample_ids_across_manifests(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Duplicate sample ids across manifest files should fail closed."""
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    payload = (
        '{"schema_version": 1, "sample_id": "dup:1", "corpus": "ravdess", '
        '"audio_path": "a.wav", "label": "happy", "speaker_id": "ravdess:1"}\n'
    )
    first.write_text(payload, encoding="utf-8")
    second.write_text(payload, encoding="utf-8")
    settings = _build_settings(max_failed_file_ratio=1.0)
    settings.dataset.manifest_paths = (first, second)
    settings.dataset.folder = tmp_path
    monkeypatch.setattr(dl, "get_settings", lambda: settings)

    with pytest.raises(RuntimeError, match="Duplicate sample_id"):
        dl.load_utterances()


def test_load_utterances_registry_uses_dataset_root_as_manifest_base_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Registry manifests should resolve relative audio paths against dataset_root."""
    dataset_root = tmp_path / "datasets" / "ravdess"
    manifest_root = tmp_path / "manifests"
    dataset_root.mkdir(parents=True, exist_ok=True)
    manifest_root.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_root / "ravdess.jsonl"
    manifest_path.write_text(
        "\n".join(
            [
                (
                    '{"schema_version": 1, "sample_id": "ravdess:a.wav", '
                    '"corpus": "ravdess", "audio_path": "clips/a.wav", '
                    '"label": "happy", "speaker_id": "ravdess:1"}'
                ),
                (
                    '{"schema_version": 1, "sample_id": "ravdess:b.wav", '
                    '"corpus": "ravdess", "audio_path": "clips/b.wav", '
                    '"label": "sad", "speaker_id": "ravdess:2"}'
                ),
            ]
        ),
        encoding="utf-8",
    )
    settings = _build_settings(max_failed_file_ratio=1.0)
    settings.dataset.manifest_paths = ()
    settings.dataset.folder = dataset_root
    monkeypatch.setattr(dl, "get_settings", lambda: settings)
    monkeypatch.setattr(
        dl,
        "load_dataset_registry",
        lambda settings: {
            "ravdess": SimpleNamespace(
                dataset_id="ravdess",
                dataset_root=dataset_root,
                manifest_path=manifest_path,
                options={},
            )
        },
    )
    monkeypatch.setattr(
        dl,
        "prepare_from_registry_entry",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("not expected")),
    )

    utterances = dl.load_utterances()

    assert utterances is not None
    assert utterances[0].audio_path == dataset_root / "clips" / "a.wav"
    assert utterances[1].audio_path == dataset_root / "clips" / "b.wav"


def test_load_utterances_registry_rebuilds_missing_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Missing registry manifests should trigger prepare_from_registry_entry."""
    dataset_root = tmp_path / "datasets" / "ravdess"
    manifest_root = tmp_path / "manifests"
    dataset_root.mkdir(parents=True, exist_ok=True)
    manifest_root.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_root / "ravdess.jsonl"
    manifest_path.write_text(
        "\n".join(
            [
                (
                    '{"schema_version": 1, "sample_id": "ravdess:a.wav", '
                    '"corpus": "ravdess", "audio_path": "clips/a.wav", '
                    '"label": "happy", "speaker_id": "ravdess:1"}'
                ),
                (
                    '{"schema_version": 1, "sample_id": "ravdess:b.wav", '
                    '"corpus": "ravdess", "audio_path": "clips/b.wav", '
                    '"label": "sad", "speaker_id": "ravdess:2"}'
                ),
            ]
        ),
        encoding="utf-8",
    )
    settings = _build_settings(max_failed_file_ratio=1.0)
    settings.dataset.manifest_paths = ()
    settings.dataset.folder = dataset_root
    monkeypatch.setattr(dl, "get_settings", lambda: settings)
    entry = SimpleNamespace(
        dataset_id="ravdess",
        dataset_root=dataset_root,
        manifest_path=tmp_path / "missing" / "ravdess.jsonl",
        options={},
    )
    monkeypatch.setattr(
        dl, "load_dataset_registry", lambda settings: {"ravdess": entry}
    )
    monkeypatch.setattr(
        dl,
        "prepare_from_registry_entry",
        lambda **_kwargs: [manifest_path],
    )

    utterances = dl.load_utterances()

    assert utterances is not None
    assert [item.sample_id for item in utterances] == ["ravdess:a.wav", "ravdess:b.wav"]
