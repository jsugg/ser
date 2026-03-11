"""Tests for MSP-Podcast mirror label generation helpers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ser.data import msp_podcast_mirror
from ser.data.msp_podcast_mirror import generate_msp_labels_csv_from_metadata_jsonl


def test_generate_msp_labels_csv_from_metadata_jsonl(tmp_path: Path) -> None:
    """Label generation should emit adapter-compatible labels with deterministic policy."""
    metadata_jsonl_path = tmp_path / "metadata.jsonl"
    labels_csv_path = tmp_path / "labels.csv"
    rows = [
        {
            "audio_relpath": "session/a.wav",
            "angry": 0.6,
            "sad": 0.2,
            "major_emotion": "sad",
        },
        {
            "audio_relpath": "session/b.wav",
            "major_emotion": "fear",
        },
        {
            "audio_relpath": "session/c.wav",
            "surprise": 0.9,
            "fear": 0.1,
        },
        {
            "audio_relpath": "session/d.wav",
            "major_emotion": "",
        },
        {
            "audio_relpath": "",
            "angry": 1.0,
        },
    ]
    metadata_jsonl_path.write_text(
        "".join(f"{json.dumps(row)}\n" for row in rows),
        encoding="utf-8",
    )

    labels_written, dropped_rows = generate_msp_labels_csv_from_metadata_jsonl(
        metadata_jsonl_path=metadata_jsonl_path,
        labels_csv_path=labels_csv_path,
    )

    assert labels_written == 3
    assert dropped_rows == 2
    with labels_csv_path.open("r", encoding="utf-8", newline="") as handle:
        parsed = list(csv.DictReader(handle))
    assert parsed == [
        {"FileName": "session/a.wav", "emotion": "angry"},
        {"FileName": "session/b.wav", "emotion": "fearful"},
        {"FileName": "session/c.wav", "emotion": "surprised"},
    ]


def test_generate_msp_labels_csv_deduplicates_and_sorts_by_filename(
    tmp_path: Path,
) -> None:
    """Duplicate filenames should be dropped and output should be deterministic."""
    metadata_jsonl_path = tmp_path / "metadata.jsonl"
    labels_csv_path = tmp_path / "labels.csv"
    rows = [
        {"audio_relpath": "session/z.wav", "major_emotion": "sad"},
        {"audio_relpath": "session/a.wav", "major_emotion": "happy"},
        {"audio_relpath": "session/z.wav", "major_emotion": "angry"},
    ]
    metadata_jsonl_path.write_text(
        "".join(f"{json.dumps(row)}\n" for row in rows),
        encoding="utf-8",
    )

    labels_written, dropped_rows = generate_msp_labels_csv_from_metadata_jsonl(
        metadata_jsonl_path=metadata_jsonl_path,
        labels_csv_path=labels_csv_path,
    )

    assert labels_written == 2
    assert dropped_rows == 1
    with labels_csv_path.open("r", encoding="utf-8", newline="") as handle:
        parsed = list(csv.DictReader(handle))
    assert parsed == [
        {"FileName": "session/a.wav", "emotion": "happy"},
        {"FileName": "session/z.wav", "emotion": "sad"},
    ]


def test_download_snapshot_fails_fast_when_disk_space_is_insufficient(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Snapshot download should fail before transfer when free space is clearly too low."""

    class _Sibling:
        def __init__(self, rfilename: str, size: int) -> None:
            self.rfilename = rfilename
            self.size = size

    class _Info:
        siblings = [_Sibling("data/train-0001.parquet", 1000)]
        sha = "abcdef1"

    class _HfApi:
        def __init__(self, token: str | None = None) -> None:
            self._token = token

        def dataset_info(
            self,
            *,
            repo_id: str,
            revision: str,
            files_metadata: bool,
        ) -> _Info:
            del repo_id, revision, files_metadata
            return _Info()

    def _should_not_download(**kwargs: object) -> None:
        del kwargs
        raise AssertionError("snapshot_download should not run on disk preflight failure")

    monkeypatch.setattr(
        msp_podcast_mirror,
        "_load_hf_clients",
        lambda: (_HfApi, _should_not_download, object()),
    )
    monkeypatch.setattr(
        msp_podcast_mirror.shutil,
        "disk_usage",
        lambda path: SimpleNamespace(
            total=10_000,
            used=9_900,
            free=100,
        ),
    )

    with pytest.raises(RuntimeError, match="insufficient disk space"):
        msp_podcast_mirror._download_snapshot(
            repo_id="AbstractTTS/PODCAST",
            revision="main",
            repo_dir=tmp_path / "repo",
            max_workers=2,
            token=None,
        )
