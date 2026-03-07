"""Tests for provider-sourced dataset preparation helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from ser.data import provider_dataset_preparation as preparation


@dataclass(frozen=True, slots=True)
class _Asset:
    name: str
    download_url: str
    size: int | None


@dataclass(frozen=True, slots=True)
class _Stats:
    files_seen: int
    labels_written: int
    dropped_files: int
    duplicate_conflicts: int


def test_build_source_manifest_payload_builds_expected_fields(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Manifest payload builder should normalize artifact and stats fields."""
    monkeypatch.setattr(preparation.time, "time", lambda: 123.5)
    payload = preparation.build_source_manifest_payload(
        dataset_root=tmp_path,
        source_payload={"provider": "zenodo", "record_id": "42"},
        labels_csv_path=tmp_path / "labels.csv",
        labels_stats=_Stats(
            files_seen=10,
            labels_written=8,
            dropped_files=2,
            duplicate_conflicts=1,
        ),
    )

    assert payload == {
        "generated_at_unix": 123.5,
        "source": {"provider": "zenodo", "record_id": "42"},
        "artifacts": {
            "dataset_root": str(tmp_path),
            "labels_csv_path": str(tmp_path / "labels.csv"),
        },
        "stats": {
            "files_seen": 10,
            "labels_written": 8,
            "dropped_files": 2,
            "duplicate_conflicts": 1,
        },
    }


def test_write_source_manifest_persists_sorted_json_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Manifest writer should persist deterministic sorted JSON with newline."""
    monkeypatch.setattr(preparation.time, "time", lambda: 77.0)
    source_manifest_path = tmp_path / "source_manifest.json"

    preparation.write_source_manifest(
        dataset_root=tmp_path,
        source_manifest_path=source_manifest_path,
        source_payload={"provider": "google-drive", "folder_url": "https://x"},
        labels_csv_path=None,
        labels_stats=None,
    )

    raw = source_manifest_path.read_text(encoding="utf-8")
    assert raw.endswith("\n")
    assert json.loads(raw) == {
        "generated_at_unix": 77.0,
        "source": {"provider": "google-drive", "folder_url": "https://x"},
        "artifacts": {
            "dataset_root": str(tmp_path),
            "labels_csv_path": None,
        },
        "stats": {},
    }


def test_generate_labels_from_audio_tree_builds_rows_and_counts(
    tmp_path: Path,
) -> None:
    """Label helper should keep deterministic rows and duplicate/drop counters."""
    audio_root = tmp_path / "raw"
    audio_root.mkdir(parents=True, exist_ok=True)
    angry_path = audio_root / "speaker1_angry.wav"
    happy_path = audio_root / "speaker2_happy.wav"
    angry_path.write_bytes(b"fake")
    happy_path.write_bytes(b"fake")
    duplicate_calls = {"speaker2_happy.wav": 0}
    captured_rows: dict[str, str] = {}

    def _collect_audio_files(
        *,
        search_root: Path,
        extensions: frozenset[str],
    ) -> list[Path]:
        assert search_root == audio_root
        assert extensions == frozenset({".wav"})
        return [angry_path, happy_path, happy_path]

    def _resolver(path: Path) -> str | None:
        if path.name == "speaker1_angry.wav":
            return "angry"
        if path.name == "speaker2_happy.wav":
            duplicate_calls[path.name] += 1
            return "happy" if duplicate_calls[path.name] == 1 else "sad"
        return None

    stats = preparation.generate_labels_from_audio_tree(
        dataset_root=tmp_path,
        search_root=audio_root,
        labels_csv_path=tmp_path / "labels.csv",
        resolver=_resolver,
        collect_audio_files=_collect_audio_files,
        compute_relative_to_dataset_root=lambda *, dataset_root, path: path.relative_to(
            dataset_root
        ).as_posix(),
        write_labels_csv=lambda *, labels_csv_path, labels_by_file: (
            captured_rows.update(labels_by_file)
        ),
        stats_factory=lambda *, files_seen, labels_written, dropped_files, duplicate_conflicts: (
            _Stats(
                files_seen=files_seen,
                labels_written=labels_written,
                dropped_files=dropped_files,
                duplicate_conflicts=duplicate_conflicts,
            )
        ),
    )

    assert stats == _Stats(
        files_seen=3,
        labels_written=2,
        dropped_files=1,
        duplicate_conflicts=1,
    )
    assert captured_rows == {
        "raw/speaker1_angry.wav": "angry",
        "raw/speaker2_happy.wav": "happy",
    }


def test_prepare_pavoque_from_github_release_downloads_filtered_assets(
    tmp_path: Path,
) -> None:
    """PAVOQUE helper should keep expected assets and persist source payload."""
    downloaded_targets: list[Path] = []
    captured_source_payload: dict[str, object] = {}

    def _read_assets(
        *,
        owner: str,
        repo: str,
    ) -> tuple[str, list[_Asset]]:
        assert owner == "marytts"
        assert repo == "pavoque-data"
        return (
            "v1.0.1",
            [
                _Asset(
                    name="pavoque-angry.flac",
                    download_url="https://example.org/a.flac",
                    size=100,
                ),
                _Asset(
                    name="pavoque-meta.yaml",
                    download_url="https://example.org/meta.yaml",
                    size=12,
                ),
                _Asset(
                    name="ignore.zip",
                    download_url="https://example.org/ignore.zip",
                    size=42,
                ),
            ],
        )

    def _download_file(
        *,
        url: str,
        destination_path: Path,
        expected_md5: str | None = None,
        expected_size: int | None = None,
        headers: dict[str, str] | None = None,
    ) -> Path:
        del url, expected_md5, expected_size, headers
        downloaded_targets.append(destination_path)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        destination_path.write_bytes(b"content")
        return destination_path

    def _generate_labels(
        *,
        dataset_root: Path,
        search_root: Path,
        labels_csv_path: Path,
        resolver: object,
        extensions: frozenset[str] = frozenset({".wav"}),
    ) -> _Stats:
        del resolver
        assert dataset_root == tmp_path
        assert search_root == tmp_path / "raw" / "pavoque"
        assert labels_csv_path == tmp_path / "labels.csv"
        assert extensions == frozenset({".wav", ".flac"})
        return _Stats(
            files_seen=3,
            labels_written=2,
            dropped_files=1,
            duplicate_conflicts=0,
        )

    def _write_source_manifest(
        *,
        dataset_root: Path,
        source_manifest_path: Path,
        source_payload: dict[str, object],
        labels_csv_path: Path | None,
        labels_stats: preparation.GeneratedLabelsStatsLike | None,
    ) -> None:
        assert dataset_root == tmp_path
        assert source_manifest_path == tmp_path / "source_manifest.json"
        assert labels_csv_path == tmp_path / "labels.csv"
        assert labels_stats is not None
        captured_source_payload.update(source_payload)

    artifacts = preparation.prepare_pavoque_from_github_release(
        dataset_root=tmp_path,
        owner="marytts",
        repo="pavoque-data",
        labels_file_name="labels.csv",
        source_manifest_file_name="source_manifest.json",
        read_github_latest_release_assets=_read_assets,
        download_file=_download_file,
        generate_labels_from_audio_tree=_generate_labels,
        infer_label_from_path_tokens=lambda _path: "angry",
        write_source_manifest=_write_source_manifest,
    )

    assert [path.name for path in downloaded_targets] == [
        "pavoque-angry.flac",
        "pavoque-meta.yaml",
    ]
    assert captured_source_payload == {
        "provider": "github-release",
        "owner": "marytts",
        "repo": "pavoque-data",
        "tag_name": "v1.0.1",
        "audio_assets": ["pavoque-angry.flac"],
        "metadata_assets": ["pavoque-meta.yaml"],
    }
    assert artifacts.dataset_root == tmp_path
    assert artifacts.labels_csv_path == tmp_path / "labels.csv"
    assert artifacts.source_manifest_path == tmp_path / "source_manifest.json"
    assert artifacts.files_seen == 3
    assert artifacts.labels_written == 2


def test_prepare_pavoque_from_github_release_requires_matching_assets(
    tmp_path: Path,
) -> None:
    """PAVOQUE helper should fail fast when no expected assets are available."""

    with pytest.raises(
        RuntimeError,
        match="does not expose expected pavoque-\\* assets",
    ):

        def _read_assets_empty(
            *,
            owner: str,
            repo: str,
        ) -> tuple[str, list[_Asset]]:
            del owner, repo
            return (
                "v0.0.1",
                [_Asset(name="readme.txt", download_url="https://x", size=1)],
            )

        preparation.prepare_pavoque_from_github_release(
            dataset_root=tmp_path,
            owner="marytts",
            repo="pavoque-data",
            labels_file_name="labels.csv",
            source_manifest_file_name="source_manifest.json",
            read_github_latest_release_assets=_read_assets_empty,
            download_file=lambda **_kwargs: tmp_path / "never",
            generate_labels_from_audio_tree=lambda **_kwargs: _Stats(
                files_seen=0,
                labels_written=0,
                dropped_files=0,
                duplicate_conflicts=0,
            ),
            infer_label_from_path_tokens=lambda _path: None,
            write_source_manifest=lambda **_kwargs: None,
        )


def test_prepare_coraa_ser_from_google_drive_generates_labels_and_manifest(
    tmp_path: Path,
) -> None:
    """CORAA helper should orchestrate download, extraction, and manifest content."""
    captured_source_payload: dict[str, object] = {}
    captured_download_root: Path | None = None

    def _download_folder(*, folder_url: str, destination_root: Path) -> list[Path]:
        nonlocal captured_download_root
        assert folder_url == "https://drive.example/folder"
        captured_download_root = destination_root
        destination_root.mkdir(parents=True, exist_ok=True)
        archive_path = destination_root / "set.zip"
        archive_path.write_bytes(b"zip")
        return [archive_path]

    def _extract_archives(*, search_root: Path, extract_root: Path) -> list[Path]:
        assert search_root == tmp_path / "downloads" / "coraa-ser"
        assert extract_root == tmp_path / "raw" / "coraa-ser"
        extract_root.mkdir(parents=True, exist_ok=True)
        archive_dir = extract_root / "set"
        archive_dir.mkdir(parents=True, exist_ok=True)
        return [archive_dir]

    def _generate_labels(
        *,
        dataset_root: Path,
        search_root: Path,
        labels_csv_path: Path,
        resolver: object,
        extensions: frozenset[str] = frozenset({".wav"}),
    ) -> _Stats:
        del resolver, extensions
        assert dataset_root == tmp_path
        assert search_root == tmp_path / "raw" / "coraa-ser"
        assert labels_csv_path == tmp_path / "labels.csv"
        return _Stats(
            files_seen=4,
            labels_written=4,
            dropped_files=0,
            duplicate_conflicts=0,
        )

    def _write_source_manifest(
        *,
        dataset_root: Path,
        source_manifest_path: Path,
        source_payload: dict[str, object],
        labels_csv_path: Path | None,
        labels_stats: preparation.GeneratedLabelsStatsLike | None,
    ) -> None:
        assert dataset_root == tmp_path
        assert source_manifest_path == tmp_path / "source_manifest.json"
        assert labels_csv_path == tmp_path / "labels.csv"
        assert labels_stats is not None
        captured_source_payload.update(source_payload)

    artifacts = preparation.prepare_coraa_ser_from_google_drive(
        dataset_root=tmp_path,
        folder_url="https://drive.example/folder",
        label_semantics="neutral_vs_non_neutral_by_gender",
        labels_file_name="labels.csv",
        source_manifest_file_name="source_manifest.json",
        download_google_drive_folder=_download_folder,
        extract_archives_from_tree=_extract_archives,
        generate_labels_from_audio_tree=_generate_labels,
        infer_coraa_ser_label=lambda _path: "neutral",
        write_source_manifest=_write_source_manifest,
    )

    assert captured_download_root == tmp_path / "downloads" / "coraa-ser"
    assert captured_source_payload == {
        "provider": "google-drive",
        "folder_url": "https://drive.example/folder",
        "downloaded_files_count": 1,
        "extracted_archives": [str(tmp_path / "raw" / "coraa-ser" / "set")],
        "label_semantics": "neutral_vs_non_neutral_by_gender",
    }
    assert artifacts.dataset_root == tmp_path
    assert artifacts.labels_csv_path == tmp_path / "labels.csv"
    assert artifacts.source_manifest_path == tmp_path / "source_manifest.json"
    assert artifacts.files_seen == 4
    assert artifacts.labels_written == 4
