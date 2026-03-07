"""Contract tests for Zenodo dataset preparation helpers."""

from __future__ import annotations

import csv
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from ser.data import zenodo_downloads


@dataclass(frozen=True)
class _Stats:
    files_seen: int
    labels_written: int
    dropped_files: int
    duplicate_conflicts: int


def test_parse_zenodo_md5_parses_prefixed_checksum() -> None:
    """MD5 parser should normalize Zenodo checksum tokens."""
    assert zenodo_downloads.parse_zenodo_md5("md5:ABC123") == "abc123"
    assert zenodo_downloads.parse_zenodo_md5(" sha256:abc ") is None
    assert zenodo_downloads.parse_zenodo_md5(None) is None


def test_read_zenodo_file_metadata_extracts_download_fields() -> None:
    """Metadata reader should resolve URL/checksum/size for one file key."""

    payload: object = {
        "files": [
            {
                "key": "dataset.zip",
                "links": {"self": "https://zenodo.org/record/file"},
                "checksum": "md5:abcd1234",
                "size": 123,
            }
        ]
    }
    metadata = zenodo_downloads.read_zenodo_file_metadata(
        record_id="42",
        file_key="dataset.zip",
        request_json=lambda url, headers=None: payload,
    )

    assert metadata == zenodo_downloads.ZenodoFileMetadata(
        key="dataset.zip",
        url="https://zenodo.org/record/file",
        md5="abcd1234",
        size=123,
    )


def test_download_zenodo_archive_delegates_to_download_file(tmp_path: Path) -> None:
    """Archive downloader should map Zenodo metadata into download parameters."""
    captured: dict[str, object] = {}

    def _download_file(
        *,
        url: str,
        destination_path: Path,
        expected_md5: str | None = None,
        expected_size: int | None = None,
        headers: dict[str, str] | None = None,
    ) -> Path:
        del headers
        captured["url"] = url
        captured["destination_path"] = destination_path
        captured["expected_md5"] = expected_md5
        captured["expected_size"] = expected_size
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        destination_path.write_bytes(b"ok")
        return destination_path

    archive_path = zenodo_downloads.download_zenodo_archive(
        dataset_root=tmp_path,
        record_id="42",
        file_key="dataset.zip",
        request_json=lambda url, headers=None: {
            "files": [
                {
                    "key": "dataset.zip",
                    "links": {"self": "https://zenodo.org/record/file"},
                    "checksum": "md5:abcd1234",
                    "size": 456,
                }
            ]
        },
        download_file=_download_file,
    )

    assert archive_path == tmp_path / "downloads" / "dataset.zip"
    assert captured["url"] == "https://zenodo.org/record/file"
    assert captured["destination_path"] == tmp_path / "downloads" / "dataset.zip"
    assert captured["expected_md5"] == "abcd1234"
    assert captured["expected_size"] == 456


def test_prepare_ravdess_from_zenodo_writes_manifest_and_counts_wavs(
    tmp_path: Path,
) -> None:
    """RAVDESS helper should write source manifest and count extracted wav files."""
    captured: dict[str, object] = {}

    def _download_zenodo_archive(
        *,
        dataset_root: Path,
        record_id: str,
        file_key: str,
    ) -> Path:
        assert record_id == "1188976"
        assert file_key == "Audio_Speech_Actors_01-24.zip"
        archive_path = dataset_root / "downloads" / file_key
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        archive_path.write_bytes(b"archive")
        return archive_path

    def _ensure_extracted_archive(*, archive_path: Path, extract_root: Path) -> None:
        del archive_path
        (extract_root / "raw" / "ravdess").mkdir(parents=True, exist_ok=True)
        (extract_root / "raw" / "ravdess" / "a.wav").write_bytes(b"wav")
        (extract_root / "raw" / "ravdess" / "b.wav").write_bytes(b"wav")

    def _collect_wav_files(search_root: Path) -> list[Path]:
        return sorted(search_root.rglob("*.wav"))

    def _write_source_manifest(
        *,
        dataset_root: Path,
        source_manifest_path: Path,
        source_payload: dict[str, object],
        labels_csv_path: Path | None,
        labels_stats: zenodo_downloads.GeneratedLabelsStatsLike | None,
    ) -> None:
        del dataset_root
        captured["source_manifest_path"] = source_manifest_path
        captured["source_payload"] = source_payload
        captured["labels_csv_path"] = labels_csv_path
        captured["labels_stats"] = labels_stats

    artifacts = zenodo_downloads.prepare_ravdess_from_zenodo(
        dataset_root=tmp_path,
        record_id="1188976",
        file_key="Audio_Speech_Actors_01-24.zip",
        source_manifest_file_name="source_manifest.json",
        download_zenodo_archive=_download_zenodo_archive,
        ensure_extracted_archive=_ensure_extracted_archive,
        collect_wav_files=_collect_wav_files,
        write_source_manifest=_write_source_manifest,
    )

    assert artifacts.dataset_root == tmp_path
    assert artifacts.labels_csv_path is None
    assert artifacts.audio_base_dir is None
    assert artifacts.files_seen == 2
    assert artifacts.labels_written == 0
    assert captured["labels_csv_path"] is None
    assert captured["labels_stats"] is None
    payload = captured["source_payload"]
    assert isinstance(payload, dict)
    assert payload["provider"] == "zenodo"
    assert payload["record_id"] == "1188976"


def test_prepare_cafe_from_zenodo_downloads_archives_and_generates_labels(
    tmp_path: Path,
) -> None:
    """CaFE helper should download all archives and emit labels/manifest payload."""
    downloaded_keys: list[str] = []
    extracted_archives: list[Path] = []
    captured_manifest: dict[str, object] = {}

    def _download_zenodo_archive(
        *,
        dataset_root: Path,
        record_id: str,
        file_key: str,
    ) -> Path:
        assert dataset_root == tmp_path
        assert record_id == "1478765"
        downloaded_keys.append(file_key)
        archive_path = dataset_root / "downloads" / file_key
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        archive_path.write_bytes(b"archive")
        return archive_path

    def _ensure_extracted_archive(*, archive_path: Path, extract_root: Path) -> None:
        extracted_archives.append(archive_path)
        extract_root.mkdir(parents=True, exist_ok=True)

    def _generate_labels_from_audio_tree(
        *,
        dataset_root: Path,
        search_root: Path,
        labels_csv_path: Path,
        resolver: Callable[[Path], str | None],
    ) -> _Stats:
        del resolver
        assert dataset_root == tmp_path
        assert search_root == tmp_path / "raw" / "cafe"
        assert labels_csv_path == tmp_path / "labels.csv"
        return _Stats(
            files_seen=11,
            labels_written=9,
            dropped_files=2,
            duplicate_conflicts=0,
        )

    def _write_source_manifest(
        *,
        dataset_root: Path,
        source_manifest_path: Path,
        source_payload: dict[str, object],
        labels_csv_path: Path | None,
        labels_stats: zenodo_downloads.GeneratedLabelsStatsLike | None,
    ) -> None:
        del dataset_root
        captured_manifest["source_manifest_path"] = source_manifest_path
        captured_manifest["source_payload"] = source_payload
        captured_manifest["labels_csv_path"] = labels_csv_path
        captured_manifest["labels_stats"] = labels_stats

    artifacts = zenodo_downloads.prepare_cafe_from_zenodo(
        dataset_root=tmp_path,
        record_id="1478765",
        archive_keys=("CaFE_192k_1.zip", "CaFE_192k_2.zip"),
        labels_file_name="labels.csv",
        source_manifest_file_name="source_manifest.json",
        download_zenodo_archive=_download_zenodo_archive,
        ensure_extracted_archive=_ensure_extracted_archive,
        generate_labels_from_audio_tree=_generate_labels_from_audio_tree,
        infer_label_from_path_tokens=lambda _path: "neutral",
        write_source_manifest=_write_source_manifest,
    )

    assert downloaded_keys == ["CaFE_192k_1.zip", "CaFE_192k_2.zip"]
    assert [path.name for path in extracted_archives] == downloaded_keys
    assert artifacts.files_seen == 11
    assert artifacts.labels_written == 9
    assert artifacts.labels_csv_path == tmp_path / "labels.csv"
    payload = captured_manifest["source_payload"]
    assert isinstance(payload, dict)
    assert payload["record_id"] == "1478765"
    assert payload["file_keys"] == ["CaFE_192k_1.zip", "CaFE_192k_2.zip"]


def test_prepare_emodb_2_from_zenodo_generates_labels_from_metadata(
    tmp_path: Path,
) -> None:
    """EmoDB helper should resolve metadata rows into deterministic labels."""
    captured_manifest: dict[str, object] = {}
    captured_labels: dict[str, str] = {}

    def _download_zenodo_archive(
        *,
        dataset_root: Path,
        record_id: str,
        file_key: str,
    ) -> Path:
        assert dataset_root == tmp_path
        assert record_id == "17651657"
        assert file_key == "emodb_2.0.zip"
        archive_path = dataset_root / "downloads" / file_key
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        archive_path.write_bytes(b"archive")
        return archive_path

    def _ensure_extracted_archive(*, archive_path: Path, extract_root: Path) -> None:
        del archive_path
        wav_root = extract_root / "wav" / "speaker_001"
        wav_root.mkdir(parents=True, exist_ok=True)
        (wav_root / "utt001.wav").write_bytes(b"wav")
        (wav_root / "utt002.wav").write_bytes(b"wav")
        metadata_path = extract_root / "db.emotion.categories.ambiguous.csv"
        metadata_path.write_text(
            ("file,emotion\nutt001,anger\nutt002.wav,sadness\nutt003,joy\n,anger\n"),
            encoding="utf-8",
        )

    def _compute_relative_to_dataset_root(*, dataset_root: Path, path: Path) -> str:
        return path.resolve().relative_to(dataset_root.resolve()).as_posix()

    def _write_labels_csv(
        *, labels_csv_path: Path, labels_by_file: dict[str, str]
    ) -> None:
        captured_labels.update(labels_by_file)
        labels_csv_path.parent.mkdir(parents=True, exist_ok=True)
        labels_csv_path.write_text("written", encoding="utf-8")

    def _write_source_manifest(
        *,
        dataset_root: Path,
        source_manifest_path: Path,
        source_payload: dict[str, object],
        labels_csv_path: Path | None,
        labels_stats: zenodo_downloads.GeneratedLabelsStatsLike | None,
    ) -> None:
        del dataset_root
        captured_manifest["source_manifest_path"] = source_manifest_path
        captured_manifest["source_payload"] = source_payload
        captured_manifest["labels_csv_path"] = labels_csv_path
        captured_manifest["labels_stats"] = labels_stats

    artifacts = zenodo_downloads.prepare_emodb_2_from_zenodo(
        dataset_root=tmp_path,
        record_id="17651657",
        file_key="emodb_2.0.zip",
        labels_file_name="labels.csv",
        source_manifest_file_name="source_manifest.json",
        emodb_label_map={"anger": "angry", "sadness": "sad", "joy": "happy"},
        download_zenodo_archive=_download_zenodo_archive,
        ensure_extracted_archive=_ensure_extracted_archive,
        compute_relative_to_dataset_root=_compute_relative_to_dataset_root,
        write_labels_csv=_write_labels_csv,
        write_source_manifest=_write_source_manifest,
    )

    assert artifacts.dataset_root == tmp_path
    assert artifacts.labels_csv_path == tmp_path / "labels.csv"
    assert artifacts.files_seen == 4
    assert artifacts.labels_written == 2
    assert captured_labels == {
        "raw/emodb-2.0/wav/speaker_001/utt001.wav": "angry",
        "raw/emodb-2.0/wav/speaker_001/utt002.wav": "sad",
    }
    labels_stats = cast(
        zenodo_downloads.GeneratedLabelsStatsLike,
        captured_manifest["labels_stats"],
    )
    assert labels_stats is not None
    assert labels_stats.files_seen == 4
    assert labels_stats.labels_written == 2
    assert labels_stats.dropped_files == 2
    payload = captured_manifest["source_payload"]
    assert isinstance(payload, dict)
    assert payload["record_id"] == "17651657"
    assert payload["file_key"] == "emodb_2.0.zip"


def test_prepare_spanish_meacorpus_2023_from_zenodo_writes_manifest(
    tmp_path: Path,
) -> None:
    """Spanish MEACorpus helper should stabilize metadata path and emit manifest."""
    captured_manifest: dict[str, object] = {}
    copied_paths: list[tuple[Path, Path]] = []

    def _download_zenodo_archive(
        *,
        dataset_root: Path,
        record_id: str,
        file_key: str,
    ) -> Path:
        assert dataset_root == tmp_path
        assert record_id == "18606423"
        assert file_key == "spanish-meacorpus-2023-dataset.csv"
        metadata_path = dataset_root / "downloads" / file_key
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text("filename,label\nx.wav,anger\n", encoding="utf-8")
        return metadata_path

    def _copy_file(source_path: Path, destination_path: Path) -> object:
        copied_paths.append((source_path, destination_path))
        destination_path.write_text(
            source_path.read_text(encoding="utf-8"), encoding="utf-8"
        )
        return destination_path

    def _generate_labels_from_metadata_csv(
        *,
        dataset_root: Path,
        metadata_csv_path: Path,
        labels_csv_path: Path,
        audio_search_roots: tuple[Path, ...],
        file_name_keys: tuple[str, ...],
        label_keys: tuple[str, ...],
        label_resolver: Callable[[str], str | None],
    ) -> _Stats:
        assert dataset_root == tmp_path
        assert (
            metadata_csv_path
            == tmp_path / "metadata" / "spanish-meacorpus-2023-dataset.csv"
        )
        assert labels_csv_path == tmp_path / "labels.csv"
        assert audio_search_roots == (tmp_path / "raw" / "spanish-meacorpus-2023",)
        assert file_name_keys == ("filename", "file_name", "FileName")
        assert label_keys == ("label", "emotion", "EmoClass")
        assert label_resolver("anger") == "angry"
        assert label_resolver("JOY") == "happy"
        assert label_resolver("surprise") is None
        return _Stats(
            files_seen=9,
            labels_written=7,
            dropped_files=2,
            duplicate_conflicts=0,
        )

    def _write_source_manifest(
        *,
        dataset_root: Path,
        source_manifest_path: Path,
        source_payload: dict[str, object],
        labels_csv_path: Path | None,
        labels_stats: zenodo_downloads.GeneratedLabelsStatsLike | None,
    ) -> None:
        del dataset_root
        captured_manifest["source_manifest_path"] = source_manifest_path
        captured_manifest["source_payload"] = source_payload
        captured_manifest["labels_csv_path"] = labels_csv_path
        captured_manifest["labels_stats"] = labels_stats

    artifacts = zenodo_downloads.prepare_spanish_meacorpus_2023_from_zenodo(
        dataset_root=tmp_path,
        record_id="18606423",
        metadata_key="spanish-meacorpus-2023-dataset.csv",
        labels_file_name="labels.csv",
        source_manifest_file_name="source_manifest.json",
        download_zenodo_archive=_download_zenodo_archive,
        copy_file=_copy_file,
        generate_labels_from_metadata_csv=_generate_labels_from_metadata_csv,
        write_source_manifest=_write_source_manifest,
    )

    assert copied_paths == [
        (
            tmp_path / "downloads" / "spanish-meacorpus-2023-dataset.csv",
            tmp_path / "metadata" / "spanish-meacorpus-2023-dataset.csv",
        )
    ]
    assert artifacts.dataset_root == tmp_path
    assert artifacts.labels_csv_path == tmp_path / "labels.csv"
    assert artifacts.files_seen == 9
    assert artifacts.labels_written == 7
    payload = captured_manifest["source_payload"]
    assert isinstance(payload, dict)
    assert payload["record_id"] == "18606423"
    assert payload["metadata_key"] == "spanish-meacorpus-2023-dataset.csv"


def test_generate_labels_from_metadata_csv_keeps_rows_with_present_audio(
    tmp_path: Path,
) -> None:
    """Metadata label generator should keep only rows backed by local audio clips."""
    metadata_csv_path = tmp_path / "metadata.csv"
    metadata_csv_path.write_text(
        "filename,label\nclip_a.wav,anger\nclip_missing.wav,joy\n",
        encoding="utf-8",
    )
    audio_root = tmp_path / "raw" / "spanish-meacorpus-2023"
    audio_root.mkdir(parents=True, exist_ok=True)
    (audio_root / "clip_a.wav").write_bytes(b"wav")
    labels_csv_path = tmp_path / "labels.csv"

    captured_labels: dict[str, str] = {}

    def _compute_relative_to_dataset_root(*, dataset_root: Path, path: Path) -> str:
        return path.resolve().relative_to(dataset_root.resolve()).as_posix()

    def _write_labels_csv(
        *, labels_csv_path: Path, labels_by_file: dict[str, str]
    ) -> None:
        captured_labels.update(labels_by_file)
        labels_csv_path.parent.mkdir(parents=True, exist_ok=True)
        with labels_csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["FileName", "emotion"])
            writer.writeheader()
            for file_name, label in sorted(labels_by_file.items()):
                writer.writerow({"FileName": file_name, "emotion": label})

    stats = zenodo_downloads.generate_labels_from_metadata_csv(
        dataset_root=tmp_path,
        metadata_csv_path=metadata_csv_path,
        labels_csv_path=labels_csv_path,
        audio_search_roots=(audio_root,),
        file_name_keys=("filename",),
        label_keys=("label",),
        label_resolver=lambda value: {"anger": "angry", "joy": "happy"}.get(value),
        compute_relative_to_dataset_root=_compute_relative_to_dataset_root,
        write_labels_csv=_write_labels_csv,
    )

    assert stats.files_seen == 2
    assert stats.labels_written == 1
    assert stats.dropped_files == 1
    assert stats.duplicate_conflicts == 0
    assert captured_labels == {"raw/spanish-meacorpus-2023/clip_a.wav": "angry"}
