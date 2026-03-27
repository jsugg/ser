"""Contract tests for JL-Corpus Hugging Face rows fallback download helpers."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

import ser.data.jl_corpus_downloads as jl_downloads
from ser.data.jl_corpus_downloads import (
    download_jl_corpus_via_hf_rows,
    extract_jl_corpus_audio_src,
    prepare_jl_corpus_from_hf_rows,
    prepare_jl_corpus_from_kaggle,
    sanitize_jl_corpus_index,
)


def test_sanitize_jl_corpus_index_normalizes_and_rejects_empty() -> None:
    """Sanitizer should produce deterministic file stems from rows API indexes."""
    assert sanitize_jl_corpus_index(" female1/angry 10a ") == "female1_angry_10a"
    assert sanitize_jl_corpus_index("...") is None


def test_extract_jl_corpus_audio_src_supports_list_and_dict_shapes() -> None:
    """Audio source resolver should handle both known rows API payload shapes."""
    assert (
        extract_jl_corpus_audio_src([{"src": "https://example.invalid/a.wav"}])
        == "https://example.invalid/a.wav"
    )
    assert (
        extract_jl_corpus_audio_src({"src": "https://example.invalid/b.wav"})
        == "https://example.invalid/b.wav"
    )
    assert extract_jl_corpus_audio_src([]) is None


def test_download_jl_corpus_via_hf_rows_writes_labels_csv(
    tmp_path: Path,
) -> None:
    """Rows fallback should materialize audio files and deterministic labels."""

    def _request_json(url: str, *, headers: dict[str, str] | None = None) -> object:
        del headers
        if "offset=0" in url:
            assert "dataset=CLAPv2%2FJL-Corpus" in url
            assert "config=default" in url
            assert "split=train" in url
            return {
                "num_rows_total": 2,
                "rows": [
                    {
                        "row": {
                            "index": "female1_angry_10a_1",
                            "audio": [{"src": "https://example.invalid/a.wav"}],
                        }
                    },
                    {
                        "row": {
                            "index": "male1_happy_11a_1",
                            "audio": {"src": "https://example.invalid/b.wav"},
                        }
                    },
                ],
            }
        return {"num_rows_total": 2, "rows": []}

    def _download_file(
        *,
        url: str,
        destination_path: Path,
        expected_md5: str | None = None,
        expected_size: int | None = None,
        headers: dict[str, str] | None = None,
    ) -> Path:
        del url, expected_md5, expected_size, headers
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        destination_path.write_bytes(b"wav")
        return destination_path

    def _infer_label_from_path_tokens(path: Path) -> str | None:
        token = path.as_posix().lower()
        if "angry" in token:
            return "angry"
        if "happy" in token:
            return "happy"
        return None

    def _compute_relative_to_dataset_root(*, dataset_root: Path, path: Path) -> str:
        return path.resolve().relative_to(dataset_root.resolve()).as_posix()

    def _write_labels_csv(*, labels_csv_path: Path, labels_by_file: dict[str, str]) -> None:
        labels_csv_path.parent.mkdir(parents=True, exist_ok=True)
        with labels_csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["FileName", "emotion"])
            writer.writeheader()
            for file_name, emotion in sorted(labels_by_file.items()):
                writer.writerow({"FileName": file_name, "emotion": emotion})

    labels_csv_path = tmp_path / "labels.csv"
    stats = download_jl_corpus_via_hf_rows(
        dataset_root=tmp_path,
        labels_csv_path=labels_csv_path,
        rows_api_url="https://datasets-server.huggingface.co/rows",
        dataset_id="CLAPv2/JL-Corpus",
        config="default",
        split="train",
        page_size=100,
        request_json=_request_json,
        download_file=_download_file,
        infer_label_from_path_tokens=_infer_label_from_path_tokens,
        compute_relative_to_dataset_root=_compute_relative_to_dataset_root,
        write_labels_csv=_write_labels_csv,
    )

    assert stats.files_seen == 2
    assert stats.labels_written == 2
    with labels_csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [
        {"FileName": "raw/jl-corpus/female1_angry_10a_1.wav", "emotion": "angry"},
        {"FileName": "raw/jl-corpus/male1_happy_11a_1.wav", "emotion": "happy"},
    ]


def test_download_jl_corpus_via_hf_rows_rejects_non_positive_page_size(
    tmp_path: Path,
) -> None:
    """Rows fallback should fail fast when page size is invalid."""

    def _infer_label(path: Path) -> str | None:
        del path
        return "neutral"

    with pytest.raises(RuntimeError, match="page size must be positive"):
        _ = download_jl_corpus_via_hf_rows(
            dataset_root=tmp_path,
            labels_csv_path=tmp_path / "labels.csv",
            rows_api_url="https://datasets-server.huggingface.co/rows",
            dataset_id="CLAPv2/JL-Corpus",
            config="default",
            split="train",
            page_size=0,
            request_json=lambda *_args, **_kwargs: {},
            download_file=lambda **_kwargs: tmp_path / "noop.wav",
            infer_label_from_path_tokens=_infer_label,
            compute_relative_to_dataset_root=(
                lambda *, dataset_root, path: path.relative_to(dataset_root).as_posix()
            ),
            write_labels_csv=lambda **_kwargs: None,
        )


def test_prepare_jl_corpus_from_hf_rows_writes_manifest(
    tmp_path: Path,
) -> None:
    """Rows fallback helper should write provenance manifest and return artifacts."""
    captured_manifest: dict[str, object] = {}

    def _request_json(url: str, *, headers: dict[str, str] | None = None) -> object:
        del headers
        if "offset=0" in url:
            return {
                "num_rows_total": 1,
                "rows": [
                    {
                        "row": {
                            "index": "female1_angry_10a_1",
                            "audio": {"src": "https://example.invalid/a.wav"},
                        }
                    }
                ],
            }
        return {"num_rows_total": 1, "rows": []}

    def _download_file(
        *,
        url: str,
        destination_path: Path,
        expected_md5: str | None = None,
        expected_size: int | None = None,
        headers: dict[str, str] | None = None,
    ) -> Path:
        del url, expected_md5, expected_size, headers
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        destination_path.write_bytes(b"wav")
        return destination_path

    def _compute_relative_to_dataset_root(*, dataset_root: Path, path: Path) -> str:
        return path.resolve().relative_to(dataset_root.resolve()).as_posix()

    def _write_labels_csv(*, labels_csv_path: Path, labels_by_file: dict[str, str]) -> None:
        labels_csv_path.parent.mkdir(parents=True, exist_ok=True)
        with labels_csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["FileName", "emotion"])
            writer.writeheader()
            for file_name, emotion in sorted(labels_by_file.items()):
                writer.writerow({"FileName": file_name, "emotion": emotion})

    def _write_source_manifest(
        *,
        dataset_root: Path,
        source_manifest_path: Path,
        source_payload: dict[str, object],
        labels_csv_path: Path | None,
        labels_stats: object | None,
    ) -> None:
        assert dataset_root == tmp_path
        assert source_manifest_path == tmp_path / "source_manifest.json"
        assert labels_csv_path == tmp_path / "labels.csv"
        assert labels_stats is not None
        captured_manifest.update(source_payload)

    artifacts = prepare_jl_corpus_from_hf_rows(
        dataset_root=tmp_path,
        fallback_reason="missing credentials",
        labels_file_name="labels.csv",
        source_manifest_file_name="source_manifest.json",
        dataset_id="CLAPv2/JL-Corpus",
        source_url="https://huggingface.co/datasets/CLAPv2/JL-Corpus",
        rows_api_url="https://datasets-server.huggingface.co/rows",
        config="default",
        split="train",
        page_size=100,
        request_json=_request_json,
        download_file=_download_file,
        infer_label_from_path_tokens=(lambda path: "angry" if "angry" in path.as_posix() else None),
        compute_relative_to_dataset_root=_compute_relative_to_dataset_root,
        write_labels_csv=_write_labels_csv,
        write_source_manifest=_write_source_manifest,
    )

    assert artifacts.dataset_root == tmp_path
    assert artifacts.labels_csv_path == tmp_path / "labels.csv"
    assert artifacts.source_manifest_path == tmp_path / "source_manifest.json"
    assert artifacts.files_seen == 1
    assert artifacts.labels_written == 1
    assert captured_manifest == {
        "provider": "huggingface_rows_api",
        "dataset_id": "CLAPv2/JL-Corpus",
        "source_url": "https://huggingface.co/datasets/CLAPv2/JL-Corpus",
        "fallback_reason": "missing credentials",
        "rows_api_url": "https://datasets-server.huggingface.co/rows",
    }


def test_prepare_jl_corpus_from_kaggle_writes_manifest_on_success(
    tmp_path: Path,
) -> None:
    """Kaggle JL-Corpus helper should extract, label, and persist source manifest."""
    captured_manifest: dict[str, object] = {}

    def _download_kaggle_archive(*, dataset_ref: str, destination_path: Path) -> Path:
        assert dataset_ref == "tli725/jl-corpus"
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        destination_path.write_bytes(b"zip")
        return destination_path

    def _ensure_extracted_archive(*, archive_path: Path, extract_root: Path) -> None:
        assert archive_path == tmp_path / "downloads" / "jl-corpus.zip"
        assert extract_root == tmp_path / "raw" / "jl-corpus"
        extract_root.mkdir(parents=True, exist_ok=True)

    def _generate_labels_from_audio_tree(
        *,
        dataset_root: Path,
        search_root: Path,
        labels_csv_path: Path,
        resolver: object,
        extensions: frozenset[str] = frozenset({".wav"}),
    ) -> jl_downloads.JlCorpusDownloadStats:
        del resolver, extensions
        assert dataset_root == tmp_path
        assert search_root == tmp_path / "raw" / "jl-corpus"
        assert labels_csv_path == tmp_path / "labels.csv"
        return jl_downloads.JlCorpusDownloadStats(
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
        labels_stats: object | None,
    ) -> None:
        assert dataset_root == tmp_path
        assert source_manifest_path == tmp_path / "source_manifest.json"
        assert labels_csv_path == tmp_path / "labels.csv"
        assert labels_stats is not None
        captured_manifest.update(source_payload)

    def _prepare_hf_rows_fallback(
        *,
        dataset_root: Path,
        fallback_reason: str,
    ) -> jl_downloads.AutoDownloadArtifacts:
        del dataset_root, fallback_reason
        raise AssertionError("fallback should not run for successful Kaggle downloads")

    artifacts = prepare_jl_corpus_from_kaggle(
        dataset_root=tmp_path,
        dataset_ref="tli725/jl-corpus",
        labels_file_name="labels.csv",
        source_manifest_file_name="source_manifest.json",
        download_kaggle_archive=_download_kaggle_archive,
        ensure_extracted_archive=_ensure_extracted_archive,
        generate_labels_from_audio_tree=_generate_labels_from_audio_tree,
        infer_label_from_path_tokens=lambda path: "neutral",
        write_source_manifest=_write_source_manifest,
        prepare_hf_rows_fallback=_prepare_hf_rows_fallback,
        logger_warning=lambda msg, *args: None,
    )

    assert captured_manifest == {
        "provider": "kaggle",
        "dataset_ref": "tli725/jl-corpus",
        "archive_path": str(tmp_path / "downloads" / "jl-corpus.zip"),
    }
    assert artifacts.dataset_root == tmp_path
    assert artifacts.labels_csv_path == tmp_path / "labels.csv"
    assert artifacts.source_manifest_path == tmp_path / "source_manifest.json"
    assert artifacts.files_seen == 4
    assert artifacts.labels_written == 4


def test_prepare_jl_corpus_from_kaggle_falls_back_when_download_unavailable(
    tmp_path: Path,
) -> None:
    """Kaggle JL-Corpus helper should route to fallback and emit warning context."""
    warning_calls: list[tuple[str, tuple[object, ...]]] = []

    def _prepare_hf_rows_fallback(
        *,
        dataset_root: Path,
        fallback_reason: str,
    ) -> jl_downloads.AutoDownloadArtifacts:
        assert dataset_root == tmp_path
        assert "missing credentials" in fallback_reason
        source_manifest_path = dataset_root / "source_manifest.json"
        source_manifest_path.write_text("{}", encoding="utf-8")
        labels_csv_path = dataset_root / "labels.csv"
        labels_csv_path.write_text("FileName,emotion\n", encoding="utf-8")
        return jl_downloads.AutoDownloadArtifacts(
            dataset_root=dataset_root,
            labels_csv_path=labels_csv_path,
            audio_base_dir=dataset_root,
            source_manifest_path=source_manifest_path,
            files_seen=2,
            labels_written=2,
        )

    artifacts = prepare_jl_corpus_from_kaggle(
        dataset_root=tmp_path,
        dataset_ref="tli725/jl-corpus",
        labels_file_name="labels.csv",
        source_manifest_file_name="source_manifest.json",
        download_kaggle_archive=lambda **_kwargs: (_ for _ in ()).throw(
            RuntimeError("missing credentials")
        ),
        ensure_extracted_archive=lambda **_kwargs: None,
        generate_labels_from_audio_tree=lambda **_kwargs: (
            jl_downloads.JlCorpusDownloadStats(
                files_seen=0,
                labels_written=0,
                dropped_files=0,
                duplicate_conflicts=0,
            )
        ),
        infer_label_from_path_tokens=lambda path: "neutral",
        write_source_manifest=lambda **_kwargs: None,
        prepare_hf_rows_fallback=_prepare_hf_rows_fallback,
        logger_warning=lambda msg, *args: warning_calls.append((msg, args)),
    )

    assert artifacts.dataset_root == tmp_path
    assert artifacts.labels_written == 2
    assert warning_calls
    message, args = warning_calls[0]
    assert "falling back to public Hugging Face rows API" in message
    assert args and "missing credentials" in str(args[0])
