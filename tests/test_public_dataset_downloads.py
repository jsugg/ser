"""Tests for public dataset download helpers."""

from __future__ import annotations

import csv
import tarfile
from pathlib import Path

import pytest

from ser.data import public_dataset_downloads as downloads


def test_write_source_manifest_delegates_to_provider_helper(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Manifest wrapper should delegate persistence to provider helper seam."""
    captured: dict[str, object] = {}
    source_manifest_path = tmp_path / "source_manifest.json"
    labels_csv_path = tmp_path / "labels.csv"
    labels_stats = downloads.GeneratedLabelsStats(
        files_seen=3,
        labels_written=2,
        dropped_files=1,
        duplicate_conflicts=0,
    )

    def _fake_write_source_manifest(**kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(
        downloads.provider_dataset_preparation_helpers,
        "write_source_manifest",
        _fake_write_source_manifest,
    )

    downloads._write_source_manifest(
        dataset_root=tmp_path,
        source_manifest_path=source_manifest_path,
        source_payload={"provider": "test"},
        labels_csv_path=labels_csv_path,
        labels_stats=labels_stats,
    )

    assert captured["dataset_root"] == tmp_path
    assert captured["source_manifest_path"] == source_manifest_path
    assert captured["source_payload"] == {"provider": "test"}
    assert captured["labels_csv_path"] == labels_csv_path
    assert captured["labels_stats"] == labels_stats


def test_generate_labels_from_audio_tree_writes_deterministic_rows(
    tmp_path: Path,
) -> None:
    """Label generation should infer tokens and write deterministic CSV rows."""

    wav_root = tmp_path / "raw" / "escorpus"
    wav_root.mkdir(parents=True, exist_ok=True)
    (wav_root / "speaker1_angry_take1.wav").write_bytes(b"fake")
    (wav_root / "speaker2_happy_take2.wav").write_bytes(b"fake")
    (wav_root / "speaker3_unknown_take3.wav").write_bytes(b"fake")
    labels_csv_path = tmp_path / "labels.csv"

    stats = downloads._generate_labels_from_audio_tree(
        dataset_root=tmp_path,
        search_root=wav_root,
        labels_csv_path=labels_csv_path,
        resolver=downloads._infer_label_from_path_tokens,
    )

    assert stats.files_seen == 3
    assert stats.labels_written == 2
    assert stats.dropped_files == 1
    with labels_csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [
        {"FileName": "raw/escorpus/speaker1_angry_take1.wav", "emotion": "angry"},
        {"FileName": "raw/escorpus/speaker2_happy_take2.wav", "emotion": "happy"},
    ]


def test_infer_mesd_label_prefers_filename_prefix() -> None:
    """MESD labels should resolve from canonical filename prefix."""

    assert downloads._infer_mesd_label(Path("Anger_C_A_abajo.wav")) == "angry"
    assert downloads._infer_mesd_label(Path("Happiness_F_B_L1_abuso.wav")) == "happy"
    assert downloads._infer_mesd_label(Path("Neutral_M_A_hola.wav")) == "neutral"


def test_infer_escorpus_pe_label_maps_vad_triplets() -> None:
    """ESCorpus-PE VAD triplets should map to deterministic weak canonical labels."""

    assert downloads._infer_escorpus_pe_label(Path("Audio7_36-05-05-04.wav")) == "happy"
    assert downloads._infer_escorpus_pe_label(Path("Audio7_36-01-05-04.wav")) == "angry"
    assert (
        downloads._infer_escorpus_pe_label(Path("Audio7_36-01-05-02.wav")) == "fearful"
    )
    assert downloads._infer_escorpus_pe_label(Path("Audio7_36-01-01-03.wav")) == "sad"
    assert (
        downloads._infer_escorpus_pe_label(Path("Audio7_36-03-05-03.wav"))
        == "surprised"
    )
    assert (
        downloads._infer_escorpus_pe_label(Path("Audio7_36-01-03-03.wav")) == "disgust"
    )
    assert (
        downloads._infer_escorpus_pe_label(Path("Audio7_36-03-03-03.wav")) == "neutral"
    )


def test_generate_labels_from_audio_tree_supports_flac_extensions(
    tmp_path: Path,
) -> None:
    """Label generation should support non-WAV corpora when configured."""

    flac_root = tmp_path / "raw" / "pavoque"
    flac_root.mkdir(parents=True, exist_ok=True)
    (flac_root / "speaker1_angry.flac").write_bytes(b"fake")
    (flac_root / "speaker2_sleepy.flac").write_bytes(b"fake")
    labels_csv_path = tmp_path / "labels.csv"

    stats = downloads._generate_labels_from_audio_tree(
        dataset_root=tmp_path,
        search_root=flac_root,
        labels_csv_path=labels_csv_path,
        resolver=downloads._infer_label_from_path_tokens,
        extensions=frozenset({".flac"}),
    )

    assert stats.files_seen == 2
    assert stats.labels_written == 2
    with labels_csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [
        {"FileName": "raw/pavoque/speaker1_angry.flac", "emotion": "angry"},
        {"FileName": "raw/pavoque/speaker2_sleepy.flac", "emotion": "neutral"},
    ]


def test_generate_labels_from_audio_tree_delegates_to_provider_helper(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Labels wrapper should delegate tree orchestration to provider helper seam."""
    captured: dict[str, object] = {}
    expected = downloads.GeneratedLabelsStats(
        files_seen=4,
        labels_written=3,
        dropped_files=1,
        duplicate_conflicts=0,
    )

    def _fake_generate_labels_from_audio_tree(**kwargs: object) -> object:
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(
        downloads.provider_dataset_preparation_helpers,
        "generate_labels_from_audio_tree",
        _fake_generate_labels_from_audio_tree,
    )
    search_root = tmp_path / "raw"
    labels_csv_path = tmp_path / "labels.csv"
    result = downloads._generate_labels_from_audio_tree(
        dataset_root=tmp_path,
        search_root=search_root,
        labels_csv_path=labels_csv_path,
        resolver=downloads._infer_label_from_path_tokens,
        extensions=frozenset({".wav", ".flac"}),
    )

    assert result is expected
    assert captured["dataset_root"] == tmp_path
    assert captured["search_root"] == search_root
    assert captured["labels_csv_path"] == labels_csv_path
    assert captured["resolver"] is downloads._infer_label_from_path_tokens
    assert captured["extensions"] == frozenset({".wav", ".flac"})
    assert callable(captured["collect_audio_files"])
    assert callable(captured["compute_relative_to_dataset_root"])
    assert callable(captured["write_labels_csv"])
    assert callable(captured["stats_factory"])


def test_extract_archive_supports_tar_gz(tmp_path: Path) -> None:
    """Generic archive extraction should support tar.gz sources."""

    archive_path = tmp_path / "sample.tar.gz"
    source_dir = tmp_path / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    source_file = source_dir / "nested" / "clip.wav"
    source_file.parent.mkdir(parents=True, exist_ok=True)
    source_file.write_bytes(b"fake")
    with tarfile.open(archive_path, mode="w:gz") as handle:
        handle.add(source_file, arcname="nested/clip.wav")

    extract_root = tmp_path / "extract"
    downloads._extract_archive(archive_path=archive_path, extract_root=extract_root)

    assert (extract_root / "nested" / "clip.wav").is_file()


def test_download_file_delegates_to_provider_helper(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Download wrapper should delegate file transfer orchestration to provider helper."""
    captured: dict[str, object] = {}

    def _fake_download_file_with_retries(**kwargs: object) -> Path:
        captured.update(kwargs)
        destination_path = kwargs["destination_path"]
        if not isinstance(destination_path, Path):
            raise AssertionError("destination_path must be a Path")
        return destination_path

    monkeypatch.setattr(
        downloads.provider_download_helpers,
        "download_file_with_retries",
        _fake_download_file_with_retries,
    )
    destination_path = tmp_path / "archive.zip"

    resolved = downloads._download_file(
        url="https://example.invalid/archive.zip",
        destination_path=destination_path,
        expected_md5="abc123",
        expected_size=123,
        headers={"Authorization": "Bearer token"},
    )

    assert resolved == destination_path
    assert captured["url"] == "https://example.invalid/archive.zip"
    assert captured["destination_path"] == destination_path
    assert captured["expected_md5"] == "abc123"
    assert captured["expected_size"] == 123
    assert captured["headers"] == {"Authorization": "Bearer token"}
    assert captured["with_retries"] is downloads._with_retries
    assert captured["compute_file_md5"] is downloads._compute_file_md5
    assert captured["timeout_seconds"] == downloads.DEFAULT_HTTP_TIMEOUT_SECONDS
    assert captured["chunk_size"] == downloads.HTTP_CHUNK_SIZE


def test_download_openslr_archives_prefers_pinned_registry_with_mirror_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Pinned OpenSLR downloads should retry mirrors per artifact before failing."""

    downloaded_urls: list[str] = []

    def _fake_download_file(
        *,
        url: str,
        destination_path: Path,
        expected_md5: str | None = None,
        expected_size: int | None = None,
        headers: dict[str, str] | None = None,
    ) -> Path:
        del expected_md5, expected_size, headers
        downloaded_urls.append(url)
        if (
            url.endswith("/88/wav.tgz")
            and "openslr.org/resources" in url
            and "trmal" not in url
        ):
            raise RuntimeError("primary mirror unavailable")
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        destination_path.write_bytes(b"ok")
        return destination_path

    monkeypatch.setattr(downloads, "_download_file", _fake_download_file)
    monkeypatch.setattr(
        downloads.openslr_download_helpers,
        "read_openslr_archive_urls",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("discovery path should not run for pinned ids")
        ),
    )

    paths = downloads._download_openslr_archives(
        dataset_root=tmp_path,
        dataset_id="88",
        archive_suffixes=(".tgz",),
    )

    assert [path.name for path in paths] == ["wav.tgz", "txt.tgz"]
    assert any("openslr.org/resources/88/wav.tgz" in url for url in downloaded_urls)
    assert any("trmal.net/resources/88/wav.tgz" in url for url in downloaded_urls)


def test_prepare_emov_db_from_openslr_delegates_to_openslr_helper(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """EmoV-DB wrapper should delegate to OpenSLR preparation helper."""
    captured: dict[str, object] = {}

    def _fake_prepare_openslr_dataset(**kwargs: object) -> object:
        captured.update(kwargs)
        return downloads.openslr_dataset_preparation_helpers.AutoDownloadArtifacts(
            dataset_root=tmp_path,
            labels_csv_path=tmp_path / "labels.csv",
            audio_base_dir=tmp_path,
            source_manifest_path=tmp_path / "source_manifest.json",
            files_seen=9,
            labels_written=8,
        )

    monkeypatch.setattr(
        downloads.openslr_dataset_preparation_helpers,
        "prepare_openslr_dataset",
        _fake_prepare_openslr_dataset,
    )

    artifacts = downloads.prepare_emov_db_from_openslr(dataset_root=tmp_path)

    assert artifacts.dataset_root == tmp_path
    assert artifacts.labels_csv_path == tmp_path / "labels.csv"
    assert artifacts.files_seen == 9
    assert artifacts.labels_written == 8
    assert captured["dataset_id"] == downloads.EMOV_DB_OPENSLR_DATASET_ID
    assert captured["archive_suffixes"] == downloads.EMOV_DB_OPENSLR_ARCHIVE_SUFFIXES
    assert captured["extract_dir_name"] == "emov-db"
    assert captured["label_semantics"] is None
    assert captured["extensions"] == frozenset({".wav", ".flac"})
    assert captured["label_resolver"] is downloads._infer_label_from_path_tokens


def test_prepare_att_hack_from_openslr_delegates_to_openslr_helper(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Att-HACK wrapper should delegate to OpenSLR preparation helper."""
    captured: dict[str, object] = {}

    def _fake_prepare_openslr_dataset(**kwargs: object) -> object:
        captured.update(kwargs)
        return downloads.openslr_dataset_preparation_helpers.AutoDownloadArtifacts(
            dataset_root=tmp_path,
            labels_csv_path=tmp_path / "labels.csv",
            audio_base_dir=tmp_path,
            source_manifest_path=tmp_path / "source_manifest.json",
            files_seen=5,
            labels_written=4,
        )

    monkeypatch.setattr(
        downloads.openslr_dataset_preparation_helpers,
        "prepare_openslr_dataset",
        _fake_prepare_openslr_dataset,
    )

    artifacts = downloads.prepare_att_hack_from_openslr(dataset_root=tmp_path)

    assert artifacts.dataset_root == tmp_path
    assert artifacts.labels_csv_path == tmp_path / "labels.csv"
    assert artifacts.files_seen == 5
    assert artifacts.labels_written == 4
    assert captured["dataset_id"] == downloads.ATT_HACK_OPENSLR_DATASET_ID
    assert captured["archive_suffixes"] == downloads.ATT_HACK_OPENSLR_ARCHIVE_SUFFIXES
    assert captured["extract_dir_name"] == "att-hack"
    assert captured["label_semantics"] == "social_attitudes"
    assert captured["extensions"] is None
    assert captured["label_resolver"] is downloads._infer_att_hack_label


def test_prepare_mesd_from_mendeley_delegates_to_mendeley_helper(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """MESD wrapper should delegate to Mendeley preparation helper."""
    captured: dict[str, object] = {}

    def _fake_prepare_mesd(**kwargs: object) -> object:
        captured.update(kwargs)
        return downloads.mendeley_dataset_preparation_helpers.AutoDownloadArtifacts(
            dataset_root=tmp_path,
            labels_csv_path=tmp_path / "labels.csv",
            audio_base_dir=tmp_path,
            source_manifest_path=tmp_path / "source_manifest.json",
            files_seen=11,
            labels_written=10,
        )

    monkeypatch.setattr(
        downloads.mendeley_dataset_preparation_helpers,
        "prepare_mesd_from_mendeley",
        _fake_prepare_mesd,
    )

    artifacts = downloads.prepare_mesd_from_mendeley(dataset_root=tmp_path)

    assert artifacts.dataset_root == tmp_path
    assert artifacts.labels_csv_path == tmp_path / "labels.csv"
    assert artifacts.files_seen == 11
    assert artifacts.labels_written == 10
    assert captured["dataset_root"] == tmp_path
    assert captured["dataset_id"] == downloads.MESD_MENDELEY_DATASET_ID
    assert captured["version"] == downloads.MESD_DEFAULT_VERSION
    assert captured["extract_dir_name"] == "mesd"
    assert captured["labels_file_name"] == downloads.DEFAULT_LABELS_FILE_NAME
    assert (
        captured["source_manifest_file_name"]
        == downloads.DEFAULT_SOURCE_MANIFEST_FILE_NAME
    )
    assert (
        captured["download_mendeley_dataset_tree"]
        is downloads._download_mendeley_dataset_tree
    )
    assert (
        captured["generate_labels_from_audio_tree"]
        is downloads._generate_labels_from_audio_tree
    )
    assert captured["infer_mesd_label"] is downloads._infer_mesd_label
    assert captured["write_source_manifest"] is downloads._write_source_manifest


def test_prepare_pavoque_from_github_release_delegates_to_provider_helper(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """PAVOQUE wrapper should delegate to provider preparation helper."""
    captured: dict[str, object] = {}

    def _fake_prepare_pavoque(**kwargs: object) -> object:
        captured.update(kwargs)
        return downloads.provider_dataset_preparation_helpers.AutoDownloadArtifacts(
            dataset_root=tmp_path,
            labels_csv_path=tmp_path / "labels.csv",
            audio_base_dir=tmp_path,
            source_manifest_path=tmp_path / "source_manifest.json",
            files_seen=7,
            labels_written=6,
        )

    monkeypatch.setattr(
        downloads.provider_dataset_preparation_helpers,
        "prepare_pavoque_from_github_release",
        _fake_prepare_pavoque,
    )

    artifacts = downloads.prepare_pavoque_from_github_release(dataset_root=tmp_path)

    assert artifacts.dataset_root == tmp_path
    assert artifacts.labels_csv_path == tmp_path / "labels.csv"
    assert artifacts.files_seen == 7
    assert artifacts.labels_written == 6
    assert captured["dataset_root"] == tmp_path
    assert captured["owner"] == downloads.PAVOQUE_GITHUB_OWNER
    assert captured["repo"] == downloads.PAVOQUE_GITHUB_REPO
    assert captured["labels_file_name"] == downloads.DEFAULT_LABELS_FILE_NAME
    assert (
        captured["source_manifest_file_name"]
        == downloads.DEFAULT_SOURCE_MANIFEST_FILE_NAME
    )
    assert (
        captured["read_github_latest_release_assets"]
        is downloads._read_github_latest_release_assets
    )
    assert captured["download_file"] is downloads._download_file
    assert (
        captured["generate_labels_from_audio_tree"]
        is downloads._generate_labels_from_audio_tree
    )
    assert (
        captured["infer_label_from_path_tokens"]
        is downloads._infer_label_from_path_tokens
    )
    assert captured["write_source_manifest"] is downloads._write_source_manifest


def test_prepare_coraa_ser_from_google_drive_delegates_to_provider_helper(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """CORAA wrapper should delegate to provider preparation helper."""
    captured: dict[str, object] = {}

    def _fake_prepare_coraa(**kwargs: object) -> object:
        captured.update(kwargs)
        return downloads.provider_dataset_preparation_helpers.AutoDownloadArtifacts(
            dataset_root=tmp_path,
            labels_csv_path=tmp_path / "labels.csv",
            audio_base_dir=tmp_path,
            source_manifest_path=tmp_path / "source_manifest.json",
            files_seen=5,
            labels_written=4,
        )

    monkeypatch.setattr(
        downloads.provider_dataset_preparation_helpers,
        "prepare_coraa_ser_from_google_drive",
        _fake_prepare_coraa,
    )

    artifacts = downloads.prepare_coraa_ser_from_google_drive(dataset_root=tmp_path)

    assert artifacts.dataset_root == tmp_path
    assert artifacts.labels_csv_path == tmp_path / "labels.csv"
    assert artifacts.files_seen == 5
    assert artifacts.labels_written == 4
    assert captured["dataset_root"] == tmp_path
    assert captured["folder_url"] == downloads.CORAA_SER_GOOGLE_DRIVE_FOLDER_URL
    assert captured["label_semantics"] == "neutral_vs_non_neutral_by_gender"
    assert captured["labels_file_name"] == downloads.DEFAULT_LABELS_FILE_NAME
    assert (
        captured["source_manifest_file_name"]
        == downloads.DEFAULT_SOURCE_MANIFEST_FILE_NAME
    )
    assert (
        captured["download_google_drive_folder"]
        is downloads._download_google_drive_folder
    )
    assert (
        captured["extract_archives_from_tree"] is downloads._extract_archives_from_tree
    )
    assert (
        captured["generate_labels_from_audio_tree"]
        is downloads._generate_labels_from_audio_tree
    )
    assert captured["infer_coraa_ser_label"] is downloads._infer_coraa_ser_label
    assert captured["write_source_manifest"] is downloads._write_source_manifest


def test_read_github_latest_release_assets_parses_expected_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GitHub release parser should read tag and downloadable asset metadata."""

    payload = {
        "tag_name": "v1.0.0",
        "assets": [
            {
                "name": "pavoque-angry.flac",
                "browser_download_url": "https://example.org/pavoque-angry.flac",
                "size": 100,
            },
            {
                "name": "",
                "browser_download_url": "https://example.org/skip.bin",
            },
        ],
    }

    def _fake_request_json(
        url: str, *, headers: dict[str, str] | None = None
    ) -> object:
        del url, headers
        return payload

    monkeypatch.setattr(downloads, "_request_json", _fake_request_json)

    tag_name, assets = downloads._read_github_latest_release_assets(
        owner="marytts",
        repo="pavoque-data",
    )

    assert tag_name == "v1.0.0"
    assert len(assets) == 1
    assert assets[0].name == "pavoque-angry.flac"
    assert assets[0].download_url == "https://example.org/pavoque-angry.flac"


def test_infer_att_hack_label_from_filename_tokens() -> None:
    """Att-HACK resolver should extract social-attitude labels from filenames."""

    assert downloads._infer_att_hack_label(Path("x_friendly_1.wav")) == "friendly"
    assert downloads._infer_att_hack_label(Path("x_dominant_1.wav")) == "dominant"
    assert downloads._infer_att_hack_label(Path("x_unknown_1.wav")) is None


def test_infer_coraa_ser_label_from_filename_patterns() -> None:
    """CORAA resolver should normalize non-neutral male/female variants."""

    assert (
        downloads._infer_coraa_ser_label(Path("abc_nonneutralfemale_1.wav"))
        == "non_neutral_female"
    )
    assert (
        downloads._infer_coraa_ser_label(Path("abc_non-neutral-male_1.wav"))
        == "non_neutral_male"
    )
    assert downloads._infer_coraa_ser_label(Path("abc_neutral_1.wav")) == "neutral"


def test_generate_labels_from_metadata_csv_requires_existing_audio(
    tmp_path: Path,
) -> None:
    """Metadata-based label generation should keep rows with present local audio only."""

    metadata_csv_path = tmp_path / "metadata.csv"
    metadata_csv_path.write_text(
        "filename,label\nclip_a.wav,anger\nclip_missing.wav,joy\n",
        encoding="utf-8",
    )
    audio_root = tmp_path / "raw" / "spanish-meacorpus-2023"
    audio_root.mkdir(parents=True, exist_ok=True)
    (audio_root / "clip_a.wav").write_bytes(b"fake")
    labels_csv_path = tmp_path / "labels.csv"

    stats = downloads._generate_labels_from_metadata_csv(
        dataset_root=tmp_path,
        metadata_csv_path=metadata_csv_path,
        labels_csv_path=labels_csv_path,
        audio_search_roots=(audio_root,),
        file_name_keys=("filename",),
        label_keys=("label",),
        label_resolver=lambda value: {"anger": "angry", "joy": "happy"}.get(value),
    )

    assert stats.files_seen == 2
    assert stats.labels_written == 1
    assert stats.dropped_files == 1
    with labels_csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [
        {
            "FileName": "raw/spanish-meacorpus-2023/clip_a.wav",
            "emotion": "angry",
        }
    ]


def test_generate_labels_from_metadata_csv_delegates_to_zenodo_helper(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Metadata label wrapper should delegate orchestration to zenodo helper module."""
    captured: dict[str, object] = {}

    def _fake_generate(**kwargs: object) -> object:
        captured.update(kwargs)
        return downloads.zenodo_download_helpers.GeneratedLabelsStats(
            files_seen=4,
            labels_written=3,
            dropped_files=1,
            duplicate_conflicts=0,
        )

    monkeypatch.setattr(
        downloads.zenodo_download_helpers,
        "generate_labels_from_metadata_csv",
        _fake_generate,
    )

    result = downloads._generate_labels_from_metadata_csv(
        dataset_root=tmp_path,
        metadata_csv_path=tmp_path / "metadata.csv",
        labels_csv_path=tmp_path / "labels.csv",
        audio_search_roots=(tmp_path / "raw",),
        file_name_keys=("filename",),
        label_keys=("label",),
        label_resolver=lambda raw: raw,
    )

    assert result == downloads.GeneratedLabelsStats(
        files_seen=4,
        labels_written=3,
        dropped_files=1,
        duplicate_conflicts=0,
    )
    assert captured["dataset_root"] == tmp_path
    assert captured["metadata_csv_path"] == tmp_path / "metadata.csv"
    assert captured["labels_csv_path"] == tmp_path / "labels.csv"
    assert captured["audio_search_roots"] == (tmp_path / "raw",)
    assert captured["file_name_keys"] == ("filename",)
    assert captured["label_keys"] == ("label",)
    assert callable(captured["label_resolver"])
    assert (
        captured["compute_relative_to_dataset_root"]
        is downloads._compute_relative_to_dataset_root
    )
    assert captured["write_labels_csv"] is downloads._write_labels_csv


def test_extract_jl_corpus_audio_src_from_supported_shapes() -> None:
    """JL fallback should resolve audio URLs from list and dict row formats."""

    assert (
        downloads._extract_jl_corpus_audio_src(
            [{"src": "https://example.invalid/a.wav", "type": "audio/wav"}]
        )
        == "https://example.invalid/a.wav"
    )
    assert (
        downloads._extract_jl_corpus_audio_src({"src": "https://example.invalid/b.wav"})
        == "https://example.invalid/b.wav"
    )
    assert downloads._extract_jl_corpus_audio_src([]) is None


def test_download_jl_corpus_via_hf_rows_writes_labels(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """HF rows fallback should materialize audio and deterministic labels CSV."""

    def _fake_request_json(
        url: str, *, headers: dict[str, str] | None = None
    ) -> object:
        del headers
        if "offset=0" in url:
            return {
                "num_rows_total": 2,
                "rows": [
                    {
                        "row": {
                            "index": "female1_angry_10a_1",
                            "audio": [
                                {"src": "https://example.invalid/a.wav"},
                            ],
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

    def _fake_download_file(
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

    monkeypatch.setattr(downloads, "_request_json", _fake_request_json)
    monkeypatch.setattr(downloads, "_download_file", _fake_download_file)

    stats = downloads._download_jl_corpus_via_hf_rows(dataset_root=tmp_path)

    assert stats.files_seen == 2
    assert stats.labels_written == 2
    labels_csv_path = tmp_path / "labels.csv"
    with labels_csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [
        {
            "FileName": "raw/jl-corpus/female1_angry_10a_1.wav",
            "emotion": "angry",
        },
        {
            "FileName": "raw/jl-corpus/male1_happy_11a_1.wav",
            "emotion": "happy",
        },
    ]


def test_prepare_jl_corpus_from_hf_rows_delegates_to_jl_corpus_helper(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """HF rows fallback wrapper should delegate orchestration to JL helper module."""
    captured: dict[str, object] = {}

    def _fake_prepare_hf_rows(**kwargs: object) -> object:
        captured.update(kwargs)
        return downloads.jl_corpus_download_helpers.AutoDownloadArtifacts(
            dataset_root=tmp_path,
            labels_csv_path=tmp_path / "labels.csv",
            audio_base_dir=tmp_path,
            source_manifest_path=tmp_path / "source_manifest.json",
            files_seen=4,
            labels_written=3,
        )

    monkeypatch.setattr(
        downloads.jl_corpus_download_helpers,
        "prepare_jl_corpus_from_hf_rows",
        _fake_prepare_hf_rows,
    )

    artifacts = downloads._prepare_jl_corpus_from_hf_rows(
        dataset_root=tmp_path,
        fallback_reason="missing credentials",
    )

    assert artifacts.dataset_root == tmp_path
    assert artifacts.labels_csv_path == tmp_path / "labels.csv"
    assert artifacts.source_manifest_path == tmp_path / "source_manifest.json"
    assert artifacts.files_seen == 4
    assert artifacts.labels_written == 3
    assert captured["dataset_root"] == tmp_path
    assert captured["fallback_reason"] == "missing credentials"
    assert captured["labels_file_name"] == downloads.DEFAULT_LABELS_FILE_NAME
    assert (
        captured["source_manifest_file_name"]
        == downloads.DEFAULT_SOURCE_MANIFEST_FILE_NAME
    )
    assert captured["dataset_id"] == downloads.JL_CORPUS_HF_DATASET_ID
    assert captured["source_url"] == downloads.JL_CORPUS_HF_SOURCE_URL
    assert captured["rows_api_url"] == downloads.JL_CORPUS_HF_ROWS_API_URL
    assert captured["config"] == downloads.JL_CORPUS_HF_CONFIG
    assert captured["split"] == downloads.JL_CORPUS_HF_SPLIT
    assert captured["page_size"] == downloads.JL_CORPUS_HF_PAGE_SIZE
    assert captured["request_json"] is downloads._request_json
    assert captured["download_file"] is downloads._download_file
    assert (
        captured["infer_label_from_path_tokens"]
        is downloads._infer_label_from_path_tokens
    )
    assert (
        captured["compute_relative_to_dataset_root"]
        is downloads._compute_relative_to_dataset_root
    )
    assert captured["write_labels_csv"] is downloads._write_labels_csv
    assert captured["write_source_manifest"] is downloads._write_source_manifest
    assert captured["sanitize_index"] is downloads._sanitize_jl_corpus_index
    assert captured["extract_audio_src"] is downloads._extract_jl_corpus_audio_src


def test_prepare_jl_corpus_from_kaggle_falls_back_to_hf_rows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """JL-Corpus should use HF rows fallback when Kaggle credentials are unavailable."""

    monkeypatch.setattr(
        downloads,
        "_download_kaggle_archive",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("missing credentials")),
    )

    def _fake_hf_fallback(
        *, dataset_root: Path, fallback_reason: str
    ) -> downloads.AutoDownloadArtifacts:
        assert dataset_root == tmp_path
        assert "missing credentials" in fallback_reason
        source_manifest_path = dataset_root / "source_manifest.json"
        source_manifest_path.write_text("{}", encoding="utf-8")
        labels_csv_path = dataset_root / "labels.csv"
        labels_csv_path.write_text("FileName,emotion\n", encoding="utf-8")
        return downloads.AutoDownloadArtifacts(
            dataset_root=dataset_root,
            labels_csv_path=labels_csv_path,
            audio_base_dir=dataset_root,
            source_manifest_path=source_manifest_path,
            files_seen=1,
            labels_written=1,
        )

    monkeypatch.setattr(downloads, "_prepare_jl_corpus_from_hf_rows", _fake_hf_fallback)

    artifacts = downloads.prepare_jl_corpus_from_kaggle(dataset_root=tmp_path)

    assert artifacts.dataset_root == tmp_path
    assert artifacts.labels_written == 1


def test_prepare_jl_corpus_from_kaggle_delegates_to_jl_corpus_helper(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """JL-Corpus wrapper should delegate orchestration to JL helper module."""
    captured: dict[str, object] = {}

    def _fake_prepare_jl_corpus(**kwargs: object) -> object:
        captured.update(kwargs)
        return downloads.jl_corpus_download_helpers.AutoDownloadArtifacts(
            dataset_root=tmp_path,
            labels_csv_path=tmp_path / "labels.csv",
            audio_base_dir=tmp_path,
            source_manifest_path=tmp_path / "source_manifest.json",
            files_seen=3,
            labels_written=2,
        )

    monkeypatch.setattr(
        downloads.jl_corpus_download_helpers,
        "prepare_jl_corpus_from_kaggle",
        _fake_prepare_jl_corpus,
    )

    artifacts = downloads.prepare_jl_corpus_from_kaggle(dataset_root=tmp_path)

    assert artifacts.dataset_root == tmp_path
    assert artifacts.labels_csv_path == tmp_path / "labels.csv"
    assert artifacts.files_seen == 3
    assert artifacts.labels_written == 2
    assert captured["dataset_root"] == tmp_path
    assert captured["dataset_ref"] == downloads.JL_CORPUS_KAGGLE_DATASET_REF
    assert captured["labels_file_name"] == downloads.DEFAULT_LABELS_FILE_NAME
    assert (
        captured["source_manifest_file_name"]
        == downloads.DEFAULT_SOURCE_MANIFEST_FILE_NAME
    )
    assert captured["download_kaggle_archive"] is downloads._download_kaggle_archive
    assert captured["ensure_extracted_archive"] is downloads._ensure_extracted_archive
    assert (
        captured["generate_labels_from_audio_tree"]
        is downloads._generate_labels_from_audio_tree
    )
    assert (
        captured["infer_label_from_path_tokens"]
        is downloads._infer_label_from_path_tokens
    )
    assert captured["write_source_manifest"] is downloads._write_source_manifest
    assert (
        captured["prepare_hf_rows_fallback"]
        is downloads._prepare_jl_corpus_from_hf_rows
    )
    logger_warning = captured["logger_warning"]
    assert callable(logger_warning)


def test_download_google_drive_folder_requires_gdown(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Google Drive folder utility should fail with explicit guidance without gdown."""

    monkeypatch.setattr(downloads.shutil, "which", lambda _: None)

    with pytest.raises(RuntimeError, match="requires `gdown`"):
        downloads._download_google_drive_folder(
            folder_url="https://drive.google.com/drive/folders/example",
            destination_root=tmp_path / "downloads",
        )
