"""Tests for CSV-labeled public dataset adapters."""

from __future__ import annotations

from pathlib import Path

from ser.data.adapters.public_csv_datasets import (
    ATT_HACK_CORPUS_ID,
    ATT_HACK_DATASET_LICENSE_ID,
    ATT_HACK_DATASET_POLICY_ID,
    CAFE_CORPUS_ID,
    CAFE_DATASET_LICENSE_ID,
    CAFE_DATASET_POLICY_ID,
    CORAA_SER_CORPUS_ID,
    CORAA_SER_DATASET_LICENSE_ID,
    CORAA_SER_DATASET_POLICY_ID,
    EMODB_2_CORPUS_ID,
    EMODB_2_DATASET_LICENSE_ID,
    EMODB_2_DATASET_POLICY_ID,
    JL_CORPUS_CORPUS_ID,
    JL_CORPUS_DATASET_LICENSE_ID,
    JL_CORPUS_DATASET_POLICY_ID,
    PAVOQUE_CORPUS_ID,
    PAVOQUE_DATASET_LICENSE_ID,
    PAVOQUE_DATASET_POLICY_ID,
    SPANISH_MEACORPUS_2023_CORPUS_ID,
    SPANISH_MEACORPUS_2023_DATASET_LICENSE_ID,
    SPANISH_MEACORPUS_2023_DATASET_POLICY_ID,
    build_att_hack_manifest_jsonl,
    build_cafe_manifest_jsonl,
    build_coraa_ser_manifest_jsonl,
    build_emodb_2_manifest_jsonl,
    build_jl_corpus_manifest_jsonl,
    build_pavoque_manifest_jsonl,
    build_spanish_meacorpus_2023_manifest_jsonl,
)
from ser.data.ontology import LabelOntology


def _ontology() -> LabelOntology:
    return LabelOntology(
        ontology_id="default_v1",
        allowed_labels=frozenset(
            {"happy", "neutral", "angry", "fearful", "sad", "disgust", "surprised"}
        ),
    )


def _custom_ontology(*labels: str) -> LabelOntology:
    return LabelOntology(
        ontology_id="custom_v1",
        allowed_labels=frozenset(labels),
    )


def test_build_emodb_2_manifest_jsonl_maps_dataset_labels(tmp_path: Path) -> None:
    """EmoDB adapter should map canonicalized labels and persist metadata."""

    audio_root = tmp_path / "raw" / "emodb-2.0" / "wav"
    audio_root.mkdir(parents=True, exist_ok=True)
    (audio_root / "a.wav").write_bytes(b"fake")
    (audio_root / "b.wav").write_bytes(b"fake")
    labels_csv_path = tmp_path / "labels.csv"
    labels_csv_path.write_text(
        "FileName,emotion\n"
        "raw/emodb-2.0/wav/a.wav,happiness\n"
        "raw/emodb-2.0/wav/b.wav,boredom\n",
        encoding="utf-8",
    )

    utterances = build_emodb_2_manifest_jsonl(
        dataset_root=tmp_path,
        labels_csv_path=labels_csv_path,
        audio_base_dir=tmp_path,
        ontology=_ontology(),
        default_language="de",
        max_failed_file_ratio=0.5,
        output_path=tmp_path / "manifest.jsonl",
    )

    assert utterances is not None
    assert len(utterances) == 2
    assert {item.corpus for item in utterances} == {EMODB_2_CORPUS_ID}
    assert {item.dataset_policy_id for item in utterances} == {
        EMODB_2_DATASET_POLICY_ID
    }
    assert {item.dataset_license_id for item in utterances} == {
        EMODB_2_DATASET_LICENSE_ID
    }
    assert {item.label for item in utterances} == {"happy", "neutral"}


def test_build_jl_corpus_manifest_jsonl_maps_anxious_to_fearful(
    tmp_path: Path,
) -> None:
    """JL adapter should map anxious labels into canonical ontology values."""

    audio_root = tmp_path / "raw" / "jl-corpus"
    audio_root.mkdir(parents=True, exist_ok=True)
    (audio_root / "female1_angry_10a_1.wav").write_bytes(b"fake")
    (audio_root / "female1_anxious_10a_1.wav").write_bytes(b"fake")
    labels_csv_path = tmp_path / "labels.csv"
    labels_csv_path.write_text(
        "FileName,emotion\n"
        "raw/jl-corpus/female1_angry_10a_1.wav,angry\n"
        "raw/jl-corpus/female1_anxious_10a_1.wav,anxious\n",
        encoding="utf-8",
    )

    utterances = build_jl_corpus_manifest_jsonl(
        dataset_root=tmp_path,
        labels_csv_path=labels_csv_path,
        audio_base_dir=tmp_path,
        ontology=_ontology(),
        default_language="en",
        max_failed_file_ratio=0.5,
        output_path=tmp_path / "manifest.jsonl",
    )

    assert utterances is not None
    assert len(utterances) == 2
    assert {item.corpus for item in utterances} == {JL_CORPUS_CORPUS_ID}
    assert {item.dataset_policy_id for item in utterances} == {
        JL_CORPUS_DATASET_POLICY_ID
    }
    assert {item.dataset_license_id for item in utterances} == {
        JL_CORPUS_DATASET_LICENSE_ID
    }
    assert {item.label for item in utterances} == {"angry", "fearful"}


def test_build_cafe_manifest_jsonl_maps_french_labels(tmp_path: Path) -> None:
    """CaFE adapter should map French labels into canonical ontology values."""

    audio_root = tmp_path / "raw" / "cafe"
    audio_root.mkdir(parents=True, exist_ok=True)
    (audio_root / "speaker1_colere_001.wav").write_bytes(b"fake")
    (audio_root / "speaker2_joie_001.wav").write_bytes(b"fake")
    labels_csv_path = tmp_path / "labels.csv"
    labels_csv_path.write_text(
        "FileName,emotion\n"
        "raw/cafe/speaker1_colere_001.wav,colere\n"
        "raw/cafe/speaker2_joie_001.wav,joie\n",
        encoding="utf-8",
    )

    utterances = build_cafe_manifest_jsonl(
        dataset_root=tmp_path,
        labels_csv_path=labels_csv_path,
        audio_base_dir=tmp_path,
        ontology=_ontology(),
        default_language="fr",
        max_failed_file_ratio=0.5,
        output_path=tmp_path / "manifest.jsonl",
    )

    assert utterances is not None
    assert len(utterances) == 2
    assert {item.corpus for item in utterances} == {CAFE_CORPUS_ID}
    assert {item.dataset_policy_id for item in utterances} == {CAFE_DATASET_POLICY_ID}
    assert {item.dataset_license_id for item in utterances} == {CAFE_DATASET_LICENSE_ID}
    assert {item.label for item in utterances} == {"angry", "happy"}


def test_build_pavoque_manifest_jsonl_maps_sleepy_to_neutral(
    tmp_path: Path,
) -> None:
    """PAVOQUE adapter should canonicalize sleepy/anxious-style labels."""

    audio_root = tmp_path / "raw" / "pavoque"
    audio_root.mkdir(parents=True, exist_ok=True)
    (audio_root / "sample_angry.flac").write_bytes(b"fake")
    (audio_root / "sample_sleepy.flac").write_bytes(b"fake")
    labels_csv_path = tmp_path / "labels.csv"
    labels_csv_path.write_text(
        "FileName,emotion\n"
        "raw/pavoque/sample_angry.flac,angry\n"
        "raw/pavoque/sample_sleepy.flac,sleepy\n",
        encoding="utf-8",
    )

    utterances = build_pavoque_manifest_jsonl(
        dataset_root=tmp_path,
        labels_csv_path=labels_csv_path,
        audio_base_dir=tmp_path,
        ontology=_ontology(),
        default_language="en",
        max_failed_file_ratio=0.5,
        output_path=tmp_path / "manifest.jsonl",
    )

    assert utterances is not None
    assert len(utterances) == 2
    assert {item.corpus for item in utterances} == {PAVOQUE_CORPUS_ID}
    assert {item.dataset_policy_id for item in utterances} == {
        PAVOQUE_DATASET_POLICY_ID
    }
    assert {item.dataset_license_id for item in utterances} == {
        PAVOQUE_DATASET_LICENSE_ID
    }
    assert {item.label for item in utterances} == {"angry", "neutral"}


def test_build_att_hack_manifest_jsonl_preserves_social_attitudes(
    tmp_path: Path,
) -> None:
    """Att-HACK adapter should keep social-attitude labels when ontology allows them."""

    audio_root = tmp_path / "raw" / "att-hack"
    audio_root.mkdir(parents=True, exist_ok=True)
    (audio_root / "speaker_friendly_1.wav").write_bytes(b"fake")
    (audio_root / "speaker_dominant_2.wav").write_bytes(b"fake")
    labels_csv_path = tmp_path / "labels.csv"
    labels_csv_path.write_text(
        "FileName,emotion\n"
        "raw/att-hack/speaker_friendly_1.wav,friendly\n"
        "raw/att-hack/speaker_dominant_2.wav,dominant\n",
        encoding="utf-8",
    )

    utterances = build_att_hack_manifest_jsonl(
        dataset_root=tmp_path,
        labels_csv_path=labels_csv_path,
        audio_base_dir=tmp_path,
        ontology=_custom_ontology("friendly", "dominant"),
        default_language="fr",
        max_failed_file_ratio=0.5,
        output_path=tmp_path / "manifest.jsonl",
    )

    assert utterances is not None
    assert len(utterances) == 2
    assert {item.corpus for item in utterances} == {ATT_HACK_CORPUS_ID}
    assert {item.dataset_policy_id for item in utterances} == {
        ATT_HACK_DATASET_POLICY_ID
    }
    assert {item.dataset_license_id for item in utterances} == {
        ATT_HACK_DATASET_LICENSE_ID
    }
    assert {item.label for item in utterances} == {"friendly", "dominant"}


def test_build_coraa_ser_manifest_jsonl_preserves_non_neutral_labels(
    tmp_path: Path,
) -> None:
    """CORAA adapter should preserve non-neutral labels with custom ontology."""

    audio_root = tmp_path / "raw" / "coraa-ser"
    audio_root.mkdir(parents=True, exist_ok=True)
    (audio_root / "item_nonneutralfemale.wav").write_bytes(b"fake")
    (audio_root / "item_neutral.wav").write_bytes(b"fake")
    labels_csv_path = tmp_path / "labels.csv"
    labels_csv_path.write_text(
        "FileName,emotion\n"
        "raw/coraa-ser/item_nonneutralfemale.wav,non_neutral_female\n"
        "raw/coraa-ser/item_neutral.wav,neutral\n",
        encoding="utf-8",
    )

    utterances = build_coraa_ser_manifest_jsonl(
        dataset_root=tmp_path,
        labels_csv_path=labels_csv_path,
        audio_base_dir=tmp_path,
        ontology=_custom_ontology("neutral", "non_neutral_female"),
        default_language="pt",
        max_failed_file_ratio=0.5,
        output_path=tmp_path / "manifest.jsonl",
    )

    assert utterances is not None
    assert len(utterances) == 2
    assert {item.corpus for item in utterances} == {CORAA_SER_CORPUS_ID}
    assert {item.dataset_policy_id for item in utterances} == {
        CORAA_SER_DATASET_POLICY_ID
    }
    assert {item.dataset_license_id for item in utterances} == {
        CORAA_SER_DATASET_LICENSE_ID
    }
    assert {item.label for item in utterances} == {"neutral", "non_neutral_female"}


def test_build_spanish_meacorpus_manifest_jsonl_maps_ekman_labels(
    tmp_path: Path,
) -> None:
    """Spanish MEACorpus adapter should map anger/joy labels into ontology."""

    audio_root = tmp_path / "raw" / "spanish-meacorpus-2023"
    audio_root.mkdir(parents=True, exist_ok=True)
    (audio_root / "clip_a.mp3").write_bytes(b"fake")
    (audio_root / "clip_b.mp3").write_bytes(b"fake")
    labels_csv_path = tmp_path / "labels.csv"
    labels_csv_path.write_text(
        "FileName,emotion\n"
        "raw/spanish-meacorpus-2023/clip_a.mp3,anger\n"
        "raw/spanish-meacorpus-2023/clip_b.mp3,joy\n",
        encoding="utf-8",
    )

    utterances = build_spanish_meacorpus_2023_manifest_jsonl(
        dataset_root=tmp_path,
        labels_csv_path=labels_csv_path,
        audio_base_dir=tmp_path,
        ontology=_ontology(),
        default_language="es",
        max_failed_file_ratio=0.5,
        output_path=tmp_path / "manifest.jsonl",
    )

    assert utterances is not None
    assert len(utterances) == 2
    assert {item.corpus for item in utterances} == {SPANISH_MEACORPUS_2023_CORPUS_ID}
    assert {item.dataset_policy_id for item in utterances} == {
        SPANISH_MEACORPUS_2023_DATASET_POLICY_ID
    }
    assert {item.dataset_license_id for item in utterances} == {
        SPANISH_MEACORPUS_2023_DATASET_LICENSE_ID
    }
    assert {item.label for item in utterances} == {"angry", "happy"}
