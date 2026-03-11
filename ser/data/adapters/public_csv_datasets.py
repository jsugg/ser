"""Public adapter facade for CSV-labeled datasets."""

from __future__ import annotations

from pathlib import Path

from ser.data.adapters.csv_manifest_builder import build_csv_labeled_manifest_jsonl
from ser.data.catalog.public_datasets import (
    ASVP_ESD_CORPUS_ID,
    ASVP_ESD_DATASET_LICENSE_ID,
    ASVP_ESD_DATASET_POLICY_ID,
    ASVP_ESD_MANIFEST_SPEC,
    ATT_HACK_CORPUS_ID,
    ATT_HACK_DATASET_LICENSE_ID,
    ATT_HACK_DATASET_POLICY_ID,
    ATT_HACK_MANIFEST_SPEC,
    CAFE_CORPUS_ID,
    CAFE_DATASET_LICENSE_ID,
    CAFE_DATASET_POLICY_ID,
    CAFE_MANIFEST_SPEC,
    CORAA_SER_CORPUS_ID,
    CORAA_SER_DATASET_LICENSE_ID,
    CORAA_SER_DATASET_POLICY_ID,
    CORAA_SER_MANIFEST_SPEC,
    EMODB_2_CORPUS_ID,
    EMODB_2_DATASET_LICENSE_ID,
    EMODB_2_DATASET_POLICY_ID,
    EMODB_2_MANIFEST_SPEC,
    EMOV_DB_CORPUS_ID,
    EMOV_DB_DATASET_LICENSE_ID,
    EMOV_DB_DATASET_POLICY_ID,
    EMOV_DB_MANIFEST_SPEC,
    ESCORPUS_PE_CORPUS_ID,
    ESCORPUS_PE_DATASET_LICENSE_ID,
    ESCORPUS_PE_DATASET_POLICY_ID,
    ESCORPUS_PE_MANIFEST_SPEC,
    JL_CORPUS_CORPUS_ID,
    JL_CORPUS_DATASET_LICENSE_ID,
    JL_CORPUS_DATASET_POLICY_ID,
    JL_CORPUS_MANIFEST_SPEC,
    MESD_CORPUS_ID,
    MESD_DATASET_LICENSE_ID,
    MESD_DATASET_POLICY_ID,
    MESD_MANIFEST_SPEC,
    OREAU_FRENCH_ESD_CORPUS_ID,
    OREAU_FRENCH_ESD_DATASET_LICENSE_ID,
    OREAU_FRENCH_ESD_DATASET_POLICY_ID,
    OREAU_FRENCH_ESD_MANIFEST_SPEC,
    PAVOQUE_CORPUS_ID,
    PAVOQUE_DATASET_LICENSE_ID,
    PAVOQUE_DATASET_POLICY_ID,
    PAVOQUE_MANIFEST_SPEC,
    SPANISH_MEACORPUS_2023_CORPUS_ID,
    SPANISH_MEACORPUS_2023_DATASET_LICENSE_ID,
    SPANISH_MEACORPUS_2023_DATASET_POLICY_ID,
    SPANISH_MEACORPUS_2023_MANIFEST_SPEC,
    CsvManifestSpec,
)
from ser.data.manifest import Utterance
from ser.data.ontology import LabelOntology

__all__ = [
    "ASVP_ESD_CORPUS_ID",
    "ASVP_ESD_DATASET_LICENSE_ID",
    "ASVP_ESD_DATASET_POLICY_ID",
    "ATT_HACK_CORPUS_ID",
    "ATT_HACK_DATASET_LICENSE_ID",
    "ATT_HACK_DATASET_POLICY_ID",
    "CAFE_CORPUS_ID",
    "CAFE_DATASET_LICENSE_ID",
    "CAFE_DATASET_POLICY_ID",
    "CORAA_SER_CORPUS_ID",
    "CORAA_SER_DATASET_LICENSE_ID",
    "CORAA_SER_DATASET_POLICY_ID",
    "EMODB_2_CORPUS_ID",
    "EMODB_2_DATASET_LICENSE_ID",
    "EMODB_2_DATASET_POLICY_ID",
    "EMOV_DB_CORPUS_ID",
    "EMOV_DB_DATASET_LICENSE_ID",
    "EMOV_DB_DATASET_POLICY_ID",
    "ESCORPUS_PE_CORPUS_ID",
    "ESCORPUS_PE_DATASET_LICENSE_ID",
    "ESCORPUS_PE_DATASET_POLICY_ID",
    "JL_CORPUS_CORPUS_ID",
    "JL_CORPUS_DATASET_LICENSE_ID",
    "JL_CORPUS_DATASET_POLICY_ID",
    "MESD_CORPUS_ID",
    "MESD_DATASET_LICENSE_ID",
    "MESD_DATASET_POLICY_ID",
    "OREAU_FRENCH_ESD_CORPUS_ID",
    "OREAU_FRENCH_ESD_DATASET_LICENSE_ID",
    "OREAU_FRENCH_ESD_DATASET_POLICY_ID",
    "PAVOQUE_CORPUS_ID",
    "PAVOQUE_DATASET_LICENSE_ID",
    "PAVOQUE_DATASET_POLICY_ID",
    "SPANISH_MEACORPUS_2023_CORPUS_ID",
    "SPANISH_MEACORPUS_2023_DATASET_LICENSE_ID",
    "SPANISH_MEACORPUS_2023_DATASET_POLICY_ID",
    "build_asvp_esd_manifest_jsonl",
    "build_att_hack_manifest_jsonl",
    "build_cafe_manifest_jsonl",
    "build_coraa_ser_manifest_jsonl",
    "build_emodb_2_manifest_jsonl",
    "build_emov_db_manifest_jsonl",
    "build_escorpus_pe_manifest_jsonl",
    "build_jl_corpus_manifest_jsonl",
    "build_mesd_manifest_jsonl",
    "build_oreau_french_esd_manifest_jsonl",
    "build_pavoque_manifest_jsonl",
    "build_spanish_meacorpus_2023_manifest_jsonl",
]


def _build_manifest_jsonl(
    *,
    dataset_root: Path,
    labels_csv_path: Path,
    audio_base_dir: Path | None,
    ontology: LabelOntology,
    default_language: str,
    max_failed_file_ratio: float,
    output_path: Path,
    spec: CsvManifestSpec,
) -> list[Utterance] | None:
    return build_csv_labeled_manifest_jsonl(
        dataset_root=dataset_root,
        labels_csv_path=labels_csv_path,
        audio_base_dir=audio_base_dir,
        ontology=ontology,
        default_language=default_language,
        max_failed_file_ratio=max_failed_file_ratio,
        output_path=output_path,
        spec=spec,
    )


def build_emodb_2_manifest_jsonl(
    *,
    dataset_root: Path,
    labels_csv_path: Path,
    audio_base_dir: Path | None,
    ontology: LabelOntology,
    default_language: str,
    max_failed_file_ratio: float,
    output_path: Path,
) -> list[Utterance] | None:
    """Builds utterances and persists an EmoDB 2.0 JSONL manifest."""

    return _build_manifest_jsonl(
        dataset_root=dataset_root,
        labels_csv_path=labels_csv_path,
        audio_base_dir=audio_base_dir,
        ontology=ontology,
        default_language=default_language,
        max_failed_file_ratio=max_failed_file_ratio,
        output_path=output_path,
        spec=EMODB_2_MANIFEST_SPEC,
    )


def build_escorpus_pe_manifest_jsonl(
    *,
    dataset_root: Path,
    labels_csv_path: Path,
    audio_base_dir: Path | None,
    ontology: LabelOntology,
    default_language: str,
    max_failed_file_ratio: float,
    output_path: Path,
) -> list[Utterance] | None:
    """Builds utterances and persists an ESCorpus-PE JSONL manifest."""

    return _build_manifest_jsonl(
        dataset_root=dataset_root,
        labels_csv_path=labels_csv_path,
        audio_base_dir=audio_base_dir,
        ontology=ontology,
        default_language=default_language,
        max_failed_file_ratio=max_failed_file_ratio,
        output_path=output_path,
        spec=ESCORPUS_PE_MANIFEST_SPEC,
    )


def build_mesd_manifest_jsonl(
    *,
    dataset_root: Path,
    labels_csv_path: Path,
    audio_base_dir: Path | None,
    ontology: LabelOntology,
    default_language: str,
    max_failed_file_ratio: float,
    output_path: Path,
) -> list[Utterance] | None:
    """Builds utterances and persists a MESD JSONL manifest."""

    return _build_manifest_jsonl(
        dataset_root=dataset_root,
        labels_csv_path=labels_csv_path,
        audio_base_dir=audio_base_dir,
        ontology=ontology,
        default_language=default_language,
        max_failed_file_ratio=max_failed_file_ratio,
        output_path=output_path,
        spec=MESD_MANIFEST_SPEC,
    )


def build_oreau_french_esd_manifest_jsonl(
    *,
    dataset_root: Path,
    labels_csv_path: Path,
    audio_base_dir: Path | None,
    ontology: LabelOntology,
    default_language: str,
    max_failed_file_ratio: float,
    output_path: Path,
) -> list[Utterance] | None:
    """Builds utterances and persists an Oreau French ESD JSONL manifest."""

    return _build_manifest_jsonl(
        dataset_root=dataset_root,
        labels_csv_path=labels_csv_path,
        audio_base_dir=audio_base_dir,
        ontology=ontology,
        default_language=default_language,
        max_failed_file_ratio=max_failed_file_ratio,
        output_path=output_path,
        spec=OREAU_FRENCH_ESD_MANIFEST_SPEC,
    )


def build_jl_corpus_manifest_jsonl(
    *,
    dataset_root: Path,
    labels_csv_path: Path,
    audio_base_dir: Path | None,
    ontology: LabelOntology,
    default_language: str,
    max_failed_file_ratio: float,
    output_path: Path,
) -> list[Utterance] | None:
    """Builds utterances and persists a JL-Corpus JSONL manifest."""

    return _build_manifest_jsonl(
        dataset_root=dataset_root,
        labels_csv_path=labels_csv_path,
        audio_base_dir=audio_base_dir,
        ontology=ontology,
        default_language=default_language,
        max_failed_file_ratio=max_failed_file_ratio,
        output_path=output_path,
        spec=JL_CORPUS_MANIFEST_SPEC,
    )


def build_cafe_manifest_jsonl(
    *,
    dataset_root: Path,
    labels_csv_path: Path,
    audio_base_dir: Path | None,
    ontology: LabelOntology,
    default_language: str,
    max_failed_file_ratio: float,
    output_path: Path,
) -> list[Utterance] | None:
    """Builds utterances and persists a CaFE JSONL manifest."""

    return _build_manifest_jsonl(
        dataset_root=dataset_root,
        labels_csv_path=labels_csv_path,
        audio_base_dir=audio_base_dir,
        ontology=ontology,
        default_language=default_language,
        max_failed_file_ratio=max_failed_file_ratio,
        output_path=output_path,
        spec=CAFE_MANIFEST_SPEC,
    )


def build_asvp_esd_manifest_jsonl(
    *,
    dataset_root: Path,
    labels_csv_path: Path,
    audio_base_dir: Path | None,
    ontology: LabelOntology,
    default_language: str,
    max_failed_file_ratio: float,
    output_path: Path,
) -> list[Utterance] | None:
    """Builds utterances and persists an ASVP-ESD JSONL manifest."""

    return _build_manifest_jsonl(
        dataset_root=dataset_root,
        labels_csv_path=labels_csv_path,
        audio_base_dir=audio_base_dir,
        ontology=ontology,
        default_language=default_language,
        max_failed_file_ratio=max_failed_file_ratio,
        output_path=output_path,
        spec=ASVP_ESD_MANIFEST_SPEC,
    )


def build_emov_db_manifest_jsonl(
    *,
    dataset_root: Path,
    labels_csv_path: Path,
    audio_base_dir: Path | None,
    ontology: LabelOntology,
    default_language: str,
    max_failed_file_ratio: float,
    output_path: Path,
) -> list[Utterance] | None:
    """Builds utterances and persists an EmoV-DB JSONL manifest."""

    return _build_manifest_jsonl(
        dataset_root=dataset_root,
        labels_csv_path=labels_csv_path,
        audio_base_dir=audio_base_dir,
        ontology=ontology,
        default_language=default_language,
        max_failed_file_ratio=max_failed_file_ratio,
        output_path=output_path,
        spec=EMOV_DB_MANIFEST_SPEC,
    )


def build_pavoque_manifest_jsonl(
    *,
    dataset_root: Path,
    labels_csv_path: Path,
    audio_base_dir: Path | None,
    ontology: LabelOntology,
    default_language: str,
    max_failed_file_ratio: float,
    output_path: Path,
) -> list[Utterance] | None:
    """Builds utterances and persists a PAVOQUE JSONL manifest."""

    return _build_manifest_jsonl(
        dataset_root=dataset_root,
        labels_csv_path=labels_csv_path,
        audio_base_dir=audio_base_dir,
        ontology=ontology,
        default_language=default_language,
        max_failed_file_ratio=max_failed_file_ratio,
        output_path=output_path,
        spec=PAVOQUE_MANIFEST_SPEC,
    )


def build_att_hack_manifest_jsonl(
    *,
    dataset_root: Path,
    labels_csv_path: Path,
    audio_base_dir: Path | None,
    ontology: LabelOntology,
    default_language: str,
    max_failed_file_ratio: float,
    output_path: Path,
) -> list[Utterance] | None:
    """Builds utterances and persists an Att-HACK JSONL manifest."""

    return _build_manifest_jsonl(
        dataset_root=dataset_root,
        labels_csv_path=labels_csv_path,
        audio_base_dir=audio_base_dir,
        ontology=ontology,
        default_language=default_language,
        max_failed_file_ratio=max_failed_file_ratio,
        output_path=output_path,
        spec=ATT_HACK_MANIFEST_SPEC,
    )


def build_coraa_ser_manifest_jsonl(
    *,
    dataset_root: Path,
    labels_csv_path: Path,
    audio_base_dir: Path | None,
    ontology: LabelOntology,
    default_language: str,
    max_failed_file_ratio: float,
    output_path: Path,
) -> list[Utterance] | None:
    """Builds utterances and persists a CORAA SER JSONL manifest."""

    return _build_manifest_jsonl(
        dataset_root=dataset_root,
        labels_csv_path=labels_csv_path,
        audio_base_dir=audio_base_dir,
        ontology=ontology,
        default_language=default_language,
        max_failed_file_ratio=max_failed_file_ratio,
        output_path=output_path,
        spec=CORAA_SER_MANIFEST_SPEC,
    )


def build_spanish_meacorpus_2023_manifest_jsonl(
    *,
    dataset_root: Path,
    labels_csv_path: Path,
    audio_base_dir: Path | None,
    ontology: LabelOntology,
    default_language: str,
    max_failed_file_ratio: float,
    output_path: Path,
) -> list[Utterance] | None:
    """Builds utterances and persists a Spanish MEACorpus 2023 JSONL manifest."""

    return _build_manifest_jsonl(
        dataset_root=dataset_root,
        labels_csv_path=labels_csv_path,
        audio_base_dir=audio_base_dir,
        ontology=ontology,
        default_language=default_language,
        max_failed_file_ratio=max_failed_file_ratio,
        output_path=output_path,
        spec=SPANISH_MEACORPUS_2023_MANIFEST_SPEC,
    )
