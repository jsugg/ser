"""Adapters for CSV-labeled public datasets."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ser.data.manifest import SplitName, Utterance
from ser.data.manifest_jsonl import write_manifest_jsonl
from ser.data.ontology import LabelOntology, remap_label
from ser.utils.logger import get_logger

logger = get_logger(__name__)

EMODB_2_CORPUS_ID = "emodb-2.0"
EMODB_2_DATASET_POLICY_ID = "open"
EMODB_2_DATASET_LICENSE_ID = "cc-by-4.0"
EMODB_2_SOURCE_URL = "https://zenodo.org/records/17651657"

ESCORPUS_PE_CORPUS_ID = "escorpus-pe"
ESCORPUS_PE_DATASET_POLICY_ID = "open"
ESCORPUS_PE_DATASET_LICENSE_ID = "cc-by-4.0"
ESCORPUS_PE_SOURCE_URL = "https://zenodo.org/records/5793223"

MESD_CORPUS_ID = "mesd"
MESD_DATASET_POLICY_ID = "open"
MESD_DATASET_LICENSE_ID = "cc-by-4.0"
MESD_SOURCE_URL = "https://data.mendeley.com/datasets/cy34mh68j9/5"

OREAU_FRENCH_ESD_CORPUS_ID = "oreau-french-esd"
OREAU_FRENCH_ESD_DATASET_POLICY_ID = "open"
OREAU_FRENCH_ESD_DATASET_LICENSE_ID = "cc-by-4.0"
OREAU_FRENCH_ESD_SOURCE_URL = "https://zenodo.org/records/4405783"

JL_CORPUS_CORPUS_ID = "jl-corpus"
JL_CORPUS_DATASET_POLICY_ID = "open"
JL_CORPUS_DATASET_LICENSE_ID = "cc0-1.0"
JL_CORPUS_SOURCE_URL = "https://www.kaggle.com/datasets/tli725/jl-corpus"

CAFE_CORPUS_ID = "cafe"
CAFE_DATASET_POLICY_ID = "noncommercial"
CAFE_DATASET_LICENSE_ID = "cc-by-nc-sa-4.0"
CAFE_SOURCE_URL = "https://zenodo.org/records/1478765"

ASVP_ESD_CORPUS_ID = "asvp-esd"
ASVP_ESD_DATASET_POLICY_ID = "open"
ASVP_ESD_DATASET_LICENSE_ID = "cc-by-4.0"
ASVP_ESD_SOURCE_URL = "https://zenodo.org/records/7132783"

EMOV_DB_CORPUS_ID = "emov-db"
EMOV_DB_DATASET_POLICY_ID = "noncommercial"
EMOV_DB_DATASET_LICENSE_ID = "custom-noncommercial"
EMOV_DB_SOURCE_URL = "https://www.openslr.org/115/"

PAVOQUE_CORPUS_ID = "pavoque"
PAVOQUE_DATASET_POLICY_ID = "noncommercial"
PAVOQUE_DATASET_LICENSE_ID = "cc-by-nc-sa-4.0"
PAVOQUE_SOURCE_URL = "https://github.com/marytts/pavoque-data/releases"

ATT_HACK_CORPUS_ID = "att-hack"
ATT_HACK_DATASET_POLICY_ID = "noncommercial"
ATT_HACK_DATASET_LICENSE_ID = "cc-by-nc-nd-4.0"
ATT_HACK_SOURCE_URL = "https://www.openslr.org/88/"

CORAA_SER_CORPUS_ID = "coraa-ser"
CORAA_SER_DATASET_POLICY_ID = "research_only"
CORAA_SER_DATASET_LICENSE_ID = "custom-research-only"
CORAA_SER_SOURCE_URL = "https://github.com/rmarcacini/ser-coraa-pt-br"

SPANISH_MEACORPUS_2023_CORPUS_ID = "spanish-meacorpus-2023"
SPANISH_MEACORPUS_2023_DATASET_POLICY_ID = "noncommercial"
SPANISH_MEACORPUS_2023_DATASET_LICENSE_ID = "cc-by-nc-4.0"
SPANISH_MEACORPUS_2023_SOURCE_URL = "https://zenodo.org/records/18606423"


@dataclass(frozen=True, slots=True)
class _CsvAdapterSpec:
    corpus_id: str
    dataset_policy_id: str
    dataset_license_id: str
    source_url: str
    label_mapping: dict[str, str]


def _normalize_split(value: str | None) -> SplitName | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if not normalized:
        return None
    if normalized in {"train", "training"}:
        return "train"
    if normalized in {"dev", "valid", "validation", "development"}:
        return "dev"
    if normalized in {"test", "evaluation", "eval"}:
        return "test"
    return None


def _read_optional_float(row: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        raw = row.get(key)
        if isinstance(raw, str):
            text = raw.strip()
            if not text:
                continue
            try:
                return float(text)
            except ValueError:
                continue
    return None


def _build_sample_id(
    *,
    corpus_id: str,
    file_name: str,
    start_seconds: float | None,
    duration_seconds: float | None,
) -> str:
    normalized = file_name.replace("\\", "/")
    base = f"{corpus_id}:{normalized}"
    if start_seconds is None or duration_seconds is None:
        return base
    return f"{base}@{start_seconds:.3f}+{duration_seconds:.3f}"


def _build_csv_labeled_utterances(
    *,
    dataset_root: Path,
    labels_csv_path: Path,
    audio_base_dir: Path | None,
    ontology: LabelOntology,
    default_language: str,
    max_failed_file_ratio: float,
    spec: _CsvAdapterSpec,
) -> list[Utterance] | None:
    if not labels_csv_path.is_file():
        logger.warning("%s labels CSV not found: %s", spec.corpus_id, labels_csv_path)
        return None

    base_dir = audio_base_dir if audio_base_dir is not None else dataset_root
    utterances: list[Utterance] = []
    parse_errors: list[str] = []

    with labels_csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader):
            file_name = (
                row.get("FileName")
                or row.get("filename")
                or row.get("file")
                or row.get("path")
                or ""
            ).strip()
            if not file_name:
                parse_errors.append(f"Row {index}: missing FileName")
                continue
            raw_label = (
                row.get("emotion")
                or row.get("EmoClass")
                or row.get("label")
                or row.get("class")
                or ""
            ).strip()
            if not raw_label:
                parse_errors.append(f"Row {index}: missing emotion for {file_name}")
                continue
            normalized_raw = raw_label.lower()
            mapped_raw = spec.label_mapping.get(normalized_raw, normalized_raw)
            mapped_label = remap_label(
                raw_label=mapped_raw, mapping=None, ontology=ontology
            )
            if mapped_label is None:
                continue

            split = _normalize_split(
                row.get("Split_Set")
                or row.get("split")
                or row.get("subset")
                or row.get("partition")
            )
            start_seconds = _read_optional_float(
                row,
                "start_seconds",
                "Start",
                "start",
                "start_time",
                "StartTime",
            )
            duration_seconds = _read_optional_float(
                row,
                "duration_seconds",
                "Duration",
                "duration",
            )
            end_seconds = _read_optional_float(
                row,
                "end_seconds",
                "End",
                "end",
                "end_time",
                "EndTime",
            )
            if (
                duration_seconds is None
                and start_seconds is not None
                and end_seconds is not None
            ):
                duration = end_seconds - start_seconds
                duration_seconds = duration if duration > 0.0 else None

            language = (
                row.get("Language") or row.get("language") or row.get("lang") or ""
            ).strip() or default_language
            speaker_raw = (
                row.get("Speaker")
                or row.get("Speaker_ID")
                or row.get("speaker")
                or row.get("actor")
                or ""
            ).strip()
            speaker_id = f"{spec.corpus_id}:{speaker_raw}" if speaker_raw else None
            sample_id = _build_sample_id(
                corpus_id=spec.corpus_id,
                file_name=file_name,
                start_seconds=start_seconds,
                duration_seconds=duration_seconds,
            )
            utterances.append(
                Utterance(
                    schema_version=1,
                    sample_id=sample_id,
                    corpus=spec.corpus_id,
                    audio_path=(base_dir / file_name).expanduser(),
                    label=mapped_label,
                    raw_label=raw_label,
                    speaker_id=speaker_id,
                    language=language,
                    split=split,
                    start_seconds=start_seconds,
                    duration_seconds=duration_seconds,
                    dataset_policy_id=spec.dataset_policy_id,
                    dataset_license_id=spec.dataset_license_id,
                    source_url=spec.source_url,
                )
            )

    if parse_errors:
        logger.warning(
            "Skipped %s rows while parsing %s labels.",
            len(parse_errors),
            spec.corpus_id,
        )
        for error in parse_errors[:5]:
            logger.warning("%s", error)

    total_rows = len(utterances) + len(parse_errors)
    if total_rows > 0:
        failure_ratio = float(len(parse_errors)) / float(total_rows)
        if failure_ratio > max_failed_file_ratio:
            raise RuntimeError(
                "Aborting data load: "
                f"{failure_ratio * 100.0:.1f}% label-row failures exceeded configured "
                "limit "
                f"{max_failed_file_ratio * 100.0:.1f}%."
            )
    if not utterances:
        logger.warning("No labeled audio samples matched configured emotions.")
        return None
    labels = [utterance.label for utterance in utterances]
    if len(set(labels)) < 2:
        logger.warning("At least two emotion classes are required to train the model.")
        return None
    return utterances


def _build_csv_labeled_manifest_jsonl(
    *,
    dataset_root: Path,
    labels_csv_path: Path,
    audio_base_dir: Path | None,
    ontology: LabelOntology,
    default_language: str,
    max_failed_file_ratio: float,
    output_path: Path,
    spec: _CsvAdapterSpec,
) -> list[Utterance] | None:
    utterances = _build_csv_labeled_utterances(
        dataset_root=dataset_root,
        labels_csv_path=labels_csv_path,
        audio_base_dir=audio_base_dir,
        ontology=ontology,
        default_language=default_language,
        max_failed_file_ratio=max_failed_file_ratio,
        spec=spec,
    )
    if utterances is None:
        return None
    write_manifest_jsonl(output_path, utterances, base_dir=dataset_root)
    return utterances


_EMODB_2_SPEC = _CsvAdapterSpec(
    corpus_id=EMODB_2_CORPUS_ID,
    dataset_policy_id=EMODB_2_DATASET_POLICY_ID,
    dataset_license_id=EMODB_2_DATASET_LICENSE_ID,
    source_url=EMODB_2_SOURCE_URL,
    label_mapping={
        "anger": "angry",
        "boredom": "neutral",
        "disgust": "disgust",
        "fear": "fearful",
        "happiness": "happy",
        "neutral": "neutral",
        "sadness": "sad",
    },
)

_ESCORPUS_PE_SPEC = _CsvAdapterSpec(
    corpus_id=ESCORPUS_PE_CORPUS_ID,
    dataset_policy_id=ESCORPUS_PE_DATASET_POLICY_ID,
    dataset_license_id=ESCORPUS_PE_DATASET_LICENSE_ID,
    source_url=ESCORPUS_PE_SOURCE_URL,
    label_mapping={
        "alegria": "happy",
        "feliz": "happy",
        "enojado": "angry",
        "enojo": "angry",
        "ira": "angry",
        "miedo": "fearful",
        "triste": "sad",
        "tristeza": "sad",
        "neutral": "neutral",
        "asco": "disgust",
        "sorpresa": "surprised",
    },
)

_MESD_SPEC = _CsvAdapterSpec(
    corpus_id=MESD_CORPUS_ID,
    dataset_policy_id=MESD_DATASET_POLICY_ID,
    dataset_license_id=MESD_DATASET_LICENSE_ID,
    source_url=MESD_SOURCE_URL,
    label_mapping={
        "anger": "angry",
        "happiness": "happy",
        "sadness": "sad",
        "fear": "fearful",
        "disgust": "disgust",
        "neutral": "neutral",
    },
)

_OREAU_FRENCH_ESD_SPEC = _CsvAdapterSpec(
    corpus_id=OREAU_FRENCH_ESD_CORPUS_ID,
    dataset_policy_id=OREAU_FRENCH_ESD_DATASET_POLICY_ID,
    dataset_license_id=OREAU_FRENCH_ESD_DATASET_LICENSE_ID,
    source_url=OREAU_FRENCH_ESD_SOURCE_URL,
    label_mapping={
        "joie": "happy",
        "heureux": "happy",
        "colere": "angry",
        "peur": "fearful",
        "triste": "sad",
        "neutre": "neutral",
        "degout": "disgust",
        "surprise": "surprised",
    },
)

_JL_CORPUS_SPEC = _CsvAdapterSpec(
    corpus_id=JL_CORPUS_CORPUS_ID,
    dataset_policy_id=JL_CORPUS_DATASET_POLICY_ID,
    dataset_license_id=JL_CORPUS_DATASET_LICENSE_ID,
    source_url=JL_CORPUS_SOURCE_URL,
    label_mapping={
        "angry": "angry",
        "happy": "happy",
        "sad": "sad",
        "neutral": "neutral",
        "anxious": "fearful",
        "fearful": "fearful",
    },
)

_CAFE_SPEC = _CsvAdapterSpec(
    corpus_id=CAFE_CORPUS_ID,
    dataset_policy_id=CAFE_DATASET_POLICY_ID,
    dataset_license_id=CAFE_DATASET_LICENSE_ID,
    source_url=CAFE_SOURCE_URL,
    label_mapping={
        "colere": "angry",
        "tristesse": "sad",
        "joie": "happy",
        "peur": "fearful",
        "degout": "disgust",
        "surprise": "surprised",
        "neutre": "neutral",
    },
)

_ASVP_ESD_SPEC = _CsvAdapterSpec(
    corpus_id=ASVP_ESD_CORPUS_ID,
    dataset_policy_id=ASVP_ESD_DATASET_POLICY_ID,
    dataset_license_id=ASVP_ESD_DATASET_LICENSE_ID,
    source_url=ASVP_ESD_SOURCE_URL,
    label_mapping={
        "angry": "angry",
        "happy": "happy",
        "sad": "sad",
        "fearful": "fearful",
        "neutral": "neutral",
        "disgust": "disgust",
        "surprised": "surprised",
    },
)

_EMOV_DB_SPEC = _CsvAdapterSpec(
    corpus_id=EMOV_DB_CORPUS_ID,
    dataset_policy_id=EMOV_DB_DATASET_POLICY_ID,
    dataset_license_id=EMOV_DB_DATASET_LICENSE_ID,
    source_url=EMOV_DB_SOURCE_URL,
    label_mapping={
        "angry": "angry",
        "amused": "happy",
        "sleepy": "neutral",
        "neutral": "neutral",
    },
)

_PAVOQUE_SPEC = _CsvAdapterSpec(
    corpus_id=PAVOQUE_CORPUS_ID,
    dataset_policy_id=PAVOQUE_DATASET_POLICY_ID,
    dataset_license_id=PAVOQUE_DATASET_LICENSE_ID,
    source_url=PAVOQUE_SOURCE_URL,
    label_mapping={
        "angry": "angry",
        "amused": "happy",
        "sleepy": "neutral",
        "neutral": "neutral",
    },
)

_ATT_HACK_SPEC = _CsvAdapterSpec(
    corpus_id=ATT_HACK_CORPUS_ID,
    dataset_policy_id=ATT_HACK_DATASET_POLICY_ID,
    dataset_license_id=ATT_HACK_DATASET_LICENSE_ID,
    source_url=ATT_HACK_SOURCE_URL,
    label_mapping={
        "friendly": "friendly",
        "distant": "distant",
        "dominant": "dominant",
        "seductive": "seductive",
    },
)

_CORAA_SER_SPEC = _CsvAdapterSpec(
    corpus_id=CORAA_SER_CORPUS_ID,
    dataset_policy_id=CORAA_SER_DATASET_POLICY_ID,
    dataset_license_id=CORAA_SER_DATASET_LICENSE_ID,
    source_url=CORAA_SER_SOURCE_URL,
    label_mapping={
        "neutral": "neutral",
        "non_neutral_female": "non_neutral_female",
        "non_neutral_male": "non_neutral_male",
    },
)

_SPANISH_MEACORPUS_2023_SPEC = _CsvAdapterSpec(
    corpus_id=SPANISH_MEACORPUS_2023_CORPUS_ID,
    dataset_policy_id=SPANISH_MEACORPUS_2023_DATASET_POLICY_ID,
    dataset_license_id=SPANISH_MEACORPUS_2023_DATASET_LICENSE_ID,
    source_url=SPANISH_MEACORPUS_2023_SOURCE_URL,
    label_mapping={
        "anger": "angry",
        "angry": "angry",
        "disgust": "disgust",
        "fear": "fearful",
        "fearful": "fearful",
        "joy": "happy",
        "happy": "happy",
        "neutral": "neutral",
        "sadness": "sad",
        "sad": "sad",
    },
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
    return _build_csv_labeled_manifest_jsonl(
        dataset_root=dataset_root,
        labels_csv_path=labels_csv_path,
        audio_base_dir=audio_base_dir,
        ontology=ontology,
        default_language=default_language,
        max_failed_file_ratio=max_failed_file_ratio,
        output_path=output_path,
        spec=_EMODB_2_SPEC,
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
    return _build_csv_labeled_manifest_jsonl(
        dataset_root=dataset_root,
        labels_csv_path=labels_csv_path,
        audio_base_dir=audio_base_dir,
        ontology=ontology,
        default_language=default_language,
        max_failed_file_ratio=max_failed_file_ratio,
        output_path=output_path,
        spec=_ESCORPUS_PE_SPEC,
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
    return _build_csv_labeled_manifest_jsonl(
        dataset_root=dataset_root,
        labels_csv_path=labels_csv_path,
        audio_base_dir=audio_base_dir,
        ontology=ontology,
        default_language=default_language,
        max_failed_file_ratio=max_failed_file_ratio,
        output_path=output_path,
        spec=_MESD_SPEC,
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
    return _build_csv_labeled_manifest_jsonl(
        dataset_root=dataset_root,
        labels_csv_path=labels_csv_path,
        audio_base_dir=audio_base_dir,
        ontology=ontology,
        default_language=default_language,
        max_failed_file_ratio=max_failed_file_ratio,
        output_path=output_path,
        spec=_OREAU_FRENCH_ESD_SPEC,
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
    return _build_csv_labeled_manifest_jsonl(
        dataset_root=dataset_root,
        labels_csv_path=labels_csv_path,
        audio_base_dir=audio_base_dir,
        ontology=ontology,
        default_language=default_language,
        max_failed_file_ratio=max_failed_file_ratio,
        output_path=output_path,
        spec=_JL_CORPUS_SPEC,
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
    return _build_csv_labeled_manifest_jsonl(
        dataset_root=dataset_root,
        labels_csv_path=labels_csv_path,
        audio_base_dir=audio_base_dir,
        ontology=ontology,
        default_language=default_language,
        max_failed_file_ratio=max_failed_file_ratio,
        output_path=output_path,
        spec=_CAFE_SPEC,
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
    return _build_csv_labeled_manifest_jsonl(
        dataset_root=dataset_root,
        labels_csv_path=labels_csv_path,
        audio_base_dir=audio_base_dir,
        ontology=ontology,
        default_language=default_language,
        max_failed_file_ratio=max_failed_file_ratio,
        output_path=output_path,
        spec=_ASVP_ESD_SPEC,
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
    return _build_csv_labeled_manifest_jsonl(
        dataset_root=dataset_root,
        labels_csv_path=labels_csv_path,
        audio_base_dir=audio_base_dir,
        ontology=ontology,
        default_language=default_language,
        max_failed_file_ratio=max_failed_file_ratio,
        output_path=output_path,
        spec=_EMOV_DB_SPEC,
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
    return _build_csv_labeled_manifest_jsonl(
        dataset_root=dataset_root,
        labels_csv_path=labels_csv_path,
        audio_base_dir=audio_base_dir,
        ontology=ontology,
        default_language=default_language,
        max_failed_file_ratio=max_failed_file_ratio,
        output_path=output_path,
        spec=_PAVOQUE_SPEC,
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
    return _build_csv_labeled_manifest_jsonl(
        dataset_root=dataset_root,
        labels_csv_path=labels_csv_path,
        audio_base_dir=audio_base_dir,
        ontology=ontology,
        default_language=default_language,
        max_failed_file_ratio=max_failed_file_ratio,
        output_path=output_path,
        spec=_ATT_HACK_SPEC,
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
    return _build_csv_labeled_manifest_jsonl(
        dataset_root=dataset_root,
        labels_csv_path=labels_csv_path,
        audio_base_dir=audio_base_dir,
        ontology=ontology,
        default_language=default_language,
        max_failed_file_ratio=max_failed_file_ratio,
        output_path=output_path,
        spec=_CORAA_SER_SPEC,
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
    return _build_csv_labeled_manifest_jsonl(
        dataset_root=dataset_root,
        labels_csv_path=labels_csv_path,
        audio_base_dir=audio_base_dir,
        ontology=ontology,
        default_language=default_language,
        max_failed_file_ratio=max_failed_file_ratio,
        output_path=output_path,
        spec=_SPANISH_MEACORPUS_2023_SPEC,
    )
