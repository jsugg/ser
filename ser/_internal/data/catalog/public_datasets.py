"""Declarative catalog entries for CSV-labeled public datasets."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True, slots=True)
class CsvManifestSpec:
    """Metadata required to build a CSV-labeled manifest."""

    corpus_id: str
    dataset_policy_id: str
    dataset_license_id: str
    source_url: str
    label_mapping: Mapping[str, str]


def _freeze_mapping(entries: dict[str, str]) -> Mapping[str, str]:
    """Freeze label mappings so catalog specs remain immutable."""

    return MappingProxyType(dict(entries))


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

EMODB_2_MANIFEST_SPEC = CsvManifestSpec(
    corpus_id=EMODB_2_CORPUS_ID,
    dataset_policy_id=EMODB_2_DATASET_POLICY_ID,
    dataset_license_id=EMODB_2_DATASET_LICENSE_ID,
    source_url=EMODB_2_SOURCE_URL,
    label_mapping=_freeze_mapping(
        {
            "anger": "angry",
            "boredom": "neutral",
            "disgust": "disgust",
            "fear": "fearful",
            "happiness": "happy",
            "neutral": "neutral",
            "sadness": "sad",
        }
    ),
)

ESCORPUS_PE_MANIFEST_SPEC = CsvManifestSpec(
    corpus_id=ESCORPUS_PE_CORPUS_ID,
    dataset_policy_id=ESCORPUS_PE_DATASET_POLICY_ID,
    dataset_license_id=ESCORPUS_PE_DATASET_LICENSE_ID,
    source_url=ESCORPUS_PE_SOURCE_URL,
    label_mapping=_freeze_mapping(
        {
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
        }
    ),
)

MESD_MANIFEST_SPEC = CsvManifestSpec(
    corpus_id=MESD_CORPUS_ID,
    dataset_policy_id=MESD_DATASET_POLICY_ID,
    dataset_license_id=MESD_DATASET_LICENSE_ID,
    source_url=MESD_SOURCE_URL,
    label_mapping=_freeze_mapping(
        {
            "anger": "angry",
            "happiness": "happy",
            "sadness": "sad",
            "fear": "fearful",
            "disgust": "disgust",
            "neutral": "neutral",
        }
    ),
)

OREAU_FRENCH_ESD_MANIFEST_SPEC = CsvManifestSpec(
    corpus_id=OREAU_FRENCH_ESD_CORPUS_ID,
    dataset_policy_id=OREAU_FRENCH_ESD_DATASET_POLICY_ID,
    dataset_license_id=OREAU_FRENCH_ESD_DATASET_LICENSE_ID,
    source_url=OREAU_FRENCH_ESD_SOURCE_URL,
    label_mapping=_freeze_mapping(
        {
            "joie": "happy",
            "heureux": "happy",
            "colere": "angry",
            "peur": "fearful",
            "triste": "sad",
            "neutre": "neutral",
            "degout": "disgust",
            "surprise": "surprised",
        }
    ),
)

JL_CORPUS_MANIFEST_SPEC = CsvManifestSpec(
    corpus_id=JL_CORPUS_CORPUS_ID,
    dataset_policy_id=JL_CORPUS_DATASET_POLICY_ID,
    dataset_license_id=JL_CORPUS_DATASET_LICENSE_ID,
    source_url=JL_CORPUS_SOURCE_URL,
    label_mapping=_freeze_mapping(
        {
            "angry": "angry",
            "happy": "happy",
            "sad": "sad",
            "neutral": "neutral",
            "anxious": "fearful",
            "fearful": "fearful",
        }
    ),
)

CAFE_MANIFEST_SPEC = CsvManifestSpec(
    corpus_id=CAFE_CORPUS_ID,
    dataset_policy_id=CAFE_DATASET_POLICY_ID,
    dataset_license_id=CAFE_DATASET_LICENSE_ID,
    source_url=CAFE_SOURCE_URL,
    label_mapping=_freeze_mapping(
        {
            "colere": "angry",
            "tristesse": "sad",
            "joie": "happy",
            "peur": "fearful",
            "degout": "disgust",
            "surprise": "surprised",
            "neutre": "neutral",
        }
    ),
)

ASVP_ESD_MANIFEST_SPEC = CsvManifestSpec(
    corpus_id=ASVP_ESD_CORPUS_ID,
    dataset_policy_id=ASVP_ESD_DATASET_POLICY_ID,
    dataset_license_id=ASVP_ESD_DATASET_LICENSE_ID,
    source_url=ASVP_ESD_SOURCE_URL,
    label_mapping=_freeze_mapping(
        {
            "angry": "angry",
            "happy": "happy",
            "sad": "sad",
            "fearful": "fearful",
            "neutral": "neutral",
            "disgust": "disgust",
            "surprised": "surprised",
        }
    ),
)

EMOV_DB_MANIFEST_SPEC = CsvManifestSpec(
    corpus_id=EMOV_DB_CORPUS_ID,
    dataset_policy_id=EMOV_DB_DATASET_POLICY_ID,
    dataset_license_id=EMOV_DB_DATASET_LICENSE_ID,
    source_url=EMOV_DB_SOURCE_URL,
    label_mapping=_freeze_mapping(
        {
            "angry": "angry",
            "amused": "happy",
            "sleepy": "neutral",
            "neutral": "neutral",
        }
    ),
)

PAVOQUE_MANIFEST_SPEC = CsvManifestSpec(
    corpus_id=PAVOQUE_CORPUS_ID,
    dataset_policy_id=PAVOQUE_DATASET_POLICY_ID,
    dataset_license_id=PAVOQUE_DATASET_LICENSE_ID,
    source_url=PAVOQUE_SOURCE_URL,
    label_mapping=_freeze_mapping(
        {
            "angry": "angry",
            "amused": "happy",
            "sleepy": "neutral",
            "neutral": "neutral",
        }
    ),
)

ATT_HACK_MANIFEST_SPEC = CsvManifestSpec(
    corpus_id=ATT_HACK_CORPUS_ID,
    dataset_policy_id=ATT_HACK_DATASET_POLICY_ID,
    dataset_license_id=ATT_HACK_DATASET_LICENSE_ID,
    source_url=ATT_HACK_SOURCE_URL,
    label_mapping=_freeze_mapping(
        {
            "friendly": "friendly",
            "distant": "distant",
            "dominant": "dominant",
            "seductive": "seductive",
        }
    ),
)

CORAA_SER_MANIFEST_SPEC = CsvManifestSpec(
    corpus_id=CORAA_SER_CORPUS_ID,
    dataset_policy_id=CORAA_SER_DATASET_POLICY_ID,
    dataset_license_id=CORAA_SER_DATASET_LICENSE_ID,
    source_url=CORAA_SER_SOURCE_URL,
    label_mapping=_freeze_mapping(
        {
            "neutral": "neutral",
            "non_neutral_female": "non_neutral_female",
            "non_neutral_male": "non_neutral_male",
        }
    ),
)

SPANISH_MEACORPUS_2023_MANIFEST_SPEC = CsvManifestSpec(
    corpus_id=SPANISH_MEACORPUS_2023_CORPUS_ID,
    dataset_policy_id=SPANISH_MEACORPUS_2023_DATASET_POLICY_ID,
    dataset_license_id=SPANISH_MEACORPUS_2023_DATASET_LICENSE_ID,
    source_url=SPANISH_MEACORPUS_2023_SOURCE_URL,
    label_mapping=_freeze_mapping(
        {
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
        }
    ),
)

PUBLIC_CSV_MANIFEST_SPECS: Mapping[str, CsvManifestSpec] = MappingProxyType(
    {
        EMODB_2_CORPUS_ID: EMODB_2_MANIFEST_SPEC,
        ESCORPUS_PE_CORPUS_ID: ESCORPUS_PE_MANIFEST_SPEC,
        MESD_CORPUS_ID: MESD_MANIFEST_SPEC,
        OREAU_FRENCH_ESD_CORPUS_ID: OREAU_FRENCH_ESD_MANIFEST_SPEC,
        JL_CORPUS_CORPUS_ID: JL_CORPUS_MANIFEST_SPEC,
        CAFE_CORPUS_ID: CAFE_MANIFEST_SPEC,
        ASVP_ESD_CORPUS_ID: ASVP_ESD_MANIFEST_SPEC,
        EMOV_DB_CORPUS_ID: EMOV_DB_MANIFEST_SPEC,
        PAVOQUE_CORPUS_ID: PAVOQUE_MANIFEST_SPEC,
        ATT_HACK_CORPUS_ID: ATT_HACK_MANIFEST_SPEC,
        CORAA_SER_CORPUS_ID: CORAA_SER_MANIFEST_SPEC,
        SPANISH_MEACORPUS_2023_CORPUS_ID: SPANISH_MEACORPUS_2023_MANIFEST_SPEC,
    }
)

__all__ = [
    "ASVP_ESD_CORPUS_ID",
    "ASVP_ESD_DATASET_LICENSE_ID",
    "ASVP_ESD_DATASET_POLICY_ID",
    "ASVP_ESD_MANIFEST_SPEC",
    "ASVP_ESD_SOURCE_URL",
    "ATT_HACK_CORPUS_ID",
    "ATT_HACK_DATASET_LICENSE_ID",
    "ATT_HACK_DATASET_POLICY_ID",
    "ATT_HACK_MANIFEST_SPEC",
    "ATT_HACK_SOURCE_URL",
    "CAFE_CORPUS_ID",
    "CAFE_DATASET_LICENSE_ID",
    "CAFE_DATASET_POLICY_ID",
    "CAFE_MANIFEST_SPEC",
    "CAFE_SOURCE_URL",
    "CORAA_SER_CORPUS_ID",
    "CORAA_SER_DATASET_LICENSE_ID",
    "CORAA_SER_DATASET_POLICY_ID",
    "CORAA_SER_MANIFEST_SPEC",
    "CORAA_SER_SOURCE_URL",
    "CsvManifestSpec",
    "EMODB_2_CORPUS_ID",
    "EMODB_2_DATASET_LICENSE_ID",
    "EMODB_2_DATASET_POLICY_ID",
    "EMODB_2_MANIFEST_SPEC",
    "EMODB_2_SOURCE_URL",
    "EMOV_DB_CORPUS_ID",
    "EMOV_DB_DATASET_LICENSE_ID",
    "EMOV_DB_DATASET_POLICY_ID",
    "EMOV_DB_MANIFEST_SPEC",
    "EMOV_DB_SOURCE_URL",
    "ESCORPUS_PE_CORPUS_ID",
    "ESCORPUS_PE_DATASET_LICENSE_ID",
    "ESCORPUS_PE_DATASET_POLICY_ID",
    "ESCORPUS_PE_MANIFEST_SPEC",
    "ESCORPUS_PE_SOURCE_URL",
    "JL_CORPUS_CORPUS_ID",
    "JL_CORPUS_DATASET_LICENSE_ID",
    "JL_CORPUS_DATASET_POLICY_ID",
    "JL_CORPUS_MANIFEST_SPEC",
    "JL_CORPUS_SOURCE_URL",
    "MESD_CORPUS_ID",
    "MESD_DATASET_LICENSE_ID",
    "MESD_DATASET_POLICY_ID",
    "MESD_MANIFEST_SPEC",
    "MESD_SOURCE_URL",
    "OREAU_FRENCH_ESD_CORPUS_ID",
    "OREAU_FRENCH_ESD_DATASET_LICENSE_ID",
    "OREAU_FRENCH_ESD_DATASET_POLICY_ID",
    "OREAU_FRENCH_ESD_MANIFEST_SPEC",
    "OREAU_FRENCH_ESD_SOURCE_URL",
    "PAVOQUE_CORPUS_ID",
    "PAVOQUE_DATASET_LICENSE_ID",
    "PAVOQUE_DATASET_POLICY_ID",
    "PAVOQUE_MANIFEST_SPEC",
    "PAVOQUE_SOURCE_URL",
    "PUBLIC_CSV_MANIFEST_SPECS",
    "SPANISH_MEACORPUS_2023_CORPUS_ID",
    "SPANISH_MEACORPUS_2023_DATASET_LICENSE_ID",
    "SPANISH_MEACORPUS_2023_DATASET_POLICY_ID",
    "SPANISH_MEACORPUS_2023_MANIFEST_SPEC",
    "SPANISH_MEACORPUS_2023_SOURCE_URL",
]
