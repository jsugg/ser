"""Dataset adapters for building training manifests."""

from __future__ import annotations

from .biic_podcast import (
    BIIC_PODCAST_CORPUS_ID,
    build_biic_podcast_manifest_jsonl,
    build_biic_podcast_utterances,
)
from .crema_d import (
    CREMA_D_CORPUS_ID,
    build_crema_d_manifest_jsonl,
    build_crema_d_utterances,
)
from .msp_podcast import (
    MSP_PODCAST_CORPUS_ID,
    build_msp_podcast_manifest_jsonl,
    build_msp_podcast_utterances,
)
from .public_csv_datasets import (
    ASVP_ESD_CORPUS_ID,
    ATT_HACK_CORPUS_ID,
    CAFE_CORPUS_ID,
    CORAA_SER_CORPUS_ID,
    EMODB_2_CORPUS_ID,
    EMOV_DB_CORPUS_ID,
    ESCORPUS_PE_CORPUS_ID,
    JL_CORPUS_CORPUS_ID,
    MESD_CORPUS_ID,
    OREAU_FRENCH_ESD_CORPUS_ID,
    PAVOQUE_CORPUS_ID,
    SPANISH_MEACORPUS_2023_CORPUS_ID,
    build_asvp_esd_manifest_jsonl,
    build_att_hack_manifest_jsonl,
    build_cafe_manifest_jsonl,
    build_coraa_ser_manifest_jsonl,
    build_emodb_2_manifest_jsonl,
    build_emov_db_manifest_jsonl,
    build_escorpus_pe_manifest_jsonl,
    build_jl_corpus_manifest_jsonl,
    build_mesd_manifest_jsonl,
    build_oreau_french_esd_manifest_jsonl,
    build_pavoque_manifest_jsonl,
    build_spanish_meacorpus_2023_manifest_jsonl,
)
from .ravdess import (
    RAVDESS_CORPUS_ID,
    build_ravdess_manifest_jsonl,
    build_ravdess_utterances,
)

__all__ = [
    "ASVP_ESD_CORPUS_ID",
    "ATT_HACK_CORPUS_ID",
    "BIIC_PODCAST_CORPUS_ID",
    "CAFE_CORPUS_ID",
    "CORAA_SER_CORPUS_ID",
    "CREMA_D_CORPUS_ID",
    "EMODB_2_CORPUS_ID",
    "EMOV_DB_CORPUS_ID",
    "ESCORPUS_PE_CORPUS_ID",
    "JL_CORPUS_CORPUS_ID",
    "MESD_CORPUS_ID",
    "MSP_PODCAST_CORPUS_ID",
    "OREAU_FRENCH_ESD_CORPUS_ID",
    "PAVOQUE_CORPUS_ID",
    "RAVDESS_CORPUS_ID",
    "SPANISH_MEACORPUS_2023_CORPUS_ID",
    "build_asvp_esd_manifest_jsonl",
    "build_att_hack_manifest_jsonl",
    "build_biic_podcast_manifest_jsonl",
    "build_biic_podcast_utterances",
    "build_cafe_manifest_jsonl",
    "build_coraa_ser_manifest_jsonl",
    "build_crema_d_manifest_jsonl",
    "build_crema_d_utterances",
    "build_emodb_2_manifest_jsonl",
    "build_emov_db_manifest_jsonl",
    "build_escorpus_pe_manifest_jsonl",
    "build_jl_corpus_manifest_jsonl",
    "build_mesd_manifest_jsonl",
    "build_msp_podcast_manifest_jsonl",
    "build_msp_podcast_utterances",
    "build_oreau_french_esd_manifest_jsonl",
    "build_pavoque_manifest_jsonl",
    "build_ravdess_manifest_jsonl",
    "build_ravdess_utterances",
    "build_spanish_meacorpus_2023_manifest_jsonl",
]
