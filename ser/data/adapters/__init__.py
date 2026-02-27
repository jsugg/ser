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
from .ravdess import (
    RAVDESS_CORPUS_ID,
    build_ravdess_manifest_jsonl,
    build_ravdess_utterances,
)

__all__ = [
    "BIIC_PODCAST_CORPUS_ID",
    "CREMA_D_CORPUS_ID",
    "MSP_PODCAST_CORPUS_ID",
    "RAVDESS_CORPUS_ID",
    "build_biic_podcast_manifest_jsonl",
    "build_biic_podcast_utterances",
    "build_crema_d_manifest_jsonl",
    "build_crema_d_utterances",
    "build_msp_podcast_manifest_jsonl",
    "build_msp_podcast_utterances",
    "build_ravdess_manifest_jsonl",
    "build_ravdess_utterances",
]
