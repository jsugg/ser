"""Dataset capability catalog for pipeline-planning workflows."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DatasetCapabilityProfile:
    """Static capability profile for one supported dataset."""

    dataset_id: str
    summary: str
    modalities: tuple[str, ...]
    label_schema: str
    has_label_mapping: bool
    supervised_ser_candidate: bool
    ssl_candidate: bool
    multimodal_candidate: bool
    mergeable_with_emotion_ontology: bool
    recommended_uses: tuple[str, ...]
    notes: tuple[str, ...]


_PROFILES: dict[str, DatasetCapabilityProfile] = {
    "ravdess": DatasetCapabilityProfile(
        dataset_id="ravdess",
        summary="North American acted emotional speech (24 actors).",
        modalities=("audio",),
        label_schema="emotion_8_class",
        has_label_mapping=True,
        supervised_ser_candidate=True,
        ssl_candidate=True,
        multimodal_candidate=False,
        mergeable_with_emotion_ontology=True,
        recommended_uses=(
            "supervised_ser_training",
            "cross_corpus_emotion_merge",
            "ssl_pretraining",
        ),
        notes=("Acted speech; no transcript channel in current pipeline.",),
    ),
    "crema-d": DatasetCapabilityProfile(
        dataset_id="crema-d",
        summary="Acted emotional speech with canonical emotion categories.",
        modalities=("audio",),
        label_schema="emotion_6_class",
        has_label_mapping=True,
        supervised_ser_candidate=True,
        ssl_candidate=True,
        multimodal_candidate=False,
        mergeable_with_emotion_ontology=True,
        recommended_uses=(
            "supervised_ser_training",
            "cross_corpus_emotion_merge",
            "ssl_pretraining",
        ),
        notes=("Acted speech; no transcript channel in current pipeline.",),
    ),
    "msp-podcast": DatasetCapabilityProfile(
        dataset_id="msp-podcast",
        summary="Conversational segments with challenge-style categorical emotion labels.",
        modalities=("audio", "label_csv"),
        label_schema="emotion_8_class",
        has_label_mapping=True,
        supervised_ser_candidate=True,
        ssl_candidate=True,
        multimodal_candidate=False,
        mergeable_with_emotion_ontology=True,
        recommended_uses=(
            "supervised_ser_training",
            "cross_corpus_emotion_merge",
            "ssl_pretraining",
        ),
        notes=("Academic-license constraints apply.",),
    ),
    "emodb-2.0": DatasetCapabilityProfile(
        dataset_id="emodb-2.0",
        summary="German emotional speech corpus with provided metadata labels.",
        modalities=("audio", "metadata_csv"),
        label_schema="emotion_mapped_to_canonical",
        has_label_mapping=True,
        supervised_ser_candidate=True,
        ssl_candidate=True,
        multimodal_candidate=False,
        mergeable_with_emotion_ontology=True,
        recommended_uses=(
            "supervised_ser_training",
            "cross_lingual_emotion_transfer",
            "ssl_pretraining",
        ),
        notes=("Label mapping normalizes boredom into neutral.",),
    ),
    "escorpus-pe": DatasetCapabilityProfile(
        dataset_id="escorpus-pe",
        summary="Peruvian Spanish speech corpus with filename-encoded VAD dimensions.",
        modalities=("audio", "dimensional_annotations"),
        label_schema="vad_heuristic_to_canonical",
        has_label_mapping=False,
        supervised_ser_candidate=False,
        ssl_candidate=True,
        multimodal_candidate=False,
        mergeable_with_emotion_ontology=False,
        recommended_uses=(
            "ssl_pretraining",
            "domain_adaptation",
            "representation_learning",
        ),
        notes=(
            "VAD triplets are converted to weak categorical proxies; not recommended as a primary supervised target.",
        ),
    ),
    "mesd": DatasetCapabilityProfile(
        dataset_id="mesd",
        summary="Mexican Spanish emotional speech dataset.",
        modalities=("audio", "metadata"),
        label_schema="emotion_mapped_to_canonical",
        has_label_mapping=True,
        supervised_ser_candidate=True,
        ssl_candidate=True,
        multimodal_candidate=False,
        mergeable_with_emotion_ontology=True,
        recommended_uses=(
            "supervised_ser_training",
            "cross_lingual_emotion_transfer",
            "ssl_pretraining",
        ),
        notes=("Labels inferred from canonical filename prefixes.",),
    ),
    "oreau-french-esd": DatasetCapabilityProfile(
        dataset_id="oreau-french-esd",
        summary="French expressive speech dataset distributed as multi-part RAR.",
        modalities=("audio",),
        label_schema="emotion_inferred_from_paths",
        has_label_mapping=True,
        supervised_ser_candidate=True,
        ssl_candidate=True,
        multimodal_candidate=False,
        mergeable_with_emotion_ontology=True,
        recommended_uses=(
            "supervised_ser_training",
            "cross_lingual_emotion_transfer",
            "ssl_pretraining",
        ),
        notes=("Requires external RAR extraction backend in runtime environment.",),
    ),
    "jl-corpus": DatasetCapabilityProfile(
        dataset_id="jl-corpus",
        summary="English emotional speech corpus with anxious label variant.",
        modalities=("audio",),
        label_schema="emotion_mapped_to_canonical",
        has_label_mapping=True,
        supervised_ser_candidate=True,
        ssl_candidate=True,
        multimodal_candidate=False,
        mergeable_with_emotion_ontology=True,
        recommended_uses=(
            "supervised_ser_training",
            "cross_corpus_emotion_merge",
            "ssl_pretraining",
        ),
        notes=(
            "Primary source is Kaggle; automation falls back to Hugging Face rows API when Kaggle credentials are unavailable.",
        ),
    ),
    "cafe": DatasetCapabilityProfile(
        dataset_id="cafe",
        summary="Canadian French emotional speech corpus (CaFE).",
        modalities=("audio",),
        label_schema="emotion_mapped_to_canonical",
        has_label_mapping=True,
        supervised_ser_candidate=True,
        ssl_candidate=True,
        multimodal_candidate=False,
        mergeable_with_emotion_ontology=True,
        recommended_uses=(
            "supervised_ser_training",
            "cross_lingual_emotion_transfer",
            "ssl_pretraining",
        ),
        notes=("Noncommercial license constraints apply.",),
    ),
    "asvp-esd": DatasetCapabilityProfile(
        dataset_id="asvp-esd",
        summary="Spanish emotional speech dataset packaged on Zenodo.",
        modalities=("audio",),
        label_schema="emotion_inferred_from_paths",
        has_label_mapping=True,
        supervised_ser_candidate=True,
        ssl_candidate=True,
        multimodal_candidate=False,
        mergeable_with_emotion_ontology=True,
        recommended_uses=(
            "supervised_ser_training",
            "cross_lingual_emotion_transfer",
            "ssl_pretraining",
        ),
        notes=("Label extraction is filename/path-driven in current automation.",),
    ),
    "emov-db": DatasetCapabilityProfile(
        dataset_id="emov-db",
        summary="OpenSLR SLR115 emotional voice database.",
        modalities=("audio",),
        label_schema="emotion_mapped_to_canonical",
        has_label_mapping=True,
        supervised_ser_candidate=True,
        ssl_candidate=True,
        multimodal_candidate=False,
        mergeable_with_emotion_ontology=True,
        recommended_uses=(
            "supervised_ser_training",
            "cross_corpus_emotion_merge",
            "ssl_pretraining",
        ),
        notes=("Noncommercial license constraints apply.",),
    ),
    "pavoque": DatasetCapabilityProfile(
        dataset_id="pavoque",
        summary="PAVOQUE expressive corpus from GitHub release assets.",
        modalities=("audio", "timing_metadata"),
        label_schema="emotion_mapped_to_canonical",
        has_label_mapping=True,
        supervised_ser_candidate=True,
        ssl_candidate=True,
        multimodal_candidate=False,
        mergeable_with_emotion_ontology=True,
        recommended_uses=(
            "supervised_ser_training",
            "cross_corpus_emotion_merge",
            "ssl_pretraining",
        ),
        notes=("Current pipeline uses filename-driven labels; YAML segmentation is pending.",),
    ),
    "att-hack": DatasetCapabilityProfile(
        dataset_id="att-hack",
        summary="French social-attitude corpus (friendly/distant/dominant/seductive).",
        modalities=("audio", "text"),
        label_schema="social_attitude_classes",
        has_label_mapping=True,
        supervised_ser_candidate=False,
        ssl_candidate=True,
        multimodal_candidate=True,
        mergeable_with_emotion_ontology=False,
        recommended_uses=(
            "ssl_pretraining",
            "paralinguistic_style_modeling",
            "audio_text_representation_learning",
        ),
        notes=("Labels are not canonical SER emotions; suitable for auxiliary tasks and SSL.",),
    ),
    "coraa-ser": DatasetCapabilityProfile(
        dataset_id="coraa-ser",
        summary="Brazilian Portuguese challenge corpus with neutral/non-neutral labels.",
        modalities=("audio",),
        label_schema="binary_plus_gender_non_neutral",
        has_label_mapping=True,
        supervised_ser_candidate=False,
        ssl_candidate=True,
        multimodal_candidate=False,
        mergeable_with_emotion_ontology=False,
        recommended_uses=(
            "ssl_pretraining",
            "binary_affect_detection",
            "domain_adaptation",
        ),
        notes=("Labels are challenge-specific; not canonical emotion classes.",),
    ),
    "spanish-meacorpus-2023": DatasetCapabilityProfile(
        dataset_id="spanish-meacorpus-2023",
        summary="Spanish multimodal emotion metadata corpus from YouTube.",
        modalities=("audio", "text", "metadata"),
        label_schema="emotion_mapped_to_canonical",
        has_label_mapping=True,
        supervised_ser_candidate=True,
        ssl_candidate=True,
        multimodal_candidate=True,
        mergeable_with_emotion_ontology=True,
        recommended_uses=(
            "supervised_ser_training",
            "audio_text_emotion_fusion",
            "ssl_pretraining",
        ),
        notes=("Zenodo ships metadata/transcripts; audio needs local rehydration from YouTube.",),
    ),
    "biic-podcast": DatasetCapabilityProfile(
        dataset_id="biic-podcast",
        summary="BIIC podcast corpus with access and labeling workflow managed externally.",
        modalities=("audio", "external_labels"),
        label_schema="externally_managed",
        has_label_mapping=False,
        supervised_ser_candidate=False,
        ssl_candidate=True,
        multimodal_candidate=False,
        mergeable_with_emotion_ontology=False,
        recommended_uses=("ssl_pretraining", "domain_adaptation"),
        notes=("Manual access and custom label pipeline are required.",),
    ),
}


def resolve_dataset_capability_profile(dataset_id: str) -> DatasetCapabilityProfile:
    """Resolves one static capability profile for a supported dataset id."""

    profile = _PROFILES.get(dataset_id)
    if profile is None:
        raise ValueError(f"No capability profile registered for dataset {dataset_id!r}.")
    return profile


def list_dataset_capability_profiles() -> tuple[DatasetCapabilityProfile, ...]:
    """Returns all capability profiles ordered by dataset id."""

    return tuple(_PROFILES[key] for key in sorted(_PROFILES))
