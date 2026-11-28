"""Dataset-specific label inference helpers for public SER corpora."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from ser._internal.data.catalog.public_datasets import MESD_MANIFEST_SPEC

_TOKEN_LABEL_MAP: dict[str, str] = {
    "anger": "angry",
    "angry": "angry",
    "enojado": "angry",
    "enojo": "angry",
    "ira": "angry",
    "rabia": "angry",
    "furieux": "angry",
    "colere": "angry",
    "sad": "sad",
    "sadness": "sad",
    "triste": "sad",
    "tristeza": "sad",
    "happy": "happy",
    "happiness": "happy",
    "feliz": "happy",
    "alegre": "happy",
    "alegria": "happy",
    "joie": "happy",
    "fear": "fearful",
    "fearful": "fearful",
    "anxious": "fearful",
    "anxiety": "fearful",
    "miedo": "fearful",
    "temor": "fearful",
    "peur": "fearful",
    "disgust": "disgust",
    "disgusted": "disgust",
    "asco": "disgust",
    "degout": "disgust",
    "surprise": "surprised",
    "surprised": "surprised",
    "sorpresa": "surprised",
    "neutral": "neutral",
    "neutro": "neutral",
    "neutre": "neutral",
    "calm": "neutral",
    "calme": "neutral",
    "boredom": "neutral",
    "sleepy": "neutral",
    "sleepiness": "neutral",
    "amused": "happy",
    "tristesse": "sad",
    "contempt": "contempt",
    "desprecio": "contempt",
    "mepris": "contempt",
}

_MESD_NAME = re.compile(
    r"(Anger|Disgust|Fear|Happiness|Neutral|Sadness)_([FMC])_([AB])_(?:(L[12])_)?(.+)"
)

_ESCORPUS_PE_VAD_PATTERN = re.compile(r"-(\d{2})-(\d{2})-(\d{2})$")


def infer_label_from_path_tokens(path: Path) -> str | None:
    """Infers one canonical label from filename/parent-folder token hints."""
    joined = " ".join([path.stem.lower(), *[part.lower() for part in path.parent.parts if part]])
    tokens = [token for token in re.split(r"[^a-z0-9]+", joined) if token]
    for token in tokens:
        mapped = _TOKEN_LABEL_MAP.get(token)
        if mapped is not None:
            return mapped
    return None


def infer_escorpus_pe_label(path: Path) -> str | None:
    """Infers ESCorpus-PE labels from VAD suffixes using a conservative heuristic."""
    match = _ESCORPUS_PE_VAD_PATTERN.search(path.stem)
    if match is None:
        return infer_label_from_path_tokens(path)
    valence_str, arousal_str, dominance_str = match.groups()
    valence = int(valence_str)
    arousal = int(arousal_str)
    dominance = int(dominance_str)
    if valence >= 4 and arousal >= 4:
        return "happy"
    if valence <= 2 and arousal >= 4:
        return "angry" if dominance >= 3 else "fearful"
    if valence <= 2 and arousal <= 2:
        return "sad"
    if arousal >= 4 and valence == 3:
        return "surprised"
    if valence <= 2 and arousal == 3:
        return "disgust"
    return "neutral"


@dataclass(frozen=True)
class MesdFilename:
    """Source metadata encoded in a MESD v5 WAV filename."""

    emotion: str
    voice_type: str
    reduction_level: str | None

    @property
    def label(self) -> str:
        return MESD_MANIFEST_SPEC.label_mapping[self.emotion.lower()]


def parse_mesd_filename(path: Path) -> MesdFilename | None:
    """Parses the documented human and naturalness-reduced filename formats."""
    match = _MESD_NAME.fullmatch(path.stem) if path.suffix == ".wav" else None
    if match is None:
        return None
    emotion, voice_type, corpus, reduction_level, word = match.groups()
    if not all(token.isalpha() and token.islower() for token in word.split("_")):
        return None
    if reduction_level is not None and (voice_type, corpus) != ("F", "B"):
        return None
    return MesdFilename(emotion, voice_type, reduction_level)


def infer_mesd_label(path: Path) -> str | None:
    """Returns an exact categorical label for a human MESD clip only."""
    parsed = parse_mesd_filename(path)
    return parsed.label if parsed is not None and parsed.reduction_level is None else None


def infer_att_hack_label(path: Path) -> str | None:
    """Infers one ATT-HACK label from known keyword tokens."""
    joined = " ".join([path.stem.lower(), *[part.lower() for part in path.parent.parts if part]])
    tokens = [token for token in re.split(r"[^a-z0-9]+", joined) if token]
    known_labels = {"friendly", "distant", "dominant", "seductive"}
    for token in tokens:
        if token in known_labels:
            return token
    return None


def infer_coraa_ser_label(path: Path) -> str | None:
    """Infers one CORAA-SER label from canonical filename patterns."""
    normalized = path.stem.lower().replace("-", "_")
    compact = normalized.replace("_", "")
    if "nonneutralfemale" in compact:
        return "non_neutral_female"
    if "nonneutralmale" in compact:
        return "non_neutral_male"
    tokens = [token for token in re.split(r"[^a-z0-9]+", normalized) if token]
    if "neutral" in tokens:
        return "neutral"
    return None


__all__ = [
    "infer_att_hack_label",
    "infer_coraa_ser_label",
    "infer_escorpus_pe_label",
    "infer_label_from_path_tokens",
    "infer_mesd_label",
]
