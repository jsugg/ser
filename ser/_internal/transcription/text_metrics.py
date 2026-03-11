"""Text-metric helpers for transcription profiling."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence

from ser.domain import TranscriptWord


def normalize_words(text: str) -> list[str]:
    """Normalize transcript text into comparable token lists."""

    lowered = text.strip().lower()
    normalized = re.sub(r"[^a-z0-9 ]+", " ", lowered)
    return [token for token in normalized.split() if token]


def levenshtein_distance(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    """Compute token-level Levenshtein distance."""

    if not reference:
        return len(hypothesis)
    if not hypothesis:
        return len(reference)

    previous_row = list(range(len(hypothesis) + 1))
    for ref_index, ref_token in enumerate(reference, start=1):
        current_row = [ref_index]
        for hyp_index, hyp_token in enumerate(hypothesis, start=1):
            insert_cost = current_row[hyp_index - 1] + 1
            delete_cost = previous_row[hyp_index] + 1
            substitute_cost = previous_row[hyp_index - 1] + (0 if ref_token == hyp_token else 1)
            current_row.append(min(insert_cost, delete_cost, substitute_cost))
        previous_row = current_row
    return previous_row[-1]


def compute_word_error_rate(reference_text: str, hypothesis_text: str) -> float:
    """Compute word error rate using normalized token sequences."""

    reference_tokens = normalize_words(reference_text)
    hypothesis_tokens = normalize_words(hypothesis_text)
    if not reference_tokens:
        return 0.0 if not hypothesis_tokens else 1.0
    distance = levenshtein_distance(reference_tokens, hypothesis_tokens)
    return distance / float(len(reference_tokens))


def transcript_words_to_text(words: Sequence[TranscriptWord]) -> str:
    """Convert per-word transcript entries into plain normalized text."""

    return " ".join(word.word.strip() for word in words if word.word.strip())


def percentile(values: Sequence[float], percentile: float) -> float:
    """Return a nearest-rank percentile for possibly empty numeric samples."""

    if not values:
        return 1.0
    rank = max(0, math.ceil(percentile * len(values)) - 1)
    return sorted(values)[rank]


__all__ = [
    "compute_word_error_rate",
    "levenshtein_distance",
    "normalize_words",
    "percentile",
    "transcript_words_to_text",
]
