"""Utterance-level corpus/class sampling and bounded window selection."""

from __future__ import annotations

import hashlib
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class UtteranceSamplingItem:
    """Minimal utterance metadata needed by the balanced sampler."""

    sample_id: str
    corpus: str
    label: str
    window_count: int
    duration_seconds: float | None = None

    def validate(self) -> None:
        """Validates item identity and bounded integer window count."""
        if not self.sample_id.strip() or not self.corpus.strip() or not self.label.strip():
            raise ValueError("Sampling item identifiers and label must be non-empty.")
        if self.window_count <= 0:
            raise ValueError("Sampling item window_count must be positive.")
        if self.duration_seconds is not None and self.duration_seconds <= 0.0:
            raise ValueError("Sampling item duration_seconds must be positive when provided.")


@dataclass(frozen=True)
class SamplingProbability:
    """Expected contribution of one utterance under hierarchical sampling."""

    sample_id: str
    corpus: str
    label: str
    probability: float


def utterance_sampling_distribution(
    items: list[UtteranceSamplingItem],
) -> tuple[SamplingProbability, ...]:
    """Computes ``sqrt(corpus)`` and inverse-``sqrt(class)`` sampling probabilities."""
    if not items:
        raise ValueError("Cannot build a sampling distribution for an empty dataset.")
    sample_ids: set[str] = set()
    corpus_counts: Counter[str] = Counter()
    class_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for item in items:
        item.validate()
        if item.sample_id in sample_ids:
            raise ValueError(f"Duplicate sampling sample_id {item.sample_id!r}.")
        sample_ids.add(item.sample_id)
        corpus_counts[item.corpus] += 1
        class_counts[item.corpus][item.label] += 1

    corpus_normalizer = sum(math.sqrt(count) for count in corpus_counts.values())
    class_normalizers = {
        corpus: sum(1.0 / math.sqrt(count) for count in counts.values())
        for corpus, counts in class_counts.items()
    }
    probabilities = []
    for item in items:
        corpus_probability = math.sqrt(corpus_counts[item.corpus]) / corpus_normalizer
        label_count = class_counts[item.corpus][item.label]
        class_probability = (1.0 / math.sqrt(label_count)) / class_normalizers[item.corpus]
        item_probability = corpus_probability * class_probability / label_count
        probabilities.append(
            SamplingProbability(item.sample_id, item.corpus, item.label, item_probability)
        )
    total = sum(row.probability for row in probabilities)
    if not math.isclose(total, 1.0, rel_tol=1e-12, abs_tol=1e-12):
        raise RuntimeError(f"Sampling probabilities do not sum to one: {total!r}.")
    return tuple(sorted(probabilities, key=lambda row: row.sample_id))


def select_training_windows(
    *,
    sample_id: str,
    window_count: int,
    max_windows: int,
    seed: int,
    epoch: int = 0,
) -> tuple[int, ...]:
    """Selects a deterministic random bounded window subset for one epoch."""
    if not sample_id.strip():
        raise ValueError("sample_id must be non-empty.")
    if window_count <= 0 or max_windows <= 0:
        raise ValueError("window_count and max_windows must be positive.")
    if epoch < 0:
        raise ValueError("epoch must be non-negative.")
    if window_count <= max_windows:
        return tuple(range(window_count))
    digest = hashlib.sha256(f"{seed}:{epoch}:{sample_id}".encode()).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    return tuple(sorted(rng.sample(range(window_count), max_windows)))


def sampling_contributions(
    items: list[UtteranceSamplingItem],
) -> dict[str, dict[str, float]]:
    """Reports expected sample and duration contributions by corpus and class."""
    item_by_id = {item.sample_id: item for item in items}
    probabilities = utterance_sampling_distribution(items)
    corpus: defaultdict[str, float] = defaultdict(float)
    classes: defaultdict[str, float] = defaultdict(float)
    duration: defaultdict[str, float] = defaultdict(float)
    for row in probabilities:
        corpus[row.corpus] += row.probability
        classes[f"{row.corpus}:{row.label}"] += row.probability
        seconds = item_by_id[row.sample_id].duration_seconds
        if seconds is not None:
            duration[row.corpus] += row.probability * seconds
    return {
        "corpus": dict(sorted(corpus.items())),
        "class": dict(sorted(classes.items())),
        "expected_duration_seconds": dict(sorted(duration.items())),
    }
