"""Explicit, versioned dataset recipes and task routing."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from ser._internal.data.manifest import Utterance
from ser._internal.data.ontology import normalize_label

DATASET_RECIPE_SCHEMA_VERSION = 1
CANONICAL_EMOTIONS = frozenset(
    {"neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"}
)

type TaskName = Literal[
    "primary_emotion",
    "raw_emotion",
    "vad",
    "attitude",
    "binary_affect",
    "language",
    "text_alignment",
    "ssl",
]
type RouteDisposition = Literal["accepted", "remapped", "weak", "dropped", "missing", "quarantined"]
_TASK_NAMES = frozenset(
    {
        "primary_emotion",
        "raw_emotion",
        "vad",
        "attitude",
        "binary_affect",
        "language",
        "text_alignment",
        "ssl",
    }
)


@dataclass(frozen=True)
class CorpusRecipe:
    """Task policy for one corpus."""

    corpus: str
    exact_primary_labels: frozenset[str] = frozenset()
    approximate_labels: frozenset[str] = frozenset()
    auxiliary_tasks: tuple[TaskName, ...] = ()

    def validate(self) -> None:
        """Validates one corpus routing policy."""
        if not self.corpus.strip():
            raise ValueError("Corpus recipe id must be non-empty.")
        if self.exact_primary_labels - CANONICAL_EMOTIONS:
            raise ValueError(f"Corpus {self.corpus!r} contains non-canonical primary labels.")
        if self.exact_primary_labels & self.approximate_labels:
            raise ValueError(f"Corpus {self.corpus!r} has labels marked exact and approximate.")
        if "primary_emotion" in self.auxiliary_tasks:
            raise ValueError("primary_emotion must be configured through exact_primary_labels.")
        if not set(self.auxiliary_tasks).issubset(_TASK_NAMES):
            raise ValueError(f"Corpus {self.corpus!r} contains unsupported auxiliary tasks.")

    def to_record(self) -> dict[str, object]:
        """Returns a deterministic JSON-compatible recipe record."""
        return {
            "corpus": self.corpus,
            "exact_primary_labels": sorted(self.exact_primary_labels),
            "approximate_labels": sorted(self.approximate_labels),
            "auxiliary_tasks": list(self.auxiliary_tasks),
        }


@dataclass(frozen=True)
class DatasetRecipe:
    """Versioned declaration of corpora, ontology, and training tasks."""

    recipe_id: str
    revision: str
    ontology_version: str
    corpora: tuple[CorpusRecipe, ...]
    schema_version: int = DATASET_RECIPE_SCHEMA_VERSION

    def validate(self) -> None:
        """Validates required revisions and unique corpus policies."""
        if self.schema_version != DATASET_RECIPE_SCHEMA_VERSION:
            raise ValueError(f"Unsupported dataset recipe schema {self.schema_version!r}.")
        for field_name, value in (
            ("recipe_id", self.recipe_id),
            ("revision", self.revision),
            ("ontology_version", self.ontology_version),
        ):
            if not value.strip():
                raise ValueError(f"Dataset recipe {field_name} must be non-empty.")
        corpus_ids: set[str] = set()
        for corpus in self.corpora:
            corpus.validate()
            if corpus.corpus in corpus_ids:
                raise ValueError(f"Duplicate corpus recipe {corpus.corpus!r}.")
            corpus_ids.add(corpus.corpus)
        if not corpus_ids:
            raise ValueError("Dataset recipe must include at least one corpus.")

    def to_record(self) -> dict[str, object]:
        """Returns the canonical JSON-compatible representation."""
        return {
            "schema_version": self.schema_version,
            "recipe_id": self.recipe_id,
            "revision": self.revision,
            "ontology_version": self.ontology_version,
            "corpora": [
                corpus.to_record() for corpus in sorted(self.corpora, key=lambda row: row.corpus)
            ],
        }

    @property
    def digest(self) -> str:
        """Returns the SHA-256 digest of the canonical recipe representation."""
        self.validate()
        payload = json.dumps(self.to_record(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def corpus_policy(self, corpus: str) -> CorpusRecipe | None:
        """Returns the policy for one corpus, if declared."""
        return next((policy for policy in self.corpora if policy.corpus == corpus), None)


@dataclass(frozen=True)
class RoutedUtterance:
    """Exhaustive routing result for one manifest row."""

    utterance: Utterance
    disposition: RouteDisposition
    tasks: frozenset[TaskName]
    reason: str


def route_utterance(utterance: Utterance, recipe: DatasetRecipe) -> RoutedUtterance:
    """Routes one row without forcing incompatible labels into the primary head."""
    policy = recipe.corpus_policy(utterance.corpus)
    if policy is None:
        return RoutedUtterance(utterance, "quarantined", frozenset(), "corpus_not_in_recipe")

    tasks: set[TaskName] = {"ssl"}
    if utterance.vad is not None and "vad" in policy.auxiliary_tasks:
        tasks.add("vad")
    label = utterance.label
    if (
        utterance.social_attitude is not None or label is not None
    ) and "attitude" in policy.auxiliary_tasks:
        tasks.add("attitude")
    if (
        utterance.binary_affect is not None or label is not None
    ) and "binary_affect" in policy.auxiliary_tasks:
        tasks.add("binary_affect")
    if utterance.language is not None and "language" in policy.auxiliary_tasks:
        tasks.add("language")
    if utterance.transcript is not None and "text_alignment" in policy.auxiliary_tasks:
        tasks.add("text_alignment")

    raw_label = normalize_label(utterance.raw_label) if utterance.raw_label else label
    if (
        label is not None
        and label in policy.exact_primary_labels
        and raw_label not in policy.approximate_labels
    ):
        tasks.add("primary_emotion")
        disposition: RouteDisposition = "remapped" if raw_label != label else "accepted"
        return RoutedUtterance(utterance, disposition, frozenset(tasks), "exact_primary_label")
    if raw_label is not None and raw_label in policy.approximate_labels:
        tasks.add("raw_emotion")
        return RoutedUtterance(
            utterance,
            "weak",
            frozenset(tasks),
            "approximate_label_is_auxiliary_only",
        )
    if label is not None and "raw_emotion" in policy.auxiliary_tasks:
        tasks.add("raw_emotion")
    if len(tasks) > 1:
        return RoutedUtterance(utterance, "accepted", frozenset(tasks), "auxiliary_targets")
    if label is None and all(
        target is None
        for target in (
            utterance.vad,
            utterance.social_attitude,
            utterance.binary_affect,
            utterance.language,
            utterance.transcript,
        )
    ):
        return RoutedUtterance(utterance, "missing", frozenset(tasks), "no_usable_targets")
    return RoutedUtterance(utterance, "dropped", frozenset(tasks), "target_not_enabled_by_recipe")


def research_recipe_v1() -> DatasetRecipe:
    """Returns the common leakage-safe cross-domain research recipe."""
    exact_corpora = (
        "ravdess",
        "crema-d",
        "msp-podcast",
        "mesd",
        "oreau-french-esd",
        "cafe",
        "asvp-esd",
        "spanish-meacorpus-2023",
        "biic-podcast",
    )
    policies = [
        CorpusRecipe(corpus=corpus, exact_primary_labels=CANONICAL_EMOTIONS)
        for corpus in exact_corpora
    ]
    policies.extend(
        (
            CorpusRecipe(
                corpus="escorpus-pe",
                approximate_labels=frozenset({"boredom", "neutral"}),
                auxiliary_tasks=("vad", "language"),
            ),
            CorpusRecipe(
                corpus="att-hack",
                auxiliary_tasks=("attitude", "language", "text_alignment"),
            ),
            CorpusRecipe(
                corpus="coraa-ser",
                auxiliary_tasks=("binary_affect", "language", "text_alignment"),
            ),
            CorpusRecipe(
                corpus="emodb-2.0",
                exact_primary_labels=CANONICAL_EMOTIONS,
                approximate_labels=frozenset({"boredom"}),
            ),
            CorpusRecipe(
                corpus="emov-db",
                exact_primary_labels=CANONICAL_EMOTIONS,
                approximate_labels=frozenset({"anxious", "amused", "sleepy"}),
            ),
            CorpusRecipe(corpus="pavoque", auxiliary_tasks=("raw_emotion", "language")),
            CorpusRecipe(corpus="jl-corpus", exact_primary_labels=CANONICAL_EMOTIONS),
        )
    )
    return DatasetRecipe(
        recipe_id="cross-domain-common",
        revision="1",
        ontology_version="canonical-eight-v1",
        corpora=tuple(policies),
    )


def load_dataset_recipe(value: str | Path) -> DatasetRecipe:
    """Loads a built-in recipe id or a versioned JSON recipe file."""
    if str(value) == "research-v1":
        return research_recipe_v1()
    path = Path(value).expanduser()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as err:
        raise ValueError(f"Unable to load dataset recipe {path}: {err}") from err
    if not isinstance(payload, dict):
        raise ValueError("Dataset recipe root must be a JSON object.")
    corpora_raw = payload.get("corpora")
    if not isinstance(corpora_raw, list):
        raise ValueError("Dataset recipe 'corpora' must be a list.")
    corpora: list[CorpusRecipe] = []
    for raw in corpora_raw:
        if not isinstance(raw, dict):
            raise ValueError("Dataset recipe corpora must contain objects.")
        corpus = raw.get("corpus")
        exact = raw.get("exact_primary_labels", [])
        approximate = raw.get("approximate_labels", [])
        tasks = raw.get("auxiliary_tasks", [])
        if not isinstance(corpus, str) or not corpus.strip():
            raise ValueError("Dataset recipe corpus id must be non-empty.")
        for field_name, field_value in (
            ("exact_primary_labels", exact),
            ("approximate_labels", approximate),
            ("auxiliary_tasks", tasks),
        ):
            if not isinstance(field_value, list) or any(
                not isinstance(item, str) or not item.strip() for item in field_value
            ):
                raise ValueError(f"Dataset recipe {field_name!r} must be a list of strings.")
        corpora.append(
            CorpusRecipe(
                corpus=corpus.strip(),
                exact_primary_labels=frozenset(normalize_label(item) for item in exact),
                approximate_labels=frozenset(normalize_label(item) for item in approximate),
                auxiliary_tasks=tuple(cast(TaskName, item.strip()) for item in tasks),
            )
        )
    schema_version = payload.get("schema_version")
    recipe_id = payload.get("recipe_id")
    revision = payload.get("revision")
    ontology_version = payload.get("ontology_version")
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or not isinstance(recipe_id, str)
        or not isinstance(revision, str)
        or not isinstance(ontology_version, str)
    ):
        raise ValueError("Dataset recipe is missing required schema/id/revision/ontology fields.")
    recipe = DatasetRecipe(
        schema_version=schema_version,
        recipe_id=recipe_id,
        revision=revision,
        ontology_version=ontology_version,
        corpora=tuple(corpora),
    )
    recipe.validate()
    return recipe
