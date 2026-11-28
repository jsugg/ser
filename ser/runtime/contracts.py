"""Runtime pipeline contracts for train/inference orchestration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from ser.domain import EmotionSegment, TimelineEntry, TranscriptWord
from ser.profiles import ProfileName
from ser.runtime.schema import InferenceResult

type SubtitleFormat = Literal["ass", "srt", "vtt"]


@dataclass(frozen=True)
class InferenceRequest:
    """Input contract for one inference execution."""

    file_path: str
    language: str
    save_transcript: bool = False
    include_transcript: bool = True
    subtitle_output_path: str | None = None
    subtitle_format: SubtitleFormat | None = None


@dataclass(frozen=True)
class InferenceExecution:
    """Output contract for one inference execution."""

    profile: ProfileName
    output_schema_version: str
    backend_id: str
    emotions: list[EmotionSegment]
    transcript: list[TranscriptWord]
    timeline: list[TimelineEntry]
    used_backend_path: bool = False
    timeline_csv_path: str | None = None
    subtitle_path: str | None = None
    detailed_result: InferenceResult | None = None
    phase_timings_seconds: dict[str, float] = field(default_factory=dict)


type BackendInferenceCallable = Callable[[InferenceRequest], InferenceResult]


type LanguageHint = Literal["en", "es", "pt"]
type AnalysisIsolation = Literal["worker", "inline"]
type ModelFormat = Literal["safe", "legacy-pickle"]
type LicenceClass = Literal["open", "non-commercial", "research-only", "academic-only"]


@dataclass(frozen=True)
class AnalysisTargets:
    """Performance goals for utterance analysis, measured by the benchmark harness.

    Analysis never enforces these; a slow call still returns its record. The
    defaults stand until measured otherwise (owner decision of 2026-11-28).
    """

    p95_real_time_factor: float = 0.5
    cold_start_s: float = 20.0
    steady_memory_bytes: int = 1_000_000_000


@dataclass(frozen=True)
class AnalysisConfig:
    """Limits that analysis enforces, and how the model process runs.

    Analysis is CPU-only; this is policy, not an option. In ``"worker"`` isolation
    the model runs in one persistent child process at ``worker_nice`` with at most
    ``max_threads`` native threads and no visible GPU, and ``analyze`` returns
    ``timeout`` no later than its deadline. ``"inline"`` runs in the caller's
    process and checks the deadline only between stages; it is for callers that
    isolate SER themselves. Inline never changes process-wide settings (priority,
    thread pools, environment): the caller owns its process, so ``max_threads`` and
    ``worker_nice`` apply in the worker only.
    """

    min_duration_s: float = 0.3
    max_duration_s: float = 120.0
    timeout_floor_s: float = 10.0
    timeout_per_audio_second: float = 2.0
    silence_rms_dbfs: float = -60.0
    isolation: AnalysisIsolation = "worker"
    max_threads: int = 2
    worker_nice: int = 19
    worker_start_timeout_s: float = 60.0
    targets: AnalysisTargets = field(default_factory=AnalysisTargets)


@dataclass(frozen=True)
class ModelIdentity:
    """Identity of the model an analyser loaded, as written into every record.

    ``model_sha256`` identifies the model. ``model_id`` is a label chosen by whoever
    built the model and is not promised unique; key stored results on the pair.
    """

    profile: ProfileName
    model_id: str
    model_sha256: str
    label_set_id: str
    labels: tuple[str, ...]


@dataclass(frozen=True)
class TrainingAudioForm:
    """Audio form a model was trained on, as recorded in its artifact."""

    sample_rate_hz: int
    channels: int
    canonical_conversion: bool


@dataclass(frozen=True)
class LicenceRecord:
    """Licence of one encoder or dataset that went into a model.

    ``name`` is the identifier an allow-list matches on, such as a catalogue
    dataset id. A safe artifact whose records leave any field empty does not load.
    """

    name: str
    licence: str
    licence_class: LicenceClass
    source: str


@dataclass(frozen=True)
class ModelProvenance:
    """Licence and origin record bound to the payload it describes.

    ``encoder`` is the pretrained encoder's licence, or ``"none"`` when the artifact
    states that no pretrained encoder was used; ``None`` means the artifact did not
    say. ``datasets`` lists every training dataset. A safe artifact with ``None``
    as ``encoder`` or no datasets is incomplete and does not load, so an inspected
    or loaded manifest never carries either.
    """

    payload_sha256: str
    encoder: LicenceRecord | Literal["none"] | None
    datasets: tuple[LicenceRecord, ...]


@dataclass(frozen=True)
class ModelManifest:
    """What a verified model file is, readable before any analysis runs.

    ``baseline`` models (legacy pickles, or audio forms other than canonical) load
    only on explicit opt-in and carry a ``baseline:`` prefix in their ``model_id``.
    """

    identity: ModelIdentity
    artifact_format: ModelFormat
    training_audio: TrainingAudioForm | None
    provenance: ModelProvenance | None
    baseline: bool
