"""Analysis adapter for the trusted, uncalibrated fast baseline."""

from __future__ import annotations

import pickle
import time
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ser._internal.analysis.backend import (
    BackendFrame,
    BackendLoadRequest,
    BackendScores,
    coverage_weighted_mean,
    frame_layout,
)
from ser._internal.models.artifact_envelope import (
    DEFAULT_BACKEND_ID,
    DEFAULT_FRAME_SIZE_SECONDS,
    DEFAULT_FRAME_STRIDE_SECONDS,
    EmotionClassifier,
    deserialize_model_artifact,
)
from ser._internal.repr.handcrafted import HandcraftedBackend
from ser._internal.utils.audio_utils import CanonicalAudio
from ser.runtime.contracts import LanguageHint


@dataclass(frozen=True)
class _FastFeatures:
    spans: tuple[tuple[float, float], ...]
    vectors: NDArray[np.float64]
    duration_s: float


class FastBackend:
    def __init__(
        self, model: EmotionClassifier, labels: tuple[str, ...], extractor: HandcraftedBackend
    ) -> None:
        self._model: EmotionClassifier | None = model
        self._labels = labels
        self._extractor = extractor

    def extract(self, audio: CanonicalAudio, *, deadline_monotonic_s: float) -> _FastFeatures:
        spans = frame_layout(
            audio.duration_seconds,
            frame_size_s=DEFAULT_FRAME_SIZE_SECONDS,
            stride_s=DEFAULT_FRAME_STRIDE_SECONDS,
        )
        vectors: list[NDArray[np.float64]] = []
        for start, end in spans:
            if time.monotonic() >= deadline_monotonic_s:
                raise TimeoutError
            first = round(start * audio.sample_rate)
            last = (
                audio.samples.size
                if end == audio.duration_seconds
                else round(end * audio.sample_rate)
            )
            vectors.append(
                self._extractor.extract_vector(audio.samples[first:last], audio.sample_rate)
            )
        return _FastFeatures(spans, np.stack(vectors), audio.duration_seconds)

    def score(
        self,
        features: object,
        *,
        language: LanguageHint | None,
        deadline_monotonic_s: float,
    ) -> BackendScores:
        if not isinstance(features, _FastFeatures) or self._model is None:
            raise ValueError("Invalid fast analysis state.")
        values = np.asarray(self._model.predict_proba(features.vectors), dtype=np.float64)
        if values.shape != (len(features.spans), len(self._labels)):
            raise ValueError("Invalid fast probability shape.")
        frames = tuple(
            BackendFrame(start, end, dict(zip(self._labels, map(float, row), strict=True)))
            for (start, end), row in zip(features.spans, values, strict=True)
        )
        return BackendScores(coverage_weighted_mean(frames, features.duration_s), frames)

    def close(self) -> None:
        self._model = None


def load_fast(request: BackendLoadRequest) -> FastBackend:
    if request.artifact_format != "legacy-pickle" or not request.manifest.baseline:
        raise ValueError("No compatible fast payload format is registered.")
    loaded = deserialize_model_artifact(pickle.loads(request.payload))
    metadata = loaded.artifact_metadata
    extractor = HandcraftedBackend(feature_flags=request.feature_flags)
    if (
        metadata is None
        or metadata["backend_id"] != DEFAULT_BACKEND_ID
        or extractor.feature_dim != loaded.expected_feature_size
        or getattr(loaded.model, "n_features_in_", None) != extractor.feature_dim
    ):
        raise ValueError("The model is incompatible with fast analysis features.")
    labels = tuple(map(str, loaded.model.classes_))
    if len(labels) != len(request.manifest.identity.labels) or set(labels) != set(
        request.manifest.identity.labels
    ):
        raise ValueError("The model labels do not match its manifest.")
    return FastBackend(loaded.model, labels, extractor)
