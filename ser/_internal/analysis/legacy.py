"""Quiet deserialization and scoring for trusted legacy models."""

from __future__ import annotations

import pickle
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from threading import RLock
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ser._internal.models.artifact_envelope import LoadedModel

_WARNING_LOCK = RLock()


@contextmanager
def capture_legacy_warnings() -> Iterator[list[warnings.WarningMessage]]:
    # Python 3.12 warning filters are process-wide; serialize these legacy scopes.
    with _WARNING_LOCK, warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        yield caught


def load_legacy(data: bytes) -> tuple[LoadedModel, bool]:
    with capture_legacy_warnings() as caught:
        from sklearn.exceptions import InconsistentVersionWarning

        from ser._internal.models.artifact_envelope import deserialize_model_artifact

        loaded = deserialize_model_artifact(pickle.loads(data))
    return loaded, any(isinstance(item.message, InconsistentVersionWarning) for item in caught)
