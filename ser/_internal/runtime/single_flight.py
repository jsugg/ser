"""Ref-counted single-flight registry for profile/model-scoped operations."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import Lock

_DEFAULT_MODEL_KEY = "__default__"


@dataclass(slots=True)
class _SingleFlightState:
    """Tracks one keyed lock plus the number of active holders and waiters."""

    lock: Lock = field(default_factory=Lock)
    references: int = 0


class SingleFlightRegistry:
    """Serializes same-key work while pruning idle lock entries."""

    def __init__(self) -> None:
        """Initializes the keyed lock registry."""
        self._guard = Lock()
        self._states: dict[tuple[str, str], _SingleFlightState] = {}

    def active_key_count(self) -> int:
        """Returns the number of retained lock keys."""
        with self._guard:
            return len(self._states)

    @contextmanager
    def lock(
        self,
        *,
        profile: str,
        backend_model_id: str | None,
    ) -> Iterator[None]:
        """Serializes work for one profile/model tuple."""
        key = (profile, self._normalize_model_key(backend_model_id))
        with self._guard:
            state = self._states.get(key)
            if state is None:
                state = _SingleFlightState()
                self._states[key] = state
            state.references += 1
        state.lock.acquire()
        try:
            yield
        finally:
            state.lock.release()
            with self._guard:
                state.references -= 1
                if state.references == 0:
                    self._states.pop(key, None)

    @staticmethod
    def _normalize_model_key(backend_model_id: str | None) -> str:
        """Returns the canonical model key used by the registry."""
        if isinstance(backend_model_id, str):
            normalized = backend_model_id.strip()
            if normalized:
                return normalized
        return _DEFAULT_MODEL_KEY
