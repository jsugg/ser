"""Simulated host resources for training-readiness unit tests."""

from __future__ import annotations

import os
import resource
import shutil
from dataclasses import dataclass
from types import SimpleNamespace

import psutil
import pytest

_MIB = 1024 * 1024
_PAGE_SIZE = 4096


@dataclass
class ReadinessHost:
    """Host readings seen by readiness; defaults are ample so no resource check fires."""

    free_memory_bytes: int = 8 * 1024 * _MIB
    available_memory_bytes: int = 12 * 1024 * _MIB
    memory_error: Exception | None = None
    disk_free_bytes: int = 1024 * 1024 * _MIB
    fd_soft_limit: int = 4096


@pytest.fixture
def readiness_host(monkeypatch: pytest.MonkeyPatch) -> ReadinessHost:
    """Replaces every host reading made by readiness resource checks with controlled values."""
    host = ReadinessHost()
    real_sysconf = os.sysconf
    real_getrlimit = resource.getrlimit

    def _virtual_memory() -> SimpleNamespace:
        if host.memory_error is not None:
            raise host.memory_error
        return SimpleNamespace(
            total=16 * 1024 * _MIB,
            free=host.free_memory_bytes,
            available=host.available_memory_bytes,
        )

    def _sysconf(name: str | int) -> int:
        if name == "SC_AVPHYS_PAGES":
            return host.free_memory_bytes // _PAGE_SIZE
        if name in {"SC_PAGE_SIZE", "SC_PAGESIZE"}:
            return _PAGE_SIZE
        return real_sysconf(name)

    def _disk_usage(_path: object) -> SimpleNamespace:
        return SimpleNamespace(total=host.disk_free_bytes, used=0, free=host.disk_free_bytes)

    def _getrlimit(kind: int) -> tuple[int, int]:
        if kind == resource.RLIMIT_NOFILE:
            return host.fd_soft_limit, host.fd_soft_limit
        return real_getrlimit(kind)

    monkeypatch.setattr(psutil, "virtual_memory", _virtual_memory)
    monkeypatch.setattr(os, "sysconf", _sysconf)
    monkeypatch.setattr(shutil, "disk_usage", _disk_usage)
    monkeypatch.setattr(resource, "getrlimit", _getrlimit)
    return host
