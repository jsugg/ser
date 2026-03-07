"""Contracts for accurate runtime model-compatibility helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from types import SimpleNamespace
from typing import cast

import pytest

from ser.config import AppConfig
from ser.runtime.accurate_model_contract import (
    ensure_accurate_compatible_model,
    validate_accurate_loaded_model_runtime_contract,
    warn_on_runtime_selector_mismatch,
)


@dataclass(frozen=True)
class _LoadedModelStub:
    artifact_metadata: dict[str, object] | None


@dataclass(frozen=True)
class _RuntimePolicyStub:
    device: str
    dtype: str


def test_ensure_accurate_compatible_model_rejects_backend_model_mismatch() -> None:
    """Compatibility check should fail closed on backend model-id mismatch."""
    loaded_model = _LoadedModelStub(
        artifact_metadata={
            "backend_id": "hf_whisper",
            "profile": "accurate",
            "backend_model_id": "unexpected/model-id",
        }
    )

    with pytest.raises(RuntimeError, match="backend_model_id"):
        ensure_accurate_compatible_model(
            loaded_model,
            expected_backend_id="hf_whisper",
            expected_profile="accurate",
            expected_backend_model_id="configured/model-id",
            unavailable_error_factory=RuntimeError,
        )


def test_warn_on_runtime_selector_mismatch_emits_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Mismatch diagnostics should log one warning with device/dtype details."""
    loaded_model = _LoadedModelStub(
        artifact_metadata={
            "torch_device": "mps",
            "torch_dtype": "float16",
        }
    )
    logger = logging.getLogger("ser.tests.accurate_model_contract")
    caplog.set_level(logging.WARNING, logger=logger.name)

    warn_on_runtime_selector_mismatch(
        loaded_model=loaded_model,
        backend_id="hf_whisper",
        requested_device="auto",
        requested_dtype="auto",
        backend_override_device=None,
        backend_override_dtype=None,
        profile="accurate",
        logger=logger,
        resolve_runtime_policy=lambda **_kwargs: _RuntimePolicyStub(
            device="cpu",
            dtype="float32",
        ),
    )

    warning_messages = [record.getMessage() for record in caplog.records]
    assert len(warning_messages) == 1
    assert "profile" in warning_messages[0]
    assert "device artifact='mps' runtime='cpu'" in warning_messages[0]
    assert "dtype artifact='float16' runtime='float32'" in warning_messages[0]


def test_validate_loaded_model_runtime_contract_combines_guards(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Composite helper should validate compatibility and emit selector mismatch once."""
    loaded_model = _LoadedModelStub(
        artifact_metadata={
            "backend_id": "hf_whisper",
            "profile": "accurate",
            "backend_model_id": "configured/model-id",
            "torch_device": "mps",
            "torch_dtype": "float16",
        }
    )
    settings = cast(
        AppConfig,
        SimpleNamespace(
            torch_runtime=SimpleNamespace(device="auto", dtype="auto"),
            feature_runtime_policy=SimpleNamespace(
                for_backend=lambda _backend_id: SimpleNamespace(
                    device="cpu",
                    dtype="float32",
                )
            ),
        ),
    )
    logger = logging.getLogger("ser.tests.accurate_model_contract.composite")
    caplog.set_level(logging.WARNING, logger=logger.name)

    validate_accurate_loaded_model_runtime_contract(
        loaded_model,
        settings=settings,
        expected_backend_id="hf_whisper",
        expected_profile="accurate",
        expected_backend_model_id="configured/model-id",
        unavailable_error_factory=RuntimeError,
        logger=logger,
        resolve_runtime_policy=lambda **_kwargs: _RuntimePolicyStub(
            device="cpu",
            dtype="float32",
        ),
    )

    warning_messages = [record.getMessage() for record in caplog.records]
    assert len(warning_messages) == 1
    assert "embedding distribution may shift" in warning_messages[0]
