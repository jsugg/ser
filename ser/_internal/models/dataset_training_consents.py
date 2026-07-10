"""Training-time dataset consent orchestration helpers."""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from typing import Any

from ser.config import AppConfig
from ser.data import Utterance
from ser.data.dataset_consents import (
    DatasetConsentError,
    compute_missing_dataset_consents,
    ensure_dataset_consents,
    persist_dataset_consents,
)


def ensure_dataset_training_consents(
    *,
    utterances: Sequence[Utterance],
    settings: AppConfig,
    logger_warning: Callable[..., object],
    stdin_isatty: Callable[[int], bool] = os.isatty,
    prompt_input: Callable[[], str] = input,
    prompt_print: Callable[..., Any] = print,
) -> None:
    """Enforces dataset acknowledgements for training with interactive fallback."""
    try:
        ensure_dataset_consents(settings=settings, utterances=list(utterances))
        return
    except DatasetConsentError as err:
        message = str(err)
        interactive = stdin_isatty(0) and stdin_isatty(2)
        if not interactive:
            raise

    logger_warning("%s", message)
    prompt_print("To acknowledge and continue, type 'accept': ", end="", flush=True)
    try:
        response = prompt_input().strip().lower()
    except EOFError:
        response = ""
    if response != "accept":
        raise DatasetConsentError(message)

    missing_policies, missing_licenses = compute_missing_dataset_consents(
        settings=settings,
        utterances=list(utterances),
    )
    persist_dataset_consents(
        settings=settings,
        accept_policy_ids=sorted(missing_policies),
        accept_license_ids=sorted(missing_licenses),
        source="training",
    )


__all__ = ["ensure_dataset_training_consents"]
