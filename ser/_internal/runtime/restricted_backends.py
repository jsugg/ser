"""Restricted-backend consent and policy helpers for runtime workflows."""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import ser.license_check as license_check
from ser.config import AppConfig
from ser.license_check import (
    BackendLicensePolicy,
    BackendLicensePolicyError,
    LicenseDecision,
    evaluate_backend_access,
    get_backend_policy,
    load_persisted_backend_consents,
    parse_allowed_restricted_backends_env,
    persist_backend_consent,
)
from ser.profiles import get_profile_catalog, resolve_profile_name

type RestrictedBackendPrompt = Callable[[BackendLicensePolicy], bool]
type RuntimeCliLogLevel = Literal["info", "error"]
type RuntimeCliLogRecord = tuple[RuntimeCliLogLevel, str]

type _PrepareRestrictedBackendOptInState = Callable[..., RestrictedBackendOptInState]
type _EnforceRestrictedBackendsForCli = Callable[..., None]


@dataclass(frozen=True, slots=True)
class RestrictedBackendOptInState:
    """Restricted-backend opt-in state derived from one CLI invocation."""

    required_backend_ids: tuple[str, ...]
    persisted_all_count: int
    persisted_profile_backend_ids: tuple[str, ...]
    should_exit_zero: bool


def _profile_resolution_requested(
    *,
    use_profile_pipeline: bool,
    file_path: str | None,
) -> bool:
    """Returns whether profile resolution should run for this invocation."""
    return bool(use_profile_pipeline or file_path)


def required_restricted_backends_for_current_profile(
    settings: AppConfig,
    *,
    use_profile_pipeline: bool,
) -> tuple[str, ...]:
    """Returns restricted backend ids required by the active runtime profile."""
    if not use_profile_pipeline:
        return ()
    profile_name = resolve_profile_name(settings)
    backend_id = get_profile_catalog()[profile_name].backend_id
    policy = get_backend_policy(backend_id)
    if policy is None or not policy.restricted:
        return ()
    return (backend_id,)


def persist_required_restricted_backends(
    settings: AppConfig,
    *,
    use_profile_pipeline: bool,
    consent_source: str,
) -> tuple[str, ...]:
    """Persists consent for restricted backends required by the active profile."""
    required_backends = required_restricted_backends_for_current_profile(
        settings,
        use_profile_pipeline=use_profile_pipeline,
    )
    persisted: list[str] = []
    for backend_id in required_backends:
        persist_backend_consent(
            settings=settings,
            backend_id=backend_id,
            consent_source=consent_source,
        )
        persisted.append(backend_id)
    return tuple(persisted)


def persist_all_restricted_backend_consents(
    settings: AppConfig,
    *,
    consent_source: str,
) -> int:
    """Persists consent for all known restricted backends and returns count."""
    records = license_check.persist_all_restricted_backend_consents(
        settings=settings,
        consent_source=consent_source,
    )
    return len(records)


def prepare_restricted_backend_opt_in_state(
    *,
    settings: AppConfig,
    use_profile_pipeline: bool,
    train_requested: bool,
    file_path: str | None,
    accept_restricted_backends: bool,
    accept_all_restricted_backends: bool,
    all_consent_source: str = "cli_flag_accept_all",
    profile_consent_source: str = "cli_flag_accept_restricted",
) -> RestrictedBackendOptInState:
    """Prepares restricted-backend state transitions for one CLI invocation."""
    profile_resolution_enabled = _profile_resolution_requested(
        use_profile_pipeline=use_profile_pipeline,
        file_path=file_path,
    )
    required_backend_ids = required_restricted_backends_for_current_profile(
        settings,
        use_profile_pipeline=profile_resolution_enabled,
    )
    persisted_all_count = 0
    persisted_profile_backend_ids: tuple[str, ...] = ()
    if accept_all_restricted_backends:
        persisted_all_count = persist_all_restricted_backend_consents(
            settings,
            consent_source=all_consent_source,
        )
    if accept_restricted_backends:
        persisted_profile_backend_ids = persist_required_restricted_backends(
            settings,
            use_profile_pipeline=profile_resolution_enabled,
            consent_source=profile_consent_source,
        )
    should_exit_zero = (accept_restricted_backends or accept_all_restricted_backends) and (
        not train_requested and not file_path
    )
    return RestrictedBackendOptInState(
        required_backend_ids=required_backend_ids,
        persisted_all_count=persisted_all_count,
        persisted_profile_backend_ids=persisted_profile_backend_ids,
        should_exit_zero=should_exit_zero,
    )


def enforce_restricted_backends_for_cli(
    *,
    settings: AppConfig,
    train_requested: bool,
    file_path: str | None,
    required_backend_ids: tuple[str, ...],
    is_interactive: bool,
    prompt_for_policy: RestrictedBackendPrompt | None = None,
    consent_source: str = "interactive_prompt",
) -> None:
    """Enforces restricted-backend access policy for executable CLI paths."""
    if not (train_requested or file_path):
        return
    if not required_backend_ids:
        return
    ensure_restricted_backends_ready_for_command(
        settings,
        required_backend_ids=required_backend_ids,
        is_interactive=is_interactive,
        prompt_for_policy=prompt_for_policy,
        consent_source=consent_source,
    )


def run_restricted_backend_cli_gate(
    *,
    settings: AppConfig,
    use_profile_pipeline: bool,
    train_requested: bool,
    file_path: str | None,
    accept_restricted_backends: bool,
    accept_all_restricted_backends: bool,
    is_interactive: bool,
    prepare_opt_in_state: _PrepareRestrictedBackendOptInState,
    enforce_for_cli: _EnforceRestrictedBackendsForCli,
) -> tuple[tuple[RuntimeCliLogRecord, ...], int | None]:
    """Evaluates restricted-backend CLI gate and returns logs plus optional exit code."""
    opt_in_state = prepare_opt_in_state(
        settings=settings,
        use_profile_pipeline=use_profile_pipeline,
        train_requested=train_requested,
        file_path=file_path,
        accept_restricted_backends=accept_restricted_backends,
        accept_all_restricted_backends=accept_all_restricted_backends,
    )
    logs: list[RuntimeCliLogRecord] = []
    if opt_in_state.persisted_all_count > 0:
        logs.append(
            (
                "info",
                "Persisted restricted-backend consent for "
                f"{opt_in_state.persisted_all_count} backend(s).",
            )
        )
    if opt_in_state.persisted_profile_backend_ids:
        logs.append(
            (
                "info",
                "Persisted restricted-backend consent for active profile backend(s): "
                + ", ".join(opt_in_state.persisted_profile_backend_ids),
            )
        )
    if opt_in_state.should_exit_zero:
        return (tuple(logs), 0)

    try:
        enforce_for_cli(
            settings=settings,
            train_requested=train_requested,
            file_path=file_path,
            required_backend_ids=opt_in_state.required_backend_ids,
            is_interactive=is_interactive,
        )
    except BackendLicensePolicyError as err:
        logs.append(("error", str(err)))
        return (tuple(logs), 2)
    return (tuple(logs), None)


def collect_missing_restricted_backend_consents(
    settings: AppConfig,
    *,
    required_backend_ids: tuple[str, ...],
) -> tuple[LicenseDecision, ...]:
    """Returns restricted backend decisions that still require explicit consent."""
    if not required_backend_ids:
        return ()
    persisted_consents = load_persisted_backend_consents(settings=settings)
    allowed_restricted_backends = parse_allowed_restricted_backends_env()
    missing: list[LicenseDecision] = []
    for backend_id in required_backend_ids:
        decision = evaluate_backend_access(
            backend_id=backend_id,
            restricted_backends_enabled=settings.runtime_flags.restricted_backends,
            allowed_restricted_backends=allowed_restricted_backends,
            persisted_consents=persisted_consents,
        )
        if decision.allowed:
            continue
        missing.append(decision)
    return tuple(missing)


def _default_restricted_backend_prompt(policy: BackendLicensePolicy) -> bool:
    """Prompts for one restricted-backend acknowledgement in interactive shells."""
    print(
        "Restricted backend acknowledgement required:",
        file=sys.stderr,
    )
    print(f"  backend: {policy.backend_id}", file=sys.stderr)
    print(f"  license: {policy.license_id}", file=sys.stderr)
    print(f"  source: {policy.source_url}", file=sys.stderr)
    answer = input("Persist opt-in for this backend now? [y/N]: ").strip().lower()
    return answer in {"y", "yes"}


def ensure_restricted_backends_ready_for_command(
    settings: AppConfig,
    *,
    required_backend_ids: tuple[str, ...],
    is_interactive: bool,
    prompt_for_policy: RestrictedBackendPrompt | None = None,
    consent_source: str = "interactive_prompt",
) -> None:
    """Ensures required restricted backends have explicit opt-in before execution."""
    while True:
        missing_decisions = collect_missing_restricted_backend_consents(
            settings,
            required_backend_ids=required_backend_ids,
        )
        if not missing_decisions:
            return
        decision = missing_decisions[0]
        policy = get_backend_policy(decision.policy.backend_id)
        if policy is None:
            raise BackendLicensePolicyError(decision.reason)
        if not is_interactive:
            raise BackendLicensePolicyError(
                f"{decision.reason} Non-interactive shell cannot prompt for consent. "
                "Use `--accept-restricted-backends`, "
                "`--accept-all-restricted-backends`, "
                "`SER_ALLOWED_RESTRICTED_BACKENDS`, or "
                "`SER_ENABLE_RESTRICTED_BACKENDS=true`."
            )
        prompt = (
            prompt_for_policy
            if prompt_for_policy is not None
            else _default_restricted_backend_prompt
        )
        if not prompt(policy):
            raise BackendLicensePolicyError(
                f"Restricted backend {policy.backend_id!r} requires explicit "
                "acknowledgement before execution."
            )
        persist_backend_consent(
            settings=settings,
            backend_id=policy.backend_id,
            consent_source=consent_source,
        )


__all__ = [
    "RestrictedBackendOptInState",
    "RestrictedBackendPrompt",
    "RuntimeCliLogRecord",
    "collect_missing_restricted_backend_consents",
    "enforce_restricted_backends_for_cli",
    "ensure_restricted_backends_ready_for_command",
    "persist_all_restricted_backend_consents",
    "persist_required_restricted_backends",
    "prepare_restricted_backend_opt_in_state",
    "required_restricted_backends_for_current_profile",
    "run_restricted_backend_cli_gate",
]
