"""Stable-whisper transcription adapter implementation."""

from __future__ import annotations

import importlib
import inspect
import logging
import os
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast
from urllib.parse import urlparse

from ser._internal.runtime.environment_plan import build_runtime_environment_plan
from ser._internal.runtime.process_env import temporary_process_env
from ser.domain import TranscriptWord
from ser.profiles import TranscriptionBackendId
from ser.runtime.phase_contract import (
    PHASE_TRANSCRIPTION,
    PHASE_TRANSCRIPTION_MODEL_LOAD,
)
from ser.transcript.backends import stable_whisper_torio_probe
from ser.transcript.backends.base import (
    BackendRuntimeRequest,
    CompatibilityIssue,
    CompatibilityReport,
    TranscriptionBackendAdapter,
)
from ser.transcript.backends.stable_whisper_admission_runtime import (
    is_compatibility_activation_error,
    is_retryable_precision_failure,
    mps_compatibility_fallback_reason,
    resolve_stable_whisper_mps_admission_decision,
    should_enforce_stable_whisper_transcribe_admission,
)
from ser.transcript.backends.stable_whisper_mps_compat import (
    enable_stable_whisper_mps_compatibility,
    get_stable_whisper_runtime_device,
    is_stable_whisper_mps_compatibility_enabled,
    set_stable_whisper_mps_compatibility_enabled,
    set_stable_whisper_runtime_device,
    stable_whisper_mps_timing_compatibility_context,
)
from ser.transcript.backends.stable_whisper_transcribe_admission import (
    resolve_transcribe_runtime_device as resolve_stable_whisper_transcribe_runtime_device,
)
from ser.transcript.backends.stable_whisper_transcribe_execution import (
    run_stable_whisper_transcribe_with_retry,
)
from ser.transcript.backends.stable_whisper_transcribe_kwargs import (
    build_stable_whisper_transcribe_kwargs,
)
from ser.transcript.backends.stable_whisper_transcribe_runtime import (
    classify_transcription_failure_for_runtime,
    effective_precision_candidates,
    release_torch_runtime_memory_for_retry,
    summarize_runtime_error,
)
from ser.transcript.mps_admission import (
    MpsAdmissionDecision,
    decide_mps_admission_for_transcription,
    log_mps_admission_control_fallback,
    mps_admission_control_enabled,
    mps_hard_oom_shortcut_enabled,
)
from ser.transcript.runtime_failures import (
    TranscriptionFailureClassification,
)
from ser.utils.logger import (
    DependencyLogPolicy,
    DependencyPolicyContext,
    WarningPolicy,
    scoped_dependency_log_policy,
)
from ser.utils.transcription_compat import (
    has_known_stable_whisper_sparse_mps_incompatibility,
)

if TYPE_CHECKING:
    from ser.config import AppConfig

_INVALID_ESCAPE_POLICY_ID = "stable_whisper.invalid_escape_sequence"
_INVALID_ESCAPE_MESSAGE_REGEX = r"^invalid escape sequence '\\,'$"
_INVALID_ESCAPE_MODULE_REGEX = r"^stable_whisper\.result$"
_FP16_CPU_WARNING_POLICY_ID = "stable_whisper.fp16_cpu_fallback_warning"
_FP16_CPU_WARNING_MESSAGE_REGEX = r"^FP16 is not supported on CPU; using FP32 instead$"
_DEMUCS_DEPRECATED_POLICY_ID = "stable_whisper.demucs_deprecated_warning"
_DEMUCS_DEPRECATED_MESSAGE_REGEX = (
    r"^``demucs`` is deprecated and will be removed in future versions\. "
    r'Use ``denoiser="demucs"`` instead\.$'
)
_TRANSCRIBE_WARNING_MODULE_REGEX = r"^stable_whisper\.whisper_word_level\.original_whisper$"
_TORIO_FFMPEG_PROBE_POLICY_ID = "torio.ffmpeg_probe_debug_traceback"
_STABLE_WHISPER_IMPORT_POLICY = DependencyLogPolicy(
    logger_prefixes=frozenset({"torio"}),
    demote_from_level=logging.DEBUG,
    demote_to_level=logging.DEBUG,
    backend_ids=frozenset({"stable_whisper"}),
    warning_policies=(
        WarningPolicy(
            policy_id=_INVALID_ESCAPE_POLICY_ID,
            action="ignore",
            message_regex=_INVALID_ESCAPE_MESSAGE_REGEX,
            module_regex=_INVALID_ESCAPE_MODULE_REGEX,
            category=SyntaxWarning,
            backend_ids=frozenset({"stable_whisper"}),
            op_tags=frozenset(
                {
                    "stable_whisper.import",
                    "stable_whisper.result.import",
                }
            ),
        ),
    ),
)
_STABLE_WHISPER_TRANSCRIBE_POLICY = DependencyLogPolicy(
    logger_prefixes=frozenset(),
    backend_ids=frozenset({"stable_whisper"}),
    phase_names=frozenset({PHASE_TRANSCRIPTION}),
    op_tags=frozenset({"stable_whisper.transcribe"}),
    warning_policies=(
        WarningPolicy(
            policy_id=_FP16_CPU_WARNING_POLICY_ID,
            action="ignore",
            message_regex=_FP16_CPU_WARNING_MESSAGE_REGEX,
            module_regex=_TRANSCRIBE_WARNING_MODULE_REGEX,
            category=UserWarning,
            backend_ids=frozenset({"stable_whisper"}),
            phase_names=frozenset({PHASE_TRANSCRIPTION}),
            op_tags=frozenset({"stable_whisper.transcribe"}),
        ),
        WarningPolicy(
            policy_id=_DEMUCS_DEPRECATED_POLICY_ID,
            action="ignore",
            message_regex=_DEMUCS_DEPRECATED_MESSAGE_REGEX,
            module_regex=_TRANSCRIBE_WARNING_MODULE_REGEX,
            category=UserWarning,
            backend_ids=frozenset({"stable_whisper"}),
            phase_names=frozenset({PHASE_TRANSCRIPTION}),
            op_tags=frozenset({"stable_whisper.transcribe"}),
        ),
    ),
)


class StableWhisperAdapter(TranscriptionBackendAdapter):
    """Adapter for stable-whisper backend behavior and compatibility checks."""

    @property
    def backend_id(self) -> TranscriptionBackendId:
        """Returns canonical backend identifier."""
        return "stable_whisper"

    def check_compatibility(
        self,
        *,
        runtime_request: BackendRuntimeRequest,
        settings: AppConfig,
    ) -> CompatibilityReport:
        """Checks stable-whisper dependency/runtime compatibility."""
        del runtime_request
        del settings
        functional_issues: list[CompatibilityIssue] = []
        operational_issues: list[CompatibilityIssue] = []
        noise_issues: list[CompatibilityIssue] = [
            CompatibilityIssue(
                code="stable_whisper_invalid_escape_sequence",
                message=(
                    "stable-whisper may emit SyntaxWarning invalid escape sequence "
                    "during module import; scoped warning policy is required."
                ),
                impact="informational",
            ),
            CompatibilityIssue(
                code="stable_whisper_fp16_cpu_fallback_warning",
                message=(
                    "stable-whisper may emit a CPU fp16 fallback UserWarning "
                    "during transcription; adapter resolves device-aware precision "
                    "and uses a "
                    "scoped warning policy fallback."
                ),
                impact="informational",
            ),
            CompatibilityIssue(
                code="stable_whisper_demucs_deprecated_warning",
                message=(
                    "stable-whisper may emit a demucs deprecation UserWarning "
                    "when legacy demucs argument is used; adapter prefers "
                    "denoiser='demucs' and keeps a scoped warning policy fallback."
                ),
                impact="informational",
            ),
        ]
        with scoped_dependency_log_policy(
            policy=_STABLE_WHISPER_IMPORT_POLICY,
            context=DependencyPolicyContext(
                backend_id=self.backend_id,
                phase_name=PHASE_TRANSCRIPTION_MODEL_LOAD,
                op_tag="stable_whisper.compatibility",
            ),
            keep_demoted=False,
        ):
            torio_issue = stable_whisper_torio_probe.detect_default_torio_ffmpeg_operational_issue()
        if torio_issue is not None:
            operational_issues.append(torio_issue)
            noise_issues.append(
                CompatibilityIssue(
                    code="torio_ffmpeg_probe_debug_traceback",
                    message=(
                        "torio may emit DEBUG traceback probe logs while checking "
                        "FFmpeg extension availability; adapter applies scoped "
                        "suppression to keep dependency noise bounded."
                    ),
                    impact="informational",
                )
            )
        if not stable_whisper_torio_probe.is_module_available("stable_whisper"):
            functional_issues.append(
                CompatibilityIssue(
                    code="missing_dependency_stable_whisper",
                    message=(
                        "Missing stable-whisper dependencies. Ensure project dependencies "
                        "are installed."
                    ),
                    impact="blocking",
                )
            )
        return CompatibilityReport(
            backend_id="stable_whisper",
            functional_issues=tuple(functional_issues),
            operational_issues=tuple(operational_issues),
            noise_issues=tuple(noise_issues),
            policy_ids=(
                _INVALID_ESCAPE_POLICY_ID,
                _FP16_CPU_WARNING_POLICY_ID,
                _DEMUCS_DEPRECATED_POLICY_ID,
                _TORIO_FFMPEG_PROBE_POLICY_ID,
            ),
        )

    def setup_required(
        self,
        *,
        runtime_request: BackendRuntimeRequest,
        settings: AppConfig,
    ) -> bool:
        """Returns whether stable-whisper checkpoint assets are missing."""
        target_path = self._stable_whisper_download_target(
            model_name=runtime_request.model_name,
            settings=settings,
        )
        if target_path is None:
            return False
        return not target_path.is_file()

    def prepare_assets(
        self,
        *,
        runtime_request: BackendRuntimeRequest,
        settings: AppConfig,
    ) -> None:
        """Downloads stable-whisper checkpoint assets when required."""
        target_path = self._stable_whisper_download_target(
            model_name=runtime_request.model_name,
            settings=settings,
        )
        if target_path is None or target_path.is_file():
            return
        try:
            whisper_module = importlib.import_module("whisper")
        except ModuleNotFoundError:
            return
        model_registry = getattr(whisper_module, "_MODELS", None)
        download_fn = getattr(whisper_module, "_download", None)
        if not isinstance(model_registry, dict) or not callable(download_fn):
            return
        model_url = model_registry.get(runtime_request.model_name)
        if not isinstance(model_url, str):
            return
        os.makedirs(settings.models.whisper_download_root, exist_ok=True)
        download_fn(
            model_url,
            str(settings.models.whisper_download_root),
            False,
        )

    def load_model(
        self,
        *,
        runtime_request: BackendRuntimeRequest,
        settings: AppConfig,
    ) -> object:
        """Loads one stable-whisper model for resolved runtime device inference."""
        with scoped_dependency_log_policy(
            policy=_STABLE_WHISPER_IMPORT_POLICY,
            context=DependencyPolicyContext(
                backend_id=self.backend_id,
                phase_name=PHASE_TRANSCRIPTION_MODEL_LOAD,
                op_tag="stable_whisper.import",
            ),
            keep_demoted=False,
        ):
            download_root: Path = settings.models.whisper_download_root
            torch_cache_root: Path = settings.models.torch_cache_root
            os.makedirs(download_root, exist_ok=True)
            os.makedirs(torch_cache_root, exist_ok=True)
            runtime_environment = build_runtime_environment_plan(settings)
            with temporary_process_env(
                runtime_environment.torch_runtime.merged(runtime_environment.stable_whisper)
            ):
                try:
                    stable_whisper = importlib.import_module("stable_whisper")
                except ModuleNotFoundError as err:
                    raise RuntimeError(
                        "Missing stable-whisper dependencies. Ensure project dependencies "
                        "are installed."
                    ) from err

                load_model = getattr(stable_whisper, "load_model", None)
                if not callable(load_model):
                    raise RuntimeError(
                        "stable-whisper package does not expose a callable load_model()."
                    )
                if runtime_request.device_type == "mps":
                    load_admission = self._resolve_mps_admission_decision(
                        settings=settings,
                        runtime_request=runtime_request,
                        phase="model_load",
                    )
                    if load_admission is not None and not load_admission.allow_mps:
                        self._log_mps_admission_control_fallback(
                            phase="model_load",
                            decision=load_admission,
                        )
                        model = load_model(
                            name=runtime_request.model_name,
                            device="cpu",
                            dq=False,
                            download_root=str(download_root),
                            in_memory=True,
                        )
                        set_stable_whisper_mps_compatibility_enabled(
                            model,
                            enabled=False,
                        )
                        set_stable_whisper_runtime_device(model, device_type="cpu")
                    else:
                        model = load_model(
                            name=runtime_request.model_name,
                            device="cpu",
                            dq=False,
                            download_root=str(download_root),
                            in_memory=True,
                        )
                        set_stable_whisper_runtime_device(model, device_type="cpu")
                        try:
                            model = enable_stable_whisper_mps_compatibility(model)
                        except Exception as err:
                            fallback_reason = self._mps_compatibility_fallback_reason(err)
                            if fallback_reason is None:
                                raise
                            if fallback_reason == "retryable_runtime_error":
                                self._release_torch_runtime_memory_for_retry()
                            self._move_model_to_cpu_runtime(model)
                            error_summary = self._summarize_runtime_error(err)
                            logging.getLogger(__name__).warning(
                                "MPS compatibility mode unavailable; using cpu runtime "
                                "(reason=%s, error=%s).",
                                fallback_reason,
                                error_summary,
                            )
                            set_stable_whisper_mps_compatibility_enabled(
                                model,
                                enabled=False,
                            )
                            set_stable_whisper_runtime_device(model, device_type="cpu")
                        else:
                            set_stable_whisper_runtime_device(model, device_type="mps")
                            if has_known_stable_whisper_sparse_mps_incompatibility():
                                logging.getLogger(__name__).info(
                                    "MPS compatibility mode enabled "
                                    "(sparse_mps_operator_gap_detected)."
                                )
                            else:
                                logging.getLogger(__name__).info("MPS compatibility mode enabled.")
                else:
                    try:
                        model = load_model(
                            name=runtime_request.model_name,
                            device=runtime_request.device_spec,
                            dq=False,
                            download_root=str(download_root),
                            in_memory=True,
                        )
                        set_stable_whisper_runtime_device(
                            model,
                            device_type=runtime_request.device_type,
                        )
                    except Exception as err:
                        should_retry_on_cpu = runtime_request.device_type in {
                            "mps",
                            "cuda",
                        } and self._is_retryable_precision_failure(err)
                        if not should_retry_on_cpu:
                            raise
                        self._release_torch_runtime_memory_for_retry()
                        error_summary = self._summarize_runtime_error(err)
                        logging.getLogger(__name__).warning(
                            "Model load failed on %s due to retryable runtime "
                            "error; retrying on cpu (%s).",
                            runtime_request.device_spec,
                            error_summary,
                        )
                        model = load_model(
                            name=runtime_request.model_name,
                            device="cpu",
                            dq=False,
                            download_root=str(download_root),
                            in_memory=True,
                        )
                        set_stable_whisper_mps_compatibility_enabled(
                            model,
                            enabled=False,
                        )
                        set_stable_whisper_runtime_device(model, device_type="cpu")
            resolved_device_type = get_stable_whisper_runtime_device(
                model,
                default_device_type=runtime_request.device_type,
            )
            logging.getLogger(__name__).info(
                "Runtime device ready (%s).",
                resolved_device_type,
            )
            return model

    def transcribe(
        self,
        *,
        model: object,
        runtime_request: BackendRuntimeRequest,
        file_path: str,
        language: str,
        settings: AppConfig,
    ) -> list[TranscriptWord]:
        """Runs one stable-whisper transcription call and formats word timings."""
        transcribe = getattr(model, "transcribe", None)
        if not callable(transcribe):
            raise RuntimeError(
                "Loaded stable-whisper model does not expose a callable transcribe()."
            )
        typed_transcribe = cast(Callable[..., object], transcribe)
        logger = logging.getLogger(__name__)
        runtime_device_type = get_stable_whisper_runtime_device(
            model,
            default_device_type=runtime_request.device_type,
        )
        runtime_device_type = resolve_stable_whisper_transcribe_runtime_device(
            model=model,
            runtime_request=runtime_request,
            settings=settings,
            runtime_device_type=runtime_device_type,
            resolve_mps_admission_decision=self._resolve_mps_admission_decision,
            should_enforce_transcribe_admission=self._should_enforce_transcribe_admission,
            log_mps_admission_control_fallback=self._log_mps_admission_control_fallback,
            move_model_to_cpu_runtime=self._move_model_to_cpu_runtime,
            set_mps_compatibility_enabled=set_stable_whisper_mps_compatibility_enabled,
            set_runtime_device=set_stable_whisper_runtime_device,
            logger=logger,
        )

        precision_candidates = self._effective_precision_candidates(
            runtime_request=runtime_request,
            runtime_device_type=runtime_device_type,
        )

        def _build_transcribe_kwargs_for_runtime(
            request: BackendRuntimeRequest,
            precision: str,
        ) -> dict[str, object]:
            return self._build_transcribe_kwargs(
                transcribe_callable=typed_transcribe,
                runtime_request=request,
                file_path=file_path,
                language=language,
                precision=precision,
            )

        def _invoke_runtime_transcribe(
            transcribe_kwargs: dict[str, object],
            runtime_device: str,
        ) -> object:
            with scoped_dependency_log_policy(
                policy=_STABLE_WHISPER_TRANSCRIBE_POLICY,
                context=DependencyPolicyContext(
                    backend_id=self.backend_id,
                    phase_name=PHASE_TRANSCRIPTION,
                    op_tag="stable_whisper.transcribe",
                ),
            ):
                mps_compat_context = (
                    stable_whisper_mps_timing_compatibility_context()
                    if (
                        runtime_device == "mps"
                        and is_stable_whisper_mps_compatibility_enabled(model)
                    )
                    else nullcontext()
                )
                with mps_compat_context:
                    return typed_transcribe(**transcribe_kwargs)

        return run_stable_whisper_transcribe_with_retry(
            model=model,
            runtime_request=runtime_request,
            settings=settings,
            runtime_device_type=runtime_device_type,
            precision_candidates=precision_candidates,
            typed_transcribe=typed_transcribe,
            build_transcribe_kwargs=_build_transcribe_kwargs_for_runtime,
            invoke_runtime_transcribe=_invoke_runtime_transcribe,
            classify_failure=(
                lambda err, precision, current_settings: (
                    self._classify_transcription_failure(
                        err=err,
                        runtime_device_type=runtime_device_type,
                        precision=precision,
                        settings=current_settings,
                    )
                )
            ),
            release_runtime_memory_for_retry=self._release_torch_runtime_memory_for_retry,
            summarize_runtime_error=self._summarize_runtime_error,
            move_model_to_cpu_runtime=self._move_model_to_cpu_runtime,
            set_mps_compatibility_disabled=(
                lambda runtime_model: set_stable_whisper_mps_compatibility_enabled(
                    runtime_model,
                    enabled=False,
                )
            ),
            set_runtime_device_cpu=(
                lambda runtime_model: set_stable_whisper_runtime_device(
                    runtime_model,
                    device_type="cpu",
                )
            ),
            normalize_result=self._normalize_result,
            format_transcript=self._format_transcript,
            logger=logger,
        )

    @classmethod
    def _build_transcribe_kwargs(
        cls,
        *,
        transcribe_callable: Callable[..., object],
        runtime_request: BackendRuntimeRequest,
        file_path: str,
        language: str,
        precision: str,
    ) -> dict[str, object]:
        """Builds stable-whisper transcribe kwargs for cross-version compatibility."""
        return build_stable_whisper_transcribe_kwargs(
            transcribe_callable=transcribe_callable,
            runtime_request=runtime_request,
            file_path=file_path,
            language=language,
            precision=precision,
            supports_keyword_argument=(
                lambda callable_obj, parameter_name: cls._supports_keyword_argument(
                    callable_obj,
                    parameter_name=parameter_name,
                )
            ),
        )

    @staticmethod
    def _is_retryable_precision_failure(err: Exception) -> bool:
        """Returns whether one transcription failure may succeed on fallback precision."""
        return is_retryable_precision_failure(err)

    @staticmethod
    def _mps_compatibility_fallback_reason(err: Exception) -> str | None:
        """Returns one CPU-fallback reason for MPS compatibility activation failures."""
        return mps_compatibility_fallback_reason(
            err,
            retryable_precision_checker=(StableWhisperAdapter._is_retryable_precision_failure),
            compatibility_activation_checker=(
                StableWhisperAdapter._is_compatibility_activation_error
            ),
        )

    @staticmethod
    def _is_compatibility_activation_error(err: Exception) -> bool:
        """Returns whether one MPS compatibility activation failure is fail-safe."""
        return is_compatibility_activation_error(err)

    @staticmethod
    def _classify_transcription_failure(
        *,
        err: Exception,
        runtime_device_type: str,
        precision: str,
        settings: AppConfig,
    ) -> TranscriptionFailureClassification:
        """Classifies one stable-whisper transcription failure into action buckets."""
        return classify_transcription_failure_for_runtime(
            err=err,
            runtime_device_type=runtime_device_type,
            precision=precision,
            settings=settings,
            hard_oom_shortcut_enabled=(
                StableWhisperAdapter._mps_hard_oom_shortcut_enabled(settings)
            ),
        )

    @staticmethod
    def _move_model_to_cpu_runtime(model: object) -> bool:
        """Moves one loaded model to CPU runtime when supported by model object."""
        to_method = getattr(model, "to", None)
        if not callable(to_method):
            return False
        to_method(device="cpu")
        return True

    @staticmethod
    def _effective_precision_candidates(
        *,
        runtime_request: BackendRuntimeRequest,
        runtime_device_type: str,
    ) -> tuple[str, ...]:
        """Resolves one runtime precision order using actual loaded model device."""
        return effective_precision_candidates(
            runtime_request=runtime_request,
            runtime_device_type=runtime_device_type,
        )

    @staticmethod
    def _resolve_mps_admission_decision(
        *,
        settings: AppConfig,
        runtime_request: BackendRuntimeRequest,
        phase: Literal["model_load", "transcribe"],
    ) -> MpsAdmissionDecision | None:
        """Returns one dynamic MPS admission decision when control is enabled."""
        return resolve_stable_whisper_mps_admission_decision(
            settings=settings,
            runtime_request=runtime_request,
            phase=phase,
            logger=logging.getLogger(__name__),
            heuristic_resolver=decide_mps_admission_for_transcription,
        )

    @staticmethod
    def _mps_admission_control_enabled(settings: AppConfig) -> bool:
        """Returns whether dynamic MPS admission control is enabled by config."""
        return mps_admission_control_enabled(settings)

    @staticmethod
    def _mps_hard_oom_shortcut_enabled(settings: AppConfig) -> bool:
        """Returns whether hard MPS OOM shortcut is enabled by config."""
        return mps_hard_oom_shortcut_enabled(settings)

    @staticmethod
    def _log_mps_admission_control_fallback(
        *,
        phase: str,
        decision: MpsAdmissionDecision,
    ) -> None:
        """Logs one concise MPS admission-control fallback message."""
        log_mps_admission_control_fallback(
            phase=phase,
            decision=decision,
            logger=logging.getLogger(__name__),
        )

    @staticmethod
    def _should_enforce_transcribe_admission(decision: MpsAdmissionDecision) -> bool:
        """Returns whether one pre-transcribe admission decision should be enforced."""
        return should_enforce_stable_whisper_transcribe_admission(decision)

    @staticmethod
    def _release_torch_runtime_memory_for_retry() -> None:
        """Releases best-effort torch runtime caches before retry attempts."""
        release_torch_runtime_memory_for_retry(logger=logging.getLogger(__name__))

    @staticmethod
    def _summarize_runtime_error(err: Exception, max_chars: int = 180) -> str:
        """Returns one single-line runtime error summary for retry logs."""
        return summarize_runtime_error(err, max_chars=max_chars)

    @staticmethod
    def _supports_keyword_argument(
        callable_obj: Callable[..., object],
        *,
        parameter_name: str,
    ) -> bool:
        """Returns whether one callable supports one keyword argument."""
        try:
            callable_signature = inspect.signature(callable_obj)
        except (TypeError, ValueError):
            return True
        parameter = callable_signature.parameters.get(parameter_name)
        if parameter is not None:
            return parameter.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
                inspect.Parameter.VAR_KEYWORD,
            )
        return any(
            existing_parameter.kind is inspect.Parameter.VAR_KEYWORD
            for existing_parameter in callable_signature.parameters.values()
        )

    @staticmethod
    def _stable_whisper_download_target(
        *,
        model_name: str,
        settings: AppConfig,
    ) -> Path | None:
        """Returns expected checkpoint path for registry-backed stable-whisper models."""
        normalized_name = model_name.strip()
        if not normalized_name:
            raise RuntimeError("Transcription model name must be a non-empty string.")
        if Path(normalized_name).is_file():
            return None
        try:
            whisper_module = importlib.import_module("whisper")
        except ModuleNotFoundError:
            return None
        model_registry = getattr(whisper_module, "_MODELS", None)
        if not isinstance(model_registry, dict):
            return None
        model_url = model_registry.get(normalized_name)
        if not isinstance(model_url, str):
            return None
        filename = Path(urlparse(model_url).path).name
        if not filename:
            return None
        return settings.models.whisper_download_root / filename

    def _normalize_result(self, raw_transcript: object) -> object:
        """Normalizes raw stable-whisper outputs to a WhisperResult instance."""
        with scoped_dependency_log_policy(
            policy=_STABLE_WHISPER_IMPORT_POLICY,
            context=DependencyPolicyContext(
                backend_id=self.backend_id,
                phase_name=PHASE_TRANSCRIPTION,
                op_tag="stable_whisper.result.import",
            ),
            keep_demoted=False,
        ):
            stable_result_module = importlib.import_module("stable_whisper.result")
        whisper_result_ctor = getattr(stable_result_module, "WhisperResult", None)
        if whisper_result_ctor is None:
            raise RuntimeError(
                "stable-whisper package does not expose stable_whisper.result.WhisperResult."
            )
        if isinstance(raw_transcript, whisper_result_ctor):
            return raw_transcript
        if isinstance(raw_transcript, dict | list | str):
            return whisper_result_ctor(raw_transcript)
        raise RuntimeError("Unexpected transcription result type from stable-whisper.")

    @staticmethod
    def _format_transcript(result: object) -> list[TranscriptWord]:
        """Formats a stable-whisper result into transcript words."""
        all_words = getattr(result, "all_words", None)
        if not callable(all_words):
            raise RuntimeError("Invalid Whisper result object.")
        words = all_words()
        if not isinstance(words, list | tuple):
            raise RuntimeError("Invalid stable-whisper words payload.")
        transcript: list[TranscriptWord] = []
        for word in words:
            start = getattr(word, "start", None)
            end = getattr(word, "end", None)
            if start is None or end is None:
                continue
            transcript.append(
                TranscriptWord(
                    word=str(getattr(word, "word", "")),
                    start_seconds=float(start),
                    end_seconds=float(end),
                )
            )
        return transcript
