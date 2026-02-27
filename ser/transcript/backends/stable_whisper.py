"""Stable-whisper transcription adapter implementation."""

from __future__ import annotations

import gc
import importlib
import importlib.util
import inspect
import logging
import os
import sys
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Literal, cast
from urllib.parse import urlparse

from ser.domain import TranscriptWord
from ser.profiles import TranscriptionBackendId
from ser.runtime.phase_contract import (
    PHASE_TRANSCRIPTION,
    PHASE_TRANSCRIPTION_MODEL_LOAD,
)
from ser.transcript.backends.base import (
    BackendRuntimeRequest,
    CompatibilityIssue,
    CompatibilityReport,
    TranscriptionBackendAdapter,
)
from ser.transcript.backends.stable_whisper_mps_compat import (
    enable_stable_whisper_mps_compatibility,
    get_stable_whisper_runtime_device,
    is_stable_whisper_mps_compatibility_enabled,
    set_stable_whisper_mps_compatibility_enabled,
    set_stable_whisper_runtime_device,
    stable_whisper_mps_timing_compatibility_context,
)
from ser.transcript.mps_admission import (
    MpsAdmissionDecision,
    decide_mps_admission_for_transcription,
    format_gib_short,
)
from ser.transcript.mps_admission_overrides import (
    apply_calibrated_mps_admission_override,
)
from ser.transcript.runtime_failures import (
    FailureDisposition,
    TranscriptionFailureClassification,
    classify_stable_whisper_transcription_failure,
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
_TRANSCRIBE_WARNING_MODULE_REGEX = (
    r"^stable_whisper\.whisper_word_level\.original_whisper$"
)
_STABLE_WHISPER_IMPORT_POLICY = DependencyLogPolicy(
    logger_prefixes=frozenset(),
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
        noise_issues: list[CompatibilityIssue] = [
            CompatibilityIssue(
                code="stable_whisper_invalid_escape_sequence",
                message=(
                    "stable-whisper may emit SyntaxWarning invalid escape sequence "
                    "during module import; scoped warning policy is required."
                ),
            ),
            CompatibilityIssue(
                code="stable_whisper_fp16_cpu_fallback_warning",
                message=(
                    "stable-whisper may emit a CPU fp16 fallback UserWarning "
                    "during transcription; adapter resolves device-aware precision "
                    "and uses a "
                    "scoped warning policy fallback."
                ),
            ),
            CompatibilityIssue(
                code="stable_whisper_demucs_deprecated_warning",
                message=(
                    "stable-whisper may emit a demucs deprecation UserWarning "
                    "when legacy demucs argument is used; adapter prefers "
                    "denoiser='demucs' and keeps a scoped warning policy fallback."
                ),
            ),
        ]
        if not self._is_module_available("stable_whisper"):
            functional_issues.append(
                CompatibilityIssue(
                    code="missing_dependency_stable_whisper",
                    message=(
                        "Missing stable-whisper dependencies. Ensure project dependencies "
                        "are installed."
                    ),
                )
            )
        return CompatibilityReport(
            backend_id="stable_whisper",
            functional_issues=tuple(functional_issues),
            noise_issues=tuple(noise_issues),
            policy_ids=(
                _INVALID_ESCAPE_POLICY_ID,
                _FP16_CPU_WARNING_POLICY_ID,
                _DEMUCS_DEPRECATED_POLICY_ID,
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
        ):
            try:
                stable_whisper = importlib.import_module("stable_whisper")
            except ModuleNotFoundError as err:
                raise RuntimeError(
                    "Missing stable-whisper dependencies. Ensure project dependencies "
                    "are installed."
                ) from err

            download_root: Path = settings.models.whisper_download_root
            torch_cache_root: Path = settings.models.torch_cache_root
            os.makedirs(download_root, exist_ok=True)
            os.makedirs(torch_cache_root, exist_ok=True)
            os.environ["TORCH_HOME"] = str(torch_cache_root)
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
                        if not self._is_retryable_precision_failure(err):
                            raise
                        self._release_torch_runtime_memory_for_retry()
                        error_summary = self._summarize_runtime_error(err)
                        logging.getLogger(__name__).warning(
                            "MPS compatibility mode failed; using cpu runtime (%s).",
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
                            logging.getLogger(__name__).info(
                                "MPS compatibility mode enabled."
                            )
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
        runtime_device_type = get_stable_whisper_runtime_device(
            model,
            default_device_type=runtime_request.device_type,
        )
        if runtime_device_type == "mps":
            transcribe_admission = self._resolve_mps_admission_decision(
                settings=settings,
                runtime_request=runtime_request,
                phase="transcribe",
            )
            if transcribe_admission is not None and not transcribe_admission.allow_mps:
                if self._should_enforce_transcribe_admission(transcribe_admission):
                    self._log_mps_admission_control_fallback(
                        phase="transcribe",
                        decision=transcribe_admission,
                    )
                    if self._move_model_to_cpu_runtime(model):
                        set_stable_whisper_mps_compatibility_enabled(
                            model,
                            enabled=False,
                        )
                        set_stable_whisper_runtime_device(model, device_type="cpu")
                        runtime_device_type = "cpu"
                else:
                    logging.getLogger(__name__).info(
                        "MPS admission estimate below budget but confidence=%s; "
                        "allowing one MPS attempt and relying on runtime fallback "
                        "(reason=%s).",
                        transcribe_admission.confidence,
                        transcribe_admission.reason_code,
                    )

        precision_candidates = self._effective_precision_candidates(
            runtime_request=runtime_request,
            runtime_device_type=runtime_device_type,
        )
        last_error: Exception | None = None
        for index, precision in enumerate(precision_candidates):
            transcribe_kwargs = self._build_transcribe_kwargs(
                transcribe_callable=typed_transcribe,
                runtime_request=runtime_request,
                file_path=file_path,
                language=language,
                precision=precision,
            )
            raw_transcript: object
            try:
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
                            runtime_device_type == "mps"
                            and is_stable_whisper_mps_compatibility_enabled(model)
                        )
                        else nullcontext()
                    )
                    with mps_compat_context:
                        raw_transcript = typed_transcribe(**transcribe_kwargs)
                return self._format_transcript(self._normalize_result(raw_transcript))
            except Exception as err:
                last_error = err
                failure_classification = self._classify_transcription_failure(
                    err=err,
                    runtime_device_type=runtime_device_type,
                    precision=precision,
                    settings=settings,
                )
                if failure_classification.is_retryable:
                    self._release_torch_runtime_memory_for_retry()
                should_force_cpu_now = (
                    failure_classification.disposition
                    == FailureDisposition.FAILOVER_CPU_NOW
                    and runtime_device_type in {"mps", "cuda"}
                )
                is_final_candidate = index >= len(precision_candidates) - 1
                is_terminal_candidate = is_final_candidate or should_force_cpu_now
                if (
                    is_terminal_candidate
                    and failure_classification.is_retryable
                    and runtime_device_type in {"mps", "cuda"}
                ):
                    error_summary = self._summarize_runtime_error(err)
                    if should_force_cpu_now:
                        logging.getLogger(__name__).info(
                            "Transcription hard MPS OOM on %s; switching directly "
                            "to cpu (reason=%s, error=%s).",
                            precision,
                            failure_classification.reason_code,
                            error_summary,
                        )
                    else:
                        logging.getLogger(__name__).warning(
                            "Transcription retrying on cpu runtime after %s "
                            "failure (%s).",
                            runtime_device_type,
                            error_summary,
                        )
                    if self._move_model_to_cpu_runtime(model):
                        set_stable_whisper_mps_compatibility_enabled(
                            model,
                            enabled=False,
                        )
                        set_stable_whisper_runtime_device(model, device_type="cpu")
                        runtime_device_type = "cpu"
                        cpu_kwargs = self._build_transcribe_kwargs(
                            transcribe_callable=typed_transcribe,
                            runtime_request=BackendRuntimeRequest(
                                model_name=runtime_request.model_name,
                                use_demucs=runtime_request.use_demucs,
                                use_vad=runtime_request.use_vad,
                                device_spec="cpu",
                                device_type="cpu",
                                precision_candidates=("float32",),
                                memory_tier="not_applicable",
                            ),
                            file_path=file_path,
                            language=language,
                            precision="float32",
                        )
                        try:
                            raw_transcript = typed_transcribe(**cpu_kwargs)
                            return self._format_transcript(
                                self._normalize_result(raw_transcript)
                            )
                        except Exception as cpu_err:
                            raise RuntimeError(
                                "Failed to transcribe audio."
                            ) from cpu_err

                if (
                    is_terminal_candidate
                    or failure_classification.disposition
                    == FailureDisposition.FAIL_FAST
                ):
                    raise RuntimeError("Failed to transcribe audio.") from err
                logging.getLogger(__name__).warning(
                    "Transcription retrying with fallback precision after "
                    "failure using %s on %s: %s",
                    precision,
                    runtime_device_type,
                    err,
                )
        if last_error is not None:
            raise RuntimeError("Failed to transcribe audio.") from last_error
        raise RuntimeError("Failed to transcribe audio.")

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
        kwargs: dict[str, object] = {
            "audio": file_path,
            "language": language,
            "verbose": False,
            "word_timestamps": True,
            "no_speech_threshold": None,
            "vad": runtime_request.use_vad,
        }
        if cls._supports_keyword_argument(
            transcribe_callable,
            parameter_name="fp16",
        ):
            kwargs["fp16"] = precision == "float16"
        if runtime_request.use_demucs:
            if cls._supports_keyword_argument(
                transcribe_callable,
                parameter_name="denoiser",
            ):
                kwargs["denoiser"] = "demucs"
            elif cls._supports_keyword_argument(
                transcribe_callable,
                parameter_name="demucs",
            ):
                kwargs["demucs"] = True
        elif cls._supports_keyword_argument(
            transcribe_callable,
            parameter_name="demucs",
        ):
            kwargs["demucs"] = False
        return kwargs

    @staticmethod
    def _is_retryable_precision_failure(err: Exception) -> bool:
        """Returns whether one transcription failure may succeed on fallback precision."""
        message = str(err).strip().lower()
        if isinstance(err, NotImplementedError):
            not_implemented_markers = (
                "sparsemps",
                "could not run",
                "aten::",
                "backend",
                "mps",
            )
            if all(marker in message for marker in not_implemented_markers):
                return True
            if "std_mean" in message and "mps" in message:
                return True
        retryable_markers = (
            "out of memory",
            "mps backend out of memory",
            "fp16 is not supported on cpu",
            "fp16 is not supported",
            "half precision is not supported",
            "bfloat16 is not supported",
            "unsupported dtype",
            "sparsemps",
            "aten::empty.memory_format",
            "cannot convert a mps tensor to float64 dtype",
            "std_mean.correction",
        )
        return any(marker in message for marker in retryable_markers)

    @staticmethod
    def _classify_transcription_failure(
        *,
        err: Exception,
        runtime_device_type: str,
        precision: str,
        settings: AppConfig,
    ) -> TranscriptionFailureClassification:
        """Classifies one stable-whisper transcription failure into action buckets."""
        classification = classify_stable_whisper_transcription_failure(
            err=err,
            runtime_device_type=runtime_device_type,
            precision=precision,
        )
        if (
            classification.disposition == FailureDisposition.FAILOVER_CPU_NOW
            and not StableWhisperAdapter._mps_hard_oom_shortcut_enabled(settings)
        ):
            return TranscriptionFailureClassification(
                disposition=FailureDisposition.RETRY_NEXT_PRECISION,
                reason_code=f"{classification.reason_code}_disabled",
                is_retryable=True,
            )
        return classification

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
        if runtime_device_type == "cpu":
            return ("float32",)
        return runtime_request.precision_candidates or ("float32",)

    @staticmethod
    def _resolve_mps_admission_decision(
        *,
        settings: AppConfig,
        runtime_request: BackendRuntimeRequest,
        phase: Literal["model_load", "transcribe"],
    ) -> MpsAdmissionDecision | None:
        """Returns one dynamic MPS admission decision when control is enabled."""
        if not StableWhisperAdapter._mps_admission_control_enabled(settings):
            return None
        transcription_settings = getattr(settings, "transcription", None)
        configured_min_headroom_mb = getattr(
            transcription_settings,
            "mps_admission_min_headroom_mb",
            64.0,
        )
        configured_safety_margin_mb = getattr(
            transcription_settings,
            "mps_admission_safety_margin_mb",
            64.0,
        )
        min_headroom_mb = (
            configured_min_headroom_mb
            if isinstance(configured_min_headroom_mb, int | float)
            and not isinstance(configured_min_headroom_mb, bool)
            else 64.0
        )
        safety_margin_mb = (
            configured_safety_margin_mb
            if isinstance(configured_safety_margin_mb, int | float)
            and not isinstance(configured_safety_margin_mb, bool)
            else 64.0
        )
        heuristic_decision = decide_mps_admission_for_transcription(
            model_name=runtime_request.model_name,
            phase=phase,
            min_headroom_mb=float(min_headroom_mb),
            safety_margin_mb=float(safety_margin_mb),
        )
        resolved_decision = apply_calibrated_mps_admission_override(
            settings=settings,
            runtime_request=runtime_request,
            phase=phase,
            heuristic_decision=heuristic_decision,
        )
        if resolved_decision.reason_code != heuristic_decision.reason_code:
            logging.getLogger(__name__).info(
                "MPS admission calibrated override applied for %s "
                "(model=%s, reason=%s, confidence=%s).",
                phase,
                runtime_request.model_name,
                resolved_decision.reason_code,
                resolved_decision.confidence,
            )
        return resolved_decision

    @staticmethod
    def _mps_admission_control_enabled(settings: AppConfig) -> bool:
        """Returns whether dynamic MPS admission control is enabled by config."""
        transcription_settings = getattr(settings, "transcription", None)
        enabled = getattr(transcription_settings, "mps_admission_control_enabled", True)
        return bool(enabled)

    @staticmethod
    def _mps_hard_oom_shortcut_enabled(settings: AppConfig) -> bool:
        """Returns whether hard MPS OOM shortcut is enabled by config."""
        transcription_settings = getattr(settings, "transcription", None)
        enabled = getattr(transcription_settings, "mps_hard_oom_shortcut_enabled", True)
        return bool(enabled)

    @staticmethod
    def _log_mps_admission_control_fallback(
        *,
        phase: str,
        decision: MpsAdmissionDecision,
    ) -> None:
        """Logs one concise MPS admission-control fallback message."""
        logging.getLogger(__name__).info(
            "MPS admission control switched %s to cpu "
            "(reason=%s, confidence=%s, required_%s=%s, available_%s=%s).",
            phase,
            decision.reason_code,
            decision.confidence,
            decision.required_metric,
            format_gib_short(decision.required_bytes),
            decision.available_metric,
            format_gib_short(decision.available_bytes),
        )

    @staticmethod
    def _should_enforce_transcribe_admission(decision: MpsAdmissionDecision) -> bool:
        """Returns whether one pre-transcribe admission decision should be enforced."""
        if decision.confidence == "high":
            return True
        return decision.reason_code == "mps_headroom_unknown_large_model"

    @staticmethod
    def _release_torch_runtime_memory_for_retry() -> None:
        """Releases best-effort torch runtime caches before retry attempts."""
        gc.collect()
        torch_module = sys.modules.get("torch")
        if not isinstance(torch_module, ModuleType):
            return
        mps_module = getattr(torch_module, "mps", None)
        if isinstance(mps_module, ModuleType):
            is_available = getattr(mps_module, "is_available", None)
            empty_cache = getattr(mps_module, "empty_cache", None)
            try:
                if callable(is_available) and is_available() and callable(empty_cache):
                    empty_cache()
            except Exception:
                logging.getLogger(__name__).debug(
                    "Ignored failure while emptying torch MPS cache before retry.",
                    exc_info=True,
                )
        cuda_module = getattr(torch_module, "cuda", None)
        if isinstance(cuda_module, ModuleType):
            is_available = getattr(cuda_module, "is_available", None)
            empty_cache = getattr(cuda_module, "empty_cache", None)
            try:
                if callable(is_available) and is_available() and callable(empty_cache):
                    empty_cache()
            except Exception:
                logging.getLogger(__name__).debug(
                    "Ignored failure while emptying torch CUDA cache before retry.",
                    exc_info=True,
                )

    @staticmethod
    def _summarize_runtime_error(err: Exception, max_chars: int = 180) -> str:
        """Returns one single-line runtime error summary for retry logs."""
        normalized = " ".join(str(err).split())
        if len(normalized) <= max_chars:
            return normalized
        return f"{normalized[: max_chars - 3]}..."

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

    @staticmethod
    def _is_module_available(module_name: str) -> bool:
        """Returns whether one Python module is available or already loaded."""
        if module_name in sys.modules:
            return True
        try:
            return importlib.util.find_spec(module_name) is not None
        except ValueError:
            return module_name in sys.modules
