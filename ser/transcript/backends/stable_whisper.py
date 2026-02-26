"""Stable-whisper transcription adapter implementation."""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, cast
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
from ser.utils.logger import (
    DependencyLogPolicy,
    DependencyPolicyContext,
    WarningPolicy,
    scoped_dependency_log_policy,
)

if TYPE_CHECKING:
    from ser.config import AppConfig

_INVALID_ESCAPE_POLICY_ID = "stable_whisper.invalid_escape_sequence"
_INVALID_ESCAPE_MESSAGE_REGEX = r"^invalid escape sequence '\\,'$"
_INVALID_ESCAPE_MODULE_REGEX = r"^stable_whisper\.result$"
_FP16_CPU_WARNING_POLICY_ID = "stable_whisper.fp16_cpu_fallback_warning"
_FP16_CPU_WARNING_MESSAGE_REGEX = (
    r"^FP16 is not supported on CPU; using FP32 instead$"
)
_DEMUCS_DEPRECATED_POLICY_ID = "stable_whisper.demucs_deprecated_warning"
_DEMUCS_DEPRECATED_MESSAGE_REGEX = (
    r'^``demucs`` is deprecated and will be removed in future versions\. '
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
                    "during transcription; adapter sets fp16=False and uses a "
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
        """Loads one stable-whisper model for CPU inference."""
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
            model = load_model(
                name=runtime_request.model_name,
                device="cpu",
                dq=False,
                download_root=str(download_root),
                in_memory=True,
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
        del settings
        transcribe = getattr(model, "transcribe", None)
        if not callable(transcribe):
            raise RuntimeError(
                "Loaded stable-whisper model does not expose a callable transcribe()."
            )
        typed_transcribe = cast(Callable[..., object], transcribe)
        transcribe_kwargs = self._build_transcribe_kwargs(
            transcribe_callable=typed_transcribe,
            runtime_request=runtime_request,
            file_path=file_path,
            language=language,
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
                raw_transcript = typed_transcribe(**transcribe_kwargs)
        except Exception as err:
            raise RuntimeError("Failed to transcribe audio.") from err
        return self._format_transcript(self._normalize_result(raw_transcript))

    @classmethod
    def _build_transcribe_kwargs(
        cls,
        *,
        transcribe_callable: Callable[..., object],
        runtime_request: BackendRuntimeRequest,
        file_path: str,
        language: str,
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
            kwargs["fp16"] = False
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
