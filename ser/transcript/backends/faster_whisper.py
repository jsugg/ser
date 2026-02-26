"""Faster-whisper transcription adapter implementation."""

from __future__ import annotations

import importlib
import importlib.util
import logging
import os
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, cast

from ser.domain import TranscriptWord
from ser.profiles import TranscriptionBackendId
from ser.runtime.phase_contract import PHASE_TRANSCRIPTION
from ser.transcript.backends.base import (
    BackendRuntimeRequest,
    CompatibilityIssue,
    CompatibilityReport,
    TranscriptionBackendAdapter,
)
from ser.utils.logger import (
    DependencyLogPolicy,
    DependencyPolicyContext,
    scoped_dependency_log_policy,
)

if TYPE_CHECKING:
    from ser.config import AppConfig

_FASTER_WHISPER_INFO_POLICY_ID = "faster_whisper.info_demotion"
_FASTER_WHISPER_INFO_POLICY = DependencyLogPolicy(
    logger_prefixes=frozenset({"faster_whisper"}),
    backend_ids=frozenset({"faster_whisper"}),
    phase_names=frozenset({PHASE_TRANSCRIPTION}),
    op_tags=frozenset({"faster_whisper.transcribe"}),
)


class FasterWhisperAdapter(TranscriptionBackendAdapter):
    """Adapter for faster-whisper backend behavior and compatibility checks."""

    @property
    def backend_id(self) -> TranscriptionBackendId:
        """Returns canonical backend identifier."""
        return "faster_whisper"

    def check_compatibility(
        self,
        *,
        runtime_request: BackendRuntimeRequest,
        settings: AppConfig,
    ) -> CompatibilityReport:
        """Checks faster-whisper dependency/runtime compatibility."""
        del settings
        functional_issues: list[CompatibilityIssue] = []
        operational_issues: list[CompatibilityIssue] = []
        noise_issues: list[CompatibilityIssue] = [
            CompatibilityIssue(
                code="faster_whisper_info_chatter",
                message=(
                    "faster-whisper may emit verbose INFO dependency logs during "
                    "transcription; scoped log demotion policy is required."
                ),
            )
        ]
        if not self._is_module_available("faster_whisper"):
            functional_issues.append(
                CompatibilityIssue(
                    code="missing_dependency_faster_whisper",
                    message=(
                        "Missing faster-whisper dependencies for transcription backend "
                        "'faster_whisper'. Install faster-whisper (for example, "
                        "`uv sync --extra full`) or switch to `stable_whisper`."
                    ),
                )
            )
        if runtime_request.use_demucs:
            operational_issues.append(
                CompatibilityIssue(
                    code="faster_whisper_demucs_unsupported",
                    message=(
                        "faster-whisper backend does not support demucs preprocessing; "
                        "demucs flag will be ignored."
                    ),
                )
            )
        return CompatibilityReport(
            backend_id="faster_whisper",
            functional_issues=tuple(functional_issues),
            operational_issues=tuple(operational_issues),
            noise_issues=tuple(noise_issues),
            policy_ids=(_FASTER_WHISPER_INFO_POLICY_ID,),
        )

    def setup_required(
        self,
        *,
        runtime_request: BackendRuntimeRequest,
        settings: AppConfig,
    ) -> bool:
        """Returns whether local faster-whisper cache is missing for model."""
        model_name = runtime_request.model_name.strip()
        if not model_name or Path(model_name).is_dir():
            return False
        try:
            fw_utils = importlib.import_module("faster_whisper.utils")
        except ModuleNotFoundError:
            return False
        download_model = getattr(fw_utils, "download_model", None)
        if not callable(download_model):
            return False
        try:
            model_path = download_model(
                model_name,
                local_files_only=True,
                cache_dir=str(settings.models.whisper_download_root),
            )
        except Exception:
            return True
        if isinstance(model_path, str):
            return not Path(model_path).is_dir()
        path_getter = getattr(model_path, "__fspath__", None)
        if not callable(path_getter):
            return False
        resolved_path = path_getter()
        if not isinstance(resolved_path, str):
            return False
        return not Path(resolved_path).is_dir()

    def prepare_assets(
        self,
        *,
        runtime_request: BackendRuntimeRequest,
        settings: AppConfig,
    ) -> None:
        """Downloads faster-whisper assets when absent from local cache."""
        model_name = runtime_request.model_name.strip()
        if not model_name or Path(model_name).is_dir():
            return
        try:
            fw_utils = importlib.import_module("faster_whisper.utils")
        except ModuleNotFoundError:
            return
        download_model = getattr(fw_utils, "download_model", None)
        if not callable(download_model):
            return
        os.makedirs(settings.models.whisper_download_root, exist_ok=True)
        download_model(
            model_name,
            local_files_only=False,
            cache_dir=str(settings.models.whisper_download_root),
        )

    def load_model(
        self,
        *,
        runtime_request: BackendRuntimeRequest,
        settings: AppConfig,
    ) -> object:
        """Loads one faster-whisper model for CPU inference."""
        try:
            faster_whisper_module = importlib.import_module("faster_whisper")
        except ModuleNotFoundError as err:
            raise RuntimeError(
                "Missing faster-whisper dependencies for transcription backend "
                "'faster_whisper'. Install faster-whisper (for example, "
                "`uv sync --extra full`) or switch to `stable_whisper`."
            ) from err
        whisper_model = getattr(faster_whisper_module, "WhisperModel", None)
        if whisper_model is None:
            raise RuntimeError(
                "faster-whisper package does not expose WhisperModel."
            )
        download_root: Path = settings.models.whisper_download_root
        os.makedirs(download_root, exist_ok=True)
        return whisper_model(
            runtime_request.model_name,
            device="cpu",
            compute_type="int8",
            download_root=str(download_root),
        )

    def transcribe(
        self,
        *,
        model: object,
        runtime_request: BackendRuntimeRequest,
        file_path: str,
        language: str,
        settings: AppConfig,
    ) -> list[TranscriptWord]:
        """Runs one faster-whisper transcription call and formats words."""
        del settings
        transcribe = getattr(model, "transcribe", None)
        if not callable(transcribe):
            raise RuntimeError(
                "Loaded faster-whisper model does not expose a callable transcribe()."
            )
        if runtime_request.use_demucs:
            logging.getLogger(__name__).warning(
                "faster-whisper backend does not support demucs preprocessing; "
                "demucs flag is ignored."
            )
        raw_transcribe_result: object
        try:
            with scoped_dependency_log_policy(
                policy=_FASTER_WHISPER_INFO_POLICY,
                context=DependencyPolicyContext(
                    backend_id=self.backend_id,
                    phase_name=PHASE_TRANSCRIPTION,
                    op_tag="faster_whisper.transcribe",
                ),
                keep_demoted=True,
            ):
                raw_transcribe_result = transcribe(
                    audio=file_path,
                    language=language,
                    word_timestamps=True,
                    vad_filter=runtime_request.use_vad,
                    beam_size=5,
                )
        except Exception as err:
            raise RuntimeError("Failed to transcribe audio.") from err
        if not isinstance(raw_transcribe_result, tuple) or len(raw_transcribe_result) != 2:
            raise RuntimeError(
                "Unexpected result envelope returned by faster-whisper transcribe()."
            )
        segments = raw_transcribe_result[0]
        if not isinstance(segments, Iterable):
            raise RuntimeError(
                "Unexpected segment stream type returned by faster-whisper."
            )
        transcript_words: list[TranscriptWord] = []
        for segment in cast(Iterable[object], segments):
            words = getattr(segment, "words", None)
            if not isinstance(words, list | tuple):
                continue
            for word in words:
                start = getattr(word, "start", None)
                end = getattr(word, "end", None)
                if start is None or end is None:
                    continue
                transcript_words.append(
                    TranscriptWord(
                        word=str(getattr(word, "word", "")),
                        start_seconds=float(start),
                        end_seconds=float(end),
                    )
                )
        return transcript_words

    @staticmethod
    def _is_module_available(module_name: str) -> bool:
        """Returns whether one Python module is available or already loaded."""
        if module_name in sys.modules:
            return True
        try:
            return importlib.util.find_spec(module_name) is not None
        except ValueError:
            return module_name in sys.modules
