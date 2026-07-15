"""Runtime command and exception-classification helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING

from ser._internal.utils.subtitles import resolve_subtitle_export_request
from ser.config import AppConfig

if TYPE_CHECKING:
    from ser._internal.transcript.profiling import RuntimeCalibrationResult
    from ser.runtime.contracts import InferenceExecution, SubtitleFormat

type _TrainingWorkflow = Callable[..., None]
type _InferenceWorkflow = Callable[..., InferenceExecution]
type _CalibrationWorkflow = Callable[..., RuntimeCalibrationResult]
type _TrainingErrorClassifier = Callable[[Exception], WorkflowErrorDisposition]
type _InferenceErrorClassifier = Callable[[Exception], WorkflowErrorDisposition]
type _CalibrationRunner = Callable[..., RuntimeCalibrationResult]


@dataclass(frozen=True, slots=True)
class WorkflowErrorDisposition:
    """How one CLI workflow error should be logged and exited."""

    exit_code: int
    message: str
    include_traceback: bool = False
    exception: Exception | None = field(default=None, repr=False, compare=False)

    @property
    def exc_info(self) -> tuple[type[BaseException], BaseException, TracebackType | None] | bool:
        """Returns explicit logging traceback data without consulting active exception state."""
        if not self.include_traceback or self.exception is None:
            return False
        return (type(self.exception), self.exception, self.exception.__traceback__)


def classify_training_exception(err: Exception) -> WorkflowErrorDisposition:
    """Classifies one training exception into CLI logging/exit behavior."""
    from ser._internal.models.training_readiness import (
        PreparedPlanError,
        QuarantineBudgetExceeded,
        TrainingReadinessError,
    )

    if isinstance(
        err,
        (PreparedPlanError, QuarantineBudgetExceeded, TrainingReadinessError),
    ):
        return WorkflowErrorDisposition(exit_code=2, message=str(err))
    return WorkflowErrorDisposition(
        exit_code=1,
        message=f"Training workflow failed: {err}",
        include_traceback=True,
        exception=err,
    )


def classify_inference_exception(err: Exception) -> WorkflowErrorDisposition:
    """Classifies one inference exception into CLI logging/exit behavior."""
    from ser._internal.license_check import BackendLicensePolicyError
    from ser._internal.runtime.accurate_inference import (
        AccurateInferenceExecutionError,
        AccurateInferenceTimeoutError,
        AccurateModelLoadError,
        AccurateModelUnavailableError,
        AccurateRuntimeDependencyError,
    )
    from ser._internal.runtime.fast_inference import (
        FastInferenceExecutionError,
        FastInferenceTimeoutError,
        FastModelLoadError,
        FastModelUnavailableError,
    )
    from ser._internal.runtime.medium_inference import (
        MediumInferenceExecutionError,
        MediumInferenceTimeoutError,
        MediumModelLoadError,
        MediumModelUnavailableError,
        MediumRuntimeDependencyError,
    )
    from ser._internal.runtime.registry import UnsupportedProfileError
    from ser._internal.transcript.transcript_extractor import TranscriptionError

    if isinstance(err, UnsupportedProfileError):
        return WorkflowErrorDisposition(exit_code=2, message=str(err))
    if isinstance(
        err,
        (
            BackendLicensePolicyError,
            AccurateRuntimeDependencyError,
            AccurateModelLoadError,
            AccurateModelUnavailableError,
            AccurateInferenceTimeoutError,
            FastModelLoadError,
            FastModelUnavailableError,
            FastInferenceTimeoutError,
            MediumRuntimeDependencyError,
            MediumModelLoadError,
            MediumModelUnavailableError,
            MediumInferenceTimeoutError,
            FileNotFoundError,
        ),
    ):
        return WorkflowErrorDisposition(exit_code=2, message=str(err))
    if isinstance(err, AccurateInferenceExecutionError):
        return WorkflowErrorDisposition(
            exit_code=1,
            message=f"Accurate inference failed: {err}",
        )
    if isinstance(err, FastInferenceExecutionError):
        return WorkflowErrorDisposition(
            exit_code=1,
            message=f"Fast inference failed: {err}",
        )
    if isinstance(err, MediumInferenceExecutionError):
        return WorkflowErrorDisposition(
            exit_code=1,
            message=f"Medium inference failed: {err}",
        )
    if isinstance(err, TranscriptionError):
        return WorkflowErrorDisposition(
            exit_code=3,
            message=f"Transcription failed: {err}",
            include_traceback=True,
            exception=err,
        )
    return WorkflowErrorDisposition(
        exit_code=1,
        message=f"Prediction workflow failed: {err}",
        include_traceback=True,
        exception=err,
    )


def run_training_command(
    *,
    settings: AppConfig,
    pipeline_builder: object | None,
    run_training_workflow: _TrainingWorkflow,
    classify_training_error: _TrainingErrorClassifier,
) -> WorkflowErrorDisposition | None:
    """Runs training command and returns one exit disposition on failure."""
    try:
        run_training_workflow(
            settings=settings,
            pipeline_builder=pipeline_builder,
        )
    except Exception as err:
        return classify_training_error(err)
    return None


def run_inference_command(
    *,
    settings: AppConfig,
    file_path: str | None,
    language: str,
    save_transcript: bool,
    include_transcript: bool,
    subtitle_output_path: str | None = None,
    subtitle_format: SubtitleFormat | None = None,
    pipeline_builder: object | None,
    run_inference_workflow: _InferenceWorkflow,
    classify_inference_error: _InferenceErrorClassifier,
) -> tuple[InferenceExecution | None, WorkflowErrorDisposition | None]:
    """Runs inference command and returns execution plus optional failure disposition."""
    if not isinstance(file_path, str) or not file_path:
        return (
            None,
            WorkflowErrorDisposition(
                exit_code=1,
                message="No audio file provided for prediction.",
            ),
        )
    try:
        subtitle_export = resolve_subtitle_export_request(
            output_path=subtitle_output_path,
            subtitle_format=subtitle_format,
        )
    except ValueError as err:
        return (None, WorkflowErrorDisposition(exit_code=2, message=str(err)))
    if subtitle_export is not None and not include_transcript:
        return (
            None,
            WorkflowErrorDisposition(
                exit_code=2,
                message="Subtitle export requires transcript extraction; remove --no-transcript.",
            ),
        )
    resolved_subtitle_format: SubtitleFormat | None
    resolved_subtitle_output_path: str | None
    if subtitle_export is None:
        resolved_subtitle_format = None
        resolved_subtitle_output_path = None
    else:
        resolved_subtitle_format, resolved_subtitle_output_path = subtitle_export
    try:
        execution = run_inference_workflow(
            settings=settings,
            file_path=file_path,
            language=language,
            save_transcript=save_transcript,
            include_transcript=include_transcript,
            subtitle_output_path=resolved_subtitle_output_path,
            subtitle_format=resolved_subtitle_format,
            pipeline_builder=pipeline_builder,
        )
    except Exception as err:
        return (None, classify_inference_error(err))
    return (execution, None)


def run_transcription_runtime_calibration_workflow(
    *,
    calibration_file: Path,
    language: str,
    calibration_iterations: int,
    calibration_profiles: str,
) -> RuntimeCalibrationResult:
    """Runs runtime calibration workflow with CLI-equivalent argument handling."""
    from ser._internal.transcript.profiling import (
        parse_calibration_profiles,
        run_transcription_runtime_calibration,
    )

    if calibration_iterations <= 0:
        raise ValueError("--calibration-iterations must be greater than zero.")
    calibration_profile_names = parse_calibration_profiles(calibration_profiles)
    return run_transcription_runtime_calibration(
        calibration_file=calibration_file,
        language=language,
        iterations_per_profile=int(calibration_iterations),
        profile_names=calibration_profile_names,
    )


def run_transcription_runtime_calibration_cli(
    *,
    file_path: str | None,
    language: str,
    calibration_iterations: int,
    calibration_profiles: str,
    run_workflow: _CalibrationWorkflow,
) -> RuntimeCalibrationResult:
    """Runs CLI calibration with argument validation and workflow delegation."""
    if not isinstance(file_path, str) or not file_path.strip():
        raise ValueError(
            "Transcription runtime calibration requires --file with one sample audio path."
        )
    if calibration_iterations <= 0:
        raise ValueError("--calibration-iterations must be greater than zero.")
    return run_workflow(
        calibration_file=Path(file_path),
        language=language,
        calibration_iterations=calibration_iterations,
        calibration_profiles=calibration_profiles,
    )


def run_transcription_runtime_calibration_command(
    *,
    file_path: str | None,
    language: str,
    calibration_iterations: int,
    calibration_profiles: str,
    run_calibration_cli: _CalibrationRunner,
) -> tuple[RuntimeCalibrationResult | None, WorkflowErrorDisposition | None]:
    """Runs CLI calibration and maps failures to workflow exit dispositions."""
    try:
        calibration_result = run_calibration_cli(
            file_path=file_path,
            language=language,
            calibration_iterations=calibration_iterations,
            calibration_profiles=calibration_profiles,
        )
    except (ValueError, FileNotFoundError) as err:
        return (
            None,
            WorkflowErrorDisposition(
                exit_code=2,
                message=str(err),
            ),
        )
    except Exception as err:
        return (
            None,
            WorkflowErrorDisposition(
                exit_code=1,
                message=f"Transcription runtime calibration failed: {err}",
                include_traceback=True,
                exception=err,
            ),
        )
    return (calibration_result, None)


__all__ = [
    "WorkflowErrorDisposition",
    "classify_inference_exception",
    "classify_training_exception",
    "run_inference_command",
    "run_training_command",
    "run_transcription_runtime_calibration_cli",
    "run_transcription_runtime_calibration_command",
    "run_transcription_runtime_calibration_workflow",
]
