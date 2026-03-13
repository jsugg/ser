"""MPS compatibility helpers for stable-whisper model load and timing paths."""

from __future__ import annotations

import importlib
import inspect
import logging
import sys
import threading
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, cast

import numpy as np
import torch

_MPS_COMPAT_MODEL_ATTR = "_ser_stable_whisper_mps_compat_enabled"
_RUNTIME_DEVICE_MODEL_ATTR = "_ser_stable_whisper_runtime_device"
_TIMING_PATCH_LOCK = threading.RLock()

type _StdMeanCallable = Any
type _DtwCpuCallable = Any
type _DtwCallable = Any
type _ComputeQksCallable = Any
type _DisableSdpaCallable = Any
type _LogMelSpectrogramCallable = Any

_logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _MpsLogMelCompatibilityDecision:
    """Cached decision for whether MPS log-mel must be computed on CPU."""

    enable_cpu_offload: bool
    reason_code: str
    python_version: str
    torch_version: str


def enable_stable_whisper_mps_compatibility(model: object) -> object:
    """Moves one stable-whisper model to MPS with sparse-buffer safeguards."""
    moved_model = move_model_to_mps_with_alignment_placeholder(model)
    set_stable_whisper_mps_compatibility_enabled(moved_model, enabled=True)
    return moved_model


def move_model_to_mps_with_alignment_placeholder(model: object) -> object:
    """Moves one model to MPS while preserving sparse alignment buffers on CPU."""
    to_method = getattr(model, "to", None)
    if not callable(to_method):
        raise RuntimeError("Loaded stable-whisper model does not expose a callable to().")

    register_buffer = getattr(model, "register_buffer", None)
    alignment_heads_obj = getattr(model, "alignment_heads", None)
    has_sparse_alignment = isinstance(alignment_heads_obj, torch.Tensor) and bool(
        alignment_heads_obj.is_sparse
    )
    if not callable(register_buffer) or not has_sparse_alignment:
        return to_method(device="mps")

    original_alignment_heads = cast(torch.Tensor, alignment_heads_obj)
    dense_placeholder = torch.zeros(
        tuple(int(dim) for dim in original_alignment_heads.shape),
        dtype=torch.bool,
        device="cpu",
    )
    register_buffer("alignment_heads", dense_placeholder, persistent=False)
    try:
        moved_model = to_method(device="mps")
    except Exception as move_error:
        rollback_error: Exception | None = None
        try:
            # Best-effort transactional rollback for partial in-place MPS moves.
            to_method(device="cpu")
        except Exception as err:
            rollback_error = err
        register_buffer(
            "alignment_heads",
            original_alignment_heads,
            persistent=False,
        )
        if rollback_error is not None:
            raise RuntimeError(
                "Failed to move stable-whisper model to MPS and failed to roll back to CPU."
            ) from move_error
        raise

    moved_register_buffer = getattr(moved_model, "register_buffer", None)
    if not callable(moved_register_buffer):
        raise RuntimeError(
            "Loaded stable-whisper model does not expose register_buffer() after MPS move."
        )
    moved_register_buffer(
        "alignment_heads",
        original_alignment_heads.cpu(),
        persistent=False,
    )
    return moved_model


def set_stable_whisper_mps_compatibility_enabled(
    model: object,
    *,
    enabled: bool,
) -> None:
    """Marks one model as requiring MPS timing compatibility patches."""
    try:
        setattr(model, _MPS_COMPAT_MODEL_ATTR, enabled)
    except Exception:
        return


def is_stable_whisper_mps_compatibility_enabled(model: object) -> bool:
    """Returns whether one model needs stable-whisper MPS timing compatibility."""
    return bool(getattr(model, _MPS_COMPAT_MODEL_ATTR, False))


def set_stable_whisper_runtime_device(
    model: object,
    *,
    device_type: str,
) -> None:
    """Stores one model-level runtime device hint for transcription execution."""
    try:
        setattr(model, _RUNTIME_DEVICE_MODEL_ATTR, device_type)
    except Exception:
        return


def get_stable_whisper_runtime_device(
    model: object,
    *,
    default_device_type: str,
) -> str:
    """Returns model runtime device hint or one conservative default."""
    runtime_device = getattr(model, _RUNTIME_DEVICE_MODEL_ATTR, default_device_type)
    if runtime_device in {"cpu", "mps", "cuda"}:
        return cast(str, runtime_device)
    return default_device_type


def _mps_backend_available() -> bool:
    """Returns whether the current torch runtime can execute MPS probes."""
    backends = getattr(torch, "backends", None)
    mps_backend = getattr(backends, "mps", None)
    is_built = getattr(mps_backend, "is_built", None)
    is_available = getattr(mps_backend, "is_available", None)
    return bool(callable(is_built) and callable(is_available) and is_built() and is_available())


@lru_cache(maxsize=1)
def _resolve_mps_log_mel_compatibility_decision() -> _MpsLogMelCompatibilityDecision:
    """Returns whether this runtime needs CPU log-mel offload on MPS."""
    decision_kwargs = {
        "python_version": sys.version.split()[0],
        "torch_version": str(getattr(torch, "__version__", "unknown")),
    }
    if not _mps_backend_available():
        return _MpsLogMelCompatibilityDecision(
            enable_cpu_offload=False,
            reason_code="mps_unavailable",
            **decision_kwargs,
        )
    try:
        _probe_mps_log_mel_frontend()
    except Exception as err:
        if _is_mps_log_mel_compatibility_error(err):
            return _MpsLogMelCompatibilityDecision(
                enable_cpu_offload=True,
                reason_code="mps_log_mel_frontend_cpu_offload_required",
                **decision_kwargs,
            )
        _logger.debug(
            "Ignored non-compatibility failure while probing MPS log-mel frontend support.",
            exc_info=True,
        )
        return _MpsLogMelCompatibilityDecision(
            enable_cpu_offload=False,
            reason_code="mps_log_mel_probe_unclassified_failure",
            **decision_kwargs,
        )
    return _MpsLogMelCompatibilityDecision(
        enable_cpu_offload=False,
        reason_code="mps_log_mel_frontend_supported",
        **decision_kwargs,
    )


def _probe_mps_log_mel_frontend() -> None:
    """Exercises the exact whisper log-mel frontend on MPS for compatibility gating."""
    whisper_audio_module = importlib.import_module("whisper.audio")
    log_mel_spectrogram = getattr(whisper_audio_module, "log_mel_spectrogram", None)
    if not callable(log_mel_spectrogram):
        raise RuntimeError("whisper.audio does not expose log_mel_spectrogram().")
    probe_audio = torch.zeros(2048, dtype=torch.float32, device="mps")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        log_mel_spectrogram(probe_audio, 80, 0)


def _is_mps_log_mel_compatibility_error(err: Exception) -> bool:
    """Returns whether one log-mel probe error requires CPU frontend offload."""
    message = " ".join(str(err).split()).lower()
    if isinstance(err, NotImplementedError):
        if "aten::_fft_r2c" in message and "mps" in message:
            return True
        if "not currently implemented" in message and "mps" in message and "aten::" in message:
            return True
    if any(marker in message for marker in ("complexfloat", "complexhalf", "complex64")):
        return "mps" in message
    return False


def _resolve_mps_log_mel_target_device(
    audio: object,
    device: object | None,
) -> torch.device | None:
    """Returns the MPS device that should receive the frontend output, when applicable."""
    if device is not None:
        if not isinstance(device, (str, int, torch.device)):
            return None
        try:
            explicit_device = torch.device(device)
        except (TypeError, RuntimeError, ValueError):
            explicit_device = None
        if explicit_device is None or explicit_device.type != "mps":
            return None
        return explicit_device
    if torch.is_tensor(audio) and cast(torch.Tensor, audio).device.type == "mps":
        return cast(torch.Tensor, audio).device
    return None


def _build_mps_safe_log_mel_spectrogram(
    original_log_mel_spectrogram: _LogMelSpectrogramCallable,
) -> _LogMelSpectrogramCallable:
    """Builds one log-mel adapter that computes the frontend on CPU for MPS callers."""

    def _log_mel_cpu_safe(
        audio: object,
        n_mels: int = 80,
        padding: int = 0,
        device: object | None = None,
    ) -> torch.Tensor:
        target_device = _resolve_mps_log_mel_target_device(audio, device)
        if target_device is None:
            return cast(
                torch.Tensor,
                original_log_mel_spectrogram(
                    audio,
                    n_mels=n_mels,
                    padding=padding,
                    device=device,
                ),
            )
        cpu_audio = cast(torch.Tensor, audio).float().cpu() if torch.is_tensor(audio) else audio
        cpu_log_mel = cast(
            torch.Tensor,
            original_log_mel_spectrogram(
                cpu_audio,
                n_mels=n_mels,
                padding=padding,
                device="cpu",
            ),
        )
        return cpu_log_mel.to(device=target_device)

    return _log_mel_cpu_safe


@contextmanager
def stable_whisper_mps_timing_compatibility_context() -> Any:
    """Patches stable-whisper MPS gaps for frontend and timing compatibility."""
    with _TIMING_PATCH_LOCK:
        timing_module = importlib.import_module("stable_whisper.timing")
        compatibility_module = importlib.import_module("stable_whisper.whisper_compatibility")
        original_whisper_module = importlib.import_module(
            "stable_whisper.whisper_word_level.original_whisper"
        )
        whisper_audio_module = importlib.import_module("whisper.audio")
        timing_module_any = cast(Any, timing_module)
        compatibility_module_any = cast(Any, compatibility_module)
        original_whisper_module_any = cast(Any, original_whisper_module)
        whisper_audio_module_any = cast(Any, whisper_audio_module)
        whisper_timing_module = importlib.import_module("whisper.timing")
        dtw_cpu = getattr(whisper_timing_module, "dtw_cpu", None)
        if not callable(dtw_cpu):
            raise RuntimeError("whisper.timing does not expose dtw_cpu().")

        original_std_mean = cast(_StdMeanCallable, torch.std_mean)
        original_compat_dtw = cast(_DtwCallable, compatibility_module_any.dtw)
        original_timing_dtw = cast(_DtwCallable, timing_module_any.dtw)
        original_timing_compute_qks = getattr(timing_module_any, "_compute_qks", None)
        original_compat_log_mel = getattr(compatibility_module_any, "log_mel_spectrogram", None)
        original_original_whisper_log_mel = getattr(
            original_whisper_module_any,
            "log_mel_spectrogram",
            None,
        )
        original_whisper_audio_log_mel = getattr(
            whisper_audio_module_any, "log_mel_spectrogram", None
        )

        safe_dtw = _build_cpu_safe_dtw(dtw_cpu)
        safe_std_mean = _build_mps_safe_std_mean(original_std_mean)

        compatibility_module_any.dtw = safe_dtw
        timing_module_any.dtw = safe_dtw
        torch.std_mean = cast(_StdMeanCallable, safe_std_mean)
        frontend_decision = _resolve_mps_log_mel_compatibility_decision()
        if frontend_decision.enable_cpu_offload and callable(original_whisper_audio_log_mel):
            safe_log_mel = _build_mps_safe_log_mel_spectrogram(original_whisper_audio_log_mel)
            if callable(original_compat_log_mel):
                compatibility_module_any.log_mel_spectrogram = safe_log_mel
            if callable(original_original_whisper_log_mel):
                original_whisper_module_any.log_mel_spectrogram = safe_log_mel
            whisper_audio_module_any.log_mel_spectrogram = safe_log_mel
            _logger.debug(
                "Enabled stable-whisper CPU log-mel fallback patch for MPS compatibility "
                "context (reason=%s, python=%s, torch=%s).",
                frontend_decision.reason_code,
                frontend_decision.python_version,
                frontend_decision.torch_version,
            )
        if callable(original_timing_compute_qks):
            disable_sdpa = getattr(compatibility_module_any, "disable_sdpa", None)
            if callable(disable_sdpa):
                timing_module_any._compute_qks = _build_cpu_offloaded_compute_qks(
                    original_compute_qks=original_timing_compute_qks,
                    disable_sdpa=disable_sdpa,
                )
                _logger.debug(
                    "Enabled stable-whisper timing QK CPU offload patch for MPS compatibility context."
                )
            else:
                _logger.debug(
                    "Skipped stable-whisper timing QK offload patch: disable_sdpa unavailable."
                )
        else:
            _logger.debug(
                "Skipped stable-whisper timing QK offload patch: _compute_qks unavailable."
            )
        try:
            yield
        finally:
            torch.std_mean = original_std_mean
            compatibility_module_any.dtw = original_compat_dtw
            timing_module_any.dtw = original_timing_dtw
            if callable(original_compat_log_mel):
                compatibility_module_any.log_mel_spectrogram = original_compat_log_mel
            if callable(original_original_whisper_log_mel):
                original_whisper_module_any.log_mel_spectrogram = original_original_whisper_log_mel
            if callable(original_whisper_audio_log_mel):
                whisper_audio_module_any.log_mel_spectrogram = original_whisper_audio_log_mel
            if original_timing_compute_qks is not None:
                timing_module_any._compute_qks = original_timing_compute_qks


def _build_cpu_safe_dtw(dtw_cpu: _DtwCpuCallable) -> _DtwCallable:
    """Builds one DTW adapter that moves MPS tensors to CPU before float64 cast."""

    def _dtw_cpu_safe(x: torch.Tensor) -> Any:
        return dtw_cpu(x.cpu().double().numpy())

    return _dtw_cpu_safe


def _build_cpu_offloaded_compute_qks(
    *,
    original_compute_qks: _ComputeQksCallable,
    disable_sdpa: _DisableSdpaCallable,
) -> _ComputeQksCallable:
    """Builds one _compute_qks adapter that offloads layer QKs to CPU immediately."""

    def _compute_qks_cpu_offload(*args: object, **kwargs: object) -> Any:
        bound = _bind_compute_qks_arguments(
            compute_qks=original_compute_qks,
            args=args,
            kwargs=kwargs,
        )
        if bound is None:
            return original_compute_qks(*args, **kwargs)

        model = bound["model"]
        tokenizer = bound["tokenizer"]
        text_tokens = bound["text_tokens"]
        mel = bound["mel"]
        tokens = bound["tokens"]
        cache = bound["cache"]
        if not (
            isinstance(mel, torch.Tensor)
            and isinstance(tokens, torch.Tensor)
            and isinstance(cache, dict)
        ):
            return original_compute_qks(*args, **kwargs)
        if not isinstance(text_tokens, list | tuple):
            return original_compute_qks(*args, **kwargs)
        text_token_ids = [int(token_id) for token_id in text_tokens]

        sot_sequence = getattr(tokenizer, "sot_sequence", None)
        eot = getattr(tokenizer, "eot", None)
        if not isinstance(sot_sequence, list | tuple) or not isinstance(eot, int):
            return original_compute_qks(*args, **kwargs)

        dims = getattr(model, "dims", None)
        n_text_layer = getattr(dims, "n_text_layer", None)
        decoder = getattr(model, "decoder", None)
        encoder = getattr(model, "encoder", None)
        if (
            not isinstance(n_text_layer, int)
            or n_text_layer <= 0
            or not callable(decoder)
            or not callable(encoder)
        ):
            return original_compute_qks(*args, **kwargs)

        decoder_blocks_obj = getattr(decoder, "blocks", None)
        if isinstance(decoder_blocks_obj, list | tuple):
            decoder_blocks: list[object] = list(decoder_blocks_obj)
        elif hasattr(decoder_blocks_obj, "__iter__"):
            decoder_blocks = list(cast(Any, decoder_blocks_obj))
        else:
            return original_compute_qks(*args, **kwargs)

        cache["qks"] = [None] * n_text_layer
        hooks: list[Any] = []
        for index, block in enumerate(decoder_blocks):
            cross_attn = getattr(block, "cross_attn", None)
            register_forward_hook = getattr(cross_attn, "register_forward_hook", None)
            if not callable(register_forward_hook):
                for hook in hooks:
                    _remove_hook_safely(hook)
                return original_compute_qks(*args, **kwargs)
            hook = register_forward_hook(
                _build_qk_offload_hook(
                    cache=cache,
                    index=index,
                )
            )
            hooks.append(hook)

        try:
            with torch.no_grad(), disable_sdpa():
                audio_features = cache.get("audio_features")
                if audio_features is None:
                    audio_features = encoder(mel.unsqueeze(0))
                    cache["audio_features"] = audio_features
                decoder_output = cast(Any, decoder(tokens.unsqueeze(0), audio_features))
                logits = cast(torch.Tensor, decoder_output[0])
                sampled_logits = logits[len(sot_sequence) :, :eot]
                token_probs = sampled_logits.softmax(dim=-1)
                token_index = np.arange(len(text_token_ids))
                cache["text_token_probs"] = token_probs[token_index, text_token_ids].tolist()
        finally:
            for hook in hooks:
                _remove_hook_safely(hook)
        return None

    return _compute_qks_cpu_offload


def _bind_compute_qks_arguments(
    *,
    compute_qks: _ComputeQksCallable,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> dict[str, object] | None:
    """Binds arguments for one stable-whisper _compute_qks-style callable."""
    try:
        signature = inspect.signature(compute_qks)
        bound = signature.bind_partial(*args, **kwargs)
    except (TypeError, ValueError):
        return None
    required = ("model", "tokenizer", "text_tokens", "mel", "tokens", "cache")
    if not all(name in bound.arguments for name in required):
        return None
    return {name: bound.arguments[name] for name in required}


def _build_qk_offload_hook(
    *,
    cache: dict[str, object],
    index: int,
) -> Any:
    """Builds one forward hook that stores detached QK tensors on CPU."""

    def _offload_qk_hook(
        _module: object,
        _inputs: object,
        outputs: object,
    ) -> None:
        qk_cache = cache.get("qks")
        if not isinstance(qk_cache, list) or index >= len(qk_cache):
            return
        if not isinstance(outputs, list | tuple) or not outputs:
            qk_cache[index] = None
            return
        qk = outputs[-1]
        if not isinstance(qk, torch.Tensor):
            qk_cache[index] = qk
            return
        detached_qk = qk.detach()
        if detached_qk.device.type != "cpu":
            detached_qk = detached_qk.to(device="cpu")
        qk_cache[index] = detached_qk

    return _offload_qk_hook


def _remove_hook_safely(hook: object) -> None:
    """Removes one torch forward-hook handle defensively."""
    remove = getattr(hook, "remove", None)
    if callable(remove):
        remove()


def _build_mps_safe_std_mean(
    original_std_mean: _StdMeanCallable,
) -> _StdMeanCallable:
    """Builds one std_mean adapter with MPS fallbacks."""

    def _std_mean_safe(
        input_tensor: torch.Tensor,
        *args: object,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        try:
            return cast(
                tuple[torch.Tensor, torch.Tensor],
                original_std_mean(input_tensor, *args, **kwargs),
            )
        except NotImplementedError as err:
            if not _is_mps_std_mean_not_implemented(err, input_tensor=input_tensor):
                raise
            try:
                return _std_mean_var_mean_fallback(
                    input_tensor=input_tensor,
                    args=args,
                    kwargs=kwargs,
                )
            except Exception:
                return _std_mean_cpu_fallback(
                    original_std_mean=original_std_mean,
                    input_tensor=input_tensor,
                    args=args,
                    kwargs=kwargs,
                )

    return _std_mean_safe


def _std_mean_var_mean_fallback(
    *,
    input_tensor: torch.Tensor,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes std/mean on MPS using var_mean+sqrt when std_mean is unsupported."""
    if args:
        raise RuntimeError("std_mean fallback only supports keyword-argument invocation.")
    dim = kwargs.get("dim", None)
    keepdim = bool(kwargs.get("keepdim", False))
    correction_arg = kwargs.get("correction", None)
    if correction_arg is None:
        unbiased = bool(kwargs.get("unbiased", True))
        correction: float = 1.0 if unbiased else 0.0
    elif isinstance(correction_arg, bool):
        correction = 1.0 if correction_arg else 0.0
    elif isinstance(correction_arg, int | float):
        correction = float(correction_arg)
    else:
        raise RuntimeError("Unsupported std_mean correction argument type.")
    variance, mean = torch.var_mean(
        input_tensor,
        dim=cast(Any, dim),
        keepdim=keepdim,
        correction=correction,
    )
    return torch.sqrt(variance), mean


def _std_mean_cpu_fallback(
    *,
    original_std_mean: _StdMeanCallable,
    input_tensor: torch.Tensor,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes std/mean on CPU and returns tensors on the original input device."""
    cpu_std, cpu_mean = cast(
        tuple[torch.Tensor, torch.Tensor],
        original_std_mean(input_tensor.float().cpu(), *args, **kwargs),
    )
    target_dtype = input_tensor.dtype if input_tensor.is_floating_point() else cpu_std.dtype
    std = cpu_std.to(device=input_tensor.device, dtype=target_dtype)
    mean = cpu_mean.to(device=input_tensor.device, dtype=target_dtype)
    return std, mean


def _is_mps_std_mean_not_implemented(
    err: NotImplementedError,
    *,
    input_tensor: torch.Tensor,
) -> bool:
    """Returns whether one error is the known std_mean MPS operator gap."""
    if input_tensor.device.type != "mps":
        return False
    message = str(err).lower()
    required_markers = (
        "std_mean",
        "not currently implemented",
        "mps",
    )
    return all(marker in message for marker in required_markers)
