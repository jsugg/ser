#!/usr/bin/env python3
"""Builds a deterministic synthetic RAVDESS-style dataset for smoke tests."""

from __future__ import annotations

import argparse
import math
import wave
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    """Parses command-line arguments for dataset synthesis."""
    parser = argparse.ArgumentParser(
        description="Generate deterministic synthetic RAVDESS smoke-test fixtures.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("ser/dataset/ravdess"),
        help="Dataset root where Actor_* folders will be created.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16_000,
        help="Sample rate (Hz) used for generated WAV files.",
    )
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=1.5,
        help="Duration in seconds for each generated clip.",
    )
    parser.add_argument(
        "--amplitude",
        type=float,
        default=0.15,
        help="Sine-wave amplitude in normalized [-1.0, 1.0] range.",
    )
    parser.add_argument(
        "--actors",
        nargs="+",
        type=int,
        default=[1, 2],
        help="Actor IDs to synthesize (e.g., 1 2).",
    )
    parser.add_argument(
        "--emotion-codes",
        nargs="+",
        type=int,
        default=list(range(1, 9)),
        help="Emotion code integers to synthesize (e.g., 1 2 3 ... 8).",
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    """Validates argument values before generating files."""
    if args.sample_rate <= 0:
        raise ValueError("--sample-rate must be positive.")
    if args.duration_seconds <= 0.0:
        raise ValueError("--duration-seconds must be positive.")
    if not (0.0 < args.amplitude <= 1.0):
        raise ValueError("--amplitude must be in (0.0, 1.0].")
    if not args.actors:
        raise ValueError("--actors must include at least one actor ID.")
    if not args.emotion_codes:
        raise ValueError("--emotion-codes must include at least one code.")
    if any(actor_id <= 0 for actor_id in args.actors):
        raise ValueError("Actor IDs must be positive integers.")
    if any(code <= 0 for code in args.emotion_codes):
        raise ValueError("Emotion codes must be positive integers.")


def _sine_pcm16_frames(
    *,
    sample_rate: int,
    duration_seconds: float,
    frequency_hz: float,
    amplitude: float,
) -> bytes:
    """Returns little-endian PCM16 mono frames for one deterministic sine clip."""
    total_samples = int(round(sample_rate * duration_seconds))
    if total_samples <= 0:
        raise ValueError("Generated clip would have zero samples.")

    two_pi = 2.0 * math.pi
    max_int16 = 32767
    frames = bytearray(total_samples * 2)
    for sample_index in range(total_samples):
        value = amplitude * math.sin(
            two_pi * frequency_hz * (sample_index / float(sample_rate))
        )
        clamped = max(-1.0, min(1.0, value))
        pcm_value = int(round(clamped * max_int16))
        frames[sample_index * 2 : sample_index * 2 + 2] = pcm_value.to_bytes(
            2,
            byteorder="little",
            signed=True,
        )
    return bytes(frames)


def _write_wav_mono_pcm16(
    *,
    output_path: Path,
    sample_rate: int,
    pcm_frames: bytes,
) -> None:
    """Writes mono PCM16 WAV data."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_frames)


def build_synthetic_dataset(args: argparse.Namespace) -> int:
    """Generates deterministic synthetic clips and returns written file count."""
    output_root = args.output_root.expanduser()
    output_root.mkdir(parents=True, exist_ok=True)

    generated = 0
    for actor_id in args.actors:
        actor_dir = output_root / f"Actor_{actor_id:02d}"
        for emotion_code in args.emotion_codes:
            frequency_hz = (
                180.0 + (float(emotion_code) * 22.0) + (float(actor_id) * 7.0)
            )
            pcm_frames = _sine_pcm16_frames(
                sample_rate=args.sample_rate,
                duration_seconds=args.duration_seconds,
                frequency_hz=frequency_hz,
                amplitude=args.amplitude,
            )
            output_path = (
                actor_dir / f"03-01-{emotion_code:02d}-01-01-01-{actor_id:02d}.wav"
            )
            _write_wav_mono_pcm16(
                output_path=output_path,
                sample_rate=args.sample_rate,
                pcm_frames=pcm_frames,
            )
            generated += 1
    return generated


def main() -> None:
    """CLI entrypoint for synthetic dataset generation."""
    args = _parse_args()
    _validate_args(args)
    generated_files = build_synthetic_dataset(args)
    print(
        "Generated synthetic RAVDESS dataset "
        f"at {args.output_root} ({generated_files} files)."
    )


if __name__ == "__main__":
    main()
