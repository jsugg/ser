"""Local benchmark helpers for SER profile latency snapshots."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from ser.models.emotion_model import predict_emotions

type BenchmarkSummary = dict[str, float | int]


def benchmark_predict(audio_path: str, runs: int) -> BenchmarkSummary:
    """Measures repeated prediction latency for one input audio file.

    Args:
        audio_path: Path to the audio file used for inference timing.
        runs: Number of benchmark iterations to execute.

    Returns:
        Aggregate latency statistics in seconds.

    Raises:
        ValueError: If `runs` is less than one.
    """
    if runs < 1:
        raise ValueError("runs must be greater than or equal to 1.")

    samples: list[float] = []
    for _ in range(runs):
        start_time = time.perf_counter()
        _ = predict_emotions(audio_path)
        samples.append(time.perf_counter() - start_time)

    ordered_samples = sorted(samples)
    p95_index = min(
        len(ordered_samples) - 1,
        int(round(0.95 * float(len(ordered_samples) - 1))),
    )
    return {
        "runs": runs,
        "mean_seconds": float(statistics.fmean(samples)),
        "median_seconds": float(statistics.median(samples)),
        "p95_seconds": float(ordered_samples[p95_index]),
        "min_seconds": float(ordered_samples[0]),
        "max_seconds": float(ordered_samples[-1]),
    }


def _parse_args() -> argparse.Namespace:
    """Parses benchmark command-line arguments."""
    parser = argparse.ArgumentParser(description="SER local benchmark utility")
    parser.add_argument("--file", required=True, help="Path to the audio input file.")
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Number of benchmark iterations to execute.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional JSON output path for benchmark summary.",
    )
    return parser.parse_args()


def main() -> None:
    """Runs the local benchmark utility and prints or writes JSON summary."""
    args = _parse_args()
    payload = benchmark_predict(audio_path=args.file, runs=args.runs)

    if args.out is None:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
