"""Validate a torch runtime lane for GitHub workflow runners."""

from __future__ import annotations

import argparse
import importlib
import platform
import sys


def _is_available(attribute: object) -> bool:
    """Return a normalized availability result for torch runtime probes."""
    if callable(attribute):
        return bool(attribute())
    return False


def main() -> int:
    """Run the requested torch runtime validation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", choices=("mps", "cuda", "xpu"), required=True)
    args = parser.parse_args()

    torch = importlib.import_module("torch")

    if args.runtime == "mps":
        backends = getattr(torch, "backends", None)
        mps = getattr(backends, "mps", None)
        is_available = getattr(mps, "is_available", None)
        is_built = getattr(mps, "is_built", None)
        available = _is_available(is_available)
        built = _is_available(is_built)
        machine = platform.machine().lower()
        print(f"platform.machine={machine}, mps_available={available}, mps_built={built}")
        if not (available and built):
            raise SystemExit(
                "MPS runtime is unavailable on this runner. Use an Apple Silicon macOS runner."
            )
        return 0

    runtime_module = getattr(torch, args.runtime, None)
    is_available = getattr(runtime_module, "is_available", None)
    available = _is_available(is_available)
    if not available:
        raise SystemExit(f"{args.runtime.upper()} runtime unavailable on the selected runner.")

    details = [f"{args.runtime}_available={available}"]
    if args.runtime == "cuda":
        device_count = getattr(runtime_module, "device_count", None)
        if callable(device_count):
            details.append(f"cuda_device_count={device_count()}")
    print(", ".join(details))
    return 0


if __name__ == "__main__":
    sys.exit(main())
