"""CLI helpers for dataset consent and dataset preparation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ser.config import get_settings
from ser.data.data_loader import _resolve_label_ontology
from ser.data.dataset_consents import (
    DatasetConsentError,
    is_policy_restricted,
    load_persisted_dataset_consents,
    persist_dataset_consents,
)
from ser.data.dataset_prepare import (
    download_dataset,
    prepare_dataset_manifest,
    resolve_dataset_descriptor,
)


def _is_interactive() -> bool:
    return sys.stdin.isatty() and sys.stderr.isatty()


def _prompt_acceptance(*, message: str) -> bool:
    print(message)
    print("Type 'accept' to continue: ", end="", flush=True)
    try:
        response = input().strip().lower()
    except EOFError:
        return False
    return response == "accept"


def _ensure_descriptor_consents(
    *,
    dataset_id: str,
    policy_id: str,
    license_id: str,
    accept_license_flag: bool,
) -> None:
    settings = get_settings()
    persisted = load_persisted_dataset_consents(settings=settings)
    normalized_policy = policy_id.strip().lower()
    normalized_license = license_id.strip().lower()
    missing_policies: list[str] = []
    missing_licenses: list[str] = []
    if (
        is_policy_restricted(normalized_policy)
        and normalized_policy not in persisted.policy_consents
    ):
        missing_policies.append(normalized_policy)
    if (
        is_policy_restricted(normalized_policy)
        and normalized_license
        and normalized_license not in persisted.license_consents
    ):
        missing_licenses.append(normalized_license)
    if not missing_policies and not missing_licenses:
        return

    interactive = _is_interactive()
    if not accept_license_flag and not interactive:
        raise DatasetConsentError(
            "Non-interactive run requires explicit acknowledgement. "
            "Re-run with `--accept-license`, or persist via `ser configure ... --persist`."
        )

    if not accept_license_flag:
        policy_text = ", ".join(missing_policies) if missing_policies else "(none)"
        license_text = ", ".join(missing_licenses) if missing_licenses else "(none)"
        accepted = _prompt_acceptance(
            message=(
                f"Dataset '{dataset_id}' requires acknowledgement.\n"
                f"Missing policy consent(s): {policy_text}\n"
                f"Missing license consent(s): {license_text}\n"
                "Your acknowledgement will be persisted locally."
            )
        )
        if not accepted:
            raise DatasetConsentError(
                "Dataset acknowledgement declined. "
                "Re-run with `--accept-license` or persist consents via `ser configure`."
            )

    persist_dataset_consents(
        settings=settings,
        accept_policy_ids=missing_policies,
        accept_license_ids=missing_licenses,
        source=f"ser data download --dataset {dataset_id}",
    )


def run_configure_command(argv: list[str]) -> int:
    """Runs `ser configure ...` command."""

    parser = argparse.ArgumentParser(prog="ser configure")
    parser.add_argument(
        "--accept-dataset-policy",
        nargs="+",
        default=[],
        help="Dataset policy IDs to acknowledge (e.g., academic_only share_alike).",
    )
    parser.add_argument(
        "--accept-dataset-license",
        nargs="+",
        default=[],
        help="Dataset license IDs to acknowledge (e.g., odbl-1.0 cc-by-nc-sa-4.0).",
    )
    parser.add_argument(
        "--persist",
        action="store_true",
        help="Persist acknowledgements to a local config file.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show currently persisted dataset consents.",
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    if args.show or (
        not args.accept_dataset_policy and not args.accept_dataset_license
    ):
        consents = load_persisted_dataset_consents(settings=settings)
        policies = ", ".join(sorted(consents.policy_consents)) or "(none)"
        licenses = ", ".join(sorted(consents.license_consents)) or "(none)"
        print(f"Persisted dataset policy consents: {policies}")
        print(f"Persisted dataset license consents: {licenses}")
        return 0

    if not args.persist:
        print("Refusing to modify local config without --persist.")
        return 2

    persist_dataset_consents(
        settings=settings,
        accept_policy_ids=list(args.accept_dataset_policy),
        accept_license_ids=list(args.accept_dataset_license),
        source="ser configure",
    )
    return 0


def run_data_command(argv: list[str]) -> int:
    """Runs `ser data ...` command."""

    parser = argparse.ArgumentParser(prog="ser data")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)
    download_parser = subparsers.add_parser(
        "download", help="Download/prepare datasets"
    )
    download_parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset id (ravdess, crema-d, msp-podcast, biic-podcast)",
    )
    download_parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Dataset install root. Defaults to SER data_root/datasets/<dataset>.",
    )
    download_parser.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
        help="Manifest output path. Defaults to SER data_root/manifests/<dataset>.jsonl.",
    )
    download_parser.add_argument(
        "--labels-csv-path",
        type=Path,
        default=None,
        help="Label/index CSV needed by segment-based corpora (MSP/BIIC).",
    )
    download_parser.add_argument(
        "--audio-base-dir",
        type=Path,
        default=None,
        help="Base directory used to resolve FileName entries in label CSV.",
    )
    download_parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download step (useful when the dataset is already present).",
    )
    download_parser.add_argument(
        "--accept-license",
        action="store_true",
        help="Acknowledge the dataset's policy/license and persist locally.",
    )

    args = parser.parse_args(argv)
    if args.subcommand != "download":
        raise RuntimeError(f"Unhandled ser data subcommand: {args.subcommand}")

    settings = get_settings()
    descriptor = resolve_dataset_descriptor(args.dataset)
    dataset_root = (
        args.dataset_root.expanduser()
        if args.dataset_root is not None
        else (settings.models.folder.parent / "datasets" / descriptor.dataset_id)
    )
    manifest_path = (
        args.manifest_path.expanduser()
        if args.manifest_path is not None
        else (
            settings.models.folder.parent
            / "manifests"
            / f"{descriptor.dataset_id}.jsonl"
        )
    )
    labels_csv_path = (
        args.labels_csv_path.expanduser() if args.labels_csv_path else None
    )
    audio_base_dir = args.audio_base_dir.expanduser() if args.audio_base_dir else None

    _ensure_descriptor_consents(
        dataset_id=descriptor.dataset_id,
        policy_id=descriptor.policy_id,
        license_id=descriptor.license_id,
        accept_license_flag=args.accept_license,
    )
    if not args.skip_download:
        download_dataset(
            settings=settings,
            dataset_id=descriptor.dataset_id,
            dataset_root=dataset_root,
        )

    ontology = _resolve_label_ontology(settings)
    built = prepare_dataset_manifest(
        settings=settings,
        dataset_id=descriptor.dataset_id,
        dataset_root=dataset_root,
        ontology=ontology,
        manifest_path=manifest_path,
        labels_csv_path=labels_csv_path,
        audio_base_dir=audio_base_dir,
        default_language=settings.default_language,
    )
    if not built:
        print("No manifest written (dataset missing or no usable samples).")
        return 2
    print("Wrote manifest(s):")
    for path in built:
        print(f"- {path}")
    return 0
