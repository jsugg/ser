"""CLI helpers for dataset consent and dataset preparation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ser.config import AppConfig
from ser.data.application import (
    build_dataset_capability_snapshot_json_payload,
    build_dataset_registry_snapshot_json_payload,
    collect_dataset_capability_snapshot,
    collect_dataset_registry_snapshot,
    compute_dataset_descriptor_missing_consents,
    persist_missing_dataset_descriptor_consents,
    run_dataset_prepare_workflow,
    run_dataset_uninstall_workflow,
)
from ser.data.dataset_consents import (
    DatasetConsentError,
    load_persisted_dataset_consents,
    persist_dataset_consents,
)
from ser.data.dataset_prepare import SUPPORTED_DATASETS


def _is_interactive() -> bool:
    return sys.stdin.isatty() and sys.stderr.isatty()


def _format_bytes(size_bytes: int) -> str:
    units = ("B", "KB", "MB", "GB", "TB")
    value = float(size_bytes)
    unit = units[0]
    for current_unit in units:
        unit = current_unit
        if value < 1024.0 or current_unit == units[-1]:
            break
        value /= 1024.0
    if unit == "B":
        return f"{int(value)} {unit}"
    return f"{value:.2f} {unit}"


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
    settings: AppConfig,
    dataset_id: str,
    accept_license_flag: bool,
) -> None:
    consent_status = compute_dataset_descriptor_missing_consents(
        settings=settings,
        dataset_id=dataset_id,
    )
    missing_policies = consent_status.missing_policy_consents
    missing_licenses = consent_status.missing_license_consents
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

    persist_missing_dataset_descriptor_consents(
        settings=settings,
        missing_policy_consents=missing_policies,
        missing_license_consents=missing_licenses,
        source=f"ser data download --dataset {dataset_id}",
    )


def run_configure_command(argv: list[str], *, settings: AppConfig) -> int:
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

    if args.show or (not args.accept_dataset_policy and not args.accept_dataset_license):
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


def run_data_command(argv: list[str], *, settings: AppConfig) -> int:
    """Runs `ser data ...` command."""

    parser = argparse.ArgumentParser(prog="ser data")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)
    supported_dataset_ids = ", ".join(sorted(SUPPORTED_DATASETS))
    download_parser = subparsers.add_parser("download", help="Download/prepare datasets")
    registry_parser = subparsers.add_parser("registry", help="Inspect persisted dataset registry")
    catalog_parser = subparsers.add_parser(
        "catalog", help="Show dataset capabilities and pipeline-use candidates"
    )
    uninstall_parser = subparsers.add_parser(
        "uninstall", help="Remove one dataset registry entry and local artifacts"
    )
    registry_parser.add_argument(
        "--show",
        action="store_true",
        help="Show registered dataset roots/manifests/source provenance.",
    )
    registry_parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Registry output format.",
    )
    registry_parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero when registry contains invalid/mismatched entries.",
    )
    catalog_parser.add_argument(
        "--all",
        action="store_true",
        help="Include non-installed supported datasets in output.",
    )
    catalog_parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Catalog output format.",
    )
    uninstall_parser.add_argument(
        "--dataset",
        required=True,
        help=f"Dataset id ({supported_dataset_ids})",
    )
    uninstall_parser.add_argument(
        "--keep-files",
        action="store_true",
        help="Only remove registry entry (keep dataset_root and manifest files).",
    )
    download_parser.add_argument(
        "--dataset",
        required=True,
        help=f"Dataset id ({supported_dataset_ids})",
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
        help=(
            "Label/index CSV path for segment-based corpora. "
            "Optional for MSP-Podcast when using built-in mirror download "
            "(auto-generates dataset_root/labels.csv)."
        ),
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
        "--source",
        type=str,
        default=None,
        help=(
            "Optional download source id override. "
            "Currently supported for MSP-Podcast mirror only "
            "(e.g., AbstractTTS/PODCAST)."
        ),
    )
    download_parser.add_argument(
        "--source-revision",
        type=str,
        default=None,
        help=(
            "Optional download source revision/tag/commit override. "
            "Currently supported for MSP-Podcast mirror only."
        ),
    )
    download_parser.add_argument(
        "--accept-license",
        action="store_true",
        help="Acknowledge the dataset's policy/license and persist locally.",
    )

    args = parser.parse_args(argv)
    if args.subcommand == "registry":
        snapshot = collect_dataset_registry_snapshot(settings=settings)
        if args.format == "json":
            payload = build_dataset_registry_snapshot_json_payload(snapshot)
            print(json.dumps(payload, indent=2, sort_keys=True))
            if args.strict and snapshot.issues:
                return 2
            return 0
        if not snapshot.entries:
            print("Dataset registry is empty.")
            if args.strict and snapshot.issues:
                return 2
            return 0
        for entry in snapshot.entries:
            source_pin = (
                f"{entry.source_repo_id}@{entry.source_revision}"
                if entry.source_repo_id and entry.source_revision
                else "(none)"
            )
            print(f"- {entry.dataset_id}")
            print(f"  dataset_root: {entry.dataset_root}")
            print(f"  manifest_path: {entry.manifest_path}")
            print(f"  source_pin: {source_pin}")
        if snapshot.issues:
            print("Registry health issues:")
            for issue in snapshot.issues:
                print(f"- [{issue.dataset_id}] {issue.code}: {issue.message}")
            if args.strict:
                return 2
        else:
            print("Registry health: ok")
        return 0

    if args.subcommand == "catalog":
        rows = collect_dataset_capability_snapshot(
            settings=settings,
            include_uninstalled=bool(args.all),
        )
        if args.format == "json":
            payload = build_dataset_capability_snapshot_json_payload(rows)
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        if not rows:
            print("No installed datasets found in registry.")
            return 0
        for row in rows:
            print(f"- {row.dataset_id} ({row.display_name})")
            print(f"  registered: {'yes' if row.registered else 'no'}")
            print(f"  installed: {'yes' if row.installed else 'no'}")
            print(f"  manifest_exists: {'yes' if row.manifest_exists else 'no'}")
            if row.dataset_root is not None:
                print(f"  dataset_root: {row.dataset_root}")
            if row.manifest_path is not None:
                print(f"  manifest_path: {row.manifest_path}")
            print(
                "  audio_files: "
                f"referenced={row.referenced_audio_files}, "
                f"present={row.present_audio_files}, "
                f"nonempty={row.nonempty_audio_files}"
            )
            print(
                "  dataset_size: "
                f"{row.dataset_size_bytes} bytes ({_format_bytes(row.dataset_size_bytes)})"
            )
            print(f"  source_url: {row.source_url}")
            print(f"  policy/license: {row.policy_id} / {row.license_id}")
            print(f"  modalities: {', '.join(row.modalities)}")
            print(f"  label_schema: {row.label_schema}")
            print(
                "  candidates: "
                f"supervised_ser={'yes' if row.supervised_ser_candidate else 'no'}, "
                f"ssl={'yes' if row.ssl_candidate else 'no'}, "
                f"multimodal={'yes' if row.multimodal_candidate else 'no'}, "
                "emotion_merge="
                f"{'yes' if row.mergeable_with_emotion_ontology else 'no'}"
            )
            print(f"  recommended_uses: {', '.join(row.recommended_uses)}")
            if row.notes:
                print(f"  notes: {'; '.join(row.notes)}")
        return 0

    if args.subcommand == "uninstall":
        try:
            result = run_dataset_uninstall_workflow(
                settings=settings,
                dataset_id=args.dataset,
                remove_files=not args.keep_files,
            )
        except ValueError as err:
            print(str(err), file=sys.stderr)
            return 2
        if not result.removed_from_registry:
            print(
                f"Dataset `{result.descriptor.dataset_id}` is not registered.",
                file=sys.stderr,
            )
            return 2
        print(
            f"Uninstalled dataset `{result.descriptor.dataset_id}` "
            f"(keep_files={'yes' if args.keep_files else 'no'})."
        )
        for manifest_path in result.removed_manifest_paths:
            print(f"- removed_manifest: {manifest_path}")
        for dataset_root in result.removed_dataset_roots:
            print(f"- removed_dataset_root: {dataset_root}")
        return 0

    if args.subcommand != "download":
        raise RuntimeError(f"Unhandled ser data subcommand: {args.subcommand}")

    try:
        if args.skip_download and (args.source or args.source_revision):
            raise ValueError("Download source overrides cannot be used with --skip-download.")
        _ensure_descriptor_consents(
            settings=settings,
            dataset_id=args.dataset,
            accept_license_flag=args.accept_license,
        )
        workflow_result = run_dataset_prepare_workflow(
            settings=settings,
            dataset_id=args.dataset,
            dataset_root=args.dataset_root,
            manifest_path=args.manifest_path,
            labels_csv_path=args.labels_csv_path,
            audio_base_dir=args.audio_base_dir,
            source_repo_id=args.source,
            source_revision=args.source_revision,
            default_language=settings.default_language,
            skip_download=args.skip_download,
        )
    except (DatasetConsentError, ValueError, FileNotFoundError) as err:
        print(str(err), file=sys.stderr)
        return 2
    except RuntimeError as err:
        print(str(err), file=sys.stderr)
        return 1

    if not workflow_result.manifest_paths:
        print("No manifest written (dataset missing or no usable samples).")
        return 2
    print("Wrote manifest(s):")
    for path in workflow_result.manifest_paths:
        print(f"- {path}")
    return 0
