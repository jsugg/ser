"""OpenSLR metadata resolution helpers for dataset download workflows."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from urllib import parse

OPENSLR_HF_SCRIPT_URL = "https://huggingface.co/datasets/openslr/openslr/raw/main/openslr.py"
OPENSLR_CANONICAL_RESOURCE_BASE_URL = "https://openslr.org/resources"


@dataclass(frozen=True, slots=True)
class OpenSlrPinnedArtifact:
    """Pinned OpenSLR artifact name and ordered mirror URLs."""

    file_name: str
    urls: tuple[str, ...]


_OPENSLR_MIRROR_TEMPLATES: tuple[str, ...] = (
    "https://openslr.org/resources/{dataset_id}/{file_name}",
    "https://openslr.trmal.net/resources/{dataset_id}/{file_name}",
    "https://openslr.elda.org/resources/{dataset_id}/{file_name}",
    "https://openslr.magicdatatech.com/resources/{dataset_id}/{file_name}",
)


def build_openslr_pinned_artifact(
    *,
    dataset_id: str,
    file_name: str,
) -> OpenSlrPinnedArtifact:
    """Builds one pinned OpenSLR artifact entry from a dataset id and file name."""
    urls = tuple(
        template.format(dataset_id=dataset_id, file_name=file_name)
        for template in _OPENSLR_MIRROR_TEMPLATES
    )
    return OpenSlrPinnedArtifact(file_name=file_name, urls=urls)


_OPENSLR_PINNED_ARTIFACTS: dict[str, tuple[OpenSlrPinnedArtifact, ...]] = {
    "88": tuple(
        build_openslr_pinned_artifact(dataset_id="88", file_name=file_name)
        for file_name in ("wav.tgz", "txt.tgz")
    ),
    "115": tuple(
        build_openslr_pinned_artifact(dataset_id="115", file_name=file_name)
        for file_name in (
            "bea_Amused.tar.gz",
            "bea_Angry.tar.gz",
            "bea_Disgusted.tar.gz",
            "bea_Neutral.tar.gz",
            "bea_Sleepy.tar.gz",
            "jenie_Amused.tar.gz",
            "jenie_Angry.tar.gz",
            "jenie_Disgusted.tar.gz",
            "jenie_Neutral.tar.gz",
            "jenie_Sleepy.tar.gz",
            "josh_Amused.tar.gz",
            "josh_Neutral.tar.gz",
            "josh_Sleepy.tar.gz",
            "sam_Amused.tar.gz",
            "sam_Angry.tar.gz",
            "sam_Disgusted.tar.gz",
            "sam_Neutral.tar.gz",
            "sam_Sleepy.tar.gz",
        )
    ),
}


def extract_href_values(html_text: str) -> list[str]:
    """Extracts non-empty href values from one HTML payload."""
    href_values = re.findall(
        r"""href\s*=\s*["']([^"']+)["']""",
        html_text,
        flags=re.IGNORECASE,
    )
    cleaned: list[str] = []
    for value in href_values:
        candidate = value.strip()
        if candidate:
            cleaned.append(candidate)
    return cleaned


def extract_openslr_files_from_hf_script(
    *,
    script_text: str,
    dataset_id: str,
) -> list[str]:
    """Extracts file names for one SLR id from Hugging Face OpenSLR script text."""
    resource_key = f"SLR{dataset_id}"
    module = ast.parse(script_text, mode="exec")
    resources_payload: object | None = None
    for statement in module.body:
        if not isinstance(statement, ast.Assign):
            continue
        for target in statement.targets:
            if isinstance(target, ast.Name) and target.id == "_RESOURCES":
                resources_payload = ast.literal_eval(statement.value)
                break
        if resources_payload is not None:
            break
    if not isinstance(resources_payload, dict):
        return []
    resource_entry = resources_payload.get(resource_key)
    if not isinstance(resource_entry, dict):
        return []
    raw_files = resource_entry.get("Files")
    if not isinstance(raw_files, list):
        return []
    files: list[str] = []
    for value in raw_files:
        if isinstance(value, str) and value.strip():
            files.append(value.strip())
    return files


def build_canonical_archive_urls(
    *,
    dataset_id: str,
    file_names: list[str],
    archive_suffixes: tuple[str, ...],
) -> list[str]:
    """Builds canonical OpenSLR archive URLs from file names and suffix filters."""
    normalized_suffixes = tuple(suffix.lower() for suffix in archive_suffixes)
    urls: list[str] = []
    seen: set[str] = set()
    for file_name in file_names:
        if not file_name.lower().endswith(normalized_suffixes):
            continue
        url = f"{OPENSLR_CANONICAL_RESOURCE_BASE_URL}/{dataset_id}/{file_name}"
        if url in seen:
            continue
        seen.add(url)
        urls.append(url)
    return urls


def extract_archive_urls_from_listing_html(
    *,
    listing_url: str,
    html_text: str,
    archive_suffixes: tuple[str, ...],
) -> list[str]:
    """Extracts archive URLs from one OpenSLR listing HTML payload."""
    raw_links = extract_href_values(html_text)
    normalized_suffixes = tuple(suffix.lower() for suffix in archive_suffixes)
    seen: set[str] = set()
    urls: list[str] = []
    for href in raw_links:
        url = parse.urljoin(listing_url, href)
        lower_url = url.lower()
        if not lower_url.startswith("http://") and not lower_url.startswith("https://"):
            continue
        if not lower_url.endswith(normalized_suffixes):
            continue
        if url in seen:
            continue
        seen.add(url)
        urls.append(url)
    return urls


def resolve_openslr_pinned_artifacts(
    *,
    dataset_id: str,
    archive_suffixes: tuple[str, ...],
) -> tuple[OpenSlrPinnedArtifact, ...]:
    """Resolves pinned OpenSLR artifacts filtered by required archive suffixes."""
    artifacts = _OPENSLR_PINNED_ARTIFACTS.get(dataset_id)
    if artifacts is None:
        return ()
    normalized_suffixes = tuple(suffix.lower() for suffix in archive_suffixes)
    filtered: list[OpenSlrPinnedArtifact] = []
    for artifact in artifacts:
        if artifact.file_name.lower().endswith(normalized_suffixes):
            filtered.append(artifact)
    return tuple(filtered)


__all__ = [
    "OPENSLR_HF_SCRIPT_URL",
    "OpenSlrPinnedArtifact",
    "build_canonical_archive_urls",
    "extract_archive_urls_from_listing_html",
    "extract_openslr_files_from_hf_script",
    "resolve_openslr_pinned_artifacts",
]
