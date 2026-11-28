"""Tests for manifest provenance, curated manifest output, and load-time curation."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import soundfile as sf

import ser._internal.data.adapters.crema_d as crema_d
import ser._internal.data.adapters.ravdess as ravdess
import ser._internal.data.data_loader as dl
import ser._internal.data.dataset_prepare as dp
import ser._internal.data.strategies.default as strategies
import ser.config as config
from ser._internal.data.curation_ledger import (
    CurationLedger,
    CurationLedgerError,
    LedgerEntry,
    git_blob_sha1,
    git_lfs_pointer_sha1,
    git_tree_sha1,
    load_builtin_curation_ledger,
    normalized_audio_sha256,
    source_manifest_path,
)
from ser._internal.data.dataset_audit import audit_dataset_recipe
from ser._internal.data.manifest_jsonl import load_manifest_jsonl
from ser._internal.data.ontology import LabelOntology
from ser._internal.data.recipe import CorpusRecipe, DatasetRecipe
from ser.config import AppConfig


def _write_wav(path: Path, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    sf.write(path, rng.uniform(-0.5, 0.5, 400).astype(np.float32), 16_000)


def _ontology() -> LabelOntology:
    return LabelOntology(ontology_id="default_v1", allowed_labels=frozenset({"happy", "sad"}))


def _recipe(corpus: str) -> DatasetRecipe:
    return DatasetRecipe(
        recipe_id="fixture",
        revision="1",
        ontology_version="canonical-eight-v1",
        corpora=(CorpusRecipe(corpus=corpus, exact_primary_labels=frozenset({"happy", "sad"})),),
    )


def _entry(
    corpus: str, revision: str, root: Path, rel: str, reason: str, kept: str | None = None
) -> LedgerEntry:
    return LedgerEntry(
        corpus=corpus,
        dataset_revision=revision,
        sample_id=f"{corpus}:{rel}",
        normalized_audio_sha256=normalized_audio_sha256(root / rel),
        reason=reason,  # type: ignore[arg-type]
        kept_sample_id=None if kept is None else f"{corpus}:{kept}",
    )


def _ravdess_tree(root: Path) -> tuple[CurationLedger, str]:
    for actor in range(1, 7):
        _write_wav(root / f"Actor_0{actor}" / f"03-01-03-01-01-01-0{actor}.wav", 10 * actor)
        _write_wav(root / f"Actor_0{actor}" / f"03-01-04-01-01-01-0{actor}.wav", 10 * actor + 1)
    repeat = root / "Actor_01" / "03-01-03-01-01-02-01.wav"
    repeat.write_bytes((root / "Actor_01" / "03-01-03-01-01-01-01.wav").read_bytes())
    conflict = root / "Actor_02" / "03-01-03-01-02-01-02.wav"
    conflict.write_bytes((root / "Actor_02" / "03-01-04-01-01-01-02.wav").read_bytes())
    revision = ravdess.RAVDESS_DATASET_REVISION
    ledger = CurationLedger(
        ledger_id="fixture",
        version="1",
        entries=(
            _entry(
                "ravdess",
                revision,
                root,
                "Actor_01/03-01-03-01-01-02-01.wav",
                "exact_repeat_same_label",
                kept="Actor_01/03-01-03-01-01-01-01.wav",
            ),
            _entry(
                "ravdess",
                revision,
                root,
                "Actor_02/03-01-03-01-02-01-02.wav",
                "conflicting_label_duplicate",
            ),
            _entry(
                "ravdess",
                revision,
                root,
                "Actor_02/03-01-04-01-01-01-02.wav",
                "conflicting_label_duplicate",
            ),
        ),
    )
    tree = git_tree_sha1(
        {path.relative_to(root).as_posix(): git_blob_sha1(path) for path in root.rglob("*.wav")}
    )
    return ledger, tree


def _build_ravdess(root: Path, output: Path, ledger: CurationLedger | None) -> object:
    return ravdess.build_ravdess_manifest_jsonl(
        dataset_root=root,
        dataset_glob_pattern=str(root / "Actor_*" / "*.wav"),
        emotion_code_map={"03": "happy", "04": "sad"},
        default_language="en",
        ontology=_ontology(),
        max_failed_file_ratio=0.5,
        output_path=output,
        curation_ledger=ledger,
    )


def test_ravdess_builder_writes_curated_and_source_manifests_with_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "ravdess"
    ledger, tree = _ravdess_tree(root)
    monkeypatch.setattr(ravdess, "RAVDESS_AUDIO_TREE_SHA1", tree)
    output = tmp_path / "manifests" / "ravdess.jsonl"

    _build_ravdess(root, output, ledger)

    curated = load_manifest_jsonl(output, ontology=_ontology(), base_dir=root)
    source = load_manifest_jsonl(source_manifest_path(output), ontology=_ontology(), base_dir=root)
    assert len(source) == 14
    assert {row.sample_id for row in source} - {row.sample_id for row in curated} == {
        entry.sample_id for entry in ledger.entries
    }
    assert {row.dataset_revision for row in source} == {ravdess.RAVDESS_DATASET_REVISION}
    assert all(
        row.normalized_audio_sha256 == normalized_audio_sha256(row.audio_path) for row in source
    )
    assert str(tmp_path) not in output.read_text(encoding="utf-8")

    report = audit_dataset_recipe(
        curated, recipe=_recipe("ravdess"), seed=3, strict=True, curation=ledger
    )
    assert len(report.curation_excluded) == 3

    _write_wav(root / "Actor_01" / "03-01-03-01-01-01-01.wav", 999)
    with pytest.raises(CurationLedgerError, match="changed"):
        audit_dataset_recipe(
            curated, recipe=_recipe("ravdess"), seed=3, strict=True, curation=ledger
        )


def test_ravdess_curated_build_from_an_unverified_tree_fails_before_writing(
    tmp_path: Path,
) -> None:
    root = tmp_path / "ravdess"
    ledger, _tree = _ravdess_tree(root)
    output = tmp_path / "manifests" / "ravdess.jsonl"

    with pytest.raises(CurationLedgerError, match="does not match Zenodo record 1188976"):
        _build_ravdess(root, output, ledger)
    assert not output.exists() and not source_manifest_path(output).exists()

    _build_ravdess(root, output, None)

    rows = load_manifest_jsonl(output, ontology=_ontology(), base_dir=root)
    assert len(rows) == 14
    assert {row.dataset_revision for row in rows} == {None}
    assert all(row.normalized_audio_sha256 is not None for row in rows)


def test_ravdess_builder_without_ledger_writes_the_raw_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "ravdess"
    _ledger, tree = _ravdess_tree(root)
    monkeypatch.setattr(ravdess, "RAVDESS_AUDIO_TREE_SHA1", tree)
    output = tmp_path / "manifests" / "ravdess.jsonl"

    _build_ravdess(root, output, None)

    assert len(load_manifest_jsonl(output, ontology=_ontology(), base_dir=root)) == 14
    assert not source_manifest_path(output).exists()


def _crema_d_tree(root: Path) -> tuple[CurationLedger, str]:
    audio = root / "AudioWAV"
    for actor in range(1001, 1007):
        _write_wav(audio / f"{actor}_IEO_HAP_XX.wav", actor)
        _write_wav(audio / f"{actor}_IEO_SAD_XX.wav", actor + 100)
    (audio / "1001_TIE_SAD_XX.wav").write_bytes((audio / "1001_IEO_HAP_XX.wav").read_bytes())
    ledger = CurationLedger(
        ledger_id="fixture",
        version="1",
        entries=tuple(
            _entry(
                "crema-d",
                crema_d.CREMA_D_DATASET_REVISION,
                root,
                rel,
                "conflicting_label_duplicate",
            )
            for rel in ("AudioWAV/1001_IEO_HAP_XX.wav", "AudioWAV/1001_TIE_SAD_XX.wav")
        ),
    )
    tree = git_tree_sha1({path.name: git_lfs_pointer_sha1(path) for path in audio.glob("*.wav")})
    return ledger, tree


def test_crema_d_builder_derives_revision_from_the_lfs_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "crema-d"
    ledger, tree = _crema_d_tree(root)
    revision = crema_d.CREMA_D_DATASET_REVISION
    output = tmp_path / "manifests" / "crema-d.jsonl"

    def build() -> list[str | None]:
        crema_d.build_crema_d_manifest_jsonl(
            dataset_root=root,
            dataset_glob_pattern="AudioWAV/**/*.wav",
            emotion_code_map={"HAP": "happy", "SAD": "sad"},
            default_language="en",
            ontology=_ontology(),
            max_failed_file_ratio=0.5,
            output_path=output,
            curation_ledger=ledger,
        )
        rows = load_manifest_jsonl(output, ontology=_ontology(), base_dir=root)
        return [row.dataset_revision for row in rows]

    with pytest.raises(CurationLedgerError, match="does not match release v1.0"):
        build()
    assert not output.exists() and not source_manifest_path(output).exists()
    monkeypatch.setattr(crema_d, "CREMA_D_AUDIO_TREE_SHA1", tree)
    revisions = build()
    assert set(revisions) == {revision} and len(revisions) == 11
    assert len(source_manifest_path(output).read_text(encoding="utf-8").splitlines()) == 13


def _prepare_crema_d(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ledger: CurationLedger,
    registered: list[dict[str, object]],
) -> Path:
    monkeypatch.setattr(strategies, "load_builtin_curation_ledger", lambda: ledger)
    monkeypatch.setattr(
        dp, "upsert_dataset_registry_entry", lambda **kwargs: registered.append(kwargs)
    )
    manifest = tmp_path / "manifests" / "crema-d.jsonl"
    dp.prepare_dataset_manifest(
        settings=cast(
            AppConfig,
            SimpleNamespace(
                data_loader=SimpleNamespace(max_failed_file_ratio=0.5), default_language="en"
            ),
        ),
        dataset_id="crema-d",
        dataset_root=tmp_path / "crema-d",
        ontology=_ontology(),
        manifest_path=manifest,
    )
    return manifest


def test_crema_d_curated_build_from_an_unverified_tree_writes_and_registers_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger, _tree = _crema_d_tree(tmp_path / "crema-d")
    registered: list[dict[str, object]] = []

    with pytest.raises(CurationLedgerError, match="does not match release v1.0"):
        _prepare_crema_d(tmp_path, monkeypatch, ledger, registered)

    assert list((tmp_path / "manifests").iterdir()) == []
    assert registered == []


def test_crema_d_curated_build_ignores_a_non_audio_file_in_the_release_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger, tree = _crema_d_tree(tmp_path / "crema-d")
    (tmp_path / "crema-d" / "AudioWAV" / "notes.txt").write_text("notes", encoding="utf-8")
    monkeypatch.setattr(crema_d, "CREMA_D_AUDIO_TREE_SHA1", tree)

    registered: list[dict[str, object]] = []

    manifest = _prepare_crema_d(tmp_path, monkeypatch, ledger, registered)

    assert [entry["manifest_path"] for entry in registered] == [manifest]
    rows = load_manifest_jsonl(manifest, ontology=_ontology(), base_dir=tmp_path / "crema-d")
    assert len(rows) == 11
    assert {row.dataset_revision for row in rows} == {crema_d.CREMA_D_DATASET_REVISION}


@pytest.mark.parametrize("corpus", ["ravdess", "crema-d"])
def test_curated_single_class_subset_gets_the_release_diagnostic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, corpus: str
) -> None:
    root = tmp_path / corpus
    output = tmp_path / "manifests" / f"{corpus}.jsonl"
    if corpus == "ravdess":
        ledger, _tree = _ravdess_tree(root)
        for path in root.glob("Actor_*/03-01-04-*.wav"):
            path.unlink()
        subset = git_tree_sha1(
            {path.relative_to(root).as_posix(): git_blob_sha1(path) for path in root.rglob("*.wav")}
        )
        module, constant, release = ravdess, "RAVDESS_AUDIO_TREE_SHA1", "Zenodo record 1188976"

        def build() -> object:
            return _build_ravdess(root, output, ledger)

    else:
        ledger, _tree = _crema_d_tree(root)
        for path in (root / "AudioWAV").glob("*_SAD_*.wav"):
            path.unlink()
        subset = git_tree_sha1(
            {path.name: git_lfs_pointer_sha1(path) for path in (root / "AudioWAV").glob("*.wav")}
        )
        module, constant, release = crema_d, "CREMA_D_AUDIO_TREE_SHA1", "release v1.0"

        def build() -> object:
            return crema_d.build_crema_d_manifest_jsonl(
                dataset_root=root,
                dataset_glob_pattern="AudioWAV/**/*.wav",
                emotion_code_map={"HAP": "happy", "SAD": "sad"},
                default_language="en",
                ontology=_ontology(),
                max_failed_file_ratio=0.5,
                output_path=output,
                curation_ledger=ledger,
            )

    with pytest.raises(CurationLedgerError, match=f"does not match {release}"):
        build()
    assert not output.exists() and not source_manifest_path(output).exists()

    monkeypatch.setattr(module, constant, subset)

    assert build() is None
    assert not output.exists() and not source_manifest_path(output).exists()


@pytest.mark.parametrize("corpus", ["ravdess", "crema-d"])
def test_unverified_tree_message_names_a_fresh_download(tmp_path: Path, corpus: str) -> None:
    root = tmp_path / corpus
    output = tmp_path / "manifests" / f"{corpus}.jsonl"
    if corpus == "ravdess":
        ledger, _tree = _ravdess_tree(root)
        files = "Audio_Speech_Actors_01-24"
    else:
        ledger, _tree = _crema_d_tree(root)
        files = "v1.0"

    with pytest.raises(CurationLedgerError) as error:
        if corpus == "ravdess":
            _build_ravdess(root, output, ledger)
        else:
            crema_d.build_crema_d_manifest_jsonl(
                dataset_root=root,
                dataset_glob_pattern="AudioWAV/**/*.wav",
                emotion_code_map={"HAP": "happy", "SAD": "sad"},
                default_language="en",
                ontology=_ontology(),
                max_failed_file_ratio=0.5,
                output_path=output,
                curation_ledger=ledger,
            )

    assert str(error.value).endswith(
        f"no manifest was written. Restore the {files} WAV files (remove added ones, restore "
        f"changed ones) and rerun the build, or download a fresh copy with `ser data download "
        f"--dataset {corpus} --dataset-root <new directory>`; downloading again into {root} "
        "keeps the files already there."
    )


def test_ravdess_curated_build_without_audio_files_writes_nothing(tmp_path: Path) -> None:
    output = tmp_path / "manifests" / "ravdess.jsonl"

    assert _build_ravdess(tmp_path / "ravdess", output, load_builtin_curation_ledger()) is None
    assert not output.exists()


def test_builtin_ledger_names_the_pinned_source_revisions() -> None:
    revisions = {entry.dataset_revision for entry in load_builtin_curation_ledger().entries}

    assert revisions == {ravdess.RAVDESS_DATASET_REVISION, crema_d.CREMA_D_DATASET_REVISION}
    assert crema_d.CREMA_D_DATASET_REVISION == (
        "v1.0@f3b8611a309886568dfa957141775b2e05add04a"
        ":AudioWAV=387d831b6b63df157bef95de05ec61dbffc95da3"
    )
    assert ravdess.RAVDESS_DATASET_REVISION.startswith("10.5281/zenodo.1188976:")
    assert ravdess.RAVDESS_AUDIO_TREE_SHA1 in ravdess.RAVDESS_DATASET_REVISION


def test_loading_does_not_invent_hashes_or_revisions(tmp_path: Path) -> None:
    records = []
    for index, code in enumerate(("03", "04", "03", "04"), start=1):
        rel = f"Actor_0{index}/03-01-{code}-01-01-01-0{index}.wav"
        _write_wav(tmp_path / rel, index)
        records.append(
            {
                "sample_id": f"ravdess:{rel}",
                "corpus": "ravdess",
                "audio_path": rel,
                "label": "happy" if code == "03" else "sad",
                "speaker_id": f"ravdess:0{index}",
            }
        )
    manifest = tmp_path / "ravdess.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    settings = SimpleNamespace(
        emotions={"03": "happy", "04": "sad"},
        feature_flags=config.FeatureFlags(),
        audio_read=config.AudioReadConfig(),
        dataset=SimpleNamespace(
            glob_pattern="unused",
            folder=tmp_path,
            manifest_paths=(manifest,),
            recipe="research-v1",
            strict_audit=False,
        ),
        models=SimpleNamespace(num_cores=1, folder=tmp_path / "models"),
        data_loader=SimpleNamespace(max_workers=1, max_failed_file_ratio=0.5),
        training=SimpleNamespace(test_size=0.5, random_state=42, stratify_split=True),
        default_language="en",
    )

    rows = dl.load_utterances(settings=cast(AppConfig, settings))

    assert rows
    assert {(row.dataset_revision, row.normalized_audio_sha256) for row in rows} == {(None, None)}
