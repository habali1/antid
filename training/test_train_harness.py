#!/usr/bin/env python3
"""test_train_harness.py — HELPER / UNIT tests for the Phase 4A training
harness (train.py, checkpoint.py, numerics.py, data_provenance.py,
evaluate.py's config-resolution helper): pure functions and small,
isolated write/verify boundaries. These tests do NOT exercise the real
epoch-commit/resume orchestration end to end -- see
test_train_orchestration.py for the interrupted-vs-uninterrupted
integration test and crash-boundary fault injection, which is the only
place "resume" is actually proven, not just its individual pieces.

Never initializes or trains the real B4 model and never downloads
pretrained weights. Run directly: python test_train_harness.py
"""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import os
import pickle
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import checkpoint as ckpt_mod  # noqa: E402
import data_provenance  # noqa: E402
import numerics  # noqa: E402
import train  # noqa: E402
from data import Sample  # noqa: E402
from data_provenance import DataIntegrityError  # noqa: E402
import run_manifest_schema  # noqa: E402
from evaluate import (  # noqa: E402
    load_and_validate_run_manifest,
    resolve_eval_config,
    verify_final_artifacts_against_run_manifest,
    verify_run_manifest_matches_checkpoint_provenance,
)

REAL_MANIFEST = HERE.parent / "data" / "northeast_expansion_v1" / "manifest_all_northeast_v1.csv"
REAL_TAXONOMY = HERE.parent / "data" / "northeast_expansion_v1" / "northeast_taxonomy_v1.json"
REAL_LOCAL_DATA_DIR = HERE.parent / "data" / "clean"
REAL_MANIFEST_SHA256 = "b4f39115c80f955755987c0959dc7ed019620f1f143c64b62523f12bc8894000"
REAL_TAXONOMY_SHA256 = "5e9671fddaa1fa46dcb76a4a94787533f67c5dfb54e90841ff60432a17aec27e"


# --------------------------------------------------------------- fixtures
SOURCE_FIELDS = ["species", "slug", "taxon_id", "genus", "genus_id", "photo_id",
                 "lat", "lon", "split", "common_name", "sha256"]


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SOURCE_FIELDS)
        w.writeheader()
        w.writerows(rows)


def _tiny_rows() -> list[dict]:
    rows = []
    for i in range(4):
        rows.append({
            "species": "Bus novus", "slug": "bus-novus", "taxon_id": "2", "genus": "Bus",
            "genus_id": "9", "photo_id": f"N{i}", "lat": "42.0", "lon": "-76.0",
            "split": "train" if i < 3 else "val", "common_name": "", "sha256": "",
        })
    for i in range(2):
        rows.append({
            "species": "Aus testus", "slug": "aus-testus", "taxon_id": "1", "genus": "Aus",
            "genus_id": "8", "photo_id": f"A{i}", "lat": "41.0", "lon": "-75.0",
            "split": "train" if i < 1 else "val", "common_name": "Test ant", "sha256": "",
        })
    return rows


def _committed_taxonomy_for(rows: list[dict]) -> dict:
    slugs = sorted({r["slug"] for r in rows})
    out = {}
    for i, slug in enumerate(slugs):
        r = next(rr for rr in rows if rr["slug"] == slug)
        out[str(i)] = {
            "species_name": r["species"],
            "common_name": (r.get("common_name") or "").strip() or None,
            "taxon_id": int(r["taxon_id"]),
            "slug": slug,
            "genus": r["genus"],
        }
    return out


class TinyFixture:
    """A minimal, fully isolated manifest/local-dir/taxonomy triple -- NOT
    the real Northeast catalog -- with real image bytes and matching
    per-row sha256 columns, for testing explicit-source loading and
    image-byte verification without touching real repo data."""

    def __init__(self, td: Path):
        self.td = td
        self.rows = _tiny_rows()
        self.local_dir = td / "clean"
        for r in self.rows:
            p = self.local_dir / r["slug"] / f"{r['photo_id']}.jpg"
            p.parent.mkdir(parents=True, exist_ok=True)
            content = f"fake-image-{r['slug']}-{r['photo_id']}".encode()
            p.write_bytes(content)
            r["sha256"] = hashlib.sha256(content).hexdigest()
        self.manifest_path = td / "manifest.csv"
        _write_csv(self.manifest_path, self.rows)
        self.committed_taxonomy = _committed_taxonomy_for(self.rows)
        self.taxonomy_path = td / "taxonomy.json"
        self.taxonomy_path.write_text(json.dumps(self.committed_taxonomy, indent=2))
        self.manifest_sha256 = hashlib.sha256(self.manifest_path.read_bytes()).hexdigest()
        self.taxonomy_sha256 = hashlib.sha256(self.taxonomy_path.read_bytes()).hexdigest()

    def load(self, **overrides):
        kwargs = dict(
            manifest_csv=self.manifest_path, local_data_dir=self.local_dir,
            taxonomy_json=self.taxonomy_path,
            expected_manifest_sha256=self.manifest_sha256,
            expected_taxonomy_sha256=self.taxonomy_sha256,
            database_url=None,
        )
        kwargs.update(overrides)
        return data_provenance.load_explicit_manifest_source(**kwargs)


# ------------------------------------------------------- explicit data source
class TestExplicitDataSource(unittest.TestCase):
    def test_explicit_source_wins_without_db_fallback(self):
        with tempfile.TemporaryDirectory() as td:
            fx = TinyFixture(Path(td))
            with mock.patch.object(data_provenance, "_manifest_from_csv",
                                   wraps=data_provenance._manifest_from_csv) as spy:
                samples, taxonomy, m_sha, t_sha, committed = fx.load()
                spy.assert_called_once_with(fx.manifest_path, fx.local_dir)
            self.assertEqual(len(samples), 6)
            self.assertEqual(len(taxonomy), 2)
            self.assertEqual(m_sha, fx.manifest_sha256)
            self.assertEqual(t_sha, fx.taxonomy_sha256)
            self.assertEqual(committed, fx.committed_taxonomy)

    def test_database_url_present_is_a_hard_failure(self):
        with tempfile.TemporaryDirectory() as td:
            fx = TinyFixture(Path(td))
            with mock.patch.object(data_provenance, "_manifest_from_csv") as spy:
                with self.assertRaises(DataIntegrityError) as cm:
                    fx.load(database_url="postgres://irrelevant/db")
                spy.assert_not_called()
            self.assertIn("DATABASE_URL", str(cm.exception))

    def test_manifest_hash_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            fx = TinyFixture(Path(td))
            with self.assertRaises(DataIntegrityError):
                fx.load(expected_manifest_sha256="0" * 64)

    def test_taxonomy_hash_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            fx = TinyFixture(Path(td))
            with self.assertRaises(DataIntegrityError):
                fx.load(expected_taxonomy_sha256="0" * 64)

    def test_taxonomy_content_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            fx = TinyFixture(Path(td))
            tampered = copy.deepcopy(fx.committed_taxonomy)
            tampered["0"]["common_name"] = "Deliberately wrong"
            fx.taxonomy_path.write_text(json.dumps(tampered, indent=2))
            new_hash = hashlib.sha256(fx.taxonomy_path.read_bytes()).hexdigest()
            with self.assertRaises(DataIntegrityError) as cm:
                fx.load(expected_taxonomy_sha256=new_hash)
            self.assertIn("does not exactly match", str(cm.exception))


class TestDatasetShapeAssertion(unittest.TestCase):
    def _samples(self, n_train, n_val):
        out = []
        for i in range(n_train):
            s = Sample(f"/x/t{i}.jpg", 0, "bus-novus")
            s.__dict__["split"] = "train"
            out.append(s)
        for i in range(n_val):
            s = Sample(f"/x/v{i}.jpg", 0, "bus-novus")
            s.__dict__["split"] = "val"
            out.append(s)
        return out

    def test_wrong_total_fails_closed(self):
        samples = self._samples(2, 1)
        taxonomy = {0: {"slug": "bus-novus"}}
        with self.assertRaises(DataIntegrityError) as cm:
            data_provenance.assert_dataset_shape(samples, taxonomy)
        self.assertIn(str(data_provenance.EXPECTED_SAMPLE_COUNT), str(cm.exception))

    def test_wrong_class_count_fails_closed(self):
        n, m = data_provenance.EXPECTED_TRAIN_COUNT, data_provenance.EXPECTED_VAL_COUNT
        samples = self._samples(n, m)
        taxonomy = {0: {"slug": "bus-novus"}}
        with self.assertRaises(DataIntegrityError) as cm:
            data_provenance.assert_dataset_shape(samples, taxonomy)
        self.assertIn(str(data_provenance.EXPECTED_CLASS_COUNT), str(cm.exception))

    def test_wrong_train_val_split_fails_closed(self):
        samples = self._samples(data_provenance.EXPECTED_TRAIN_COUNT - 1,
                                data_provenance.EXPECTED_VAL_COUNT + 1)
        taxonomy = {i: {"slug": f"s{i}"} for i in range(data_provenance.EXPECTED_CLASS_COUNT)}
        with self.assertRaises(DataIntegrityError):
            data_provenance.assert_dataset_shape(samples, taxonomy)


class TestCheckPerClassCounts(unittest.TestCase):
    def _full_valid_counts(self) -> dict:
        counts = {slug: {"train": 200, "val": 40}
                 for slug in data_provenance.NORTHEAST_NEW_SPECIES_SLUGS}
        counts["some-legacy-species"] = {"train": 159, "val": 40}
        return counts

    def test_valid_counts_report_no_problems(self):
        self.assertEqual(data_provenance.check_per_class_counts(self._full_valid_counts()), [])

    def test_northeast_species_wrong_count_is_reported(self):
        counts = self._full_valid_counts()
        a_new_slug = next(iter(data_provenance.NORTHEAST_NEW_SPECIES_SLUGS))
        counts[a_new_slug] = {"train": 199, "val": 40}
        problems = data_provenance.check_per_class_counts(counts)
        self.assertTrue(any(a_new_slug in p for p in problems))

    def test_legacy_train_count_out_of_range_is_reported(self):
        counts = self._full_valid_counts()
        counts["some-legacy-species"] = {"train": 100, "val": 40}
        problems = data_provenance.check_per_class_counts(counts)
        self.assertTrue(any("some-legacy-species" in p for p in problems))

    def test_legacy_empty_val_is_reported(self):
        counts = self._full_valid_counts()
        counts["some-legacy-species"] = {"train": 159, "val": 0}
        problems = data_provenance.check_per_class_counts(counts)
        self.assertTrue(any("nonempty" in p for p in problems))

    def test_missing_northeast_species_is_reported(self):
        counts = self._full_valid_counts()
        a_new_slug = next(iter(data_provenance.NORTHEAST_NEW_SPECIES_SLUGS))
        del counts[a_new_slug]
        problems = data_provenance.check_per_class_counts(counts)
        self.assertTrue(any("missing Northeast species" in p for p in problems))


class TestVerifyImageBytes(unittest.TestCase):
    def test_success_counts_files_and_bytes(self):
        with tempfile.TemporaryDirectory() as td:
            fx = TinyFixture(Path(td))
            stats = data_provenance.verify_image_bytes(fx.manifest_path, fx.local_dir)
            self.assertEqual(stats["files_verified"], 6)
            self.assertGreater(stats["total_bytes"], 0)

    def test_missing_file_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            fx = TinyFixture(Path(td))
            (fx.local_dir / "bus-novus" / "N0.jpg").unlink()
            with self.assertRaises(DataIntegrityError) as cm:
                data_provenance.verify_image_bytes(fx.manifest_path, fx.local_dir)
            self.assertIn("no resolved file", str(cm.exception))

    def test_sha256_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            fx = TinyFixture(Path(td))
            (fx.local_dir / "bus-novus" / "N0.jpg").write_bytes(b"tampered content")
            with self.assertRaises(DataIntegrityError) as cm:
                data_provenance.verify_image_bytes(fx.manifest_path, fx.local_dir)
            self.assertIn("sha256 mismatch", str(cm.exception))

    def test_duplicate_logical_key_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            fx = TinyFixture(Path(td))
            dup_rows = fx.rows + [dict(fx.rows[0])]
            _write_csv(fx.manifest_path, dup_rows)
            with self.assertRaises(DataIntegrityError) as cm:
                data_provenance.verify_image_bytes(fx.manifest_path, fx.local_dir)
            self.assertIn("duplicate logical key", str(cm.exception))

    def test_ambiguous_extension_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            fx = TinyFixture(Path(td))
            (fx.local_dir / "bus-novus" / "N0.png").write_bytes(b"a second file for the same key")
            with self.assertRaises(DataIntegrityError) as cm:
                data_provenance.verify_image_bytes(fx.manifest_path, fx.local_dir)
            self.assertIn("ambiguous", str(cm.exception))

    def test_missing_recorded_sha256_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            fx = TinyFixture(Path(td))
            blanked = [dict(r) for r in fx.rows]
            blanked[0]["sha256"] = ""
            _write_csv(fx.manifest_path, blanked)
            with self.assertRaises(DataIntegrityError) as cm:
                data_provenance.verify_image_bytes(fx.manifest_path, fx.local_dir)
            self.assertIn("no recorded sha256", str(cm.exception))


# ------------------------------------------------------------ output-dir safety
class TestOutputDirSafety(unittest.TestCase):
    def test_nonempty_dir_without_resume_refused(self):
        with tempfile.TemporaryDirectory() as td:
            art = Path(td)
            (art / "stray.txt").write_text("x")
            with self.assertRaises(SystemExit) as cm:
                train.check_output_dir_safety(art, resume=False)
            self.assertIn("not empty", str(cm.exception))

    def test_resume_without_existing_contents_refused(self):
        with tempfile.TemporaryDirectory() as td:
            art = Path(td) / "does_not_exist_yet"
            with self.assertRaises(SystemExit) as cm:
                train.check_output_dir_safety(art, resume=True)
            self.assertIn("nothing to", str(cm.exception))

    def test_empty_dir_fresh_start_allowed(self):
        with tempfile.TemporaryDirectory() as td:
            train.check_output_dir_safety(Path(td), resume=False)

    def test_nonempty_dir_with_resume_allowed(self):
        with tempfile.TemporaryDirectory() as td:
            art = Path(td)
            (art / "checkpoint_last.pth").write_bytes(b"x")
            train.check_output_dir_safety(art, resume=True)


class TestResumableStatus(unittest.TestCase):
    def test_completed_status_refused(self):
        with self.assertRaises(SystemExit) as cm:
            train.check_resumable_status({"status": "completed"}, Path("run_manifest.json"))
        self.assertIn("completed", str(cm.exception))

    def test_paused_for_smoke_status_accepted(self):
        train.check_resumable_status({"status": "paused_for_smoke"}, Path("x"))  # must not raise

    def test_unknown_status_refused(self):
        with self.assertRaises(SystemExit):
            train.check_resumable_status({"status": "some_made_up_status"}, Path("x"))

    def test_resumable_statuses_allowed(self):
        for status in train.RESUMABLE_STATUSES:
            with self.subTest(status=status):
                train.check_resumable_status({"status": status}, Path("x"))  # must not raise

    def test_paused_for_smoke_is_in_resumable_statuses(self):
        self.assertIn("paused_for_smoke", train.RESUMABLE_STATUSES)


class TestGitState(unittest.TestCase):
    def test_missing_commit_refused(self):
        with self.assertRaises(SystemExit) as cm:
            train.require_clean_git_state(None, False)
        self.assertIn("HEAD", str(cm.exception))

    def test_unavailable_dirty_state_refused(self):
        with self.assertRaises(SystemExit) as cm:
            train.require_clean_git_state("abc123", None)
        self.assertIn("cleanliness", str(cm.exception).lower())

    def test_dirty_tree_refused(self):
        with self.assertRaises(SystemExit) as cm:
            train.require_clean_git_state("abc123", True)
        self.assertIn("dirty", str(cm.exception))

    def test_clean_available_state_allowed(self):
        train.require_clean_git_state("abc123", False)  # must not raise


# ------------------------------------------------------------ selection rule
class TestSelectionRule(unittest.TestCase):
    def test_higher_top1_wins(self):
        self.assertGreater(train.selection_key(0.61, 0.70, 5), train.selection_key(0.60, 0.90, 3))

    def test_tie_top1_higher_top3_wins(self):
        self.assertGreater(train.selection_key(0.60, 0.81, 5), train.selection_key(0.60, 0.80, 3))

    def test_full_tie_earlier_epoch_wins(self):
        self.assertGreater(train.selection_key(0.60, 0.80, 2), train.selection_key(0.60, 0.80, 5))

    def test_identical_is_not_an_improvement(self):
        self.assertFalse(train.selection_key(0.60, 0.80, 5) > train.selection_key(0.60, 0.80, 5))


# ------------------------------------------------------------ versioned checkpoint commit
class TestCommitEpoch(unittest.TestCase):
    def _commit(self, art, epoch, is_best, canonical_history, previous_best, top1=0.5, top3=0.7):
        return ckpt_mod.commit_epoch(
            art, epoch=epoch, is_best=is_best, model_state={"w": torch.tensor([float(epoch)])},
            optimizer_state={"step": epoch}, provenance={"p": 1}, resolved_config={"c": 1},
            rng_state={"r": 1}, train_generator_state=torch.Generator().get_state(),
            metrics={"top1": top1, "top3": top3}, canonical_history=canonical_history,
            history_row={"epoch": epoch, "top1": top1}, previous_best=previous_best,
        )

    def test_first_epoch_is_always_best(self):
        with tempfile.TemporaryDirectory() as td:
            art = Path(td)
            best = self._commit(art, 0, True, [], None)
            self.assertEqual(best["epoch"], 0)
            self.assertTrue((art / best["filename"]).exists())
            last = ckpt_mod.torch_load_trusted(art / "checkpoint_last.pth", map_location="cpu")
            self.assertEqual(last["completed_epoch"], 0)
            self.assertEqual(last["best"], best)
            self.assertEqual(len(last["history"]), 1)

    def test_history_jsonl_written_after_checkpoint_last_and_matches(self):
        with tempfile.TemporaryDirectory() as td:
            art = Path(td)
            self._commit(art, 0, True, [], None)
            history_path = art / "history.jsonl"
            self.assertTrue(history_path.exists())
            rows = [json.loads(line) for line in history_path.read_text().splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["epoch"], 0)

    def test_superseded_best_file_is_removed(self):
        with tempfile.TemporaryDirectory() as td:
            art = Path(td)
            best0 = self._commit(art, 0, True, [], None, top1=0.5)
            history = [{"epoch": 0, "top1": 0.5}]
            best1 = self._commit(art, 1, True, history, best0, top1=0.9)
            self.assertFalse((art / best0["filename"]).exists())
            self.assertTrue((art / best1["filename"]).exists())

    def test_non_best_epoch_keeps_previous_best_referenced(self):
        with tempfile.TemporaryDirectory() as td:
            art = Path(td)
            best0 = self._commit(art, 0, True, [], None, top1=0.9)
            history = [{"epoch": 0, "top1": 0.9}]
            best1 = self._commit(art, 1, False, history, best0, top1=0.1)
            self.assertEqual(best1, best0)
            self.assertTrue((art / best0["filename"]).exists())
            last = ckpt_mod.torch_load_trusted(art / "checkpoint_last.pth", map_location="cpu")
            self.assertEqual(last["best"], best0)
            self.assertEqual(len(last["history"]), 2)

    def test_crash_between_best_file_and_checkpoint_last_leaves_orphan(self):
        with tempfile.TemporaryDirectory() as td:
            art = Path(td)
            real_atomic_torch_save = ckpt_mod.atomic_torch_save

            def side_effect(obj, path):
                if Path(path).name == "checkpoint_last.pth":
                    raise OSError("simulated crash before checkpoint_last commits")
                real_atomic_torch_save(obj, path)  # let the best-file write really happen

            with mock.patch.object(ckpt_mod, "atomic_torch_save", side_effect=side_effect):
                with self.assertRaises(OSError):
                    self._commit(art, 0, True, [], None)
            best_files = list(art.glob("checkpoint_best_epoch_*.pth"))
            self.assertEqual(len(best_files), 1)  # orphaned, but present
            self.assertFalse((art / "checkpoint_last.pth").exists())

    def test_crash_between_checkpoint_last_and_history_leaves_stale_history(self):
        with tempfile.TemporaryDirectory() as td:
            art = Path(td)
            self._commit(art, 0, True, [], None)  # epoch 0 commits fully
            history0 = [{"epoch": 0, "top1": 0.5}]
            with mock.patch.object(ckpt_mod, "atomic_write_bytes",
                                   side_effect=OSError("simulated crash writing history.jsonl")):
                with self.assertRaises(OSError):
                    self._commit(art, 1, True, history0, {"epoch": 0, "top1": 0.5,
                                                          "filename": "x", "sha256": "y"})
            # checkpoint_last committed epoch 1, but history.jsonl is stale (still epoch 0 only)
            last = ckpt_mod.torch_load_trusted(art / "checkpoint_last.pth", map_location="cpu")
            self.assertEqual(last["completed_epoch"], 1)
            stale_rows = [json.loads(l) for l in (art / "history.jsonl").read_text().splitlines()]
            self.assertEqual(len(stale_rows), 1)

            repaired = ckpt_mod.repair_history_if_needed(art / "history.jsonl", last["history"])
            self.assertTrue(repaired)
            fixed_rows = [json.loads(l) for l in (art / "history.jsonl").read_text().splitlines()]
            self.assertEqual(len(fixed_rows), 2)

    def test_repair_is_a_noop_when_already_consistent(self):
        with tempfile.TemporaryDirectory() as td:
            art = Path(td)
            self._commit(art, 0, True, [], None)
            last = ckpt_mod.torch_load_trusted(art / "checkpoint_last.pth", map_location="cpu")
            self.assertFalse(ckpt_mod.repair_history_if_needed(art / "history.jsonl", last["history"]))

    def test_orphan_cleanup_failure_is_logged_not_fatal(self):
        with tempfile.TemporaryDirectory() as td:
            art = Path(td)
            best0 = self._commit(art, 0, True, [], None, top1=0.5)
            history = [{"epoch": 0, "top1": 0.5}]
            with mock.patch("pathlib.Path.unlink", side_effect=OSError("locked")):
                best1 = self._commit(art, 1, True, history, best0, top1=0.9)  # must not raise
            self.assertTrue((art / best0["filename"]).exists())  # orphan survives
            self.assertTrue((art / best1["filename"]).exists())

    def test_list_orphan_best_files(self):
        with tempfile.TemporaryDirectory() as td:
            art = Path(td)
            best0 = self._commit(art, 0, True, [], None, top1=0.5)
            (art / "checkpoint_best_epoch_099.pth").write_bytes(b"orphan")
            orphans = ckpt_mod.list_orphan_best_files(art, best0["filename"])
            self.assertEqual([p.name for p in orphans], ["checkpoint_best_epoch_099.pth"])


class TestVerifyReferencedBest(unittest.TestCase):
    def _committed(self, art):
        return ckpt_mod.commit_epoch(
            art, epoch=0, is_best=True, model_state={"w": torch.tensor([1.0])},
            optimizer_state={}, provenance={"p": 1}, resolved_config={"c": 1},
            rng_state={}, train_generator_state=torch.Generator().get_state(),
            metrics={"top1": 0.5, "top3": 0.7}, canonical_history=[],
            history_row={"epoch": 0}, previous_best=None,
        )

    def test_valid_reference_verifies(self):
        with tempfile.TemporaryDirectory() as td:
            art = Path(td)
            self._committed(art)
            last = ckpt_mod.torch_load_trusted(art / "checkpoint_last.pth", map_location="cpu")
            payload = ckpt_mod.verify_referenced_best(art, last)
            self.assertTrue(torch.equal(payload["model"]["w"], torch.tensor([1.0])))

    def test_missing_best_file_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            art = Path(td)
            self._committed(art)
            last = ckpt_mod.torch_load_trusted(art / "checkpoint_last.pth", map_location="cpu")
            (art / last["best"]["filename"]).unlink()
            with self.assertRaises(ckpt_mod.CheckpointIntegrityError) as cm:
                ckpt_mod.verify_referenced_best(art, last)
            self.assertIn("does not exist", str(cm.exception))

    def test_tampered_best_file_hash_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            art = Path(td)
            self._committed(art)
            last = ckpt_mod.torch_load_trusted(art / "checkpoint_last.pth", map_location="cpu")
            (art / last["best"]["filename"]).write_bytes(b"tampered")
            with self.assertRaises(ckpt_mod.CheckpointIntegrityError) as cm:
                ckpt_mod.verify_referenced_best(art, last)
            self.assertIn("sha256", str(cm.exception))

    def test_no_best_reference_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ckpt_mod.CheckpointIntegrityError):
                ckpt_mod.verify_referenced_best(Path(td), {"best": None})


# ------------------------------------------------------------ resume provenance
class TestResumeProvenance(unittest.TestCase):
    def _provenance(self, **overrides) -> dict:
        base = {
            "manifest_sha256": "m" * 64, "taxonomy_sha256": "t" * 64,
            "val_split_sha256": "v" * 64, "resolved_config_sha256": "c" * 64,
            "backbone": "tf_efficientnet_b4", "num_classes": 65,
            "git_commit": "abc123", "numerical_policy": dict(numerics.NUMERICAL_POLICY),
            "run_kind": "full", "limit_batches": None, "wandb_enabled": False,
        }
        base.update(overrides)
        return base

    def test_identical_provenance_has_no_mismatches(self):
        p = self._provenance()
        self.assertEqual(ckpt_mod.provenance_mismatches(p, dict(p)), [])

    def test_each_key_mismatch_is_individually_reported(self):
        current = self._provenance()
        for key in ckpt_mod.PROVENANCE_KEYS:
            with self.subTest(key=key):
                saved = self._provenance()
                if key == "numerical_policy":
                    saved[key] = {**saved[key], "cudnn_deterministic": False}
                elif key == "wandb_enabled":
                    saved[key] = True
                elif key == "limit_batches":
                    saved[key] = 2
                elif isinstance(saved[key], str):
                    saved[key] = saved[key][::-1] + "x"
                elif isinstance(saved[key], int):
                    saved[key] = saved[key] + 1
                mismatches = ckpt_mod.provenance_mismatches(current, saved)
                self.assertEqual(len(mismatches), 1, mismatches)
                self.assertTrue(mismatches[0].startswith(key + ":"))

    def test_smoke_vs_full_run_kind_mismatch_detected(self):
        current = self._provenance(run_kind="full", limit_batches=None)
        saved = self._provenance(run_kind="smoke", limit_batches=2)
        mismatches = ckpt_mod.provenance_mismatches(current, saved)
        self.assertEqual(len(mismatches), 2)  # run_kind AND limit_batches differ

    def test_build_provenance_excludes_artifacts_dir_from_config_hash(self):
        cfg_a = {"seed": 42, "artifacts_dir": "artifacts/run_a"}
        cfg_b = {"seed": 42, "artifacts_dir": "artifacts/run_b"}
        pa = train.build_provenance(manifest_sha256="m", taxonomy_sha256="t",
                                    val_split_sha256="v", cfg=cfg_a, backbone="x",
                                    num_classes=1, git_commit="c",
                                    numerical_policy=numerics.NUMERICAL_POLICY,
                                    run_kind="full", limit_batches=None, wandb_enabled=False)
        pb = train.build_provenance(manifest_sha256="m", taxonomy_sha256="t",
                                    val_split_sha256="v", cfg=cfg_b, backbone="x",
                                    num_classes=1, git_commit="c",
                                    numerical_policy=numerics.NUMERICAL_POLICY,
                                    run_kind="full", limit_batches=None, wandb_enabled=False)
        self.assertEqual(pa["resolved_config_sha256"], pb["resolved_config_sha256"])

    def test_build_provenance_changes_hash_on_hyperparameter_change(self):
        cfg_a = {"seed": 42, "lr": 0.0001}
        cfg_b = {"seed": 42, "lr": 0.0002}
        pa = train.build_provenance(manifest_sha256="m", taxonomy_sha256="t",
                                    val_split_sha256="v", cfg=cfg_a, backbone="x",
                                    num_classes=1, git_commit="c",
                                    numerical_policy=numerics.NUMERICAL_POLICY,
                                    run_kind="full", limit_batches=None, wandb_enabled=False)
        pb = train.build_provenance(manifest_sha256="m", taxonomy_sha256="t",
                                    val_split_sha256="v", cfg=cfg_b, backbone="x",
                                    num_classes=1, git_commit="c",
                                    numerical_policy=numerics.NUMERICAL_POLICY,
                                    run_kind="full", limit_batches=None, wandb_enabled=False)
        self.assertNotEqual(pa["resolved_config_sha256"], pb["resolved_config_sha256"])


# ------------------------------------------------------------ RNG determinism
class TestRNGDeterminism(unittest.TestCase):
    def test_capture_restore_round_trip(self):
        numerics.seed_everything(123)
        state = numerics.capture_rng_state()

        py_first = random.random()
        np_first = np.random.rand()
        torch_first = torch.rand(3)
        cuda_first = torch.cuda.FloatTensor(3).uniform_() if torch.cuda.is_available() else None

        numerics.restore_rng_state(state)

        py_second = random.random()
        np_second = np.random.rand()
        torch_second = torch.rand(3)

        self.assertEqual(py_first, py_second)
        self.assertEqual(np_first, np_second)
        self.assertTrue(torch.equal(torch_first, torch_second))

        if cuda_first is not None:
            numerics.restore_rng_state(state)
            cuda_second = torch.cuda.FloatTensor(3).uniform_()
            self.assertTrue(torch.equal(cuda_first, cuda_second))

    def test_seed_everything_returns_a_seeded_generator(self):
        g1 = numerics.seed_everything(7)
        g2 = numerics.seed_everything(7)
        t1 = torch.randperm(20, generator=g1)
        t2 = torch.randperm(20, generator=g2)
        self.assertTrue(torch.equal(t1, t2))

    def test_worker_init_fn_is_picklable(self):
        pickled = pickle.dumps(numerics.worker_init_fn)
        restored = pickle.loads(pickled)
        self.assertIs(restored, numerics.worker_init_fn)

    def test_worker_init_fn_derives_from_torch_seed_context(self):
        torch.manual_seed(1)
        numerics.worker_init_fn(0)
        state_a = random.getstate()

        torch.manual_seed(2)
        numerics.worker_init_fn(0)
        state_b = random.getstate()

        self.assertNotEqual(state_a, state_b)


# ------------------------------------------------------------ taxonomy equality
class TestTaxonomyEquality(unittest.TestCase):
    def test_matches_when_equal(self):
        committed = {"0": {"slug": "bus-novus", "taxon_id": 2}}
        taxonomy = {0: {"slug": "bus-novus", "taxon_id": 2}}
        self.assertTrue(data_provenance.taxonomy_matches_committed(taxonomy, committed))

    def test_does_not_match_on_any_field_difference(self):
        committed = {"0": {"slug": "bus-novus", "taxon_id": 2}}
        taxonomy = {0: {"slug": "bus-novus", "taxon_id": 999}}
        self.assertFalse(data_provenance.taxonomy_matches_committed(taxonomy, committed))

    def test_does_not_match_on_extra_or_missing_class(self):
        committed = {"0": {"slug": "bus-novus", "taxon_id": 2},
                    "1": {"slug": "aus-testus", "taxon_id": 1}}
        taxonomy = {0: {"slug": "bus-novus", "taxon_id": 2}}
        self.assertFalse(data_provenance.taxonomy_matches_committed(taxonomy, committed))


# ------------------------------------------------------------ numerical policy
class TestNumericalPolicy(unittest.TestCase):
    def test_apply_sets_full_fp32_flags(self):
        applied = numerics.apply_numerical_policy()
        self.assertEqual(applied, numerics.NUMERICAL_POLICY)
        current = numerics.current_numerical_policy()
        self.assertEqual(current, numerics.NUMERICAL_POLICY)
        self.assertFalse(torch.backends.cudnn.allow_tf32)
        self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
        self.assertEqual(torch.get_float32_matmul_precision(), "highest")
        self.assertFalse(torch.backends.cudnn.benchmark)
        self.assertTrue(torch.backends.cudnn.deterministic)

    def test_apply_is_idempotent(self):
        a = numerics.apply_numerical_policy()
        b = numerics.apply_numerical_policy()
        self.assertEqual(a, b)


# ------------------------------------------------------------ evaluate.py config resolution
class TestResolveEvalConfig(unittest.TestCase):
    def test_default_config_path_uses_run_config(self):
        run_cfg = {"model": {"backbone": "tf_efficientnet_b4"}}
        result = resolve_eval_config(None, run_cfg, "irrelevant-hash")
        self.assertIs(result, run_cfg)

    def test_matching_supplied_config_is_accepted(self):
        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "custom.yaml"
            cfg_path.write_text("seed: 42\n")
            supplied_cfg = {"seed": 42}
            expected_hash = data_provenance.resolved_config_sha256(supplied_cfg)
            run_cfg = {"seed": 999}  # deliberately different, to prove supplied wins
            result = resolve_eval_config(cfg_path, run_cfg, expected_hash)
            self.assertEqual(result, supplied_cfg)

    def test_mismatched_supplied_config_is_refused(self):
        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "custom.yaml"
            cfg_path.write_text("seed: 1\n")
            with self.assertRaises(SystemExit) as cm:
                resolve_eval_config(cfg_path, {"seed": 999}, "0" * 64)
            self.assertIn("does not match", str(cm.exception))


# ------------------------------------------------------------ evaluate.py run_manifest binding
def _eval_valid_run_manifest(*, completed: bool = False) -> dict:
    rm = {
        "run_manifest_schema_version": run_manifest_schema.RUN_MANIFEST_SCHEMA_VERSION,
        "status": "completed" if completed else "running", "run_kind": "full",
        "git_head": "abc123", "git_dirty": False,
        "invocations": [{"timestamp_utc": "2026-01-01T00:00:00Z", "argv": [],
                         "resume": False, "pause_after_epoch": None}],
        "started_at_utc": "2026-01-01T00:00:00Z", "updated_at_utc": "2026-01-01T00:00:00Z",
        "finished_at_utc": "2026-01-02T00:00:00Z" if completed else None,
        "manifest": {"path": "/data/manifest.csv", "sha256": "a" * 64, "rows": 13581},
        "taxonomy_source": {"path": "/data/taxonomy.json", "sha256": "b" * 64, "num_classes": 65},
        "val_split": {"path": "/art/val_split.json", "sha256": "c" * 64,
                     "n_train": 10985, "n_val": 2596, "n_total": 13581},
        "last_completed_epoch": 5,
        "best": {"epoch": 3, "metrics": {"top1": 0.6, "top3": 0.8},
                "filename": "checkpoint_best_epoch_003.pth", "sha256": "d" * 64},
        "final_artifact_hashes": (
            {name: "e" * 64 for name in run_manifest_schema.FINAL_ARTIFACT_NAMES}
            if completed else None
        ),
    }
    return rm


def _eval_matching_provenance() -> dict:
    return {"manifest_sha256": "a" * 64, "taxonomy_sha256": "b" * 64, "val_split_sha256": "c" * 64}


class TestLoadAndValidateRunManifest(unittest.TestCase):
    def test_missing_file_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(SystemExit) as cm:
                load_and_validate_run_manifest(Path(td) / "run_manifest.json")
            self.assertIn("does not exist", str(cm.exception))

    def test_schema_invalid_content_fails_closed_with_specific_message(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "run_manifest.json"
            path.write_text(json.dumps({"status": "running"}))  # missing everything else
            with self.assertRaises(SystemExit) as cm:
                load_and_validate_run_manifest(path)
            self.assertIn("failed schema validation", str(cm.exception))

    def test_null_manifest_source_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "run_manifest.json"
            rm = _eval_valid_run_manifest()
            rm["manifest"] = None
            path.write_text(json.dumps(rm))
            with self.assertRaises(SystemExit) as cm:
                load_and_validate_run_manifest(path)
            self.assertIn("no recorded manifest/taxonomy_source", str(cm.exception))

    def test_valid_manifest_loads(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "run_manifest.json"
            rm = _eval_valid_run_manifest()
            path.write_text(json.dumps(rm))
            loaded = load_and_validate_run_manifest(path)
            self.assertEqual(loaded["status"], "running")


class TestVerifyRunManifestMatchesCheckpointProvenance(unittest.TestCase):
    def test_matching_hashes_pass(self):
        verify_run_manifest_matches_checkpoint_provenance(
            _eval_valid_run_manifest(), _eval_matching_provenance())  # must not raise

    def test_manifest_hash_disagreement_fails_closed(self):
        provenance = _eval_matching_provenance()
        provenance["manifest_sha256"] = "0" * 64
        with self.assertRaises(SystemExit) as cm:
            verify_run_manifest_matches_checkpoint_provenance(_eval_valid_run_manifest(), provenance)
        self.assertIn("manifest.sha256", str(cm.exception))

    def test_taxonomy_hash_disagreement_fails_closed(self):
        provenance = _eval_matching_provenance()
        provenance["taxonomy_sha256"] = "0" * 64
        with self.assertRaises(SystemExit) as cm:
            verify_run_manifest_matches_checkpoint_provenance(_eval_valid_run_manifest(), provenance)
        self.assertIn("taxonomy_source.sha256", str(cm.exception))

    def test_val_split_hash_disagreement_fails_closed(self):
        provenance = _eval_matching_provenance()
        provenance["val_split_sha256"] = "0" * 64
        with self.assertRaises(SystemExit) as cm:
            verify_run_manifest_matches_checkpoint_provenance(_eval_valid_run_manifest(), provenance)
        self.assertIn("val_split.sha256", str(cm.exception))


class TestVerifyFinalArtifactsAgainstRunManifest(unittest.TestCase):
    def _write_artifacts(self, art: Path, final_hashes: dict) -> Path:
        art.mkdir(parents=True, exist_ok=True)
        ckpt_path = art / "model.pth"
        contents = {
            "model.pth": b"model-bytes", "prototypes.npy": b"proto-bytes",
            "taxonomy.json": b"tax-bytes", "val_split.json": b"split-bytes",
        }
        for name, data in contents.items():
            (art / name).write_bytes(data)
            final_hashes[name] = hashlib.sha256(data).hexdigest()
        return ckpt_path

    def test_noop_when_not_completed(self):
        rm = _eval_valid_run_manifest(completed=False)
        with tempfile.TemporaryDirectory() as td:
            art = Path(td)
            # No artifact files exist at all -- must not even be read.
            verify_final_artifacts_against_run_manifest(rm, art, art / "model.pth")

    def test_matching_hashes_pass_when_completed(self):
        rm = _eval_valid_run_manifest(completed=True)
        with tempfile.TemporaryDirectory() as td:
            art = Path(td)
            ckpt_path = self._write_artifacts(art, rm["final_artifact_hashes"])
            verify_final_artifacts_against_run_manifest(rm, art, ckpt_path)  # must not raise

    def test_tampered_model_pth_fails_closed(self):
        rm = _eval_valid_run_manifest(completed=True)
        with tempfile.TemporaryDirectory() as td:
            art = Path(td)
            ckpt_path = self._write_artifacts(art, rm["final_artifact_hashes"])
            ckpt_path.write_bytes(b"tampered-model-bytes")
            with self.assertRaises(SystemExit) as cm:
                verify_final_artifacts_against_run_manifest(rm, art, ckpt_path)
            self.assertIn("model.pth", str(cm.exception))

    def test_tampered_taxonomy_json_fails_closed(self):
        rm = _eval_valid_run_manifest(completed=True)
        with tempfile.TemporaryDirectory() as td:
            art = Path(td)
            ckpt_path = self._write_artifacts(art, rm["final_artifact_hashes"])
            (art / "taxonomy.json").write_bytes(b"tampered-taxonomy-bytes")
            with self.assertRaises(SystemExit) as cm:
                verify_final_artifacts_against_run_manifest(rm, art, ckpt_path)
            self.assertIn("taxonomy.json", str(cm.exception))


# ------------------------------------------------------------ preflight (subprocess)
def _run_preflight(args: list[str], env_overrides: dict | None = None,
                   database_url: str | None = "__unset__") -> subprocess.CompletedProcess:
    env = dict(os.environ)
    if database_url == "__unset__":
        env.pop("DATABASE_URL", None)
    elif database_url is not None:
        env["DATABASE_URL"] = database_url
    if env_overrides:
        env.update(env_overrides)
    return subprocess.run(
        [sys.executable, str(HERE / "train.py"), "--preflight-only", *args],
        cwd=HERE, capture_output=True, text=True, env=env, timeout=180,
    )


class TestPreflightOnly(unittest.TestCase):
    def test_preflight_reports_dataset_shape_mismatch_for_a_non_northeast_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            fx = TinyFixture(Path(td))
            art = Path(td) / "art"
            result = _run_preflight([
                "--manifest-csv", str(fx.manifest_path),
                "--local-data-dir", str(fx.local_dir),
                "--taxonomy-json", str(fx.taxonomy_path),
                "--expected-manifest-sha256", fx.manifest_sha256,
                "--expected-taxonomy-sha256", fx.taxonomy_sha256,
                "--artifacts-dir", str(art),
            ])
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("PASS: explicit manifest source", result.stdout)
            self.assertIn("PASS: image bytes verified (6 files", result.stdout)
            self.assertIn("FAIL: dataset shape", result.stdout)
            self.assertFalse(art.exists())  # preflight writes nothing

    def test_preflight_reports_manifest_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as td:
            fx = TinyFixture(Path(td))
            result = _run_preflight([
                "--manifest-csv", str(fx.manifest_path),
                "--local-data-dir", str(fx.local_dir),
                "--taxonomy-json", str(fx.taxonomy_path),
                "--expected-manifest-sha256", "0" * 64,
                "--expected-taxonomy-sha256", fx.taxonomy_sha256,
                "--artifacts-dir", str(Path(td) / "art"),
            ])
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("FAIL: data source", result.stdout)

    def test_preflight_reports_database_url_hard_failure(self):
        with tempfile.TemporaryDirectory() as td:
            fx = TinyFixture(Path(td))
            result = _run_preflight([
                "--manifest-csv", str(fx.manifest_path),
                "--local-data-dir", str(fx.local_dir),
                "--taxonomy-json", str(fx.taxonomy_path),
                "--expected-manifest-sha256", fx.manifest_sha256,
                "--expected-taxonomy-sha256", fx.taxonomy_sha256,
                "--artifacts-dir", str(Path(td) / "art"),
            ], database_url="postgres://irrelevant/db")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("DATABASE_URL", result.stdout)

    def test_preflight_reports_image_byte_mismatch(self):
        with tempfile.TemporaryDirectory() as td:
            fx = TinyFixture(Path(td))
            (fx.local_dir / "bus-novus" / "N0.jpg").write_bytes(b"tampered")
            result = _run_preflight([
                "--manifest-csv", str(fx.manifest_path),
                "--local-data-dir", str(fx.local_dir),
                "--taxonomy-json", str(fx.taxonomy_path),
                "--expected-manifest-sha256", fx.manifest_sha256,
                "--expected-taxonomy-sha256", fx.taxonomy_sha256,
                "--artifacts-dir", str(Path(td) / "art"),
            ])
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("FAIL: image bytes", result.stdout)

    def test_preflight_reports_nonempty_output_dir_refusal(self):
        with tempfile.TemporaryDirectory() as td:
            art = Path(td) / "art"
            art.mkdir()
            (art / "stray.txt").write_text("x")
            result = _run_preflight(["--artifacts-dir", str(art)])
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("FAIL: output-directory safety", result.stdout)

    def test_preflight_reports_numerical_policy_and_estimated_sizes(self):
        with tempfile.TemporaryDirectory() as td:
            result = _run_preflight(["--artifacts-dir", str(Path(td) / "art")])
            self.assertIn("numerical policy applied", result.stdout)
            self.assertIn("cudnn_allow_tf32': False", result.stdout)
            self.assertIn("estimated artifact/checkpoint sizes", result.stdout)
            self.assertIn("NOT measured", result.stdout)

    def test_preflight_reports_git_state(self):
        with tempfile.TemporaryDirectory() as td:
            result = _run_preflight(["--artifacts-dir", str(Path(td) / "art")])
            self.assertIn("[preflight] git: commit=", result.stdout)

    def test_pause_after_epoch_requires_limit_batches(self):
        with tempfile.TemporaryDirectory() as td:
            result = _run_preflight([
                "--artifacts-dir", str(Path(td) / "art"), "--pause-after-epoch", "1",
            ])
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("--pause-after-epoch requires --limit-batches", result.stderr)


@unittest.skipUnless(
    REAL_MANIFEST.exists() and REAL_TAXONOMY.exists() and REAL_LOCAL_DATA_DIR.exists(),
    "real Northeast manifest/taxonomy/clean-data-dir not present in this checkout",
)
class TestRealPreflight(unittest.TestCase):
    """--preflight-only against the ACTUAL frozen 65-species catalog. Still
    initializes no model and writes nothing. Git-state gating means this only
    achieves a full PASS on a clean working tree -- while the harness change
    itself is uncommitted, the git-dirty check is EXPECTED to be the sole
    reported problem; every other check must still independently pass."""

    def test_real_preflight(self):
        with tempfile.TemporaryDirectory() as td:
            art = Path(td) / "northeast_v1_b4_dev"
            result = _run_preflight([
                "--manifest-csv", str(REAL_MANIFEST),
                "--local-data-dir", str(REAL_LOCAL_DATA_DIR),
                "--taxonomy-json", str(REAL_TAXONOMY),
                "--expected-manifest-sha256", REAL_MANIFEST_SHA256,
                "--expected-taxonomy-sha256", REAL_TAXONOMY_SHA256,
                "--artifacts-dir", str(art),
                "--num-workers", "0",
            ], database_url="__unset__")
            output = result.stdout + result.stderr
            self.assertIn("PASS: explicit manifest source", output, output)
            self.assertIn("PASS: dataset shape (13581 samples, 65 classes, 10985 train, "
                         "2596 val)", output)
            self.assertIn("PASS: image bytes verified (13581 files", output)
            self.assertIn("PASS: per-class counts", output)
            self.assertFalse(art.exists())  # preflight writes nothing regardless

            git_commit, git_dirty = train.get_git_state()
            if git_commit is not None and git_dirty is False:
                self.assertEqual(result.returncode, 0, output)
                self.assertIn("[preflight] PASS: all checks passed", output)
            else:
                # Expected during active development: git-state is the ONLY
                # failure, every data/hash/shape/image-byte check still passes.
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("FAIL: git state", output)


if __name__ == "__main__":
    unittest.main()
