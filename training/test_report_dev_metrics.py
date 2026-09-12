#!/usr/bin/env python3
"""test_report_dev_metrics.py — unit + integration tests for
report_dev_metrics.py.

Never touches the real northeast_v1_b4_dev_v2 run, never downloads a real
timm backbone: the integration fixture uses a tiny, parameter-free
"embed = channel mean" model over 4x4 solid-color PNGs, so the whole pipeline
(manifest/image hash verification, val-split resolution, provenance
cross-checks, raw-cosine inference, genus metrics, deterministic
serialization, byte-identical verification) is exercised deterministically
and fast, without ever constructing a real EfficientNet backbone.

Run directly: python test_report_dev_metrics.py
"""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import report_dev_metrics as rdm  # noqa: E402
from data import Sample  # noqa: E402

DEVICE = torch.device("cpu")


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _sha256_file(p: Path) -> str:
    return _sha256_bytes(p.read_bytes())


class TinyReportModel(nn.Module):
    """No parameters at all: embed() is the per-channel spatial mean. Fully
    deterministic given a fixed input -- no seeding needed."""

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=(2, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - unused
        return self.embed(x)


# --------------------------------------------------------------- fixtures
def _write_solid_png(path: Path, rgb: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), rgb).save(path, format="PNG")


def _write_manifest_csv(path: Path, rows: list[dict]) -> None:
    import csv
    fields = ["slug", "photo_id", "species", "common_name", "taxon_id", "lat", "lon",
             "split", "sha256"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({f: r.get(f, "") for f in fields})


def build_full_fixture(tmp_path: Path):
    """A complete, self-consistent, 2-species/6-image COMPLETED run, built so
    that raw-cosine inference over the val split is perfectly separable
    (alpha=red, beta=green) -- deterministic without any seeding. Returns
    (run_dir, local_data_dir, manifest_csv, taxonomy_json)."""
    local_data_dir = tmp_path / "clean"
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)

    colors = {"alpha-ant": (255, 0, 0), "beta-ant": (0, 255, 0)}
    photo_ids = {"alpha-ant": ["train1", "train2", "val1"],
                "beta-ant": ["train1", "train2", "val1"]}
    splits = {"train1": "train", "train2": "train", "val1": "val"}

    rows = []
    for slug, color in colors.items():
        for pid in photo_ids[slug]:
            img_path = local_data_dir / slug / f"{pid}.png"
            _write_solid_png(img_path, color)
            rows.append({
                "slug": slug, "photo_id": pid,
                "species": "Alpha ant" if slug == "alpha-ant" else "Beta ant",
                "common_name": "", "taxon_id": "", "lat": "", "lon": "",
                "split": splits[pid], "sha256": _sha256_file(img_path),
            })

    manifest_csv = tmp_path / "manifest.csv"
    _write_manifest_csv(manifest_csv, rows)
    manifest_sha256 = _sha256_file(manifest_csv)

    committed_taxonomy = {
        "0": {"species_name": "Alpha ant", "common_name": None, "taxon_id": None,
             "slug": "alpha-ant", "genus": "Alpha"},
        "1": {"species_name": "Beta ant", "common_name": None, "taxon_id": None,
             "slug": "beta-ant", "genus": "Beta"},
    }
    taxonomy_json = tmp_path / "taxonomy.json"
    taxonomy_json.write_text(json.dumps(committed_taxonomy))
    taxonomy_sha256 = _sha256_file(taxonomy_json)

    val_split_record = {
        "seed": 42, "val_fraction": 0.5, "n_total": 6, "n_train": 4, "n_val": 2,
        "train": ["alpha-ant/train1", "alpha-ant/train2",
                 "beta-ant/train1", "beta-ant/train2"],
        "val": ["alpha-ant/val1", "beta-ant/val1"],
    }
    val_split_bytes = (json.dumps(val_split_record, indent=2, sort_keys=True) + "\n").encode()
    (run_dir / "val_split.json").write_bytes(val_split_bytes)
    val_split_sha256 = _sha256_bytes(val_split_bytes)

    cfg = {
        "model": {"backbone": "tiny-test-backbone", "embedding_dim": 3, "pretrained": False},
        "dropout": 0.0, "batch_size": 2, "image_size": 8,
        "normalize": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
    }
    provenance = {
        "manifest_sha256": manifest_sha256, "taxonomy_sha256": taxonomy_sha256,
        "val_split_sha256": val_split_sha256, "resolved_config_sha256": "c" * 64,
        "backbone": "tiny-test-backbone", "num_classes": 2, "git_commit": "f" * 40,
        "numerical_policy": {}, "run_kind": "full", "limit_batches": None,
        "wandb_enabled": False, "validation_cadence": 1,
    }
    torch.save({"model": {}, "config": cfg, "provenance": provenance, "best_epoch": 0},
              run_dir / "model.pth")

    protos = np.array([[1.0, -1.0, -1.0], [-1.0, 1.0, -1.0]], dtype=np.float32)
    protos = protos / np.linalg.norm(protos, axis=1, keepdims=True)
    np.save(run_dir / "prototypes.npy", protos)

    (run_dir / "taxonomy.json").write_text(json.dumps(committed_taxonomy))
    (run_dir / "geo_index.json").write_text(json.dumps({"cell_size_deg": 1.0, "cells": {}}))
    eval_payload = {"overall": {"n": 2, "top1": 1.0, "top3": 1.0}, "per_species": {}}
    (run_dir / "eval.json").write_text(json.dumps(eval_payload))
    (run_dir / "backbone.onnx").write_bytes(b"not a real onnx file, hash-only fixture")

    history_row = {"epoch": 0, "train_loss": 0.0, "val_top1": 1.0, "val_top3": 1.0,
                  "duration_seconds": {"train": 0.0, "prototypes": 0.0, "validation": 0.0},
                  "validation_ran": True, "is_best": True, "best_epoch_so_far": 0,
                  "timestamp_utc": "2026-01-01T00:00:00Z"}
    (run_dir / "history.jsonl").write_text(json.dumps(history_row) + "\n")

    final_artifact_hashes = {
        name: _sha256_file(run_dir / name) for name in rdm.FINAL_ARTIFACT_NAMES
    }

    run_manifest = {
        "run_manifest_schema_version": 2, "status": "completed", "run_kind": "full",
        "validation_cadence": 1, "git_head": "f" * 40, "git_dirty": False,
        "invocations": [{"timestamp_utc": "2026-01-01T00:00:00Z", "argv": [],
                         "resume": False, "pause_after_epoch": None}],
        "started_at_utc": "2026-01-01T00:00:00Z", "updated_at_utc": "2026-01-01T00:00:00Z",
        "finished_at_utc": "2026-01-01T00:00:00Z",
        "manifest": {"path": str(manifest_csv), "sha256": manifest_sha256, "rows": 6},
        "taxonomy_source": {"path": str(taxonomy_json), "sha256": taxonomy_sha256,
                            "num_classes": 2},
        "val_split": {"path": str(run_dir / "val_split.json"), "sha256": val_split_sha256,
                     "n_train": 4, "n_val": 2, "n_total": 6},
        "last_completed_epoch": 0,
        "best": {"epoch": 0, "metrics": {"top1": 1.0, "top3": 1.0},
                 "filename": "checkpoint_best_epoch_000.pth", "sha256": "a" * 64},
        "final_artifact_hashes": final_artifact_hashes,
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2, sort_keys=True))

    return run_dir, local_data_dir, manifest_csv, taxonomy_json


def _generate(run_dir, local_data_dir, manifest_csv, taxonomy_json):
    # require_clean_git/assert_frozen_catalog_shape=False: this fixture is a
    # tiny 2-species synthetic catalog, not the real 65-species one, and
    # tests must not depend on this repo's own live git working-tree state.
    with mock.patch.object(rdm, "build_model", lambda cfg, num_classes: TinyReportModel()):
        return rdm.generate_report(
            run_dir=run_dir, local_data_dir=local_data_dir, manifest_csv=manifest_csv,
            taxonomy_json=taxonomy_json, database_url=None, device=DEVICE,
            require_clean_git=False, assert_frozen_catalog_shape=False,
        )


# ----------------------------------------------------------------- unit tests
class TestVerifyFinalArtifactHashes(unittest.TestCase):
    def _run_dir(self, tmp):
        run_dir = Path(tmp) / "run"
        run_dir.mkdir()
        for name in rdm.FINAL_ARTIFACT_NAMES:
            (run_dir / name).write_bytes(name.encode())
        return run_dir

    def test_matches(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = self._run_dir(tmp)
            expected = {n: _sha256_file(run_dir / n) for n in rdm.FINAL_ARTIFACT_NAMES}
            rm = {"final_artifact_hashes": expected}
            result = rdm.verify_final_artifact_hashes(run_dir, rm)
            self.assertEqual(result, expected)

    def test_mismatch_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = self._run_dir(tmp)
            expected = {n: _sha256_file(run_dir / n) for n in rdm.FINAL_ARTIFACT_NAMES}
            expected["model.pth"] = "0" * 64  # corrupt one recorded hash
            rm = {"final_artifact_hashes": expected}
            with self.assertRaisesRegex(rdm.DevReportError, "artifact hash mismatch"):
                rdm.verify_final_artifact_hashes(run_dir, rm)

    def test_missing_file_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = self._run_dir(tmp)
            expected = {n: _sha256_file(run_dir / n) for n in rdm.FINAL_ARTIFACT_NAMES}
            (run_dir / "eval.json").unlink()
            rm = {"final_artifact_hashes": expected}
            with self.assertRaisesRegex(rdm.DevReportError, "artifact hash mismatch"):
                rdm.verify_final_artifact_hashes(run_dir, rm)


class TestLoadCompletedRunManifest(unittest.TestCase):
    def _base_manifest(self) -> dict:
        # Shaped to satisfy stage="completed" schema validation (which checks
        # structure regardless of the status VALUE) while status itself is
        # not "completed" -- exactly the case load_completed_run_manifest
        # must reject on its own explicit status check.
        return {
            "run_manifest_schema_version": 2, "status": "failed", "run_kind": "full",
            "validation_cadence": 1, "git_head": "f" * 40, "git_dirty": False,
            "invocations": [{"timestamp_utc": "2026-01-01T00:00:00Z", "argv": [],
                             "resume": False, "pause_after_epoch": None}],
            "started_at_utc": "2026-01-01T00:00:00Z", "updated_at_utc": "2026-01-01T00:00:00Z",
            "finished_at_utc": None,
            "manifest": {"path": "m.csv", "sha256": "a" * 64, "rows": 1},
            "taxonomy_source": {"path": "t.json", "sha256": "b" * 64, "num_classes": 1},
            "val_split": {"path": "v.json", "sha256": "c" * 64, "n_train": 1,
                         "n_val": 1, "n_total": 2},
            "last_completed_epoch": 0,
            "best": {"epoch": 0, "metrics": {"top1": 0.5, "top3": 0.5},
                     "filename": "checkpoint_best_epoch_000.pth", "sha256": "d" * 64},
            "final_artifact_hashes": {n: "e" * 64 for n in rdm.FINAL_ARTIFACT_NAMES},
        }

    def test_non_completed_status_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            rm = self._base_manifest()
            (run_dir / "run_manifest.json").write_text(json.dumps(rm))
            with self.assertRaisesRegex(rdm.DevReportError, "expected 'completed'"):
                rdm.load_completed_run_manifest(run_dir)

    def test_missing_file_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(rdm.DevReportError, "does not exist"):
                rdm.load_completed_run_manifest(Path(tmp))


class TestAssertTaxonomyHasGenus(unittest.TestCase):
    def test_rejects_blank_genus(self):
        taxonomy = {0: {"slug": "alpha-ant", "genus": "Alpha"},
                   1: {"slug": "beta-ant", "genus": ""}}
        with self.assertRaisesRegex(rdm.DevReportError, "missing/blank genus"):
            rdm.assert_taxonomy_has_genus(taxonomy)

    def test_accepts_nonblank(self):
        taxonomy = {0: {"slug": "alpha-ant", "genus": "Alpha"}}
        rdm.assert_taxonomy_has_genus(taxonomy)  # must not raise


class TestResolvePinnedValSplit(unittest.TestCase):
    def _samples(self):
        return [
            Sample("alpha-ant/val1.png", 0, "alpha-ant"),
            Sample("beta-ant/val1.png", 1, "beta-ant"),
            Sample("alpha-ant/train1.png", 0, "alpha-ant"),
        ]

    def test_hash_mismatch_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            record = {"n_val": 2, "val": ["alpha-ant/val1", "beta-ant/val1"]}
            (run_dir / "val_split.json").write_text(json.dumps(record))
            rm = {"val_split": {"sha256": "0" * 64}}  # wrong on purpose
            with self.assertRaisesRegex(rdm.DevReportError, "val-split mismatch"):
                rdm.resolve_pinned_val_split(run_dir, rm, self._samples())

    def test_resolves_expected_val_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            record = {"n_val": 2, "val": ["alpha-ant/val1", "beta-ant/val1"]}
            payload = json.dumps(record).encode()
            (run_dir / "val_split.json").write_bytes(payload)
            rm = {"val_split": {"sha256": _sha256_bytes(payload)}}
            val_samples, split_record = rdm.resolve_pinned_val_split(
                run_dir, rm, self._samples())
            self.assertEqual(len(val_samples), 2)
            self.assertEqual({s.slug for s in val_samples}, {"alpha-ant", "beta-ant"})

    def test_count_mismatch_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            # n_val claims 3 but only 2 samples will actually resolve.
            record = {"n_val": 3, "val": ["alpha-ant/val1", "beta-ant/val1"]}
            payload = json.dumps(record).encode()
            (run_dir / "val_split.json").write_bytes(payload)
            rm = {"val_split": {"sha256": _sha256_bytes(payload)}}
            with self.assertRaisesRegex(rdm.DevReportError, "val-split mismatch"):
                rdm.resolve_pinned_val_split(run_dir, rm, self._samples())


class TestComputeGenusMetrics(unittest.TestCase):
    def test_rates_and_counts(self):
        # 3 species: two share genus "Alpha" (idx 0, 1), one is "Beta" (idx 2).
        taxonomy = {
            0: {"slug": "alpha-one", "genus": "Alpha"},
            1: {"slug": "alpha-two", "genus": "Alpha"},
            2: {"slug": "beta-one", "genus": "Beta"},
        }
        predictions = [
            (0, [0, 1, 1]),   # top1 correct species -> genus correct, unanimous true genus
            (0, [1, 0, 2]),   # top1 wrong species (1) but same genus (Alpha) -> counts
            (2, [0, 1, 2]),   # top1 wrong species+genus; genus never in top3 either
        ]
        result = rdm.compute_genus_metrics(predictions, taxonomy, frozenset())
        overall = result["overall"]
        self.assertEqual(overall["n"], 3)
        # genus_top1: rows 0 and 1 have top1 genus == true genus (Alpha); row 2 does not.
        self.assertEqual(overall["genus_top1"], {"count": 2, "total": 3, "rate": 2 / 3})
        # wrong_species_but_correct_genus_top1: among the 2 species-wrong rows
        # (row 1 and row 2), only row 1 has correct genus.
        self.assertEqual(overall["wrong_species_but_correct_genus_top1"],
                         {"count": 1, "total": 2, "rate": 0.5})
        # top3_unanimous_true_genus: only row 0 has all three preds in genus Alpha
        # matching its true genus Alpha; row 1's preds are [1,0,2] -- idx2 is
        # genus Beta, not unanimous; row 2's true genus is Beta but its preds
        # are all Alpha/Alpha/Beta -- not unanimous either.
        self.assertEqual(overall["top3_unanimous_true_genus"]["count"], 1)
        per_genus = result["per_genus"]
        self.assertEqual(per_genus["Alpha"]["n"], 2)
        self.assertEqual(per_genus["Beta"]["n"], 1)

    def test_zero_denominator_rate_is_null(self):
        taxonomy = {0: {"slug": "alpha-one", "genus": "Alpha"}}
        predictions = [(0, [0])]  # top1 always correct -> species_wrong_top1 total is 0
        result = rdm.compute_genus_metrics(predictions, taxonomy, frozenset())
        rate = result["overall"]["wrong_species_but_correct_genus_top1"]
        self.assertEqual(rate, {"count": 0, "total": 0, "rate": None})


class TestComputeSelectionNuance(unittest.TestCase):
    def _write_history(self, run_dir: Path, rows: list[dict]) -> None:
        with (run_dir / "history.jsonl").open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")

    def test_runner_up_and_margin(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            self._write_history(run_dir, [
                {"epoch": 5, "val_top1": 0.6798921417565486, "val_top3": 0.84,
                 "validation_ran": True},
                {"epoch": 26, "val_top1": 0.6802773497688752, "val_top3": 0.83,
                 "validation_ran": True},
                {"epoch": 10, "val_top1": 0.5, "val_top3": 0.7, "validation_ran": False},
            ])
            run_manifest = {"best": {"epoch": 26}}
            result = rdm.compute_selection_nuance(run_dir, run_manifest,
                                                   top1_correct=1766, top1_n=2596)
            self.assertEqual(result["selected"]["human_epoch"], 27)
            self.assertEqual(result["runner_up"]["human_epoch"], 6)
            self.assertEqual(result["runner_up"]["top1_correct_derived"], 1765)
            self.assertEqual(result["margin_top1_predictions"], 1)

    def test_disagreement_with_run_manifest_best_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            self._write_history(run_dir, [
                {"epoch": 5, "val_top1": 0.9, "val_top3": 0.95, "validation_ran": True},
            ])
            run_manifest = {"best": {"epoch": 999}}  # disagrees with history's top rank
            with self.assertRaises(rdm.DevReportError):
                rdm.compute_selection_nuance(run_dir, run_manifest,
                                             top1_correct=1, top1_n=1)

    def test_missing_history_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = rdm.compute_selection_nuance(Path(tmp), {"best": {"epoch": 0}}, 1, 1)
            self.assertIsNone(result)


class TestNewSpeciesMembership(unittest.TestCase):
    def test_constant_has_exactly_fifteen_slugs(self):
        from data_provenance import NORTHEAST_NEW_SPECIES_SLUGS
        rdm.assert_new_species_constant_shape(NORTHEAST_NEW_SPECIES_SLUGS)  # must not raise
        self.assertEqual(len(NORTHEAST_NEW_SPECIES_SLUGS), 15)

    def test_constant_wrong_count_raises(self):
        with self.assertRaisesRegex(rdm.DevReportError, "exactly 15"):
            rdm.assert_new_species_constant_shape(frozenset({"a", "b"}))

    def test_present_in_taxonomy_passes(self):
        taxonomy = {0: {"slug": "lasius-neoniger"}, 1: {"slug": "camponotus-americanus"}}
        rdm.assert_new_species_present_in_taxonomy(
            taxonomy, frozenset({"lasius-neoniger", "camponotus-americanus"}))  # no raise

    def test_missing_slug_raises(self):
        taxonomy = {0: {"slug": "alpha-ant"}, 1: {"slug": "beta-ant"}}
        with self.assertRaisesRegex(rdm.DevReportError, "not present in this run's taxonomy"):
            rdm.assert_new_species_present_in_taxonomy(
                taxonomy, frozenset({"alpha-ant", "lasius-neoniger"}))


class TestGroupDenominatorsAndKnownB4Counts(unittest.TestCase):
    def test_group_denominators_pass_on_exact_match(self):
        aggregates = {"new_15": {"n": 600}, "legacy_50": {"n": 1996}, "all_65": {"n": 2596}}
        rdm.assert_group_denominators(aggregates)  # must not raise

    def test_group_denominators_reject_off_by_one(self):
        aggregates = {"new_15": {"n": 599}, "legacy_50": {"n": 1996}, "all_65": {"n": 2596}}
        with self.assertRaisesRegex(rdm.DevReportError, "group denominator mismatch"):
            rdm.assert_group_denominators(aggregates)

    def test_known_b4_counts_pass_on_exact_match(self):
        report = {"aggregates": rdm.KNOWN_B4_AGGREGATE_COUNTS}
        rdm.assert_known_b4_aggregate_counts(report)  # must not raise

    def test_known_b4_counts_reject_mismatch(self):
        aggregates = json.loads(json.dumps(rdm.KNOWN_B4_AGGREGATE_COUNTS))
        aggregates["all_65"]["top1_correct"] = 1765  # one off from the known 1766
        with self.assertRaisesRegex(rdm.DevReportError, "known-B4-result mismatch"):
            rdm.assert_known_b4_aggregate_counts({"aggregates": aggregates})


class TestGitDirtyExcluding(unittest.TestCase):
    def test_clean_status_is_not_dirty(self):
        with mock.patch("subprocess.check_output", return_value=b""):
            self.assertFalse(rdm._git_dirty_excluding(frozenset()))

    def test_dirt_outside_ignore_set_is_dirty(self):
        status = b" M training/data.py\n"
        with mock.patch("subprocess.check_output", return_value=status):
            self.assertTrue(rdm._git_dirty_excluding(frozenset()))

    def test_dirt_confined_to_ignored_path_is_not_dirty(self):
        status = b"?? training/reports/northeast_v1_b4_dev_report.json\n"
        ignore = frozenset({"training/reports/northeast_v1_b4_dev_report.json"})
        with mock.patch("subprocess.check_output", return_value=status):
            self.assertFalse(rdm._git_dirty_excluding(ignore))

    def test_mixed_dirt_one_outside_ignore_set_is_dirty(self):
        status = (b"?? training/reports/northeast_v1_b4_dev_report.json\n"
                 b" M training/data.py\n")
        ignore = frozenset({"training/reports/northeast_v1_b4_dev_report.json"})
        with mock.patch("subprocess.check_output", return_value=status):
            self.assertTrue(rdm._git_dirty_excluding(ignore))

    def test_git_unavailable_returns_none(self):
        with mock.patch("subprocess.check_output", side_effect=FileNotFoundError):
            self.assertIsNone(rdm._git_dirty_excluding(frozenset()))


class TestSerializeReport(unittest.TestCase):
    def test_deterministic_sorted_lf(self):
        report = {"b": 1, "a": [3, 2, 1]}
        payload = rdm.serialize_report(report)
        self.assertNotIn(b"\r\n", payload)
        self.assertTrue(payload.endswith(b"\n"))
        self.assertEqual(payload, rdm.serialize_report(report))
        text = payload.decode()
        self.assertLess(text.index('"a"'), text.index('"b"'))


# ----------------------------------------------------------- integration tests
class TestGenerateReportIntegration(unittest.TestCase):
    def test_success_and_byte_identical_verification(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, local_data_dir, manifest_csv, taxonomy_json = build_full_fixture(Path(tmp))
            report = _generate(run_dir, local_data_dir, manifest_csv, taxonomy_json)

            self.assertEqual(report["label"], rdm.LABEL)
            self.assertEqual(report["report_schema_version"], rdm.REPORT_SCHEMA_VERSION)
            self.assertEqual(report["generator"]["name"], rdm.GENERATOR_NAME)
            self.assertEqual(report["generator"]["version"], rdm.GENERATOR_VERSION)
            self.assertEqual(len(report["generator"]["sha256"]), 64)
            self.assertEqual(report["generator"]["sha256"],
                             _sha256_file(Path(rdm.__file__).resolve()))
            checkpoint = report["source"]["checkpoint"]
            self.assertEqual(checkpoint["backbone"], "tiny-test-backbone")
            self.assertEqual(checkpoint["embedding_dim"], 3)
            self.assertEqual(checkpoint["image_size"], 8)
            self.assertEqual(checkpoint["normalize"], {"mean": [0.5, 0.5, 0.5],
                                                        "std": [0.5, 0.5, 0.5]})
            self.assertEqual(checkpoint["interpolation_effective"], "bilinear")
            self.assertEqual(report["dataset"]["species_count"], 2)
            self.assertEqual(report["dataset"]["val_n"], 2)
            self.assertEqual(report["reproduction"]["top1"], 1.0)
            self.assertEqual(report["reproduction"]["top1_correct"], 2)
            self.assertEqual(report["reproduction"]["top3"], 1.0)
            self.assertTrue(report["reproduction"]["matches_shipped_eval_json"])
            self.assertNotIn(str(run_dir), json.dumps(report))  # no absolute machine path

            payload_1 = rdm.serialize_report(report)
            report_2 = _generate(run_dir, local_data_dir, manifest_csv, taxonomy_json)
            payload_2 = rdm.serialize_report(report_2)
            self.assertEqual(payload_1, payload_2)  # deterministic regeneration

            existing = Path(tmp) / "committed_report.json"
            existing.write_bytes(payload_1)
            rdm.verify_byte_identical(payload_2, existing)  # must not raise

            existing.write_bytes(payload_1 + b" ")  # simulate drift
            with self.assertRaises(rdm.DevReportError):
                rdm.verify_byte_identical(payload_2, existing)

    def test_reproduction_mismatch_aborts_before_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, local_data_dir, manifest_csv, taxonomy_json = build_full_fixture(Path(tmp))
            rm_path = run_dir / "run_manifest.json"
            run_manifest = json.loads(rm_path.read_text())
            run_manifest["best"]["metrics"]["top1"] = 0.1234  # disagrees with live inference
            rm_path.write_text(json.dumps(run_manifest, indent=2, sort_keys=True))

            with self.assertRaisesRegex(rdm.DevReportError, "Reproduction mismatch"):
                _generate(run_dir, local_data_dir, manifest_csv, taxonomy_json)

    def test_frozen_catalog_shape_rejects_tiny_fixture(self):
        # This fixture is a 2-species/2-val-image synthetic catalog -- with
        # assert_frozen_catalog_shape left at its default (True), it must be
        # rejected as not the real 65-species Northeast catalog.
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, local_data_dir, manifest_csv, taxonomy_json = build_full_fixture(Path(tmp))
            with mock.patch.object(rdm, "build_model",
                                   lambda cfg, num_classes: TinyReportModel()):
                with self.assertRaisesRegex(rdm.DevReportError,
                                            "not present in this run's taxonomy"):
                    rdm.generate_report(
                        run_dir=run_dir, local_data_dir=local_data_dir,
                        manifest_csv=manifest_csv, taxonomy_json=taxonomy_json,
                        database_url=None, device=DEVICE, require_clean_git=False,
                    )

    def test_artifact_hash_mismatch_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, local_data_dir, manifest_csv, taxonomy_json = build_full_fixture(Path(tmp))
            with (run_dir / "eval.json").open("ab") as fh:
                fh.write(b" ")  # corrupt after hashes were recorded

            with self.assertRaisesRegex(rdm.DevReportError, "artifact hash mismatch"):
                _generate(run_dir, local_data_dir, manifest_csv, taxonomy_json)


if __name__ == "__main__":
    unittest.main()
