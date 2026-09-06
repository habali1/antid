#!/usr/bin/env python3
"""Focused synthetic tests for the Northeast-expansion Milestone 1 geo/split
fixes: train.py building geo_index.json from the train split only, replacing
(never leaving stale) an empty sidecar, val_split.json pinning both train and
val membership, and evaluate.py's --geo-source train requiring a verified
pinned split (fail closed otherwise). No training, model evaluation, or
network access -- everything here is synthetic Sample fixtures and small
JSON files under a temp directory. Run directly: python test_geo_split.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import train  # noqa: E402
import evaluate  # noqa: E402
import data  # noqa: E402
from data import Sample  # noqa: E402


def make_sample(slug: str, stem: str, label: int, *,
                lat: float | None = None, lon: float | None = None,
                split: str | None = None) -> Sample:
    s = Sample(storage_path=f"/data/{slug}/{stem}.jpg", label=label, slug=slug)
    if lat is not None:
        s.lat = lat
    if lon is not None:
        s.lon = lon
    if split is not None:
        s.split = split
    return s


class TestBuildGeoIndexTrainOnly(unittest.TestCase):
    def test_uses_train_split_only_excludes_val_coordinates(self):
        # Same species, two disjoint coordinate clusters: one seen only by
        # train-labeled samples, one seen only by val-labeled samples.
        train_pair = [
            make_sample("aus", f"train{i}", 0, lat=10.0, lon=-75.0, split="train")
            for i in range(2)
        ]
        val_pair = [
            make_sample("aus", f"val{i}", 0, lat=50.0, lon=-50.0, split="val")
            for i in range(2)
        ]
        samples = train_pair + val_pair
        taxonomy = {0: {"slug": "aus"}}

        train_s, val_s = train.split_samples(samples, val_fraction=0.5, seed=0)
        self.assertEqual(len(train_s), 2)
        self.assertEqual(len(val_s), 2)

        cells_train_only = train.build_geo_index(train_s, taxonomy,
                                                  cell_size_deg=1.0, min_obs_per_cell=2)
        self.assertEqual(cells_train_only, {"aus": [[10, -75]]})

        # Documents the bug this fixes: passing the full (train+val) sample
        # list -- what train.py used to do -- lets the val-only coordinate
        # leak into the shipped index.
        cells_everything = train.build_geo_index(samples, taxonomy,
                                                  cell_size_deg=1.0, min_obs_per_cell=2)
        self.assertIn([50, -50], cells_everything["aus"])
        self.assertNotIn([50, -50], cells_train_only["aus"])

    def test_negative_coordinate_flooring(self):
        # US longitudes are negative. Cell math must floor toward -inf, not
        # truncate toward zero (int(-75.1) == -75, which is the wrong cell).
        samples = [
            make_sample("aus", "p1", 0, lat=42.5, lon=-75.5),
            make_sample("aus", "p2", 0, lat=42.9, lon=-75.1),
        ]
        taxonomy = {0: {"slug": "aus"}}
        cells = train.build_geo_index(samples, taxonomy,
                                      cell_size_deg=1.0, min_obs_per_cell=2)
        self.assertEqual(cells, {"aus": [[42, -76]]})


class TestSplitSamplesIntegrity(unittest.TestCase):
    def test_rejects_partial_explicit_split_metadata(self):
        samples = [
            make_sample("aus", "a", 0, split="train"),
            make_sample("aus", "b", 0),
        ]
        with self.assertRaises(ValueError):
            train.split_samples(samples, val_fraction=0.5, seed=0)

    def test_rejects_unknown_explicit_split_value(self):
        samples = [
            make_sample("aus", "a", 0, split="train"),
            make_sample("aus", "b", 0, split="test"),
        ]
        with self.assertRaises(ValueError):
            train.split_samples(samples, val_fraction=0.5, seed=0)


class TestWriteGeoIndexSidecar(unittest.TestCase):
    def test_empty_replaces_stale_content(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "geo_index.json"
            path.write_text(json.dumps({
                "cell_size_deg": 2.0,
                "cells": {"stale-species": [[5, 5]]},
            }))
            n = train.write_geo_index_sidecar(path, {}, 1.0)
            self.assertEqual(n, 0)
            written = json.loads(path.read_text())
            self.assertEqual(written, {
                "cell_size_deg": 1.0, "cells": {}, "source_split": "train",
            })

    def test_populated_includes_source_split(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "geo_index.json"
            cells = {"aus": [[10, -75], [11, -75]]}
            n = train.write_geo_index_sidecar(path, cells, 1.0)
            self.assertEqual(n, 2)
            written = json.loads(path.read_text())
            self.assertEqual(written, {
                "cell_size_deg": 1.0, "cells": cells, "source_split": "train",
            })


class TestBuildValSplitRecord(unittest.TestCase):
    def test_pins_sorted_disjoint_train_and_val(self):
        train_s = [make_sample("aus", "b", 0), make_sample("aus", "a", 0)]
        val_s = [make_sample("aus", "z", 0)]
        cfg = {"seed": 7, "val_fraction": 0.2}
        record = train.build_val_split_record(cfg, train_s, val_s)
        self.assertEqual(record["n_train"], 2)
        self.assertEqual(record["n_val"], 1)
        self.assertEqual(record["n_total"], 3)
        self.assertEqual(record["train"], ["aus/a", "aus/b"])  # sorted
        self.assertEqual(record["val"], ["aus/z"])
        self.assertFalse(set(record["train"]) & set(record["val"]))

    def test_rejects_duplicate_logical_key_within_train(self):
        train_s = [make_sample("aus", "a", 0), make_sample("aus", "a", 0)]
        with self.assertRaises(ValueError):
            train.build_val_split_record(
                {"seed": 7, "val_fraction": 0.2}, train_s,
                [make_sample("aus", "v", 0)])

    def test_rejects_logical_key_shared_across_splits(self):
        with self.assertRaises(ValueError):
            train.build_val_split_record(
                {"seed": 7, "val_fraction": 0.2},
                [make_sample("aus", "same", 0)],
                [make_sample("aus", "same", 0)])


class TestResolvePinnedTrainSplit(unittest.TestCase):
    def _samples(self):
        return [
            make_sample("aus", "t1", 0), make_sample("aus", "t2", 0),
            make_sample("bee", "t1", 1), make_sample("bee", "v1", 1),
        ]

    def _write_split(self, td, data):
        path = Path(td) / "val_split.json"
        path.write_text(json.dumps(data))
        return path

    @staticmethod
    def _complete_record(train_keys, val_keys):
        return {
            "n_total": len(train_keys) + len(val_keys),
            "n_train": len(train_keys),
            "n_val": len(val_keys),
            "train": train_keys,
            "val": val_keys,
        }

    def test_success_resolves_verified_train_only(self):
        samples = self._samples()
        with tempfile.TemporaryDirectory() as td:
            split_file = self._write_split(td, self._complete_record(
                ["aus/t1", "aus/t2", "bee/t1"], ["bee/v1"]))
            resolved = evaluate.resolve_pinned_train_split(samples, split_file)
        self.assertEqual(len(resolved), 3)
        self.assertEqual({s.storage_path for s in resolved},
                         {samples[0].storage_path, samples[1].storage_path,
                          samples[2].storage_path})

    def test_fails_closed_when_split_file_missing(self):
        with tempfile.TemporaryDirectory() as td:
            missing = Path(td) / "does_not_exist.json"
            with self.assertRaises(SystemExit):
                evaluate.resolve_pinned_train_split(self._samples(), missing)

    def test_fails_closed_on_legacy_val_only_split(self):
        with tempfile.TemporaryDirectory() as td:
            split_file = self._write_split(td, {"val": ["bee/v1"]})
            with self.assertRaises(SystemExit):
                evaluate.resolve_pinned_train_split(self._samples(), split_file)

    def test_fails_closed_when_val_key_entirely_missing(self):
        with tempfile.TemporaryDirectory() as td:
            split_file = self._write_split(td, {"train": ["aus/t1"]})
            with self.assertRaises(SystemExit):
                evaluate.resolve_pinned_train_split(self._samples(), split_file)

    def test_fails_closed_on_train_val_overlap(self):
        with tempfile.TemporaryDirectory() as td:
            split_file = self._write_split(td, self._complete_record(
                ["aus/t1", "aus/t2", "bee/t1"],
                ["bee/t1"]))  # also pinned as train -- not disjoint
            with self.assertRaises(SystemExit):
                evaluate.resolve_pinned_train_split(self._samples(), split_file)

    def test_fails_closed_on_duplicate_key_within_list(self):
        with tempfile.TemporaryDirectory() as td:
            split_file = self._write_split(td, self._complete_record(
                ["aus/t1", "aus/t1", "bee/t1"], ["bee/v1"]))
            with self.assertRaises(SystemExit):
                evaluate.resolve_pinned_train_split(self._samples(), split_file)

    def test_fails_closed_on_unresolvable_key(self):
        with tempfile.TemporaryDirectory() as td:
            split_file = self._write_split(td, self._complete_record(
                ["aus/t1", "aus/t2", "ghost/nope"], ["bee/v1"]))
            with self.assertRaises(SystemExit):
                evaluate.resolve_pinned_train_split(self._samples(), split_file)

    def test_fails_closed_on_key_resolving_to_multiple_samples(self):
        samples = self._samples() + [make_sample("aus", "t1", 0)]  # duplicate
        with tempfile.TemporaryDirectory() as td:
            split_file = self._write_split(td, self._complete_record(
                ["aus/t1", "aus/t2", "bee/t1"], ["bee/v1"]))
            with self.assertRaises(SystemExit):
                evaluate.resolve_pinned_train_split(samples, split_file)

    def test_fails_closed_when_recorded_count_disagrees(self):
        with tempfile.TemporaryDirectory() as td:
            record = self._complete_record(["aus/t1", "aus/t2", "bee/t1"],
                                           ["bee/v1"])
            record["n_train"] = 99
            split_file = self._write_split(td, record)
            with self.assertRaises(SystemExit):
                evaluate.resolve_pinned_train_split(self._samples(), split_file)

    def test_fails_closed_on_non_array_membership(self):
        with tempfile.TemporaryDirectory() as td:
            record = self._complete_record(["aus/t1"], ["bee/v1"])
            record["train"] = {"aus/t1": True}
            split_file = self._write_split(td, record)
            with self.assertRaises(SystemExit):
                evaluate.resolve_pinned_train_split(self._samples(), split_file)


class TestDescribeGeoFileSource(unittest.TestCase):
    def test_labels_train_only_as_an_unverified_declaration(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "geo_index.json"
            path.write_text(json.dumps(
                {"cell_size_deg": 1.0, "cells": {}, "source_split": "train"}))
            desc = evaluate.describe_geo_file_source(path)
        self.assertIn("declares source_split=train", desc)
        self.assertIn("not independently verified", desc)

    def test_labels_missing_field_as_unknown_provenance(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "geo_index.json"
            path.write_text(json.dumps({"cell_size_deg": 1.0, "cells": {}}))
            desc = evaluate.describe_geo_file_source(path)
        self.assertIn("provenance unknown", desc)
        self.assertNotIn("source_split=train", desc)

    def test_labels_unreadable_file_as_unknown_provenance(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "geo_index.json"
            path.write_text("not json")
            desc = evaluate.describe_geo_file_source(path)
        self.assertIn("provenance unknown", desc)

    def test_labels_unsupported_recorded_value(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "geo_index.json"
            path.write_text(json.dumps(
                {"cell_size_deg": 1.0, "cells": {}, "source_split": "all"}))
            desc = evaluate.describe_geo_file_source(path)
        self.assertIn("unsupported source_split='all'", desc)
        self.assertNotIn("source_split not recorded", desc)


class TestRequireUsableGeoCells(unittest.TestCase):
    def test_empty_cells_fail_instead_of_reporting_fake_geo_metrics(self):
        with self.assertRaises(SystemExit):
            evaluate.require_usable_geo_cells({})

    def test_nonempty_cells_are_accepted(self):
        evaluate.require_usable_geo_cells({0: {(1, 2)}})


class TestApiConsumerCompatibility(unittest.TestCase):
    """The API's geo loader (api/inference.py) is not modified by this
    change. These tests feed it exactly what write_geo_index_sidecar now
    produces and confirm it still behaves as documented, without touching
    api/inference.py itself."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(HERE.parent / "api"))
        import inference as api_inference  # noqa: E402
        cls.api_inference = api_inference

    def _bare_identifier(self):
        ident = self.api_inference.AntIdentifier.__new__(self.api_inference.AntIdentifier)
        ident.taxonomy = {0: {"slug": "known-ant"}, 1: {"slug": "other-ant"}}
        ident.geo_index_loaded = False
        ident.geo_index_reason = "missing"
        ident._geo_cells = {}
        ident._cell_size = 1.0
        return ident

    def test_empty_sidecar_reports_no_usable_cells(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "geo_index.json"
            train.write_geo_index_sidecar(path, {}, 1.0)
            ident = self._bare_identifier()
            ident._load_geo_index(path)
        self.assertFalse(ident.geo_index_loaded)
        self.assertEqual(ident.geo_index_reason, "no_usable_cells")

    def test_populated_sidecar_with_source_split_is_accepted(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "geo_index.json"
            train.write_geo_index_sidecar(path, {"known-ant": [[5, -75]]}, 1.0)
            ident = self._bare_identifier()
            ident._load_geo_index(path)
        self.assertTrue(ident.geo_index_loaded)
        self.assertEqual(ident.geo_index_reason, "active")
        self.assertEqual(ident._geo_cells, {0: {(5, -75)}})


REAL_NORTHEAST_MANIFEST = HERE.parent / "data" / "northeast_expansion_v1" / "manifest_all_northeast_v1.csv"
REAL_LOCAL_DATA_DIR = HERE.parent / "data" / "clean"
NORTHEAST_NEW_SPECIES_SLUGS = (
    "aphaenogaster-rudis", "camponotus-americanus", "camponotus-nearcticus",
    "camponotus-novaeboracensis", "camponotus-subbarbatus", "formica-exsectoides",
    "lasius-americanus", "lasius-aphidicola", "lasius-claviger", "lasius-emarginatus",
    "lasius-interjectus", "lasius-neoniger", "nylanderia-flavipes",
    "ponera-pennsylvanica", "temnothorax-curvispinosus",
)


@unittest.skipUnless(REAL_NORTHEAST_MANIFEST.exists() and REAL_LOCAL_DATA_DIR.exists(),
                     "real Northeast manifest / data/clean not present in this checkout")
class TestNortheastCatalogTrainOnlyGeoIndex(unittest.TestCase):
    """Integration: training/data.py resolves all 65 classes from the real,
    coordinate-populated 65-species manifest, and a train-only geo index built
    from it contains usable cells for every one of the 15 new species --
    proving the coordinates-sidecar integration
    (build_northeast_training_catalog.py) produces a usable geo signal, not
    just non-blank lat/lon strings. Read-only: no artifact is written."""

    @classmethod
    def setUpClass(cls):
        with mock.patch.dict(os.environ, {
            "MANIFEST_CSV": str(REAL_NORTHEAST_MANIFEST), "LOCAL_DATA_DIR": str(REAL_LOCAL_DATA_DIR),
        }, clear=False):
            os.environ.pop("DATABASE_URL", None)
            cls.samples, cls.taxonomy = data.load_manifest({})
        cls.train_s, cls.val_s = train.split_samples(cls.samples, val_fraction=0.2, seed=0)

    def test_65_classes_resolved(self):
        self.assertEqual(len(self.taxonomy), 65)

    def test_new_species_train_split_excludes_the_40_val_rows_each(self):
        # Direct object-level proof that val rows never reach train_s: each
        # new species has exactly 200 train + 40 val, not 240 train.
        for slug in NORTHEAST_NEW_SPECIES_SLUGS:
            n_train = sum(1 for s in self.train_s if s.slug == slug)
            n_val = sum(1 for s in self.val_s if s.slug == slug)
            self.assertEqual(n_train, 200, f"{slug}: train count")
            self.assertEqual(n_val, 40, f"{slug}: val count")

    def test_train_only_geo_index_has_usable_cells_for_all_15_new_species(self):
        # Combined with test_uses_train_split_only_excludes_val_coordinates
        # above (which proves build_geo_index(train_s, ...) structurally
        # cannot see val_s at all) and the exact 200/40 split proven above,
        # this closes the loop: validation coordinates cannot enter this
        # train-only index, and the train-only index is genuinely usable.
        cells = train.build_geo_index(self.train_s, self.taxonomy,
                                      cell_size_deg=1.0, min_obs_per_cell=2)
        missing = [slug for slug in NORTHEAST_NEW_SPECIES_SLUGS
                  if not cells.get(slug)]
        self.assertEqual(missing, [],
                         f"new species with no usable train-only geo cells: {missing}")


if __name__ == "__main__":
    unittest.main()
