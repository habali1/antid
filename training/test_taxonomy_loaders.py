#!/usr/bin/env python3
"""Focused tests that "genus" survives every training taxonomy loader path
(training/data.py's DB, CSV-manifest, and bare-directory loaders), and that
the real 65-species Northeast manifest + data/clean produce the intended
65-class, contiguous, slug-sorted, genus-complete schema train.py will write
to taxonomy.json on retrain. No training, evaluation, or writes to
training/artifacts/ -- this only calls the read-only data-loading functions.
Run directly: python test_taxonomy_loaders.py
"""
from __future__ import annotations

import csv
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

import data  # noqa: E402

REAL_MANIFEST = REPO / "data" / "northeast_expansion_v1" / "manifest_all_northeast_v1.csv"
REAL_LOCAL_DATA_DIR = REPO / "data" / "clean"
REAL_TAXONOMY_SNAPSHOT = REPO / "data" / "northeast_expansion_v1" / "northeast_taxonomy_v1.json"


def _write_blank_jpeg(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Minimal valid-enough placeholder; the loaders here only check existence
    # (data.py's manifest/dir loaders never decode image bytes), so content
    # doesn't matter.
    path.write_bytes(b"\xff\xd8\xff\xd9")


class TestGenusHelper(unittest.TestCase):
    def test_normal_binomial_name(self):
        self.assertEqual(data._genus_from_species_name("Camponotus pennsylvanicus"), "Camponotus")

    def test_single_word_name(self):
        self.assertEqual(data._genus_from_species_name("Formicidae"), "Formicidae")

    def test_empty_or_none_is_empty_string(self):
        self.assertEqual(data._genus_from_species_name(""), "")
        self.assertEqual(data._genus_from_species_name(None), "")

    def test_extra_whitespace(self):
        self.assertEqual(data._genus_from_species_name("  Lasius   neoniger "), "Lasius")


class TestCsvLoaderGenus(unittest.TestCase):
    def test_synthetic_csv_manifest_has_genus_for_every_class(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest = root / "manifest.csv"
            local = root / "clean"
            rows = [
                {"species": "Camponotus pennsylvanicus", "slug": "camponotus-pennsylvanicus",
                 "taxon_id": "1", "photo_id": "p1", "source": "inat_api", "lat": "", "lon": "", "split": "train"},
                {"species": "Lasius neoniger", "slug": "lasius-neoniger",
                 "taxon_id": "2", "photo_id": "p2", "source": "inat_api", "lat": "", "lon": "", "split": "val"},
            ]
            with manifest.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            for r in rows:
                _write_blank_jpeg(local / r["slug"] / f"{r['photo_id']}.jpg")

            _, taxonomy = data._manifest_from_csv(manifest, local)
            self.assertEqual(len(taxonomy), 2)
            genus_by_slug = {v["slug"]: v["genus"] for v in taxonomy.values()}
            self.assertEqual(genus_by_slug, {
                "camponotus-pennsylvanicus": "Camponotus",
                "lasius-neoniger": "Lasius",
            })

    def test_species_name_falls_back_to_slug_and_still_gets_a_genus(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest = root / "manifest.csv"
            local = root / "clean"
            row = {"species": "", "slug": "atta-mexicana", "taxon_id": "3",
                  "photo_id": "p1", "source": "inat_api", "lat": "", "lon": "", "split": "train"}
            with manifest.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(row.keys()))
                w.writeheader()
                w.writerow(row)
            _write_blank_jpeg(local / row["slug"] / f"{row['photo_id']}.jpg")

            _, taxonomy = data._manifest_from_csv(manifest, local)
            entry = next(iter(taxonomy.values()))
            self.assertEqual(entry["species_name"], "Atta mexicana")
            self.assertEqual(entry["genus"], "Atta")


class TestDirLoaderGenus(unittest.TestCase):
    def test_bare_directory_walk_has_genus_for_every_class(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for slug in ("camponotus-pennsylvanicus", "lasius-neoniger"):
                _write_blank_jpeg(root / slug / "img1.jpg")
            _, taxonomy = data._manifest_from_dir(root)
            self.assertEqual(len(taxonomy), 2)
            for v in taxonomy.values():
                self.assertTrue(v["genus"])
            genus_by_slug = {v["slug"]: v["genus"] for v in taxonomy.values()}
            self.assertEqual(genus_by_slug, {
                "camponotus-pennsylvanicus": "Camponotus",
                "lasius-neoniger": "Lasius",
            })


class TestDbLoaderGenus(unittest.TestCase):
    def test_db_loader_has_genus_for_every_class(self):
        rows = [
            ("camponotus-pennsylvanicus", "Camponotus pennsylvanicus", "Black Carpenter Ant", 129902,
             "/x/1.jpg", "train", 42.0, -83.0),
            ("lasius-neoniger", "Lasius neoniger", None, 69311,
             "/x/2.jpg", "val", None, None),
        ]

        class FakeCursor:
            def execute(self, sql):
                pass

            def fetchall(self):
                return rows

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        class FakeConn:
            def cursor(self):
                return FakeCursor()

            def close(self):
                pass

        fake_psycopg2 = types.ModuleType("psycopg2")
        fake_psycopg2.connect = lambda url: FakeConn()
        with mock.patch.dict(sys.modules, {"psycopg2": fake_psycopg2}):
            _, taxonomy = data._manifest_from_db("postgresql://fake")

        self.assertEqual(len(taxonomy), 2)
        genus_by_slug = {v["slug"]: v["genus"] for v in taxonomy.values()}
        self.assertEqual(genus_by_slug, {
            "camponotus-pennsylvanicus": "Camponotus",
            "lasius-neoniger": "Lasius",
        })


@unittest.skipUnless(REAL_MANIFEST.exists() and REAL_LOCAL_DATA_DIR.exists(),
                     "real Northeast manifest / data/clean not present in this checkout")
class TestRealNortheastManifestSchema(unittest.TestCase):
    """Loads the actual manifest train.py will read on retrain (read-only --
    no artifact is written or modified). Confirms it produces the intended
    65-class, contiguous, slug-sorted, genus-complete schema."""

    @classmethod
    def setUpClass(cls):
        with mock.patch.dict(os.environ, {
            "MANIFEST_CSV": str(REAL_MANIFEST), "LOCAL_DATA_DIR": str(REAL_LOCAL_DATA_DIR),
        }, clear=False):
            os.environ.pop("DATABASE_URL", None)
            cls.samples, cls.taxonomy = data.load_manifest({})

    def test_65_classes_contiguous_and_slug_sorted(self):
        self.assertEqual(len(self.taxonomy), 65)
        self.assertEqual(sorted(self.taxonomy), list(range(65)))
        slugs = [self.taxonomy[i]["slug"] for i in range(65)]
        self.assertEqual(slugs, sorted(slugs))

    def test_genus_present_for_every_class(self):
        for i in range(65):
            entry = self.taxonomy[i]
            self.assertTrue(entry.get("genus"), f"class {i} ({entry['slug']}) has no genus")

    def test_taxonomy_full_object_equality_with_versioned_snapshot(self):
        """Not just genus: every field (species_name, common_name, taxon_id,
        slug, genus) must match exactly, index for index, between what
        training/data.py's manifest-CSV loader derives from
        manifest_all_northeast_v1.csv and the generator's own
        northeast_taxonomy_v1.json output."""
        snapshot = json.loads(REAL_TAXONOMY_SNAPSHOT.read_text())
        self.assertEqual(len(snapshot), 65)
        for i in range(65):
            loaded_entry = self.taxonomy[i]
            snapshot_entry = snapshot[str(i)]
            self.assertEqual(loaded_entry, snapshot_entry,
                             f"class {i} ({loaded_entry.get('slug')}) full-object mismatch")

    def test_all_15_northeast_species_present_with_expected_counts(self):
        # Every one of the 15 Northeast species should resolve to 240 images
        # now that they're copied into data/clean/.
        counts = {}
        for s in self.samples:
            counts[s.slug] = counts.get(s.slug, 0) + 1
        ne_slugs_240 = ("aphaenogaster-rudis", "camponotus-americanus", "camponotus-nearcticus",
                       "camponotus-novaeboracensis", "camponotus-subbarbatus", "formica-exsectoides",
                       "lasius-americanus", "lasius-aphidicola", "lasius-claviger",
                       "lasius-emarginatus", "lasius-interjectus", "lasius-neoniger",
                       "nylanderia-flavipes", "ponera-pennsylvanica", "temnothorax-curvispinosus")
        for slug in ne_slugs_240:
            self.assertEqual(counts.get(slug), 240, f"{slug}: expected 240 resolved images, got {counts.get(slug)}")


if __name__ == "__main__":
    unittest.main()
