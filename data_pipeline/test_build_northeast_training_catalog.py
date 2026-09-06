#!/usr/bin/env python3
"""Focused tests for build_northeast_training_catalog.py: determinism,
fail-closed fault injection (missing image, incorrect hash), and the
postcondition checks the generator enforces against the real Northeast
catalog inputs (65 contiguous slug-sorted classes, all 15 new species at
200 train / 40 val, no structured-field sentinel strings). Never writes to
data/manifest_all.csv or training/artifacts/; the fault-injection tests use
fully synthetic, isolated fixtures, never the real data/clean tree.
Run directly: python test_build_northeast_training_catalog.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import build_northeast_training_catalog as catalog  # noqa: E402

def _real_inputs_available() -> bool:
    inputs = catalog.Inputs.real()
    required_paths = [
        inputs.manifest_all, inputs.northeast_train_dev, inputs.uuid_helper,
        inputs.original_taxonomy, inputs.clean_root,
        *inputs.frozen_eval_sets.values(),
    ]
    return all(p.exists() for p in required_paths)


REAL_INPUTS_AVAILABLE = _real_inputs_available()


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _empty_eval_csv(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["sha256"])


class SyntheticFixture:
    """A minimal, fully isolated Inputs fixture: one legacy species (2
    photos) and one Northeast species (2 photos), with matching
    original_taxonomy and empty frozen eval sets. Deliberately far too small
    to satisfy the real hardcoded postcondition totals (13,581 rows etc.) --
    only used for fault-injection tests where the injected fault must raise
    before those postconditions are ever checked.
    """

    def __init__(self, td: Path):
        self.td = td
        self.clean_root = td / "clean"
        self.manifest_all = td / "manifest_all.csv"
        self.original_taxonomy = td / "original_taxonomy.json"
        self.northeast_csv = td / "northeast_train_dev.csv"
        self.uuid_helper = td / "uuid_helper.json"
        self.eval_sets = {}
        for name in ("benchmark_v1", "calibration_v1", "unknown_test_v1", "northeast_final_test_v1"):
            p = td / f"{name}.csv"
            _empty_eval_csv(p)
            self.eval_sets[name] = p

        legacy_rows = [
            {"species": "Aus testus", "slug": "aus-testus", "taxon_id": "1", "photo_id": "L1",
             "source": "inat_api", "lat": "1.0", "lon": "2.0", "split": "train"},
            {"species": "Aus testus", "slug": "aus-testus", "taxon_id": "1", "photo_id": "L2",
             "source": "inat_api", "lat": "", "lon": "", "split": "val"},
        ]
        _write_csv(self.manifest_all, legacy_rows)
        for r in legacy_rows:
            self._write_image(r["slug"], r["photo_id"], f"legacy-content-{r['photo_id']}".encode())

        self.original_taxonomy.write_text(json.dumps({
            "0": {"species_name": "Aus testus", "common_name": "Test Ant", "taxon_id": 1,
                 "slug": "aus-testus"},
        }, indent=2))

        self.ne_rows = [
            {"species": "Bus novus", "slug": "bus-novus", "taxon_id": "2", "genus": "Bus",
             "genus_id": "9", "split": "train", "state": "NY", "observation_id": "1",
             "observation_uuid": "uuid-1", "observer_id": "1", "observed_on": "2026-01-01",
             "created_at": "2026-01-01T00:00:00Z", "geoprivacy": "open", "obscured": "false",
             "photo_id": "N1", "photo_license": "cc-by", "photo_attribution": "(c) tester",
             "source_url": "https://example.test/N1.jpg", "sha256": "", "byte_size": "10",
             "width": "10", "height": "10", "raw_relative_path": "raw/bus-novus/N1.jpg",
             "clean_relative_path": "clean/bus-novus/N1.jpg"},
            {"species": "Bus novus", "slug": "bus-novus", "taxon_id": "2", "genus": "Bus",
             "genus_id": "9", "split": "development", "state": "NY", "observation_id": "2",
             "observation_uuid": "uuid-2", "observer_id": "1", "observed_on": "2026-01-01",
             "created_at": "2026-01-01T00:00:00Z", "geoprivacy": "open", "obscured": "false",
             "photo_id": "N2", "photo_license": "cc-by", "photo_attribution": "(c) tester",
             "source_url": "https://example.test/N2.jpg", "sha256": "", "byte_size": "10",
             "width": "10", "height": "10", "raw_relative_path": "raw/bus-novus/N2.jpg",
             "clean_relative_path": "clean/bus-novus/N2.jpg"},
        ]
        for r in self.ne_rows:
            content = f"northeast-content-{r['photo_id']}".encode()
            self._write_image(r["slug"], r["photo_id"], content)
            r["sha256"] = hashlib.sha256(content).hexdigest()
        _write_csv(self.northeast_csv, self.ne_rows)

        self.uuid_helper.write_text(json.dumps({"photo_to_observation_uuid": {}}))

    def _write_image(self, slug: str, photo_id: str, content: bytes) -> Path:
        p = self.clean_root / slug / f"{photo_id}.jpg"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content)
        return p

    def rewrite_northeast_csv(self, rows: list[dict]) -> None:
        _write_csv(self.northeast_csv, rows)

    def inputs(self) -> catalog.Inputs:
        return catalog.Inputs(
            manifest_all=self.manifest_all, clean_root=self.clean_root,
            northeast_train_dev=self.northeast_csv, uuid_helper=self.uuid_helper,
            original_taxonomy=self.original_taxonomy, frozen_eval_sets=self.eval_sets,
            expected_original_taxonomy_sha256=None,
        )


class TestFailClosedFaultInjection(unittest.TestCase):
    def test_missing_northeast_image_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            fx = SyntheticFixture(Path(td))
            (fx.clean_root / "bus-novus" / "N1.jpg").unlink()
            with self.assertRaises(catalog.CatalogIntegrityError) as cm:
                catalog.build_catalog(fx.inputs())
            self.assertIn("does not resolve", str(cm.exception))

    def test_incorrect_northeast_hash_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            fx = SyntheticFixture(Path(td))
            tampered = [dict(r) for r in fx.ne_rows]
            tampered[0]["sha256"] = "0" * 64
            fx.rewrite_northeast_csv(tampered)
            with self.assertRaises(catalog.CatalogIntegrityError) as cm:
                catalog.build_catalog(fx.inputs())
            self.assertIn("does not match", str(cm.exception))

    def test_blank_provenance_field_fails_closed_before_any_output(self):
        # photo_license, photo_attribution, observation_uuid, source_url are
        # read straight from the source row and reach the blank-field guard
        # directly. photo_id and sha256 are structurally protected earlier
        # (blank photo_id can't resolve to a real file; sha256 is always a
        # freshly computed hash, never read blank from source) -- blanking
        # them still fails closed, just via those earlier checks, which is
        # exactly as fail-closed as the dedicated guard.
        directly_checked = ("photo_license", "photo_attribution", "observation_uuid", "source_url")
        for field_name in directly_checked:
            with self.subTest(field=field_name):
                with tempfile.TemporaryDirectory() as td:
                    fx = SyntheticFixture(Path(td))
                    tampered = [dict(r) for r in fx.ne_rows]
                    tampered[0][field_name] = ""
                    fx.rewrite_northeast_csv(tampered)
                    with self.assertRaises(catalog.CatalogIntegrityError) as cm:
                        catalog.build_catalog(fx.inputs())
                    self.assertIn("required provenance field", str(cm.exception))
                    self.assertIn(repr(field_name), str(cm.exception))

    def test_blank_photo_id_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            fx = SyntheticFixture(Path(td))
            tampered = [dict(r) for r in fx.ne_rows]
            tampered[0]["photo_id"] = ""
            fx.rewrite_northeast_csv(tampered)
            with self.assertRaises(catalog.CatalogIntegrityError):
                catalog.build_catalog(fx.inputs())

    def test_blank_sha256_in_postcondition_check_fails_closed(self):
        # sha256 can never actually be blank coming out of the row-building
        # loop (it's always a freshly computed 64-char digest, even for a
        # zero-byte file), so this exercises the format postcondition
        # directly rather than via the full build -- the same gate that
        # would catch it if that ever changed.
        rows = [{
            "species": "Bus novus", "slug": "bus-novus", "taxon_id": "2", "photo_id": "N1",
            "source": "inat_api", "lat": "", "lon": "", "split": "train", "common_name": "",
            "photo_license": "cc-by", "photo_attribution": "(c) tester",
            "observation_uuid": "uuid-1", "source_url": "https://example.test/N1.jpg",
            "sha256": "", "provenance_status": "northeast_v1_complete",
        }]
        taxonomy = {"0": {"species_name": "Bus novus", "common_name": None, "taxon_id": 2,
                         "slug": "bus-novus", "genus": "Bus"}}
        with tempfile.TemporaryDirectory() as td:
            fx = SyntheticFixture(Path(td))
            with self.assertRaises(catalog.CatalogIntegrityError):
                catalog._verify_postconditions(rows, taxonomy, fx.inputs())

    def test_original_taxonomy_hash_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            fx = SyntheticFixture(Path(td))
            inputs = fx.inputs()
            inputs.expected_original_taxonomy_sha256 = "f" * 64
            with self.assertRaises(catalog.CatalogIntegrityError) as cm:
                catalog.build_catalog(inputs)
            self.assertIn("sha256", str(cm.exception))

    def test_missing_original_taxonomy_input_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            fx = SyntheticFixture(Path(td))
            fx.original_taxonomy.unlink()
            with self.assertRaises(catalog.CatalogIntegrityError):
                catalog.build_catalog(fx.inputs())


def _write_full_sidecar(path: Path, source_path: Path, ne_rows: list[dict], *,
                        schema_version=1, source_sha256=None, source_rows=None,
                        rows_total=None, rows_with_coordinate=None, coverage_rate=None,
                        observations=None) -> None:
    """Writes a sidecar in the real schema (schema_version, source_manifest,
    coverage, observations) so fault-injection tests exercise the actual
    internal-validation code path, not just the outer hash binding. Every
    field defaults to a value consistent with source_path/ne_rows; pass an
    override to inject exactly one fault."""
    actual_source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if source_sha256 is None:
        source_sha256 = actual_source_sha256
    if source_rows is None:
        source_rows = len(ne_rows)
    if rows_total is None:
        rows_total = len(ne_rows)
    if rows_with_coordinate is None:
        rows_with_coordinate = len(ne_rows)
    if coverage_rate is None:
        coverage_rate = 1.0
    if observations is None:
        observations = {
            r["observation_uuid"]: {
                "slug": r["slug"], "taxon_id": int(r["taxon_id"]), "geoprivacy": "open",
                "lat": 42.0, "lon": -76.0,
            }
            for r in ne_rows
        }
    path.write_text(json.dumps({
        "schema_version": schema_version,
        "source_manifest": {"sha256": source_sha256, "rows": source_rows},
        "coverage": {"rows_total": rows_total, "rows_with_coordinate": rows_with_coordinate,
                    "coverage_rate": coverage_rate},
        "observations": observations,
    }))


class TestCoordinatesSidecarFaultInjection(unittest.TestCase):
    """Fault-injection tests only -- these fail before ever reaching the
    exact-count postconditions, so a tiny synthetic fixture works fine. The
    happy-path (successful join) is instead verified against the real
    3,600-row catalog in TestRealCatalogPostconditions below, since a tiny
    fixture can't satisfy EXPECTED_TOTAL_ROWS et al."""

    def _build(self, fx: "SyntheticFixture", sidecar_path: Path):
        inputs = fx.inputs()
        inputs.coordinates_sidecar = sidecar_path
        inputs.expected_coordinates_sidecar_sha256 = None
        return catalog.build_catalog(inputs)

    def test_sidecar_outer_hash_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            fx = SyntheticFixture(Path(td))
            sidecar_path = Path(td) / "coords.json"
            _write_full_sidecar(sidecar_path, fx.northeast_csv, fx.ne_rows)
            inputs = fx.inputs()
            inputs.coordinates_sidecar = sidecar_path
            inputs.expected_coordinates_sidecar_sha256 = "f" * 64
            with self.assertRaises(catalog.CatalogIntegrityError) as cm:
                catalog.build_catalog(inputs)
            self.assertIn("sha256", str(cm.exception))

    def test_sidecar_schema_version_not_one_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            fx = SyntheticFixture(Path(td))
            sidecar_path = Path(td) / "coords.json"
            _write_full_sidecar(sidecar_path, fx.northeast_csv, fx.ne_rows, schema_version=2)
            with self.assertRaises(catalog.CatalogIntegrityError) as cm:
                self._build(fx, sidecar_path)
            self.assertIn("schema_version", str(cm.exception))

    def test_sidecar_schema_version_bool_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            fx = SyntheticFixture(Path(td))
            sidecar_path = Path(td) / "coords.json"
            _write_full_sidecar(sidecar_path, fx.northeast_csv, fx.ne_rows, schema_version=True)
            with self.assertRaises(catalog.CatalogIntegrityError) as cm:
                self._build(fx, sidecar_path)
            self.assertIn("schema_version", str(cm.exception))

    def test_sidecar_source_manifest_sha256_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            fx = SyntheticFixture(Path(td))
            sidecar_path = Path(td) / "coords.json"
            _write_full_sidecar(sidecar_path, fx.northeast_csv, fx.ne_rows, source_sha256="0" * 64)
            with self.assertRaises(catalog.CatalogIntegrityError) as cm:
                self._build(fx, sidecar_path)
            self.assertIn("source_manifest.sha256", str(cm.exception))

    def test_sidecar_source_manifest_rows_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            fx = SyntheticFixture(Path(td))
            sidecar_path = Path(td) / "coords.json"
            _write_full_sidecar(sidecar_path, fx.northeast_csv, fx.ne_rows, source_rows=999)
            with self.assertRaises(catalog.CatalogIntegrityError) as cm:
                self._build(fx, sidecar_path)
            self.assertIn("source_manifest.rows", str(cm.exception))

    def test_sidecar_coverage_rows_total_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            fx = SyntheticFixture(Path(td))
            sidecar_path = Path(td) / "coords.json"
            _write_full_sidecar(sidecar_path, fx.northeast_csv, fx.ne_rows, rows_total=1)
            with self.assertRaises(catalog.CatalogIntegrityError) as cm:
                self._build(fx, sidecar_path)
            self.assertIn("coverage is not complete", str(cm.exception))

    def test_sidecar_coverage_rows_with_coordinate_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            fx = SyntheticFixture(Path(td))
            sidecar_path = Path(td) / "coords.json"
            _write_full_sidecar(sidecar_path, fx.northeast_csv, fx.ne_rows, rows_with_coordinate=1)
            with self.assertRaises(catalog.CatalogIntegrityError) as cm:
                self._build(fx, sidecar_path)
            self.assertIn("coverage is not complete", str(cm.exception))

    def test_sidecar_coverage_rate_not_complete_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            fx = SyntheticFixture(Path(td))
            sidecar_path = Path(td) / "coords.json"
            _write_full_sidecar(sidecar_path, fx.northeast_csv, fx.ne_rows, coverage_rate=0.5)
            with self.assertRaises(catalog.CatalogIntegrityError) as cm:
                self._build(fx, sidecar_path)
            self.assertIn("coverage_rate", str(cm.exception))

    def test_sidecar_missing_uuid_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            fx = SyntheticFixture(Path(td))
            sidecar_path = Path(td) / "coords.json"
            observations = {
                r["observation_uuid"]: {"slug": r["slug"], "taxon_id": int(r["taxon_id"]),
                                        "geoprivacy": "open", "lat": 42.0, "lon": -76.0}
                for r in fx.ne_rows[:1]  # drop the second observation
            }
            _write_full_sidecar(sidecar_path, fx.northeast_csv, fx.ne_rows, observations=observations)
            with self.assertRaises(catalog.CatalogIntegrityError) as cm:
                self._build(fx, sidecar_path)
            self.assertIn("is missing", str(cm.exception))

    def test_sidecar_unexpected_uuid_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            fx = SyntheticFixture(Path(td))
            sidecar_path = Path(td) / "coords.json"
            observations = {
                r["observation_uuid"]: {"slug": r["slug"], "taxon_id": int(r["taxon_id"]),
                                        "geoprivacy": "open", "lat": 42.0, "lon": -76.0}
                for r in fx.ne_rows
            }
            observations["uuid-9999"] = {"slug": "bus-novus", "taxon_id": 2,
                                         "geoprivacy": "open", "lat": 41.0, "lon": -75.0}
            _write_full_sidecar(sidecar_path, fx.northeast_csv, fx.ne_rows, observations=observations)
            with self.assertRaises(catalog.CatalogIntegrityError) as cm:
                self._build(fx, sidecar_path)
            self.assertIn("not present in", str(cm.exception))

    def test_sidecar_duplicate_uuid_in_source_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            fx = SyntheticFixture(Path(td))
            dup_rows = fx.ne_rows + [dict(fx.ne_rows[0])]
            fx.rewrite_northeast_csv(dup_rows)
            sidecar_path = Path(td) / "coords.json"
            _write_full_sidecar(sidecar_path, fx.northeast_csv, dup_rows)
            with self.assertRaises(catalog.CatalogIntegrityError) as cm:
                self._build(fx, sidecar_path)
            self.assertIn("duplicate observation_uuid", str(cm.exception))

    def test_sidecar_blank_uuid_in_source_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            fx = SyntheticFixture(Path(td))
            blanked = [dict(r) for r in fx.ne_rows]
            blanked[0]["observation_uuid"] = ""
            fx.rewrite_northeast_csv(blanked)
            sidecar_path = Path(td) / "coords.json"
            _write_full_sidecar(sidecar_path, fx.northeast_csv, blanked)
            with self.assertRaises(catalog.CatalogIntegrityError) as cm:
                self._build(fx, sidecar_path)
            self.assertIn("blank observation_uuid", str(cm.exception))

    def test_sidecar_slug_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            fx = SyntheticFixture(Path(td))
            sidecar_path = Path(td) / "coords.json"
            observations = {
                r["observation_uuid"]: {"slug": r["slug"], "taxon_id": int(r["taxon_id"]),
                                        "geoprivacy": "open", "lat": 42.0, "lon": -76.0}
                for r in fx.ne_rows
            }
            observations[fx.ne_rows[0]["observation_uuid"]]["slug"] = "wrong-slug"
            _write_full_sidecar(sidecar_path, fx.northeast_csv, fx.ne_rows, observations=observations)
            with self.assertRaises(catalog.CatalogIntegrityError) as cm:
                self._build(fx, sidecar_path)
            self.assertIn("slug", str(cm.exception))

    def test_sidecar_taxon_id_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            fx = SyntheticFixture(Path(td))
            sidecar_path = Path(td) / "coords.json"
            observations = {
                r["observation_uuid"]: {"slug": r["slug"], "taxon_id": int(r["taxon_id"]),
                                        "geoprivacy": "open", "lat": 42.0, "lon": -76.0}
                for r in fx.ne_rows
            }
            observations[fx.ne_rows[0]["observation_uuid"]]["taxon_id"] = 999
            _write_full_sidecar(sidecar_path, fx.northeast_csv, fx.ne_rows, observations=observations)
            with self.assertRaises(catalog.CatalogIntegrityError) as cm:
                self._build(fx, sidecar_path)
            self.assertIn("taxon_id", str(cm.exception))

    def test_sidecar_non_integer_taxon_id_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            fx = SyntheticFixture(Path(td))
            sidecar_path = Path(td) / "coords.json"
            observations = {
                r["observation_uuid"]: {"slug": r["slug"], "taxon_id": int(r["taxon_id"]),
                                        "geoprivacy": "open", "lat": 42.0, "lon": -76.0}
                for r in fx.ne_rows
            }
            observations[fx.ne_rows[0]["observation_uuid"]]["taxon_id"] = "not-a-number"
            _write_full_sidecar(sidecar_path, fx.northeast_csv, fx.ne_rows, observations=observations)
            with self.assertRaises(catalog.CatalogIntegrityError) as cm:
                self._build(fx, sidecar_path)
            self.assertIn("non-integer taxon_id", str(cm.exception))

    def test_sidecar_bool_latitude_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            fx = SyntheticFixture(Path(td))
            sidecar_path = Path(td) / "coords.json"
            observations = {
                r["observation_uuid"]: {"slug": r["slug"], "taxon_id": int(r["taxon_id"]),
                                        "geoprivacy": "open", "lat": 42.0, "lon": -76.0}
                for r in fx.ne_rows
            }
            observations[fx.ne_rows[0]["observation_uuid"]]["lat"] = True
            _write_full_sidecar(sidecar_path, fx.northeast_csv, fx.ne_rows, observations=observations)
            with self.assertRaises(catalog.CatalogIntegrityError) as cm:
                self._build(fx, sidecar_path)
            self.assertIn("non-numeric coordinate", str(cm.exception))

    def test_sidecar_non_numeric_longitude_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            fx = SyntheticFixture(Path(td))
            sidecar_path = Path(td) / "coords.json"
            observations = {
                r["observation_uuid"]: {"slug": r["slug"], "taxon_id": int(r["taxon_id"]),
                                        "geoprivacy": "open", "lat": 42.0, "lon": -76.0}
                for r in fx.ne_rows
            }
            observations[fx.ne_rows[0]["observation_uuid"]]["lon"] = "not-a-number"
            _write_full_sidecar(sidecar_path, fx.northeast_csv, fx.ne_rows, observations=observations)
            with self.assertRaises(catalog.CatalogIntegrityError) as cm:
                self._build(fx, sidecar_path)
            self.assertIn("non-numeric coordinate", str(cm.exception))

    def test_sidecar_nan_latitude_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            fx = SyntheticFixture(Path(td))
            sidecar_path = Path(td) / "coords.json"
            observations = {
                r["observation_uuid"]: {"slug": r["slug"], "taxon_id": int(r["taxon_id"]),
                                        "geoprivacy": "open", "lat": 42.0, "lon": -76.0}
                for r in fx.ne_rows
            }
            observations[fx.ne_rows[0]["observation_uuid"]]["lat"] = float("nan")
            _write_full_sidecar(sidecar_path, fx.northeast_csv, fx.ne_rows, observations=observations)
            with self.assertRaises(catalog.CatalogIntegrityError) as cm:
                self._build(fx, sidecar_path)
            self.assertIn("non-finite", str(cm.exception))

    def test_sidecar_infinite_longitude_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            fx = SyntheticFixture(Path(td))
            sidecar_path = Path(td) / "coords.json"
            observations = {
                r["observation_uuid"]: {"slug": r["slug"], "taxon_id": int(r["taxon_id"]),
                                        "geoprivacy": "open", "lat": 42.0, "lon": -76.0}
                for r in fx.ne_rows
            }
            observations[fx.ne_rows[0]["observation_uuid"]]["lon"] = float("inf")
            _write_full_sidecar(sidecar_path, fx.northeast_csv, fx.ne_rows, observations=observations)
            with self.assertRaises(catalog.CatalogIntegrityError) as cm:
                self._build(fx, sidecar_path)
            self.assertIn("non-finite", str(cm.exception))

    def test_sidecar_out_of_range_latitude_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            fx = SyntheticFixture(Path(td))
            sidecar_path = Path(td) / "coords.json"
            observations = {
                r["observation_uuid"]: {"slug": r["slug"], "taxon_id": int(r["taxon_id"]),
                                        "geoprivacy": "open", "lat": 42.0, "lon": -76.0}
                for r in fx.ne_rows
            }
            observations[fx.ne_rows[0]["observation_uuid"]]["lat"] = 91.0
            _write_full_sidecar(sidecar_path, fx.northeast_csv, fx.ne_rows, observations=observations)
            with self.assertRaises(catalog.CatalogIntegrityError) as cm:
                self._build(fx, sidecar_path)
            self.assertIn("latitude out of range", str(cm.exception))

    def test_sidecar_out_of_range_longitude_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            fx = SyntheticFixture(Path(td))
            sidecar_path = Path(td) / "coords.json"
            observations = {
                r["observation_uuid"]: {"slug": r["slug"], "taxon_id": int(r["taxon_id"]),
                                        "geoprivacy": "open", "lat": 42.0, "lon": -76.0}
                for r in fx.ne_rows
            }
            observations[fx.ne_rows[0]["observation_uuid"]]["lon"] = 181.0
            _write_full_sidecar(sidecar_path, fx.northeast_csv, fx.ne_rows, observations=observations)
            with self.assertRaises(catalog.CatalogIntegrityError) as cm:
                self._build(fx, sidecar_path)
            self.assertIn("longitude out of range", str(cm.exception))

    def test_sidecar_private_geoprivacy_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            fx = SyntheticFixture(Path(td))
            sidecar_path = Path(td) / "coords.json"
            observations = {
                r["observation_uuid"]: {"slug": r["slug"], "taxon_id": int(r["taxon_id"]),
                                        "geoprivacy": "open", "lat": 42.0, "lon": -76.0}
                for r in fx.ne_rows
            }
            observations[fx.ne_rows[0]["observation_uuid"]]["geoprivacy"] = "private"
            _write_full_sidecar(sidecar_path, fx.northeast_csv, fx.ne_rows, observations=observations)
            with self.assertRaises(catalog.CatalogIntegrityError) as cm:
                self._build(fx, sidecar_path)
            self.assertIn("geoprivacy=private", str(cm.exception))


@unittest.skipUnless(REAL_INPUTS_AVAILABLE, "real Northeast catalog inputs not present in this checkout")
class TestRealCatalogPostconditions(unittest.TestCase):
    """Exercises the actual generator against the real repo inputs. Read-only:
    build_catalog() never writes anything by itself."""

    @classmethod
    def setUpClass(cls):
        cls.result = catalog.build_catalog(catalog.Inputs.real())

    def test_deterministic_catalog_generation(self):
        result_a = catalog.build_catalog(catalog.Inputs.real())
        result_b = catalog.build_catalog(catalog.Inputs.real())
        manifest_a = catalog.serialize_manifest_csv(result_a.manifest_rows)
        manifest_b = catalog.serialize_manifest_csv(result_b.manifest_rows)
        taxonomy_a = catalog.serialize_taxonomy_json(result_a.taxonomy)
        taxonomy_b = catalog.serialize_taxonomy_json(result_b.taxonomy)
        self.assertEqual(manifest_a, manifest_b)
        self.assertEqual(taxonomy_a, taxonomy_b)

    def test_65_contiguous_slug_sorted_classes(self):
        tax = self.result.taxonomy
        self.assertEqual(sorted(int(k) for k in tax), list(range(65)))
        slugs = [tax[str(i)]["slug"] for i in range(65)]
        self.assertEqual(slugs, sorted(slugs))

    def test_all_15_new_species_at_200_train_40_val(self):
        from collections import Counter
        per_species = {}
        for r in self.result.manifest_rows:
            if r["provenance_status"] == "northeast_v1_complete":
                per_species.setdefault(r["slug"], Counter())[r["split"]] += 1
        self.assertEqual(set(per_species), catalog.NEW_SPECIES_SLUGS)
        for slug, c in per_species.items():
            self.assertEqual(c["train"], 200, slug)
            self.assertEqual(c["val"], 40, slug)

    def test_no_structured_field_sentinel_strings(self):
        for r in self.result.manifest_rows:
            for f in ("photo_license", "photo_attribution", "observation_uuid",
                     "source_url", "sha256"):
                self.assertNotEqual(r[f], "legacy_provenance_unavailable", f"{f} in row {r}")

    def test_check_mode_reproduces_frozen_outputs_byte_for_byte(self):
        manifest_out = catalog.REPO / "data" / "northeast_expansion_v1" / "manifest_all_northeast_v1.csv"
        taxonomy_out = catalog.REPO / "data" / "northeast_expansion_v1" / "northeast_taxonomy_v1.json"
        self.assertEqual(catalog.serialize_manifest_csv(self.result.manifest_rows),
                         manifest_out.read_bytes())
        self.assertEqual(catalog.serialize_taxonomy_json(self.result.taxonomy),
                         taxonomy_out.read_bytes())

    def test_all_3600_northeast_rows_have_a_coordinate_from_the_sidecar(self):
        sidecar = json.loads((catalog.REPO / "data" / "northeast_expansion_v1"
                              / "northeast_coordinates_v1.json").read_text())
        observations = sidecar["observations"]
        ne_rows = [r for r in self.result.manifest_rows
                  if r["provenance_status"] == "northeast_v1_complete"]
        self.assertEqual(len(ne_rows), 3600)
        for r in ne_rows:
            self.assertTrue(r["lat"] and r["lon"], f"blank coordinate for {r}")
            expected = observations[r["observation_uuid"]]
            self.assertEqual(float(r["lat"]), expected["lat"])
            self.assertEqual(float(r["lon"]), expected["lon"])

    def test_legacy_rows_unaffected_by_coordinates_sidecar(self):
        legacy_rows = [r for r in self.result.manifest_rows
                      if r["provenance_status"] == "legacy_partial"]
        with_coords = sum(1 for r in legacy_rows if r["lat"] and r["lon"])
        self.assertEqual(with_coords, 9974)  # unchanged from before the sidecar existed


if __name__ == "__main__":
    unittest.main()
