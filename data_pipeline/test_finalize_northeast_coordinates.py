#!/usr/bin/env python3
"""Focused tests for finalize_northeast_coordinates.py: deterministic offline
finalization and fail-closed fault injection. NEVER contacts the network --
everything here is synthetic fixtures written to a temp directory; no test
imports urllib or touches fetch_northeast_coordinates.py's network path.
Run directly: python test_finalize_northeast_coordinates.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import finalize_northeast_coordinates as fin  # noqa: E402

SOURCE_FIELDS = ["species", "slug", "taxon_id", "genus", "genus_id", "observation_id",
                "observation_uuid", "geoprivacy", "obscured"]


def _row(i: int, *, geoprivacy: str = "open", obscured: str = "false") -> dict:
    return {
        "species": "Bus novus", "slug": "bus-novus", "taxon_id": "2", "genus": "Bus",
        "genus_id": "9", "observation_id": str(1000 + i),
        "observation_uuid": f"uuid-{i:04d}", "geoprivacy": geoprivacy, "obscured": obscured,
    }


def _write_source(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SOURCE_FIELDS)
        w.writeheader()
        w.writerows(rows)


def _write_capture(path: Path, rows: list[dict], *, coords: dict | None = None,
                   retrieved_at_utc: str = "2026-01-01T00:00:00Z") -> None:
    if coords is None:
        coords = {r["observation_uuid"]: {"lat": 42.0 + i * 0.01, "lon": -76.0 - i * 0.01}
                 for i, r in enumerate(rows)}
    payload = {
        "schema_version": 1, "retrieved_at_utc": retrieved_at_utc,
        "observation_coordinates": coords,
    }
    path.write_text(json.dumps(payload, indent=2))


class Fixture:
    def __init__(self, td: Path, n: int = 3):
        self.td = td
        self.source_path = td / "source.csv"
        self.capture_path = td / "capture.json"
        self.rows = [_row(i) for i in range(n)]
        _write_source(self.source_path, self.rows)
        _write_capture(self.capture_path, self.rows)

    def finalize(self, **kw):
        return fin.finalize(self.source_path, self.capture_path,
                            expected_row_count=len(self.rows), **kw)


class TestDeterministicFinalization(unittest.TestCase):
    def test_finalize_twice_is_byte_identical(self):
        with tempfile.TemporaryDirectory() as td:
            fx = Fixture(Path(td))
            a = fx.finalize()
            b = fx.finalize()
            self.assertEqual(fin.serialize(a), fin.serialize(b))

    def test_finalized_content_shape(self):
        with tempfile.TemporaryDirectory() as td:
            fx = Fixture(Path(td), n=2)
            result = fx.finalize()
            self.assertEqual(result["coverage"], {
                "rows_total": 2, "rows_with_coordinate": 2, "coverage_rate": 1.0,
            })
            self.assertEqual(len(result["observations"]), 2)
            entry = result["observations"]["uuid-0000"]
            self.assertEqual(entry["slug"], "bus-novus")
            self.assertEqual(entry["taxon_id"], 2)
            self.assertEqual(entry["observation_id"], "1000")
            self.assertFalse(entry["obscured"])
            self.assertIn("lat", entry)
            self.assertIn("lon", entry)


class TestFailClosedFaultInjection(unittest.TestCase):
    def test_source_manifest_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as td:
            fx = Fixture(Path(td))
            with self.assertRaises(fin.FinalizeIntegrityError) as cm:
                fx.finalize(expected_source_sha256="0" * 64)
            self.assertIn("sha256", str(cm.exception))

    def test_capture_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as td:
            fx = Fixture(Path(td))
            with self.assertRaises(fin.FinalizeIntegrityError) as cm:
                fx.finalize(expected_capture_sha256="0" * 64)
            self.assertIn("sha256", str(cm.exception))
            self.assertIn("capture", str(cm.exception))

    def test_missing_uuid_in_capture(self):
        with tempfile.TemporaryDirectory() as td:
            fx = Fixture(Path(td), n=3)
            coords = {r["observation_uuid"]: {"lat": 42.0, "lon": -76.0} for r in fx.rows[:2]}
            _write_capture(fx.capture_path, fx.rows, coords=coords)
            with self.assertRaises(fin.FinalizeIntegrityError) as cm:
                fx.finalize()
            self.assertIn("missing from capture", str(cm.exception))

    def test_unexpected_uuid_in_capture(self):
        with tempfile.TemporaryDirectory() as td:
            fx = Fixture(Path(td), n=2)
            coords = {r["observation_uuid"]: {"lat": 42.0, "lon": -76.0} for r in fx.rows}
            coords["uuid-9999"] = {"lat": 41.0, "lon": -75.0}
            _write_capture(fx.capture_path, fx.rows, coords=coords)
            with self.assertRaises(fin.FinalizeIntegrityError) as cm:
                fx.finalize()
            self.assertIn("not present in the source manifest", str(cm.exception))

    def test_duplicate_uuid_in_source(self):
        with tempfile.TemporaryDirectory() as td:
            fx = Fixture(Path(td), n=2)
            dup_rows = fx.rows + [dict(fx.rows[0])]
            _write_source(fx.source_path, dup_rows)
            _write_capture(fx.capture_path, dup_rows)
            with self.assertRaises(fin.FinalizeIntegrityError) as cm:
                fin.finalize(fx.source_path, fx.capture_path, expected_row_count=3)
            self.assertIn("duplicate observation_uuid", str(cm.exception))

    def test_duplicate_uuid_key_in_capture_json(self):
        with tempfile.TemporaryDirectory() as td:
            fx = Fixture(Path(td), n=2)
            # Hand-write JSON with a literal duplicate key -- json.loads would
            # normally silently keep the last one; our custom hook must catch it.
            raw = ('{"retrieved_at_utc": "2026-01-01T00:00:00Z", '
                  '"observation_coordinates": {'
                  '"uuid-0000": {"lat": 1.0, "lon": 2.0}, '
                  '"uuid-0000": {"lat": 3.0, "lon": 4.0}, '
                  '"uuid-0001": {"lat": 5.0, "lon": 6.0}}}')
            fx.capture_path.write_text(raw)
            with self.assertRaises(fin.FinalizeIntegrityError) as cm:
                fx.finalize()
            self.assertIn("duplicate JSON key", str(cm.exception))

    def test_private_source_observation_hard_fails(self):
        with tempfile.TemporaryDirectory() as td:
            fx = Fixture(Path(td), n=2)
            rows = [dict(r) for r in fx.rows]
            rows[0]["geoprivacy"] = "private"
            _write_source(fx.source_path, rows)
            with self.assertRaises(fin.FinalizeIntegrityError) as cm:
                fx.finalize()
            self.assertIn("geoprivacy=private", str(cm.exception))

    def test_nan_coordinate_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            fx = Fixture(Path(td), n=2)
            coords = {r["observation_uuid"]: {"lat": 42.0, "lon": -76.0} for r in fx.rows}
            coords[fx.rows[0]["observation_uuid"]]["lat"] = float("nan")
            _write_capture(fx.capture_path, fx.rows, coords=coords)
            with self.assertRaises(fin.FinalizeIntegrityError) as cm:
                fx.finalize()
            self.assertIn("non-finite", str(cm.exception))

    def test_infinite_coordinate_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            fx = Fixture(Path(td), n=2)
            coords = {r["observation_uuid"]: {"lat": 42.0, "lon": -76.0} for r in fx.rows}
            coords[fx.rows[0]["observation_uuid"]]["lon"] = float("inf")
            _write_capture(fx.capture_path, fx.rows, coords=coords)
            with self.assertRaises(fin.FinalizeIntegrityError) as cm:
                fx.finalize()
            self.assertIn("non-finite", str(cm.exception))

    def test_out_of_range_latitude_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            fx = Fixture(Path(td), n=2)
            coords = {r["observation_uuid"]: {"lat": 42.0, "lon": -76.0} for r in fx.rows}
            coords[fx.rows[0]["observation_uuid"]]["lat"] = 91.0
            _write_capture(fx.capture_path, fx.rows, coords=coords)
            with self.assertRaises(fin.FinalizeIntegrityError) as cm:
                fx.finalize()
            self.assertIn("latitude out of range", str(cm.exception))

    def test_out_of_range_longitude_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            fx = Fixture(Path(td), n=2)
            coords = {r["observation_uuid"]: {"lat": 42.0, "lon": -76.0} for r in fx.rows}
            coords[fx.rows[0]["observation_uuid"]]["lon"] = -181.0
            _write_capture(fx.capture_path, fx.rows, coords=coords)
            with self.assertRaises(fin.FinalizeIntegrityError) as cm:
                fx.finalize()
            self.assertIn("longitude out of range", str(cm.exception))

    def test_incomplete_coverage_fails_closed(self):
        # Same underlying join-completeness check as "missing UUID", exercised
        # explicitly under the "incomplete coverage" framing: fewer than all
        # rows have a coordinate.
        with tempfile.TemporaryDirectory() as td:
            fx = Fixture(Path(td), n=5)
            coords = {r["observation_uuid"]: {"lat": 42.0, "lon": -76.0} for r in fx.rows[:4]}
            _write_capture(fx.capture_path, fx.rows, coords=coords)
            with self.assertRaises(fin.FinalizeIntegrityError):
                fx.finalize()

    def test_species_genus_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            fx = Fixture(Path(td), n=2)
            rows = [dict(r) for r in fx.rows]
            rows[0]["species"] = "Aus alienus"  # genus column still says "Bus"
            _write_source(fx.source_path, rows)
            with self.assertRaises(fin.FinalizeIntegrityError) as cm:
                fx.finalize()
            self.assertIn("taxonomically inconsistent", str(cm.exception))

    def test_non_integer_taxon_id_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            fx = Fixture(Path(td), n=2)
            rows = [dict(r) for r in fx.rows]
            rows[0]["taxon_id"] = "not-a-number"
            _write_source(fx.source_path, rows)
            with self.assertRaises(fin.FinalizeIntegrityError) as cm:
                fx.finalize()
            self.assertIn("non-integer", str(cm.exception))

    def test_expected_row_count_enforced(self):
        with tempfile.TemporaryDirectory() as td:
            fx = Fixture(Path(td), n=3)
            with self.assertRaises(fin.FinalizeIntegrityError) as cm:
                fin.finalize(fx.source_path, fx.capture_path, expected_row_count=3600)
            self.assertIn("expected exactly 3600", str(cm.exception))


class TestMainCliHashWiring(unittest.TestCase):
    """A helper that is tested but not used by main() is not sufficient --
    these tests exercise main() itself, not just finalize()."""

    def test_main_passes_both_mandatory_hashes_to_finalize(self):
        captured = {}

        def fake_finalize(**kwargs):
            captured.update(kwargs)
            raise fin.FinalizeIntegrityError("stop before any real I/O")

        with mock.patch.object(fin, "finalize", side_effect=fake_finalize):
            with mock.patch.object(sys, "argv", ["finalize_northeast_coordinates.py"]):
                rc = fin.main()
        self.assertEqual(rc, 1)
        self.assertEqual(captured.get("expected_source_sha256"), fin.EXPECTED_SOURCE_MANIFEST_SHA256)
        self.assertEqual(captured.get("expected_capture_sha256"), fin.EXPECTED_CAPTURE_SHA256)

    @unittest.skipUnless(fin.DEFAULT_SOURCE.exists() and fin.DEFAULT_CAPTURE.exists(),
                         "real Northeast source manifest / capture not present in this checkout")
    def test_main_fails_closed_when_source_hash_constant_is_tampered(self):
        with mock.patch.object(fin, "EXPECTED_SOURCE_MANIFEST_SHA256", "0" * 64):
            with mock.patch.object(sys, "argv", ["finalize_northeast_coordinates.py"]):
                rc = fin.main()
        self.assertEqual(rc, 1)

    @unittest.skipUnless(fin.DEFAULT_SOURCE.exists() and fin.DEFAULT_CAPTURE.exists(),
                         "real Northeast source manifest / capture not present in this checkout")
    def test_main_fails_closed_when_capture_hash_constant_is_tampered(self):
        with mock.patch.object(fin, "EXPECTED_CAPTURE_SHA256", "0" * 64):
            with mock.patch.object(sys, "argv", ["finalize_northeast_coordinates.py"]):
                rc = fin.main()
        self.assertEqual(rc, 1)


class TestWriteAtomicAndVerification(unittest.TestCase):
    def test_overwrite_refusal_when_not_byte_identical(self):
        with tempfile.TemporaryDirectory() as td:
            fx = Fixture(Path(td))
            result = fx.finalize()
            out = Path(td) / "out.json"
            out.write_bytes(b"pre-existing different content")
            with self.assertRaises(fin.FinalizeIntegrityError) as cm:
                fin.write_atomic(result, out)
            self.assertIn("REFUSING to overwrite", str(cm.exception))
            self.assertEqual(out.read_bytes(), b"pre-existing different content")

    def test_no_op_when_already_byte_identical(self):
        with tempfile.TemporaryDirectory() as td:
            fx = Fixture(Path(td))
            result = fx.finalize()
            out = Path(td) / "out.json"
            fin.write_atomic(result, out)
            mtime_before = out.stat().st_mtime_ns
            fin.write_atomic(result, out)  # should be a silent no-op, not an error
            self.assertEqual(out.read_bytes(), fin.serialize(result))

    def test_atomic_failure_leaves_target_untouched(self):
        with tempfile.TemporaryDirectory() as td:
            fx = Fixture(Path(td))
            result = fx.finalize()
            out = Path(td) / "out.json"
            with mock.patch.object(fin.os, "replace", side_effect=OSError("simulated failure")):
                with self.assertRaises(OSError):
                    fin.write_atomic(result, out)
            self.assertFalse(out.exists())
            # no leftover temp file either
            leftovers = list(Path(td).glob("out.json.tmp*"))
            self.assertEqual(leftovers, [])

    def test_byte_identical_verification_mode(self):
        with tempfile.TemporaryDirectory() as td:
            fx = Fixture(Path(td))
            result = fx.finalize()
            out = Path(td) / "out.json"
            fin.write_atomic(result, out)
            # re-finalize and confirm serialization matches the written file byte-for-byte
            result_again = fx.finalize()
            self.assertEqual(fin.serialize(result_again), out.read_bytes())


@unittest.skipUnless(fin.DEFAULT_SOURCE.exists() and fin.DEFAULT_CAPTURE.exists(),
                     "real Northeast source manifest / capture not present in this checkout")
class TestRealCheckMode(unittest.TestCase):
    def test_check_mode_matches_frozen_sidecar(self):
        result = fin.finalize()
        self.assertEqual(result["coverage"]["rows_with_coordinate"], 3600)
        self.assertTrue(fin.DEFAULT_OUTPUT.exists())
        self.assertEqual(fin.serialize(result), fin.DEFAULT_OUTPUT.read_bytes())


if __name__ == "__main__":
    unittest.main()
