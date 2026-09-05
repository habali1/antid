#!/usr/bin/env python3
"""Focused unit tests for the metadata-only Northeast readiness audit."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from audit_northeast_candidates import (
    CSV_FIELDS,
    Exclusions,
    FUTURE_DOWNLOAD_MANIFEST_REQUIRED_FIELDS,
    license_tier,
    quota_capacity,
    select_photo,
)


class ReadinessAuditTests(unittest.TestCase):
    def test_candidate_and_future_manifest_contract_require_license_provenance(self) -> None:
        for field in ("photo_license", "photo_attribution"):
            self.assertIn(field, CSV_FIELDS)
            self.assertIn(field, FUTURE_DOWNLOAD_MANIFEST_REQUIRED_FIELDS)

    def test_photo_selection_prefers_core_license_then_lowest_id(self) -> None:
        photos = [
            {"id": 1, "url": "https://x/1.jpg", "license_code": None},
            {"id": 3, "url": "https://x/3.jpg", "license_code": "cc-by"},
            {"id": 2, "url": "https://x/2.jpg", "license_code": "cc-by"},
        ]
        chosen, n_core, n_personal, reused = select_photo(photos)
        self.assertEqual(chosen["id"], 2)
        self.assertEqual(n_core, 2)
        self.assertEqual(n_personal, 2)
        self.assertFalse(reused)

    def test_photo_selection_keeps_noncommercial_separate(self) -> None:
        photos = [
            {"id": 1, "url": "https://x/1.jpg", "license_code": "cc-by-nc"},
            {"id": 2, "url": "https://x/2.jpg", "license_code": "cc-by-nc-nd"},
        ]
        chosen, n_core, n_personal, reused = select_photo(photos)
        self.assertEqual(chosen["id"], 1)
        self.assertEqual(n_core, 0)
        self.assertEqual(n_personal, 1)
        self.assertFalse(reused)
        self.assertEqual(license_tier(chosen["license_code"]), "personal_noncommercial")

    def test_photo_selection_avoids_a_reused_photo_when_possible(self) -> None:
        photos = [
            {"id": 1, "url": "https://x/1.jpg", "license_code": "cc-by"},
            {"id": 2, "url": "https://x/2.jpg", "license_code": "cc-by-nc"},
        ]
        chosen, n_core, n_personal, reused = select_photo(photos, {1})
        self.assertEqual(chosen["id"], 2)
        self.assertEqual(n_core, 0)
        self.assertEqual(n_personal, 1)
        self.assertFalse(reused)

    def test_photo_selection_marks_unavoidable_reuse(self) -> None:
        photos = [{"id": 1, "url": "https://x/1.jpg", "license_code": "cc-by"}]
        chosen, n_core, n_personal, reused = select_photo(photos, {1})
        self.assertEqual(chosen["id"], 1)
        self.assertEqual((n_core, n_personal), (0, 0))
        self.assertTrue(reused)

    def test_any_photo_or_observation_overlap_excludes_the_observation(self) -> None:
        exclusions = Exclusions()
        exclusions.photo_sources["benchmark_v1"] = {"22"}
        exclusions.observation_sources["calibration_v1"] = {"uuid-1"}
        self.assertEqual(
            exclusions.sources_for(["11", "22"], "uuid-1"),
            ["benchmark_v1", "calibration_v1"],
        )

    def test_quota_capacity_protects_final_and_development_capacity(self) -> None:
        result = quota_capacity(60)
        self.assertFalse(result["numerically_ready"])
        self.assertEqual(
            result["illustrative_capacity_only"],
            {"final_test": 30, "development": 30, "train": 0},
        )

    def test_module_does_not_create_files_on_import(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            self.assertEqual(list(Path(temp_dir).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
