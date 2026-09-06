#!/usr/bin/env python3
from __future__ import annotations

import tempfile
import unittest
from argparse import Namespace
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch

from scrape_northeast_expansion import (
    MANIFEST_FIELDS,
    QUOTA,
    SEED,
    deterministic_key,
    group_in_run_order,
    inspect_image,
    run_restore,
    split_for_position,
    validate_source_rows,
)


def candidate(species: str, photo_id: int) -> dict[str, str]:
    return {
        "species": species,
        "taxon_id": "1",
        "genus": "Genus",
        "genus_id": "2",
        "observation_uuid": f"uuid-{photo_id}",
        "photo_id": str(photo_id),
        "photo_license": "cc-by-nc",
        "photo_attribution": "observer, CC BY-NC",
        "photo_url_medium": f"https://example.test/{photo_id}.jpg",
        "eligible_personal_nc": "true",
        "prior_overlap": "false",
        "internal_duplicate": "false",
    }


class NortheastExpansionTests(unittest.TestCase):
    def test_deterministic_order_is_stable(self) -> None:
        rows = [candidate("A a", number) for number in range(8)]
        first = sorted(rows, key=lambda row: deterministic_key(row, SEED))
        second = sorted(reversed(rows), key=lambda row: deterministic_key(row, SEED))
        self.assertEqual([r["photo_id"] for r in first], [r["photo_id"] for r in second])

    def test_species_run_order_is_ascending_reserve(self) -> None:
        thin = [candidate("Thin ant", number) for number in range(270)]
        thick = [candidate("Thick ant", 1000 + number) for number in range(275)]
        self.assertEqual([name for name, _ in group_in_run_order(thick + thin, SEED)], ["Thin ant", "Thick ant"])

    def test_split_boundaries(self) -> None:
        self.assertEqual(split_for_position(0), "train")
        self.assertEqual(split_for_position(QUOTA["train"] - 1), "train")
        self.assertEqual(split_for_position(QUOTA["train"]), "development")
        self.assertEqual(split_for_position(QUOTA["train"] + QUOTA["development"]), "final_test")
        with self.assertRaises(ValueError):
            split_for_position(sum(QUOTA.values()))

    def test_missing_provenance_fails_closed(self) -> None:
        row = candidate("A a", 1)
        row["photo_attribution"] = ""
        with self.assertRaisesRegex(RuntimeError, "photo_attribution"):
            validate_source_rows([row])

    def test_ineligible_rows_are_not_admitted(self) -> None:
        row = candidate("A a", 1)
        row["eligible_personal_nc"] = "false"
        self.assertEqual(validate_source_rows([row]), [])

    def test_manifest_has_required_provenance(self) -> None:
        for field in ("photo_license", "photo_attribution", "observation_uuid", "photo_id", "source_url", "sha256"):
            self.assertIn(field, MANIFEST_FIELDS)

    def test_decode_failure_is_classified(self) -> None:
        status, width, height, _ = inspect_image(b"not an image")
        self.assertEqual((status, width, height), ("decode_failure", None, None))

    def test_tiny_image_is_classified(self) -> None:
        from PIL import Image
        import io

        buffer = io.BytesIO()
        Image.new("RGB", (199, 300)).save(buffer, format="PNG")
        status, width, height, _ = inspect_image(buffer.getvalue())
        self.assertEqual((status, width, height), ("under_200px", 199, 300))

    def test_restore_fails_without_changing_manifest_when_source_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = root / "frozen.csv"
            row = {field: "" for field in MANIFEST_FIELDS}
            row.update({
                "species": "A a",
                "slug": "a-a",
                "split": "train",
                "observation_uuid": "missing-uuid",
                "photo_id": "1",
                "photo_license": "cc-by",
                "photo_attribution": "observer",
                "source_url": "https://example.test/1.jpg",
                "sha256": "0" * 64,
                "raw_relative_path": "raw/a-a/1.jpg",
                "clean_relative_path": "clean/a-a/1.jpg",
            })
            import csv

            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
                writer.writeheader()
                writer.writerow(row)
            before = manifest.read_bytes()
            args = Namespace(
                repo=root,
                manifest=manifest,
                out_root=root,
                request_interval=0,
            )
            with patch("scrape_northeast_expansion.restore_observations", return_value={}):
                self.assertEqual(run_restore(args), 2)
            self.assertEqual(manifest.read_bytes(), before)

    def test_restore_reuses_verified_raw_and_builds_separate_clean_copy(self) -> None:
        from PIL import Image
        import csv
        import io

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            buffer = io.BytesIO()
            Image.new("RGB", (220, 220)).save(buffer, format="JPEG")
            data = buffer.getvalue()
            digest = sha256(data).hexdigest()
            raw = root / "raw/a-a/1.jpg"
            raw.parent.mkdir(parents=True)
            raw.write_bytes(data)
            manifest = root / "frozen.csv"
            row = {field: "" for field in MANIFEST_FIELDS}
            row.update({
                "species": "A a",
                "slug": "a-a",
                "split": "train",
                "observation_uuid": "uuid-1",
                "photo_id": "1",
                "photo_license": "cc-by",
                "photo_attribution": "observer",
                "source_url": "https://example.test/1.jpg",
                "sha256": digest,
                "raw_relative_path": "raw/a-a/1.jpg",
                "clean_relative_path": "clean/a-a/1.jpg",
            })
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
                writer.writeheader()
                writer.writerow(row)
            before = manifest.read_bytes()
            args = Namespace(
                repo=root,
                manifest=manifest,
                out_root=root,
                request_interval=0,
            )
            with patch("scrape_northeast_expansion.restore_observations", return_value={}):
                self.assertEqual(run_restore(args), 0)
            self.assertEqual((root / "clean/a-a/1.jpg").read_bytes(), data)
            self.assertEqual(manifest.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
