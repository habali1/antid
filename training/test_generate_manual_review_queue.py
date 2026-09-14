#!/usr/bin/env python3
"""test_generate_manual_review_queue.py — offline/synthetic tests for
generate_manual_review_queue.py.

Every test builds its own tiny synthetic pair of source manifests under a
temporary directory -- never reads the real data/ directory, and never
opens an image (there is no image in these fixtures at all, only CSV
files).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import manual_review_contract as mrc  # noqa: E402
import generate_manual_review_queue as gq  # noqa: E402

FIELDS = ("species", "slug", "taxon_id", "split", "observation_uuid", "photo_id", "sha256")


def _write_csv(path: Path, rows: list[dict]) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path.read_bytes()


def _deterministic_digest(*parts: str) -> int:
    """A REAL sha256 digest (never Python's built-in hash(), which is
    randomized per process via PYTHONHASHSEED and would make fixture
    photo_ids/sha256s -- and therefore duplicate-collision behavior --
    flaky across runs)."""
    return int(hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest(), 16)


def _final_test_row(i: int, slug: str = "species-x") -> dict:
    return {
        "species": "Species X", "slug": slug, "taxon_id": "1000",
        "split": "final_test", "observation_uuid": f"ft-uuid-{i}",
        "photo_id": str(9000 + i),
        "sha256": f"{_deterministic_digest('final_test', slug, str(i)) % (1 << 256):064x}",
    }


def _expansion_row(slug: str, split: str, i: int) -> dict:
    # photo_id must be globally unique across the WHOLE fixture (not just
    # within one slug/split bucket) -- offset it by a stable digest of
    # (slug, split) so different buckets never collide, deterministically
    # across every process/run (no dependence on hash randomization).
    bucket_offset = _deterministic_digest(slug, split) % 100000
    return {
        "species": slug.replace("-", " ").title(), "slug": slug, "taxon_id": "2000",
        "split": split, "observation_uuid": f"{slug}-{split}-uuid-{i}",
        "photo_id": str(2_000_000 + bucket_offset * 100 + i),
        "sha256": f"{_deterministic_digest(slug, split, str(i)) % (1 << 256):064x}",
    }


TEST_SLUGS = ("test-species-0", "test-species-1")


def build_tiny_source_manifests(repo: Path, *, n_final_test: int = 6, slugs=TEST_SLUGS,
                                per_bucket: int = 6) -> tuple[bytes, bytes]:
    final_test_rows = [_final_test_row(i) for i in range(n_final_test)]
    ft_bytes = _write_csv(repo / mrc.FINAL_TEST_MANIFEST_REL_PATH, final_test_rows)

    expansion_rows = []
    for slug in slugs:
        for split in mrc.EXPANSION_SPLITS:
            for i in range(per_bucket):
                expansion_rows.append(_expansion_row(slug, split, i))
    exp_bytes = _write_csv(repo / mrc.EXPANSION_MANIFEST_REL_PATH, expansion_rows)
    return ft_bytes, exp_bytes


def patched_constants(n_final_test: int, slugs, per_slug: int):
    n_species = len(slugs)
    n_sampled = n_species * per_slug * len(mrc.EXPANSION_SPLITS)
    n_total = n_final_test + n_sampled
    rows_per_session = n_total // 2
    return mock.patch.multiple(
        mrc,
        APPROVED_SPECIES_SLUGS=tuple(slugs),
        N_NEW_SPECIES=n_species, PER_SLUG_PER_SPLIT=per_slug,
        N_SAMPLED_ROWS=n_sampled, N_QUEUE_ROWS=n_total, ROWS_PER_SUGGESTED_SESSION=rows_per_session,
    )


def patched_frozen_manifests(ft_bytes: bytes, ft_rows: int, exp_bytes: bytes, exp_rows: int):
    frozen = {
        mrc.FINAL_TEST_DATASET: {"path": mrc.FINAL_TEST_MANIFEST_REL_PATH,
                                 "byte_sha256": mrc.sha256_bytes(ft_bytes), "row_count": ft_rows},
        mrc.EXPANSION_DATASET: {"path": mrc.EXPANSION_MANIFEST_REL_PATH,
                                "byte_sha256": mrc.sha256_bytes(exp_bytes), "row_count": exp_rows},
    }
    return mock.patch.object(mrc, "FROZEN_SOURCE_MANIFESTS", frozen)


def _args(repo: Path, **overrides):
    d = dict(repo=repo)
    d.update(overrides)
    return argparse.Namespace(**d)


class _FixtureTestCase(unittest.TestCase):
    """Common setup: tiny synthetic manifests + patched constants +
    patched frozen-manifest bindings, all torn down automatically."""

    N_FINAL_TEST = 6
    SLUGS = TEST_SLUGS
    PER_BUCKET = 6
    PER_SLUG = 3

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.repo = Path(self._tmpdir.name)
        ft_bytes, exp_bytes = build_tiny_source_manifests(
            self.repo, n_final_test=self.N_FINAL_TEST, slugs=self.SLUGS, per_bucket=self.PER_BUCKET)
        n_exp_rows = len(self.SLUGS) * len(mrc.EXPANSION_SPLITS) * self.PER_BUCKET
        self.const_patcher = patched_constants(self.N_FINAL_TEST, self.SLUGS, self.PER_SLUG)
        self.const_patcher.start()
        self.addCleanup(self.const_patcher.stop)
        self.frozen_patcher = patched_frozen_manifests(ft_bytes, self.N_FINAL_TEST, exp_bytes, n_exp_rows)
        self.frozen_patcher.start()
        self.addCleanup(self.frozen_patcher.stop)

    def tearDown(self):
        self._tmpdir.cleanup()


class TestQueueComposition(_FixtureTestCase):
    def test_exact_composition_counts(self):
        pre = gq.run_preflight(_args(self.repo))
        rows = pre["rows"]
        self.assertEqual(len(rows), 6 + 2 * 3 * 2)  # 6 final_test + 12 sampled = 18
        n_ft = sum(1 for r in rows if r["dataset"] == mrc.FINAL_TEST_DATASET)
        n_train = sum(1 for r in rows if r["dataset"] == mrc.EXPANSION_DATASET and r["split"] == "train")
        n_dev = sum(1 for r in rows if r["dataset"] == mrc.EXPANSION_DATASET and r["split"] == "development")
        self.assertEqual(n_ft, 6)
        self.assertEqual(n_train, 6)
        self.assertEqual(n_dev, 6)

    def test_exact_five_per_species_when_using_five_per_slug(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            slugs = ("sp-a", "sp-b", "sp-c")
            ft_bytes, exp_bytes = build_tiny_source_manifests(repo, n_final_test=4, slugs=slugs, per_bucket=8)
            with patched_constants(4, slugs, 5), patched_frozen_manifests(ft_bytes, 4, exp_bytes, 3 * 2 * 8):
                pre = gq.run_preflight(_args(repo))
        rows = pre["rows"]
        from collections import Counter
        per_bucket_counts = Counter(
            (r["slug"], r["split"]) for r in rows if r["dataset"] == mrc.EXPANSION_DATASET
        )
        self.assertEqual(set(per_bucket_counts.values()), {5})
        self.assertEqual(len(per_bucket_counts), 3 * 2)

    def test_session_counts_split_evenly(self):
        pre = gq.run_preflight(_args(self.repo))
        rows = pre["rows"]
        n1 = sum(1 for r in rows if r["suggested_session"] == 1)
        n2 = sum(1 for r in rows if r["suggested_session"] == 2)
        self.assertEqual(n1, len(rows) // 2)
        self.assertEqual(n2, len(rows) - len(rows) // 2)


class TestDeterminism(_FixtureTestCase):
    def test_byte_identical_regeneration(self):
        pre1 = gq.run_preflight(_args(self.repo))
        pre2 = gq.run_preflight(_args(self.repo))
        self.assertEqual(pre1["queue_csv_bytes"], pre2["queue_csv_bytes"])
        self.assertEqual(pre1["summary"], pre2["summary"])

    def test_write_then_check_round_trip(self):
        args = _args(self.repo)
        gq.cmd_write(args)
        result = gq.cmd_check(args)
        self.assertTrue(result["queue_path"].exists())

    def test_check_fails_closed_when_missing(self):
        with self.assertRaises(gq.QueueGenerationError):
            gq.cmd_check(_args(self.repo))

    def test_write_refuses_to_overwrite_queue_csv(self):
        args = _args(self.repo)
        gq.cmd_write(args)
        with self.assertRaises(gq.QueueGenerationError):
            gq.cmd_write(args)

    def test_check_compares_raw_bytes_not_semantic_json(self):
        args = _args(self.repo)
        gq.cmd_write(args)
        summary_path = self.repo / gq.SUMMARY_JSON_REL_PATH
        original = summary_path.read_bytes()
        # Reformat with different (but semantically-equal) JSON whitespace/
        # key order -- must still be rejected, since --check promises BYTE
        # identity, not semantic equality.
        parsed = json.loads(original)
        reformatted = (json.dumps(parsed, indent=4, sort_keys=False) + "\n").encode("utf-8")
        self.assertNotEqual(reformatted, original)
        summary_path.write_bytes(reformatted)
        with self.assertRaises(gq.QueueGenerationError):
            gq.cmd_check(args)

    def test_write_writes_zero_extra_files_and_check_writes_zero_bytes(self):
        args = _args(self.repo)
        gq.cmd_write(args)
        queue_path = self.repo / gq.QUEUE_CSV_REL_PATH
        summary_path = self.repo / gq.SUMMARY_JSON_REL_PATH
        before_queue = queue_path.read_bytes()
        before_summary = summary_path.read_bytes()
        gq.cmd_check(args)
        self.assertEqual(queue_path.read_bytes(), before_queue)
        self.assertEqual(summary_path.read_bytes(), before_summary)
        leftovers = list((self.repo / "training").glob("**/*.tmp*"))
        self.assertEqual(leftovers, [])


class TestPairedPublicationFailureInjection(_FixtureTestCase):
    def test_second_publish_failure_rolls_back_the_first(self):
        args = _args(self.repo)
        queue_path = self.repo / gq.QUEUE_CSV_REL_PATH
        summary_path = self.repo / gq.SUMMARY_JSON_REL_PATH

        real_link = gq.os.link
        call_count = {"n": 0}

        def flaky_link(src, dst):
            call_count["n"] += 1
            if call_count["n"] == 2:
                raise OSError("injected failure publishing the second file")
            return real_link(src, dst)

        with mock.patch.object(gq.os, "link", side_effect=flaky_link):
            with self.assertRaises(OSError):
                gq.cmd_write(args)

        self.assertFalse(queue_path.exists(), "queue CSV must be rolled back when the paired summary publish fails")
        self.assertFalse(summary_path.exists())
        leftovers = list((self.repo / "training").glob("**/*.tmp*"))
        self.assertEqual(leftovers, [], "no temp file may remain after a failed paired publish")

        # a later, un-injected write remains possible
        result = gq.cmd_write(args)
        self.assertTrue(result["queue_path"].exists())
        self.assertTrue(result["summary_path"].exists())

    def test_pre_existing_summary_refuses_and_rolls_back_queue(self):
        args = _args(self.repo)
        summary_path = self.repo / gq.SUMMARY_JSON_REL_PATH
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_bytes(b"pre-existing, not ours")

        with self.assertRaises(gq.QueueGenerationError):
            gq.cmd_write(args)

        queue_path = self.repo / gq.QUEUE_CSV_REL_PATH
        self.assertFalse(queue_path.exists(), "queue CSV must be rolled back when the summary already exists")
        self.assertEqual(summary_path.read_bytes(), b"pre-existing, not ours", "pre-existing file must be untouched")


class TestSourceManifestMismatchRejection(_FixtureTestCase):
    def test_check_fails_when_source_manifest_changes_after_write(self):
        args = _args(self.repo)
        gq.cmd_write(args)
        ft_path = self.repo / mrc.FINAL_TEST_MANIFEST_REL_PATH
        rows = list(csv.DictReader(ft_path.read_text(encoding="utf-8").splitlines()))
        rows[0]["observation_uuid"] = "mutated-uuid"
        new_bytes = _write_csv(ft_path, rows)
        # the frozen binding still expects the OLD hash -- this now
        # mismatches, which is exactly the rejection under test.
        with self.assertRaises(gq.QueueGenerationError):
            gq.cmd_check(args)

    def test_final_test_manifest_hash_mismatch_rejected(self):
        ft_path = self.repo / mrc.FINAL_TEST_MANIFEST_REL_PATH
        ft_path.write_bytes(ft_path.read_bytes() + b"\n")  # trailing garbage changes the byte hash
        with self.assertRaises(gq.QueueGenerationError) as ctx:
            gq.run_preflight(_args(self.repo))
        self.assertIn("frozen", str(ctx.exception))

    def test_final_test_row_count_mismatch_rejected(self):
        rows = [_final_test_row(i) for i in range(self.N_FINAL_TEST + 1)]
        _write_csv(self.repo / mrc.FINAL_TEST_MANIFEST_REL_PATH, rows)
        with self.assertRaises(gq.QueueGenerationError):
            gq.run_preflight(_args(self.repo))

    def test_final_test_wrong_split_rejected(self):
        rows = [_final_test_row(i) for i in range(self.N_FINAL_TEST)]
        rows[0]["split"] = "train"
        new_bytes = _write_csv(self.repo / mrc.FINAL_TEST_MANIFEST_REL_PATH, rows)
        with mock.patch.dict(mrc.FROZEN_SOURCE_MANIFESTS[mrc.FINAL_TEST_DATASET],
                             {"byte_sha256": mrc.sha256_bytes(new_bytes)}):
            with self.assertRaises(gq.QueueGenerationError):
                gq.run_preflight(_args(self.repo))

    def test_wrong_slug_set_rejected(self):
        # patched APPROVED_SPECIES_SLUGS = TEST_SLUGS but manifest uses a
        # DIFFERENT (still 2-slug) set entirely.
        wrong_slugs = ("totally-different-a", "totally-different-b")
        expansion_rows = []
        for slug in wrong_slugs:
            for split in mrc.EXPANSION_SPLITS:
                for i in range(6):
                    expansion_rows.append(_expansion_row(slug, split, i))
        new_bytes = _write_csv(self.repo / mrc.EXPANSION_MANIFEST_REL_PATH, expansion_rows)
        with mock.patch.dict(mrc.FROZEN_SOURCE_MANIFESTS[mrc.EXPANSION_DATASET],
                             {"byte_sha256": mrc.sha256_bytes(new_bytes)}):
            with self.assertRaises(gq.QueueGenerationError) as ctx:
                gq.run_preflight(_args(self.repo))
            self.assertIn("approved species slugs", str(ctx.exception))

    def test_bucket_with_too_few_rows_rejected(self):
        expansion_rows = []
        for slug in self.SLUGS:
            for split in mrc.EXPANSION_SPLITS:
                n = 2 if (slug == self.SLUGS[0] and split == "train") else 6
                for i in range(n):
                    expansion_rows.append(_expansion_row(slug, split, i))
        new_bytes = _write_csv(self.repo / mrc.EXPANSION_MANIFEST_REL_PATH, expansion_rows)
        with mock.patch.dict(mrc.FROZEN_SOURCE_MANIFESTS[mrc.EXPANSION_DATASET],
                             {"byte_sha256": mrc.sha256_bytes(new_bytes),
                             "row_count": len(expansion_rows)}):
            with self.assertRaises(gq.QueueGenerationError) as ctx:
                gq.run_preflight(_args(self.repo))
            self.assertIn("fewer than the required", str(ctx.exception))

    def test_missing_source_manifest_rejected(self):
        (self.repo / mrc.FINAL_TEST_MANIFEST_REL_PATH).unlink()
        with self.assertRaises(gq.QueueGenerationError):
            gq.run_preflight(_args(self.repo))

    def test_missing_required_column_rejected(self):
        path = self.repo / mrc.FINAL_TEST_MANIFEST_REL_PATH
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        header = lines[0].split(",")
        sha_idx = header.index("sha256")
        new_lines = []
        for line in lines:
            parts = line.split(",")
            del parts[sha_idx]
            new_lines.append(",".join(parts))
        new_bytes = ("\n".join(new_lines) + "\n").encode("utf-8")
        path.write_bytes(new_bytes)
        with mock.patch.dict(mrc.FROZEN_SOURCE_MANIFESTS[mrc.FINAL_TEST_DATASET],
                             {"byte_sha256": mrc.sha256_bytes(new_bytes)}):
            with self.assertRaises(gq.QueueGenerationError) as ctx:
                gq.run_preflight(_args(self.repo))
            self.assertIn("missing required column", str(ctx.exception))


class TestDuplicateAndMissingIdentityRejection(_FixtureTestCase):
    def test_blank_observation_uuid_rejected(self):
        path = self.repo / mrc.FINAL_TEST_MANIFEST_REL_PATH
        rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
        rows[0]["observation_uuid"] = ""
        new_bytes = _write_csv(path, rows)
        with mock.patch.dict(mrc.FROZEN_SOURCE_MANIFESTS[mrc.FINAL_TEST_DATASET],
                             {"byte_sha256": mrc.sha256_bytes(new_bytes)}):
            with self.assertRaises(gq.QueueGenerationError) as ctx:
                gq.run_preflight(_args(self.repo))
            self.assertIn("missing/blank", str(ctx.exception))

    def test_non_numeric_photo_id_rejected(self):
        path = self.repo / mrc.FINAL_TEST_MANIFEST_REL_PATH
        rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
        rows[0]["photo_id"] = "not-a-number"
        new_bytes = _write_csv(path, rows)
        with mock.patch.dict(mrc.FROZEN_SOURCE_MANIFESTS[mrc.FINAL_TEST_DATASET],
                             {"byte_sha256": mrc.sha256_bytes(new_bytes)}):
            with self.assertRaises(gq.QueueGenerationError) as ctx:
                gq.run_preflight(_args(self.repo))
            self.assertIn("non-numeric photo_id", str(ctx.exception))

    def test_duplicate_photo_id_across_datasets_rejected(self):
        # Two final_test rows (BOTH unconditionally included -- unlike
        # expansion rows, whose selection is digest-based and not
        # predictable from the test) share the same numeric photo_id.
        ft_path = self.repo / mrc.FINAL_TEST_MANIFEST_REL_PATH
        rows = list(csv.DictReader(ft_path.read_text(encoding="utf-8").splitlines()))
        rows[1]["photo_id"] = rows[0]["photo_id"]
        new_bytes = _write_csv(ft_path, rows)
        with mock.patch.dict(mrc.FROZEN_SOURCE_MANIFESTS[mrc.FINAL_TEST_DATASET],
                             {"byte_sha256": mrc.sha256_bytes(new_bytes)}):
            with self.assertRaises(gq.QueueGenerationError) as ctx:
                gq.run_preflight(_args(self.repo))
            self.assertIn("duplicate photo_id", str(ctx.exception))

    def test_duplicate_observation_uuid_across_datasets_rejected(self):
        ft_path = self.repo / mrc.FINAL_TEST_MANIFEST_REL_PATH
        rows = list(csv.DictReader(ft_path.read_text(encoding="utf-8").splitlines()))
        rows[1]["observation_uuid"] = rows[0]["observation_uuid"]
        new_bytes = _write_csv(ft_path, rows)
        with mock.patch.dict(mrc.FROZEN_SOURCE_MANIFESTS[mrc.FINAL_TEST_DATASET],
                             {"byte_sha256": mrc.sha256_bytes(new_bytes)}):
            with self.assertRaises(gq.QueueGenerationError) as ctx:
                gq.run_preflight(_args(self.repo))
            self.assertIn("duplicate observation_uuid", str(ctx.exception))

    def test_duplicate_sha256_across_datasets_rejected(self):
        ft_path = self.repo / mrc.FINAL_TEST_MANIFEST_REL_PATH
        rows = list(csv.DictReader(ft_path.read_text(encoding="utf-8").splitlines()))
        rows[1]["sha256"] = rows[0]["sha256"]
        new_bytes = _write_csv(ft_path, rows)
        with mock.patch.dict(mrc.FROZEN_SOURCE_MANIFESTS[mrc.FINAL_TEST_DATASET],
                             {"byte_sha256": mrc.sha256_bytes(new_bytes)}):
            with self.assertRaises(gq.QueueGenerationError) as ctx:
                gq.run_preflight(_args(self.repo))
            self.assertIn("duplicate image sha256", str(ctx.exception))

    def test_duplicate_identity_within_a_bucket_rejected(self):
        expansion_rows = []
        for slug in self.SLUGS:
            for split in mrc.EXPANSION_SPLITS:
                if slug == self.SLUGS[0] and split == "train":
                    same_row = _expansion_row(slug, split, 0)
                    expansion_rows.extend(dict(same_row) for _ in range(6))
                else:
                    for i in range(6):
                        expansion_rows.append(_expansion_row(slug, split, i))
        new_bytes = _write_csv(self.repo / mrc.EXPANSION_MANIFEST_REL_PATH, expansion_rows)
        with mock.patch.dict(mrc.FROZEN_SOURCE_MANIFESTS[mrc.EXPANSION_DATASET],
                             {"byte_sha256": mrc.sha256_bytes(new_bytes)}):
            with self.assertRaises(gq.QueueGenerationError) as ctx:
                gq.run_preflight(_args(self.repo))
            self.assertIn("duplicate", str(ctx.exception))


class TestNoRealImageAccess(_FixtureTestCase):
    def test_preflight_write_check_never_open_an_image_path(self):
        opened_paths = []
        real_open = Path.open

        def spying_open(self, *a, **kw):
            opened_paths.append(str(self))
            return real_open(self, *a, **kw)

        with mock.patch.object(Path, "open", spying_open):
            args = _args(self.repo)
            gq.run_preflight(args)
            gq.cmd_write(args)
            gq.cmd_check(args)

        image_like = [p for p in opened_paths if p.lower().endswith((".jpg", ".jpeg", ".png"))]
        self.assertEqual(image_like, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
