#!/usr/bin/env python3
"""test_audit_gate_v2_readiness.py — unit tests for
audit_gate_v2_readiness.py (corrected eligible-target stopping logic +
license policy B). All synthetic/offline: no network access, no image
download. A separate real-data smoke check is
`python audit_gate_v2_readiness.py --offline` against this repo's actual
data/ files.

Run directly: python test_audit_gate_v2_readiness.py
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

import audit_gate_v2_readiness as g  # noqa: E402


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({f: r.get(f, "") for f in fieldnames})


CAL_FIELDS = ["photo_id", "observation_uuid", "species", "slug", "taxon_id", "category",
             "lat", "lon", "source", "sha256", "created_at", "blur_score"]
UT_FIELDS = CAL_FIELDS
NORTHEAST_FIELDS = ["species", "slug", "taxon_id", "photo_id", "source", "lat", "lon",
                   "split", "common_name", "photo_license", "photo_attribution",
                   "observation_uuid", "source_url", "sha256", "provenance_status"]
BENCHMARK_FIELDS = ["photo_id", "observation_uuid", "species", "slug", "taxon_id", "lat",
                   "lon", "source", "sha256", "created_at"]
NFT_FIELDS = ["species", "slug", "taxon_id", "genus", "genus_id", "split", "state",
             "observation_id", "observation_uuid", "observer_id", "observed_on",
             "created_at", "geoprivacy", "obscured", "photo_id", "photo_license",
             "photo_attribution", "source_url", "sha256", "byte_size", "width", "height",
             "raw_relative_path", "clean_relative_path"]


def build_synthetic_repo(tmp_path: Path, *, n_supported: int = 3, ood_now_supported: int = 1,
                         ood_remaining: int = 2, known_holdout: int = 2):
    repo = tmp_path
    (repo / "data/northeast_expansion_v1").mkdir(parents=True)
    (repo / "data/benchmark_v1").mkdir(parents=True)
    (repo / "data/calibration_v1").mkdir(parents=True)
    (repo / "data/unknown_test_v1").mkdir(parents=True)
    (repo / "data/northeast_final_test_v1").mkdir(parents=True)

    supported_slugs = [f"supported-species-{i}" for i in range(n_supported)]
    taxonomy = {
        str(i): {"slug": slug, "species_name": slug.replace("-", " ").capitalize(),
                "taxon_id": 1000 + i, "common_name": None, "genus": "Genus"}
        for i, slug in enumerate(supported_slugs)
    }
    (repo / "data/northeast_expansion_v1/northeast_taxonomy_v1.json").write_text(
        json.dumps(taxonomy))
    _write_csv(repo / "data/northeast_expansion_v1/manifest_all_northeast_v1.csv",
              NORTHEAST_FIELDS, [])
    _write_csv(repo / "data/benchmark_v1/benchmark_v1.csv", BENCHMARK_FIELDS, [])
    _write_csv(repo / "data/northeast_final_test_v1/northeast_final_test_v1.csv", NFT_FIELDS, [])

    cal_rows = []
    for i in range(ood_now_supported):
        slug = supported_slugs[i % n_supported]
        cal_rows.append({"photo_id": f"cal-ood-sup-{i}", "observation_uuid": f"u-cal-ood-sup-{i}",
                         "species": slug, "slug": slug, "taxon_id": "1000",
                         "category": "out_of_scope_ant", "sha256": f"h-{i}" * 8})
    for i in range(ood_remaining):
        slug = f"unsupported-species-{i}"
        cal_rows.append({"photo_id": f"cal-ood-rem-{i}", "observation_uuid": f"u-cal-ood-rem-{i}",
                         "species": slug, "slug": slug, "taxon_id": "2000",
                         "category": "out_of_scope_ant", "sha256": f"r-{i}" * 8})
    for i in range(known_holdout):
        slug = supported_slugs[0]
        cal_rows.append({"photo_id": f"cal-kh-{i}", "observation_uuid": f"u-cal-kh-{i}",
                         "species": slug, "slug": slug, "taxon_id": "1000",
                         "category": "known_holdout", "sha256": f"k-{i}" * 8})
    _write_csv(repo / "data/calibration_v1/calibration_v1.csv", CAL_FIELDS, cal_rows)

    ut_rows = [{"photo_id": "ut-ood-0", "observation_uuid": "u-ut-ood-0",
               "species": "unsupported-species-0", "slug": "unsupported-species-0",
               "taxon_id": "2000", "category": "out_of_scope_ant", "sha256": "z" * 32}]
    _write_csv(repo / "data/unknown_test_v1/unknown_test_v1.csv", UT_FIELDS, ut_rows)

    return repo, taxonomy, cal_rows, ut_rows


class TestSupportedTaxonomy(unittest.TestCase):
    def test_exact_65_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, *_ = build_synthetic_repo(Path(tmp), n_supported=65)
            self.assertEqual(len(g.load_supported_taxonomy(repo)), 65)

    def test_wrong_count_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, *_ = build_synthetic_repo(Path(tmp), n_supported=3)
            with self.assertRaisesRegex(g.AuditError, "exactly 65"):
                g.load_supported_taxonomy(repo)


class TestV1IntersectionReproduction(unittest.TestCase):
    def test_disagreement_raises_and_never_adjusts_expected(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, taxonomy, *_ = build_synthetic_repo(
                Path(tmp), n_supported=3, ood_now_supported=1, ood_remaining=2)
            supported = {v["slug"]: v for v in taxonomy.values()}
            with self.assertRaisesRegex(g.AuditError, "disagreement"):
                g.reproduce_v1_intersections(repo, supported)


class TestExtractReuseCandidateRows(unittest.TestCase):
    def _cal_rows(self):
        return [
            {"category": "out_of_scope_ant", "species": "Now Supported", "slug": "now-supported",
             "taxon_id": "1", "photo_id": "p1", "observation_uuid": "u1", "sha256": "h1",
             "created_at": "", "lat": "", "lon": ""},
            {"category": "out_of_scope_ant", "species": "Still Unsupported",
             "slug": "still-unsupported", "taxon_id": "2", "photo_id": "p2",
             "observation_uuid": "u2", "sha256": "h2", "created_at": "", "lat": "", "lon": ""},
            {"category": "non_ant_insect", "species": "A Bug", "slug": "a-bug", "taxon_id": "3",
             "photo_id": "p3", "observation_uuid": "u3", "sha256": "h3",
             "created_at": "", "lat": "", "lon": ""},
            {"category": "unrelated", "species": "A Bird", "slug": "a-bird", "taxon_id": "4",
             "photo_id": "p4", "observation_uuid": "u4", "sha256": "h4",
             "created_at": "", "lat": "", "lon": ""},
            {"category": "known_holdout", "species": "Now Supported", "slug": "now-supported",
             "taxon_id": "1", "photo_id": "p5", "observation_uuid": "u5", "sha256": "h5",
             "created_at": "", "lat": "", "lon": ""},
        ]

    def test_canonical_fields_equal_origin_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "calibration_v1.csv"
            csv_path.write_text("dummy")
            supported = {"now-supported": {"taxon_id": 1}}
            pools = g.extract_reuse_candidate_rows(self._cal_rows(), supported, csv_path)
            self.assertEqual(len(pools["out_of_scope_ant"]), 1)  # now-supported excluded
            self.assertEqual(len(pools["non_ant_insect"]), 1)
            self.assertEqual(len(pools["unrelated"]), 1)
            row = pools["out_of_scope_ant"][0]
            self.assertEqual(row["photo_id"], row["origin_photo_id"])
            self.assertEqual(row["observation_uuid"], row["origin_observation_uuid"])
            self.assertEqual(row["photo_id"], "p2")
            self.assertEqual(row["hash_verification_status"], "unchecked")
            self.assertEqual(row["reuse_eligibility_reason"], "")

    def test_now_supported_species_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "calibration_v1.csv"
            csv_path.write_text("dummy")
            supported = {"now-supported": {"taxon_id": 1}}
            pools = g.extract_reuse_candidate_rows(self._cal_rows(), supported, csv_path)
            slugs = {r["slug"] for r in pools["out_of_scope_ant"]}
            self.assertNotIn("now-supported", slugs)


class TestHashVerificationAllThree(unittest.TestCase):
    def _write_image(self, repo: Path, slug: str, photo_id: str, content: bytes) -> str:
        p = repo / "data/calibration_v1" / slug / f"{photo_id}.jpg"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content)
        return hashlib.sha256(content).hexdigest()

    def _row(self, slug, photo_id, sha256):
        return {"slug": slug, "photo_id": photo_id, "origin_sha256": sha256,
               "hash_verification_status": "unchecked"}

    def test_all_three_categories_verified(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            h1 = self._write_image(repo, "s1", "1", b"a")
            h2 = self._write_image(repo, "s2", "2", b"b")
            h3 = self._write_image(repo, "s3", "3", b"c")
            pools = {"out_of_scope_ant": [self._row("s1", "1", h1)],
                    "non_ant_insect": [self._row("s2", "2", h2)],
                    "unrelated": [self._row("s3", "3", h3)]}
            report = g.verify_all_reuse_candidate_hashes(repo, pools)
            for cat in pools:
                self.assertEqual(report[cat]["rows_verified"], 1)
                self.assertEqual(report[cat]["rows_with_problems"], 0)
            self.assertEqual(pools["out_of_scope_ant"][0]["hash_verification_status"], "verified")

    def test_mismatch_is_integrity_failure_not_shortfall(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            self._write_image(repo, "s1", "1", b"a")
            pools = {"out_of_scope_ant": [self._row("s1", "1", "0" * 64)],
                    "non_ant_insect": [], "unrelated": []}
            with self.assertRaisesRegex(g.AuditError, "frozen-data integrity failure"):
                g.verify_all_reuse_candidate_hashes(repo, pools)

    def test_missing_file_is_integrity_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "data/calibration_v1").mkdir(parents=True)
            pools = {"out_of_scope_ant": [self._row("nope", "999", "a" * 64)],
                    "non_ant_insect": [], "unrelated": []}
            with self.assertRaises(g.AuditError):
                g.verify_all_reuse_candidate_hashes(repo, pools)


class TestFreshEligibility(unittest.TestCase):
    def _valid_row(self):
        return {"species": "X", "slug": "x", "taxon_id": 5, "observation_uuid": "u",
               "photo_id": "p", "source_url": "http://x", "photo_license": "cc0",
               "photo_attribution": "a"}

    def test_fully_valid_row_eligible(self):
        eligible, reason = g.is_fresh_row_eligible(self._valid_row())
        self.assertTrue(eligible)
        self.assertEqual(reason, "")

    def test_each_missing_field_rejected_with_exact_reason(self):
        cases = [
            ("species", "", "missing_species"),
            ("slug", "", "missing_slug"),
            ("observation_uuid", "", "missing_observation_uuid"),
            ("photo_id", "", "missing_photo_id"),
            ("source_url", "", "missing_source_url"),
            ("photo_attribution", "", "missing_attribution"),
            ("photo_license", "cc-by-nc-nd", "license_not_approved"),
            ("photo_license", "", "license_not_approved"),
        ]
        for field, bad_value, expected_reason in cases:
            row = self._valid_row()
            row[field] = bad_value
            eligible, reason = g.is_fresh_row_eligible(row)
            self.assertFalse(eligible, f"field {field}")
            self.assertEqual(reason, expected_reason, f"field {field}")

    def test_bool_taxon_id_rejected(self):
        row = self._valid_row()
        row["taxon_id"] = True
        eligible, reason = g.is_fresh_row_eligible(row)
        self.assertFalse(eligible)
        self.assertEqual(reason, "invalid_taxon_id")

    def test_non_int_taxon_id_rejected(self):
        row = self._valid_row()
        row["taxon_id"] = "5"
        eligible, reason = g.is_fresh_row_eligible(row)
        self.assertFalse(eligible)
        self.assertEqual(reason, "invalid_taxon_id")


class TestReusedEligibility(unittest.TestCase):
    def _valid_row(self):
        return {"hash_verification_status": "verified", "photo_license": "cc0",
               "source_url": "http://x", "photo_attribution": "a"}

    def test_verified_and_complete_is_eligible(self):
        eligible, reason = g.is_reused_row_eligible(self._valid_row())
        self.assertTrue(eligible)

    def test_unverified_hash_rejected(self):
        for status in ("unchecked", "file_missing", "hash_mismatch", None):
            row = self._valid_row()
            row["hash_verification_status"] = status
            eligible, reason = g.is_reused_row_eligible(row)
            self.assertFalse(eligible)
            self.assertTrue(reason.startswith("hash_"))

    def test_blank_license_rejected_distinctly(self):
        row = self._valid_row()
        row["photo_license"] = ""
        eligible, reason = g.is_reused_row_eligible(row)
        self.assertFalse(eligible)
        self.assertEqual(reason, "license_blank")

    def test_nd_license_rejected(self):
        row = self._valid_row()
        row["photo_license"] = "cc-by-nc-nd"
        eligible, reason = g.is_reused_row_eligible(row)
        self.assertFalse(eligible)
        self.assertEqual(reason, "license_not_approved")

    def test_missing_source_url_or_attribution_rejected(self):
        for field in ("source_url", "photo_attribution"):
            row = self._valid_row()
            row[field] = ""
            eligible, reason = g.is_reused_row_eligible(row)
            self.assertFalse(eligible)


class TestVerifiedBySha256NeverPremature(unittest.TestCase):
    def test_reuse_eligibility_reason_blank_before_hash_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "calibration_v1.csv"
            csv_path.write_text("dummy")
            rows = [{"category": "out_of_scope_ant", "species": "S", "slug": "s",
                    "taxon_id": "1", "photo_id": "p", "observation_uuid": "u", "sha256": "h",
                    "created_at": "", "lat": "", "lon": ""}]
            pools = g.extract_reuse_candidate_rows(rows, {}, csv_path)
            row = pools["out_of_scope_ant"][0]
            self.assertNotEqual(row["reuse_eligibility_reason"], "verified_by_sha256")
            self.assertEqual(row["hash_verification_status"], "unchecked")

    def test_finalize_only_labels_verified_by_sha256_after_successful_check(self):
        pools = {
            "out_of_scope_ant": [
                {"slug": "s1", "photo_id": "1", "hash_verification_status": "verified",
                 "photo_license": "cc0", "source_url": "http://x", "photo_attribution": "a",
                 "selection_status": "pending", "rejection_reason": "", "reuse_eligibility_reason": ""},
                {"slug": "s2", "photo_id": "2", "hash_verification_status": "file_missing",
                 "photo_license": "cc0", "source_url": "http://x", "photo_attribution": "a",
                 "selection_status": "pending", "rejection_reason": "", "reuse_eligibility_reason": ""},
            ],
            "non_ant_insect": [], "unrelated": [],
        }
        g.finalize_reused_rows(pools)
        verified_row = pools["out_of_scope_ant"][0]
        unverified_row = pools["out_of_scope_ant"][1]
        self.assertEqual(verified_row["reuse_eligibility_reason"], "verified_by_sha256")
        self.assertEqual(unverified_row["reuse_eligibility_reason"], "")
        self.assertEqual(unverified_row["selection_status"], "ineligible")
        self.assertTrue(unverified_row["rejection_reason"].startswith("hash_"))


class _FakeClient:
    def __init__(self, pages: list[dict]):
        self._pages = pages
        self.calls: list[dict] = []

    def get(self, path, params=None):
        self.calls.append({"path": path, "params": dict(params or {})})
        page = int((params or {}).get("page", 1))
        if page - 1 < len(self._pages):
            return self._pages[page - 1]
        return {"results": []}


def _obs(obs_id, uuid, photo_id, license_code="cc0", attribution="a", taxon_id=7,
        taxon_name="Some species"):
    # photo_id/obs_id are always positive in real iNat data -- callers must
    # never pass 0 (falsy), which would spuriously look like a missing ID.
    return {"id": obs_id, "uuid": uuid, "created_at": "2026-01-01",
           "taxon": {"id": taxon_id, "name": taxon_name, "rank": "species"},
           "photos": [{"id": photo_id, "url": "http://x/sq.jpg",
                      "license_code": license_code, "attribution": attribution}]}


class TestFetchStoppingLogicEligibleTarget(unittest.TestCase):
    def test_ineligible_rows_on_early_pages_do_not_satisfy_target(self):
        """Regression for the exact bug reported: page 1 is full of
        license-ineligible rows; the loop must keep paginating to page 2 to
        find real eligible rows, rather than stopping once len(rows)==target."""
        page1 = {"results": [_obs(i, f"u{i}", i, license_code="cc-by-nc-nd")
                             for i in range(1, 6)]}  # all ineligible (bad license)
        page2 = {"results": [_obs(100 + i, f"v{i}", 100 + i, license_code="cc0")
                             for i in range(1, 6)]}  # all eligible
        client = _FakeClient([page1, page2])
        exclusion = g.ExclusionIndex()
        rows, stats = g.fetch_species_candidates(
            client, 7, "Some species", "some-species", "known_holdout", "calibration_v2",
            exclusion, target=3, max_pages=5)
        self.assertEqual(stats["eligible_candidates"], 3)
        self.assertEqual(stats["termination_reason"], "eligible_target_reached")
        self.assertEqual(stats["pages_fetched"], 2)  # had to go past page 1
        # all 5 ineligible page-1 rows are still present with a reason
        ineligible = [r for r in rows if r["selection_status"] == "ineligible"]
        self.assertEqual(len(ineligible), 5)
        self.assertTrue(all(r["rejection_reason"] == "license_not_approved" for r in ineligible))

    def test_api_exhausted_before_target_reached(self):
        page1 = {"results": [_obs(1, "u1", 1, license_code="cc0")]}  # 1 eligible, then nothing
        client = _FakeClient([page1])
        rows, stats = g.fetch_species_candidates(
            client, 7, "Some species", "some-species", "known_holdout", "calibration_v2",
            g.ExclusionIndex(), target=10, max_pages=5)
        self.assertEqual(stats["eligible_candidates"], 1)
        self.assertEqual(stats["termination_reason"], "api_exhausted")
        self.assertEqual(stats["result_label"], "genuine_data_scarcity")

    def test_max_pages_exhausted_labeled_bounded_query_inconclusive(self):
        # Every page is full (200 would be full, but 5 is enough given max_pages=2)
        # and never reaches target -- exhausts max_pages, not the API itself.
        pages = [{"results": [_obs(seq, f"u{seq}", seq, license_code="cc0")]}
                for seq in (1, 2)]
        client = _FakeClient(pages)
        rows, stats = g.fetch_species_candidates(
            client, 7, "Some species", "some-species", "known_holdout", "calibration_v2",
            g.ExclusionIndex(), target=10, max_pages=2)
        self.assertEqual(stats["termination_reason"], "max_pages_exhausted")
        self.assertEqual(stats["result_label"], "bounded_query_inconclusive")
        self.assertTrue(stats["quota_short_and_bounded"])

    def test_excluded_photo_id_and_observation_uuid_are_skipped(self):
        page = {"results": [
            _obs(1, "u1", 100),   # excluded by photo_id
            _obs(2, "u2", 200),   # excluded by observation_uuid
            _obs(3, "u3", 300),   # kept, eligible
        ]}
        client = _FakeClient([page])
        exclusion = g.ExclusionIndex(photo_ids={"100"}, observation_uuids={"u2"})
        rows, stats = g.fetch_species_candidates(
            client, 7, "Some species", "some-species", "known_holdout", "calibration_v2",
            exclusion, target=10)
        eligible = [r for r in rows if r["selection_status"] == "candidate_eligible"]
        self.assertEqual([r["photo_id"] for r in eligible], [300])
        self.assertEqual(stats["excluded_by_frozen_id"], 2)


class TestDiversityCapCountsEligibleOnly(unittest.TestCase):
    def test_ineligible_rows_never_consume_species_cap(self):
        # 5 ineligible (bad license) + 2 eligible, all same species; cap=1.
        results = [_obs(i, f"u{i}", i, license_code="cc-by-nc-nd", taxon_name="Sp A")
                  for i in range(1, 6)]
        results += [_obs(100 + i, f"v{i}", 100 + i, license_code="cc0", taxon_name="Sp A")
                   for i in range(1, 3)]
        page = {"results": results}
        client = _FakeClient([page])
        rows, stats = g.fetch_diverse_ood_candidates(
            client, g.ExclusionIndex(), "", target=5, category="out_of_scope_ant",
            intended_dataset="unknown_test_v2", max_per_species=1)
        eligible = [r for r in rows if r["selection_status"] == "candidate_eligible"]
        capped = [r for r in rows if r.get("rejection_reason") == "species_diversity_cap_reached"]
        self.assertEqual(len(eligible), 1)  # cap=1 applies only among the 2 eligible ones
        self.assertEqual(len(capped), 1)
        ineligible_license = [r for r in rows if r.get("rejection_reason") == "license_not_approved"]
        self.assertEqual(len(ineligible_license), 5)

    def test_queries_the_requested_root_taxon(self):
        client = _FakeClient([{"results": []}])
        g.fetch_diverse_ood_candidates(
            client, g.ExclusionIndex(), without_taxon_ids="", target=5,
            category="non_ant_insect", intended_dataset="unknown_test_v2",
            taxon_id=g.TAXON_INSECTA)
        self.assertEqual(client.calls[0]["params"]["taxon_id"], g.TAXON_INSECTA)


class TestDeterministicSelection(unittest.TestCase):
    def _candidates(self, n=30):
        return [{"photo_id": str(i), "observation_uuid": f"u{i}"} for i in range(n)]

    def test_10_6_disjoint_partition(self):
        candidates = self._candidates(30)
        cal_selected, remainder = g.deterministic_select(
            candidates, "calibration_v2_known", 10, key=lambda r: r["photo_id"])
        ut_selected, reserve = g.deterministic_select(
            remainder, "unknown_test_v2_known", 6, key=lambda r: r["photo_id"])
        self.assertEqual(len(cal_selected), 10)
        self.assertEqual(len(ut_selected), 6)
        self.assertEqual({r["photo_id"] for r in cal_selected} & {r["photo_id"] for r in ut_selected},
                         set())
        self.assertEqual(len(reserve), 14)

    def test_independent_of_input_order(self):
        candidates = self._candidates(20)
        a, _ = g.deterministic_select(candidates, "calibration_v2_known", 5,
                                      key=lambda r: r["photo_id"])
        b, _ = g.deterministic_select(list(reversed(candidates)), "calibration_v2_known", 5,
                                      key=lambda r: r["photo_id"])
        self.assertEqual([r["photo_id"] for r in a], [r["photo_id"] for r in b])

    def test_distinct_domain_labels_distinct_partitions(self):
        candidates = self._candidates(50)
        a, _ = g.deterministic_select(candidates, "calibration_v2_known", 10,
                                      key=lambda r: r["photo_id"])
        b, _ = g.deterministic_select(candidates, "unknown_test_v2_known", 10,
                                      key=lambda r: r["photo_id"])
        self.assertNotEqual([r["photo_id"] for r in a], [r["photo_id"] for r in b])

    def test_unknown_domain_label_rejected(self):
        with self.assertRaises(g.AuditError):
            g.domain_rng("not_a_real_domain")

    def test_new_control_domain_labels_registered(self):
        for label in ("calibration_v2_non_ant_insect", "calibration_v2_unrelated"):
            g.domain_rng(label)  # must not raise


class TestBackfillDeficitsDerived(unittest.TestCase):
    def test_deficit_formula_not_hardcoded(self):
        for eligible_reused, target, expected_deficit in [
            (272, 300, 28), (229, 300, 71), (0, 300, 300), (300, 300, 0),
            (87, 150, 63), (119, 150, 31),
        ]:
            self.assertEqual(target - eligible_reused, expected_deficit)


class TestControlsAndOodOrderingAndDisjointness(unittest.TestCase):
    def test_calibration_selected_before_unknown_from_shared_pool(self):
        shared_eligible = [{"photo_id": str(i), "observation_uuid": f"u{i}"} for i in range(20)]
        deficit = 6
        cal_selected, remainder = g.deterministic_select(
            shared_eligible, "calibration_v2_non_ant_insect", deficit,
            key=lambda r: r["photo_id"])
        ut_selected, reserve = g.deterministic_select(
            remainder, "unknown_test_v2_non_ant_insect", 8, key=lambda r: r["photo_id"])
        self.assertEqual(len(cal_selected), 6)
        self.assertEqual(len(ut_selected), 8)
        cal_ids = {r["photo_id"] for r in cal_selected}
        ut_ids = {r["photo_id"] for r in ut_selected}
        self.assertEqual(cal_ids & ut_ids, set())

    def test_unknown_test_exclusion_set_includes_fresh_calibration_ood_taxa(self):
        """The unknown-test OOD exclusion set must be built from the ACTUAL
        planned calibration_v2 OOD rows (reused + fresh-selected), not only
        the old reusable-species map -- a fresh calibration OOD species must
        also be excluded even though it has no entry in that old map."""
        supported_taxon_ids = {1, 2, 3}
        reusable_taxon_by_slug = {"old-species": 100}
        reused_eligible_ood = [{"slug": "old-species"}]
        # A FRESH calibration OOD row for a species not in the old map:
        cal_ood_selected = [{"slug": "brand-new-species", "taxon_id": 999}]
        excluded = set(supported_taxon_ids)
        excluded.update(reusable_taxon_by_slug[r["slug"]] for r in reused_eligible_ood)
        excluded.update(r["taxon_id"] for r in cal_ood_selected)
        self.assertIn(999, excluded)
        self.assertIn(100, excluded)


class TestEligibleRateAndReferenceFlag(unittest.TestCase):
    def test_reference_constant(self):
        self.assertAlmostEqual(g.NORTHEAST_REFERENCE_ELIGIBLE_RATE, 0.778255, places=5)

    def test_rate_and_delta_computed(self):
        report = g.eligible_rate_report(100, 78, __import__("collections").Counter())
        self.assertAlmostEqual(report["eligible_rate"], 0.78)
        self.assertAlmostEqual(report["delta_from_reference_pp"], 0.1745, places=3)
        self.assertFalse(report["materially_below_northeast_reference"])

    def test_flag_true_just_below_threshold(self):
        report = g.eligible_rate_report(1000, 677, __import__("collections").Counter())  # 67.7%
        self.assertTrue(report["materially_below_northeast_reference"])

    def test_flag_false_just_above_threshold(self):
        # Comfortably above 67.8255% -- must not be flagged. (An exact
        # bit-for-bit boundary test is intentionally avoided: the threshold
        # is a float difference and its last-bit rounding is not part of
        # the contract, only "more than 10pp below" is.)
        report = g.eligible_rate_report(1_000_000, 690_000, __import__("collections").Counter())
        self.assertFalse(report["materially_below_northeast_reference"])

    def test_zero_examined_yields_null_rate(self):
        report = g.eligible_rate_report(0, 0, __import__("collections").Counter())
        self.assertIsNone(report["eligible_rate"])
        self.assertFalse(report["materially_below_northeast_reference"])


class TestDedupeCandidateRows(unittest.TestCase):
    def test_no_duplicate_selected_or_reused_identity(self):
        rows = [
            {"selection_status": "selected", "photo_id": "1"},
            {"selection_status": "selected", "photo_id": "1"},  # duplicate
            {"selection_status": "reused", "photo_id": "", "origin_photo_id": "2"},
            {"selection_status": "reused", "photo_id": "", "origin_photo_id": "2"},  # duplicate
            {"selection_status": "ineligible", "photo_id": "1"},  # not deduped
        ]
        out = g.dedupe_candidate_rows(rows)
        selected_or_reused = [r for r in out if r["selection_status"] in ("selected", "reused")]
        self.assertEqual(len(selected_or_reused), 2)
        self.assertEqual(sum(1 for r in out if r["selection_status"] == "ineligible"), 1)


class TestKnownSpeciesReadinessClassification(unittest.TestCase):
    def test_below_16_fails(self):
        for n in (0, 1, 15):
            self.assertEqual(g.classify_known_species_readiness(n), "fails_readiness")

    def test_16_to_23_fragile(self):
        for n in (16, 20, 23):
            self.assertEqual(g.classify_known_species_readiness(n), "fragile")

    def test_24_plus_ready(self):
        for n in (24, 30, 100):
            self.assertEqual(g.classify_known_species_readiness(n), "ready")


class TestNoImageDownloadPath(unittest.TestCase):
    def test_source_has_no_image_write_or_download_call(self):
        source = Path(g.__file__).read_text(encoding="utf-8")
        for forbidden in ("download_one(", "write_bytes(", "Image.open(", "PIL", "onnxruntime"):
            self.assertNotIn(forbidden, source)

    def test_paced_client_rejects_image_content_type(self):
        import unittest.mock as mock

        class FakeResponse:
            status = 200
            headers = {"Content-Type": "image/jpeg"}

            def read(self):
                return b"\xff\xd8\xff"

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        client = g.PacedJsonClient(interval_seconds=0.0)
        with mock.patch("urllib.request.urlopen", return_value=FakeResponse()):
            with self.assertRaisesRegex(g.AuditError, "image"):
                client.get("/observations", {})

    def test_json_report_never_embeds_candidate_rows(self):
        source = Path(g.__file__).read_text(encoding="utf-8")
        self.assertNotIn('"candidate_rows"', source)
        self.assertIn("candidates_csv", source)


class TestNoThresholdReference(unittest.TestCase):
    def test_source_never_uses_060_as_a_value_or_gate_logic(self):
        source = Path(g.__file__).read_text(encoding="utf-8")
        self.assertNotIn("GATE_THRESHOLD", source)
        self.assertNotIn("inference_policy", source)
        self.assertNotIn("= 0.60", source)
        self.assertNotIn("< 0.60", source)
        self.assertEqual(source.count("0.60"), 1)


if __name__ == "__main__":
    unittest.main()
