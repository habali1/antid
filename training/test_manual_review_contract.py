#!/usr/bin/env python3
"""test_manual_review_contract.py — offline tests for manual_review_contract.py."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import manual_review_contract as mrc  # noqa: E402


class TestQueueComposition(unittest.TestCase):
    def test_totals_match_requirement(self):
        self.assertEqual(mrc.N_FINAL_TEST_ROWS, 450)
        self.assertEqual(mrc.N_NEW_SPECIES, 15)
        self.assertEqual(mrc.PER_SLUG_PER_SPLIT, 5)
        self.assertEqual(mrc.N_SAMPLED_ROWS, 150)
        self.assertEqual(mrc.N_QUEUE_ROWS, 600)
        self.assertEqual(mrc.SEED, 20260905)

    def test_session_split_is_300_300(self):
        self.assertEqual(mrc.ROWS_PER_SUGGESTED_SESSION, 300)
        self.assertEqual(mrc.N_QUEUE_ROWS - mrc.ROWS_PER_SUGGESTED_SESSION, 300)
        self.assertEqual(mrc.MAX_DECISIONS_PER_SESSION, 300)


class TestSelectionDigest(unittest.TestCase):
    def test_exact_payload_format(self):
        import hashlib
        payload = "\0".join(["20260905", "myslug", "train", "uuid-1", "42"])
        expected = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        actual = mrc.compute_selection_digest(20260905, "myslug", "train", "uuid-1", 42)
        self.assertEqual(actual, expected)

    def test_photo_id_coerced_numerically(self):
        a = mrc.compute_selection_digest(mrc.SEED, "s", "train", "u", "007")
        b = mrc.compute_selection_digest(mrc.SEED, "s", "train", "u", 7)
        self.assertEqual(a, b)

    def test_non_numeric_photo_id_raises(self):
        with self.assertRaises(ValueError):
            mrc.compute_selection_digest(mrc.SEED, "s", "train", "u", "not-a-number")

    def test_deterministic_repeat(self):
        a = mrc.compute_selection_digest(mrc.SEED, "s", "development", "u1", 5)
        b = mrc.compute_selection_digest(mrc.SEED, "s", "development", "u1", 5)
        self.assertEqual(a, b)

    def test_distinct_split_changes_digest(self):
        a = mrc.compute_selection_digest(mrc.SEED, "s", "train", "u1", 5)
        b = mrc.compute_selection_digest(mrc.SEED, "s", "development", "u1", 5)
        self.assertNotEqual(a, b)


class TestReviewOrderDigest(unittest.TestCase):
    def test_distinct_from_selection_digest(self):
        sel = mrc.compute_selection_digest(mrc.SEED, "s", "train", "u1", 5)
        order = mrc.compute_review_order_digest(mrc.SEED, "northeast_expansion_v1", "s", "train", "u1", 5)
        self.assertNotEqual(sel, order)

    def test_deterministic_repeat(self):
        a = mrc.compute_review_order_digest(mrc.SEED, "d", "s", "train", "u1", 5)
        b = mrc.compute_review_order_digest(mrc.SEED, "d", "s", "train", "u1", 5)
        self.assertEqual(a, b)

    def test_dataset_changes_digest(self):
        a = mrc.compute_review_order_digest(mrc.SEED, "dataset_a", "s", "train", "u1", 5)
        b = mrc.compute_review_order_digest(mrc.SEED, "dataset_b", "s", "train", "u1", 5)
        self.assertNotEqual(a, b)


class TestEnums(unittest.TestCase):
    def test_review_states_exact(self):
        self.assertEqual(mrc.REVIEW_STATES, frozenset({
            "usable", "poor_quality_usable", "unusable_no_visible_ant",
            "unusable_corrupt", "unusable_wrong_organism", "uncertain",
        }))

    def test_label_plausibility_exact(self):
        self.assertEqual(mrc.LABEL_PLAUSIBILITY_VALUES, frozenset({"plausible", "implausible", "uncertain"}))

    def test_duplicate_suspicion_exact(self):
        self.assertEqual(mrc.DUPLICATE_SUSPICION_VALUES, frozenset({"none", "suspected", "strong"}))

    def test_adjudication_labels_exact(self):
        self.assertEqual(mrc.ADJUDICATION_LABELS, frozenset({
            "same_source_image", "different_photo_same_observation", "different_image", "uncertain",
        }))

    def test_remediation_rule_verbatim(self):
        self.assertEqual(mrc.REMEDIATION_RULE, (
            "Findings from the manual review do not modify northeast_final_test_v1 under "
            "any circumstance. Unusable images (no visible ant, corrupted file, clearly "
            "different organism) are counted and reported alongside the final-test result "
            "as a known limitation. Poor-quality but usable images remain in scope by "
            "design: they reflect real user input, and the confidence gate exists precisely "
            "to handle them. No image is removed, replaced, or relabeled."
        ))


class TestPairId(unittest.TestCase):
    def test_order_independent(self):
        a = mrc.compute_pair_id("x", "y")
        b = mrc.compute_pair_id("y", "x")
        self.assertEqual(a, b)

    def test_distinct_pairs_differ(self):
        self.assertNotEqual(mrc.compute_pair_id("x", "y"), mrc.compute_pair_id("x", "z"))


class TestImageLayoutResolver(unittest.TestCase):
    """Item 1: the image directory layout is NOT uniform -- expansion/
    final_test go through a 'clean' subdirectory, the 5 other-evidence sets
    store images directly under {dataset}/{slug}/. Both cases must resolve
    correctly from the frozen per-domain-part layout mapping."""

    def test_every_domain_part_has_a_layout_entry(self):
        self.assertEqual(set(mrc.DOMAIN_PART_IMAGE_LAYOUTS), set(mrc.DOMAIN_PARTS))

    def test_expansion_and_final_test_use_clean_subdir(self):
        for part in (mrc.DOMAIN_PART_EXPANSION_TRAIN, mrc.DOMAIN_PART_EXPANSION_DEVELOPMENT,
                    mrc.DOMAIN_PART_FINAL_TEST):
            self.assertEqual(mrc.DOMAIN_PART_IMAGE_LAYOUTS[part]["subdir"], "clean")

    def test_other_evidence_sets_use_direct_layout(self):
        for part in mrc.OTHER_EVIDENCE_SETS:
            self.assertIsNone(mrc.DOMAIN_PART_IMAGE_LAYOUTS[part]["subdir"])

    def test_clean_layout_resolves_via_subdir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p = root / mrc.FINAL_TEST_DATASET / "clean" / "some-slug" / "42.jpg"
            p.parent.mkdir(parents=True)
            p.touch()
            resolved = mrc.resolve_image_path(root, mrc.DOMAIN_PART_IMAGE_LAYOUTS,
                                              mrc.DOMAIN_PART_FINAL_TEST, "some-slug", "42")
            self.assertEqual(resolved, p)

    def test_direct_layout_resolves_without_subdir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p = root / "benchmark_v1" / "some-slug" / "42.jpg"
            p.parent.mkdir(parents=True)
            p.touch()
            resolved = mrc.resolve_image_path(root, mrc.DOMAIN_PART_IMAGE_LAYOUTS,
                                              mrc.DOMAIN_PART_BENCHMARK_V1, "some-slug", "42")
            self.assertEqual(resolved, p)

    def test_direct_layout_does_not_accidentally_match_a_clean_subdir_file(self):
        # A stray file sitting under a "clean" subdirectory must NOT satisfy
        # a direct-layout (other-evidence) lookup -- proves the two layouts
        # are not silently interchangeable.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stray = root / "benchmark_v1" / "clean" / "some-slug" / "42.jpg"
            stray.parent.mkdir(parents=True)
            stray.touch()
            with self.assertRaises(mrc.ImageResolutionError):
                mrc.resolve_image_path(root, mrc.DOMAIN_PART_IMAGE_LAYOUTS,
                                       mrc.DOMAIN_PART_BENCHMARK_V1, "some-slug", "42")

    def test_unknown_domain_part_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(mrc.ImageResolutionError):
                mrc.resolve_image_path(Path(tmp), mrc.DOMAIN_PART_IMAGE_LAYOUTS, "bogus-part", "s", "1")

    def test_domain_part_for_dataset_and_split(self):
        self.assertEqual(mrc.domain_part_for_dataset_and_split(mrc.EXPANSION_DATASET, mrc.EXPANSION_TRAIN_SPLIT),
                         mrc.DOMAIN_PART_EXPANSION_TRAIN)
        self.assertEqual(mrc.domain_part_for_dataset_and_split(mrc.EXPANSION_DATASET, mrc.EXPANSION_DEV_SPLIT),
                         mrc.DOMAIN_PART_EXPANSION_DEVELOPMENT)
        self.assertEqual(mrc.domain_part_for_dataset_and_split(mrc.FINAL_TEST_DATASET, "final_test"),
                         mrc.DOMAIN_PART_FINAL_TEST)
        with self.assertRaises(mrc.ImageResolutionError):
            mrc.domain_part_for_dataset_and_split("unknown-dataset", "train")


class TestPerceptualHashThresholds(unittest.TestCase):
    def test_thresholds_match_requirement(self):
        self.assertEqual(mrc.PHASH_MAX_HAMMING_DISTANCE, 10)
        self.assertEqual(mrc.DHASH_MAX_HAMMING_DISTANCE, 8)
        self.assertEqual(mrc.PHASH_RESIZE, (32, 32))
        self.assertEqual(mrc.PHASH_KEEP, 8)
        self.assertEqual(mrc.DHASH_RESIZE, (9, 8))
        self.assertEqual(mrc.N_DIHEDRAL_ORIENTATIONS, 8)


class TestContractContentBuild(unittest.TestCase):
    ARTIFACTS = dict(
        queue_artifact={"path": mrc.QUEUE_ARTIFACT_REL_PATH, "byte_sha256": "a" * 64,
                       "row_count": mrc.N_QUEUE_ROWS, "identity_order_sha256": "b" * 64},
        summary_artifact={"path": mrc.SUMMARY_ARTIFACT_REL_PATH, "byte_sha256": "c" * 64,
                         "content_sha256": "d" * 64},
        implementation_sources={"x": {"path": "training/x.py", "sha256": "e" * 64}},
    )

    def test_build_contract_content_is_pure_and_deterministic(self):
        a = mrc.build_contract_content(**self.ARTIFACTS)
        b = mrc.build_contract_content(**self.ARTIFACTS)
        self.assertEqual(a, b)

    def test_contract_content_is_json_serializable(self):
        import json
        content = mrc.build_contract_content(**self.ARTIFACTS)
        json.loads(json.dumps(content, sort_keys=True))

    def test_contract_includes_remediation_rule_verbatim(self):
        content = mrc.build_contract_content(**self.ARTIFACTS)
        self.assertEqual(content["remediation_rule"], mrc.REMEDIATION_RULE)

    def test_comparison_domains_cover_36_canonical_pairs(self):
        content = mrc.build_contract_content(**self.ARTIFACTS)
        names = {d["name"] for d in content["comparison_domains"]}
        self.assertEqual(len(names), 36)
        self.assertIn("expansion_development_vs_expansion_train", names)
        self.assertIn("expansion_train_vs_final_test", names)
        self.assertIn("final_test_vs_unknown_test_v1", names)
        self.assertIn("within_final_test", names)
        # A/B and B/A never both appear
        self.assertNotIn("expansion_development_vs_expansion_train".replace("_vs_", "_VS_"), names)

    def test_stop_before_inference_domains(self):
        self.assertEqual(mrc.STOP_BEFORE_INFERENCE_DOMAINS, frozenset({
            "benchmark_v1_vs_final_test", "calibration_v1_vs_final_test", "calibration_v2_vs_final_test",
            "expansion_development_vs_final_test", "expansion_train_vs_final_test",
            "final_test_vs_unknown_test_v1", "final_test_vs_unknown_test_v2",
        }))

    def test_approved_species_slugs_frozen(self):
        self.assertEqual(len(mrc.APPROVED_SPECIES_SLUGS), 15)
        self.assertEqual(len(set(mrc.APPROVED_SPECIES_SLUGS)), 15)

    def test_canonical_domain_pair_key_is_order_independent(self):
        self.assertEqual(mrc.canonical_domain_pair_key("z", "a"), mrc.canonical_domain_pair_key("a", "z"))

    def test_queue_identity_order_hash_changes_on_reorder(self):
        rows = [
            {"queue_index": 0, "dataset": "d", "split": "s", "slug": "sl", "observation_uuid": "u0",
            "photo_id": "1", "sha256": "a" * 64},
            {"queue_index": 1, "dataset": "d", "split": "s", "slug": "sl", "observation_uuid": "u1",
            "photo_id": "2", "sha256": "b" * 64},
        ]
        original = mrc.compute_queue_identity_order_sha256(rows)
        swapped = [dict(rows[1], queue_index=0), dict(rows[0], queue_index=1)]
        self.assertNotEqual(original, mrc.compute_queue_identity_order_sha256(swapped))

    def test_queue_identity_order_hash_deterministic(self):
        rows = [{"queue_index": 0, "dataset": "d", "split": "s", "slug": "sl", "observation_uuid": "u0",
                "photo_id": "1", "sha256": "a" * 64}]
        self.assertEqual(mrc.compute_queue_identity_order_sha256(rows),
                         mrc.compute_queue_identity_order_sha256(rows))


def _valid_contract() -> dict:
    content = mrc.build_contract_content(**TestContractContentBuild.ARTIFACTS)
    return {"schema_version": mrc.SCHEMA_VERSION, "status": mrc.CONTRACT_STATUS_FROZEN,
            "content": content, "content_sha256": mrc.compute_content_sha256(content),
            "generation": {"generator": "test"}}


def _envelope(content: dict) -> dict:
    return {"schema_version": mrc.SCHEMA_VERSION, "content": content,
            "content_sha256": mrc.compute_content_sha256(content)}


def _rehash(report: dict) -> dict:
    """Recomputes content_sha256 after an in-place content mutation --
    proves the validators check semantic derivation, not merely that
    content_sha256 is self-consistent."""
    report["content_sha256"] = mrc.compute_content_sha256(report["content"])
    return report


class TestScanReportValidators(unittest.TestCase):
    """Item 6: direct unit tests for the five strict scan-report schema
    validators (and the bundle orchestrator), each with a mutation that is
    rehashed afterward -- proving the validators reject semantically wrong
    content even when content_sha256 is freshly, correctly recomputed."""

    def setUp(self):
        self.contract = _valid_contract()

    def _valid_hashes_report(self):
        content = {
            "contract_content_sha256": self.contract["content_sha256"],
            "pillow_version": "10.0.0", "numpy_version": "1.26.0",
            "hashes": {
                "id-a": {"phashes": [0] * mrc.N_DIHEDRAL_ORIENTATIONS,
                        "dhashes": [0] * mrc.N_DIHEDRAL_ORIENTATIONS, "sha256": "a" * 64},
                "id-b": {"phashes": [1] * mrc.N_DIHEDRAL_ORIENTATIONS,
                        "dhashes": [1] * mrc.N_DIHEDRAL_ORIENTATIONS, "sha256": "b" * 64},
            },
        }
        return _envelope(content)

    def test_hashes_report_valid_has_no_problems(self):
        self.assertEqual(mrc.validate_hashes_report(self._valid_hashes_report(), self.contract), [])

    def test_hashes_report_wrong_contract_binding_rejected_even_after_rehash(self):
        report = self._valid_hashes_report()
        report["content"]["contract_content_sha256"] = "0" * 64
        _rehash(report)
        self.assertNotEqual(mrc.validate_hashes_report(report, self.contract), [])

    def test_hashes_report_out_of_range_hash_value_rejected_even_after_rehash(self):
        report = self._valid_hashes_report()
        report["content"]["hashes"]["id-a"]["phashes"][0] = 1 << mrc.PHASH_BITS  # out of range
        _rehash(report)
        self.assertNotEqual(mrc.validate_hashes_report(report, self.contract), [])

    def _valid_leakage_report(self):
        content = {
            "contract_content_sha256": self.contract["content_sha256"],
            "findings": [{
                "observation_uuid": "u1", "domains": ["expansion_train", "final_test"],
                "entries": [
                    {"domain": "expansion_train", "dataset": "northeast_expansion_v1", "slug": "s",
                    "split": "train", "photo_id": "1", "identity": "id-a"},
                    {"domain": "final_test", "dataset": "northeast_final_test_v1", "slug": "s",
                    "split": "final_test", "photo_id": "2", "identity": "id-b"},
                ],
            }],
        }
        return _envelope(content)

    def test_leakage_report_valid_has_no_problems(self):
        self.assertEqual(mrc.validate_metadata_leakage_report(self._valid_leakage_report(), self.contract), [])

    def test_leakage_report_unsorted_domains_rejected_even_after_rehash(self):
        report = self._valid_leakage_report()
        report["content"]["findings"][0]["domains"] = ["final_test", "expansion_train"]  # not sorted
        _rehash(report)
        self.assertNotEqual(mrc.validate_metadata_leakage_report(report, self.contract), [])

    def _all_domains_dict(self, default):
        return {d["name"]: (default() if callable(default) else default) for d in mrc.COMPARISON_DOMAINS}

    def _valid_candidate_pairs_report(self, hashes_content_sha256):
        pair = {"pair_id": mrc.compute_pair_id("id-a", "id-b"), "identity_a": "id-a", "identity_b": "id-b",
                "phash_distance": 0, "dhash_distance": 0}
        domains = self._all_domains_dict(list)
        domains["expansion_train_vs_final_test"] = [pair]
        content = {
            "contract_content_sha256": self.contract["content_sha256"],
            "hashes_report_content_sha256": hashes_content_sha256,
            "domains": domains,
        }
        return _envelope(content), pair

    def test_candidate_pairs_report_valid_has_no_problems(self):
        hashes_report = self._valid_hashes_report()
        report, _ = self._valid_candidate_pairs_report(hashes_report["content_sha256"])
        self.assertEqual(mrc.validate_candidate_pairs_report(report, self.contract, hashes_report["content_sha256"]), [])

    def test_candidate_pairs_report_wrong_pair_id_rejected_even_after_rehash(self):
        hashes_report = self._valid_hashes_report()
        report, pair = self._valid_candidate_pairs_report(hashes_report["content_sha256"])
        pair["pair_id"] = "0" * 64  # does not match compute_pair_id(identity_a, identity_b)
        _rehash(report)
        problems = mrc.validate_candidate_pairs_report(report, self.contract, hashes_report["content_sha256"])
        self.assertNotEqual(problems, [])
        self.assertTrue(any("pair_id" in p for p in problems))

    def test_candidate_pairs_report_violates_candidate_rule_rejected_even_after_rehash(self):
        hashes_report = self._valid_hashes_report()
        report, pair = self._valid_candidate_pairs_report(hashes_report["content_sha256"])
        pair["phash_distance"] = 40
        pair["dhash_distance"] = 40  # neither satisfies the candidate rule
        _rehash(report)
        self.assertNotEqual(mrc.validate_candidate_pairs_report(report, self.contract, hashes_report["content_sha256"]), [])

    def _valid_domain_summary_report(self, cp_report, leakage_report):
        cp_domains = cp_report["content"]["domains"]
        lk_findings = leakage_report["content"]["findings"]
        domains = {}
        for d in mrc.COMPARISON_DOMAINS:
            name = d["name"]
            candidate_count = len(cp_domains.get(name, []))
            if d["kind"] == "within":
                leakage_count = 0
            else:
                a, b = d["parts"]
                leakage_count = sum(1 for f in lk_findings if {a, b} <= set(f["domains"]))
            domains[name] = {"a_count": 2, "b_count": 2, "candidate_count": candidate_count,
                             "leakage_count": leakage_count}
        content = {
            "contract_content_sha256": self.contract["content_sha256"],
            "candidate_pairs_report_content_sha256": cp_report["content_sha256"],
            "metadata_leakage_report_content_sha256": leakage_report["content_sha256"],
            "domains": domains,
        }
        return _envelope(content)

    def test_domain_summary_report_valid_has_no_problems(self):
        hashes_report = self._valid_hashes_report()
        cp_report, _ = self._valid_candidate_pairs_report(hashes_report["content_sha256"])
        leakage_report = self._valid_leakage_report()
        ds_report = self._valid_domain_summary_report(cp_report, leakage_report)
        self.assertEqual(mrc.validate_domain_summary_report(ds_report, self.contract, cp_report, leakage_report), [])

    def test_domain_summary_report_wrong_candidate_count_rejected_even_after_rehash(self):
        hashes_report = self._valid_hashes_report()
        cp_report, _ = self._valid_candidate_pairs_report(hashes_report["content_sha256"])
        leakage_report = self._valid_leakage_report()
        ds_report = self._valid_domain_summary_report(cp_report, leakage_report)
        ds_report["content"]["domains"]["expansion_train_vs_final_test"]["candidate_count"] = 999
        _rehash(ds_report)
        problems = mrc.validate_domain_summary_report(ds_report, self.contract, cp_report, leakage_report)
        self.assertNotEqual(problems, [])
        self.assertTrue(any("candidate_count" in p for p in problems))

    def test_domain_summary_report_wrong_leakage_count_rejected_even_after_rehash(self):
        hashes_report = self._valid_hashes_report()
        cp_report, _ = self._valid_candidate_pairs_report(hashes_report["content_sha256"])
        leakage_report = self._valid_leakage_report()
        ds_report = self._valid_domain_summary_report(cp_report, leakage_report)
        ds_report["content"]["domains"]["expansion_train_vs_final_test"]["leakage_count"] = 999
        _rehash(ds_report)
        self.assertNotEqual(mrc.validate_domain_summary_report(ds_report, self.contract, cp_report, leakage_report), [])

    def _valid_stop_status_report(self, ds_report):
        domains = {}
        for d in mrc.COMPARISON_DOMAINS:
            name = d["name"]
            entry = ds_report["content"]["domains"][name]
            reasons = []
            if name in mrc.STOP_BEFORE_INFERENCE_DOMAINS and entry["leakage_count"]:
                reasons.append(mrc.STOP_REASON_MATCHING_OBSERVATION_UUID)
            domains[name] = {"stop_before_inference": bool(reasons), "reasons": reasons}
        content = {
            "contract_content_sha256": self.contract["content_sha256"],
            "domain_summary_report_content_sha256": ds_report["content_sha256"],
            "domains": domains,
            "overall_stop_before_inference": any(v["stop_before_inference"] for v in domains.values()),
        }
        return _envelope(content)

    def test_stop_status_report_valid_has_no_problems(self):
        hashes_report = self._valid_hashes_report()
        cp_report, _ = self._valid_candidate_pairs_report(hashes_report["content_sha256"])
        leakage_report = self._valid_leakage_report()
        ds_report = self._valid_domain_summary_report(cp_report, leakage_report)
        ss_report = self._valid_stop_status_report(ds_report)
        self.assertEqual(mrc.validate_stop_status_report(ss_report, self.contract, ds_report), [])

    def test_stop_status_report_confirmed_same_source_rejected_even_after_rehash(self):
        # A freshly generated scan report must NEVER claim a confirmed
        # same_source_image stop -- that can only come from later human
        # adjudication.
        hashes_report = self._valid_hashes_report()
        cp_report, _ = self._valid_candidate_pairs_report(hashes_report["content_sha256"])
        leakage_report = self._valid_leakage_report()
        ds_report = self._valid_domain_summary_report(cp_report, leakage_report)
        ss_report = self._valid_stop_status_report(ds_report)
        entry = ss_report["content"]["domains"]["expansion_train_vs_final_test"]
        entry["reasons"] = list(entry["reasons"]) + [mrc.STOP_REASON_SAME_SOURCE_IMAGE]
        entry["stop_before_inference"] = True
        ss_report["content"]["overall_stop_before_inference"] = True
        _rehash(ss_report)
        problems = mrc.validate_stop_status_report(ss_report, self.contract, ds_report)
        self.assertNotEqual(problems, [])
        self.assertTrue(any("same_source_image" in p for p in problems))

    def test_stop_status_report_mismatched_stop_flag_rejected_even_after_rehash(self):
        hashes_report = self._valid_hashes_report()
        cp_report, _ = self._valid_candidate_pairs_report(hashes_report["content_sha256"])
        leakage_report = self._valid_leakage_report()
        ds_report = self._valid_domain_summary_report(cp_report, leakage_report)
        ss_report = self._valid_stop_status_report(ds_report)
        ss_report["content"]["domains"]["within_final_test"]["stop_before_inference"] = True
        _rehash(ss_report)
        self.assertNotEqual(mrc.validate_stop_status_report(ss_report, self.contract, ds_report), [])

    def test_bundle_orchestrator_valid_bundle_has_no_problems(self):
        hashes_report = self._valid_hashes_report()
        cp_report, _ = self._valid_candidate_pairs_report(hashes_report["content_sha256"])
        leakage_report = self._valid_leakage_report()
        ds_report = self._valid_domain_summary_report(cp_report, leakage_report)
        ss_report = self._valid_stop_status_report(ds_report)
        reports = {
            "perceptual_hashes_report": hashes_report, "metadata_leakage_report": leakage_report,
            "candidate_pairs_report": cp_report, "domain_summary_report": ds_report,
            "stop_status_report": ss_report,
        }
        self.assertEqual(mrc.validate_scan_report_bundle(reports, self.contract), [])

    def test_bundle_orchestrator_rejects_wrong_keys(self):
        self.assertNotEqual(mrc.validate_scan_report_bundle({}, self.contract), [])

    def test_bundle_orchestrator_propagates_a_single_bad_report(self):
        hashes_report = self._valid_hashes_report()
        cp_report, pair = self._valid_candidate_pairs_report(hashes_report["content_sha256"])
        pair["pair_id"] = "0" * 64
        _rehash(cp_report)
        leakage_report = self._valid_leakage_report()
        ds_report = self._valid_domain_summary_report(cp_report, leakage_report)
        ss_report = self._valid_stop_status_report(ds_report)
        reports = {
            "perceptual_hashes_report": hashes_report, "metadata_leakage_report": leakage_report,
            "candidate_pairs_report": cp_report, "domain_summary_report": ds_report,
            "stop_status_report": ss_report,
        }
        self.assertNotEqual(mrc.validate_scan_report_bundle(reports, self.contract), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
