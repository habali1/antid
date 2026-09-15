#!/usr/bin/env python3
"""test_pair_adjudication_v2_contract.py — offline/synthetic tests for
pair_adjudication_v2_contract.py. Never opens an image; never touches
the real 3,972-row queue (that is exercised separately by the real,
read-only --preflight/--check runs reported alongside these tests)."""
from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pair_adjudication_v2_contract as v2c  # noqa: E402


def _minimal_valid_contract() -> dict:
    queue_artifact = {
        "path": v2c.QUEUE_ARTIFACT_REL_PATH, "byte_sha256": "a" * 64,
        "row_count": v2c.EXPECTED_SCOPED_TOTAL, "identity_order_sha256": "b" * 64,
    }
    implementation_sources = {name: {"path": path, "sha256": "c" * 64, "generator_git_commit": "d" * 40}
                              for name, path in v2c.APPROVED_IMPLEMENTATION_SOURCE_PATHS.items()}
    content = v2c.build_contract_content(queue_artifact=queue_artifact, implementation_sources=implementation_sources,
                                         outside_scope_identity_order_sha256="e" * 64)
    content_sha256 = v2c.compute_content_sha256(content)
    return {
        "schema_version": v2c.SCHEMA_VERSION, "status": v2c.CONTRACT_STATUS_FROZEN,
        "content": content, "content_sha256": content_sha256,
        "generation": {"note": v2c.CONTRACT_GENERATION_NOTE,
                       "generator": v2c.CONTRACT_GENERATOR_PATH},
    }


class TestSchemaValidation(unittest.TestCase):
    def test_minimal_valid_contract_passes(self):
        self.assertEqual(v2c.validate_contract_structure(_minimal_valid_contract()), [])

    def test_never_raises_on_arbitrary_json(self):
        for bad in (None, [], "x", 1, {}, {"a": 1}, {"schema_version": "x"}):
            problems = v2c.validate_contract_structure(bad)
            self.assertIsInstance(problems, list)
            self.assertTrue(problems)

    def test_missing_top_key_rejected(self):
        c = _minimal_valid_contract()
        del c["status"]
        problems = v2c.validate_contract_structure(c)
        self.assertTrue(any("missing key" in p for p in problems))

    def test_extra_top_key_rejected(self):
        c = _minimal_valid_contract()
        c["unexpected"] = 1
        problems = v2c.validate_contract_structure(c)
        self.assertTrue(any("unexpected key" in p for p in problems))

    def test_mutate_then_rehash_is_still_rejected(self):
        """A tampered content dict whose content_sha256 is freshly
        recomputed to match must STILL fail -- because it now disagrees
        with the frozen module constants (e.g. scoped_candidate_count)."""
        c = _minimal_valid_contract()
        c["content"] = copy.deepcopy(c["content"])
        c["content"]["scoped_candidate_count"] = 999999
        c["content_sha256"] = v2c.compute_content_sha256(c["content"])
        problems = v2c.validate_contract_structure(c)
        self.assertTrue(any("scoped_candidate_count" in p for p in problems))

    def test_content_sha256_mismatch_rejected(self):
        c = _minimal_valid_contract()
        c["content_sha256"] = "0" * 64
        problems = v2c.validate_contract_structure(c)
        self.assertTrue(any("content_sha256 mismatch" in p for p in problems))

    def test_wrong_phase_5f1_hash_rejected(self):
        c = _minimal_valid_contract()
        c["content"] = copy.deepcopy(c["content"])
        c["content"]["phase_5f1_contract_content_sha256"] = "0" * 64
        c["content_sha256"] = v2c.compute_content_sha256(c["content"])
        problems = v2c.validate_contract_structure(c)
        self.assertTrue(any("phase_5f1_contract_content_sha256" in p for p in problems))

    def test_bool_as_schema_version_rejected(self):
        c = _minimal_valid_contract()
        c["schema_version"] = True  # bool is not a strict int here
        problems = v2c.validate_contract_structure(c)
        self.assertTrue(any("schema_version" in p for p in problems))

    def test_implementation_source_missing_generator_git_commit_rejected(self):
        c = _minimal_valid_contract()
        c["content"] = copy.deepcopy(c["content"])
        name = "pair_adjudication_v2_contract"
        c["content"]["implementation_sources"][name] = {"path": v2c.APPROVED_IMPLEMENTATION_SOURCE_PATHS[name],
                                                         "sha256": "c" * 64}
        c["content_sha256"] = v2c.compute_content_sha256(c["content"])
        problems = v2c.validate_contract_structure(c)
        self.assertTrue(any("generator_git_commit" in p for p in problems))

    def test_implementation_sources_missing_name_rejected(self):
        c = _minimal_valid_contract()
        c["content"] = copy.deepcopy(c["content"])
        del c["content"]["implementation_sources"]["finalize_stop_status_v2"]
        c["content_sha256"] = v2c.compute_content_sha256(c["content"])
        problems = v2c.validate_contract_structure(c)
        self.assertTrue(any("implementation_sources keys must be exactly" in p for p in problems))

    def test_implementation_sources_extra_name_rejected(self):
        c = _minimal_valid_contract()
        c["content"] = copy.deepcopy(c["content"])
        c["content"]["implementation_sources"]["not_an_approved_source"] = {
            "path": "training/not_an_approved_source.py", "sha256": "c" * 64, "generator_git_commit": "d" * 40}
        c["content_sha256"] = v2c.compute_content_sha256(c["content"])
        problems = v2c.validate_contract_structure(c)
        self.assertTrue(any("implementation_sources keys must be exactly" in p for p in problems))

    def test_implementation_source_renamed_or_path_substituted_rejected(self):
        """Same approved NAME, but pointed at a different path -- must be
        rejected even though the key set is otherwise exactly right."""
        c = _minimal_valid_contract()
        c["content"] = copy.deepcopy(c["content"])
        c["content"]["implementation_sources"]["adjudicate_pairs_v2"]["path"] = "training/some_other_file.py"
        c["content_sha256"] = v2c.compute_content_sha256(c["content"])
        problems = v2c.validate_contract_structure(c)
        self.assertTrue(any("path-substituted" in p for p in problems))

    def test_outside_scope_identity_order_sha256_malformed_rejected(self):
        c = _minimal_valid_contract()
        c["content"] = copy.deepcopy(c["content"])
        c["content"]["outside_scope_identity_order_sha256"] = "not-a-hash"
        c["content_sha256"] = v2c.compute_content_sha256(c["content"])
        problems = v2c.validate_contract_structure(c)
        self.assertTrue(any("outside_scope_identity_order_sha256" in p for p in problems))


class TestQueueDerivation(unittest.TestCase):
    def test_derive_rejects_missing_domain(self):
        content = {"domains": {}}
        with self.assertRaises(ValueError):
            v2c.derive_v2_queue_rows(content)

    def test_derive_rejects_wrong_row_count_for_a_domain(self):
        content = {"domains": {d: [] for d in v2c.DOMAIN_VISIT_ORDER}}
        content["domains"]["within_final_test"] = [
            {"pair_id": f"p{i}", "identity_a": "a", "identity_b": "b", "phash_distance": 20, "dhash_distance": 20}
            for i in range(3)
        ]  # 3 != frozen expected 53
        with self.assertRaises(ValueError):
            v2c.derive_v2_queue_rows(content)

    def test_derive_full_scale_real_shape_synthetic_data(self):
        """Synthesizes fake pairs matching every frozen EXPECTED_DOMAIN_COUNTS
        exactly (arbitrary but distinct identities/distances) and confirms
        the derivation succeeds end-to-end with the real production
        constants, independent of the real candidate report on disk."""
        content = {"domains": {}}
        counter = 0
        both_rule_assigned = 0
        for domain, count in v2c.EXPECTED_DOMAIN_COUNTS.items():
            pairs = []
            for i in range(count):
                counter += 1
                # assign both-rule pairs sequentially (not necessarily i==0) until the frozen total is reached
                both_rule = both_rule_assigned < v2c.EXPECTED_BOTH_RULE_COUNT_IN_SCOPE
                if both_rule:
                    both_rule_assigned += 1
                phash = 5 if both_rule else 20
                dhash = 4 if both_rule else 20
                pairs.append({"pair_id": f"pid-{domain}-{i:05d}", "identity_a": f"a{counter}",
                             "identity_b": f"b{counter}", "phash_distance": phash, "dhash_distance": dhash})
            content["domains"][domain] = pairs
        self.assertEqual(both_rule_assigned, v2c.EXPECTED_BOTH_RULE_COUNT_IN_SCOPE)
        rows = v2c.derive_v2_queue_rows(content)
        self.assertEqual(len(rows), v2c.EXPECTED_SCOPED_TOTAL)
        self.assertEqual(sum(1 for r in rows if r["both_rule_match"]), v2c.EXPECTED_BOTH_RULE_COUNT_IN_SCOPE)
        # every session holds exactly one domain
        by_session_domain = {r["suggested_session"]: r["domain"] for r in rows}
        for session, domain in by_session_domain.items():
            self.assertTrue(session.startswith(domain))
        # no session exceeds the max
        counts = {}
        for r in rows:
            counts[r["suggested_session"]] = counts.get(r["suggested_session"], 0) + 1
        self.assertTrue(all(c <= v2c.MAX_PAIRS_PER_SESSION for c in counts.values()))
        # all 18 both-rule rows are GLOBALLY first (literal frozen priority order)
        both_rule_indices = sorted(r["queue_index"] for r in rows if r["both_rule_match"])
        self.assertEqual(both_rule_indices, list(range(v2c.EXPECTED_BOTH_RULE_COUNT_IN_SCOPE)))
        # a both-rule row never shares a session block with a standard row
        for r in rows:
            block_rows = [x for x in rows if x["suggested_session"] == r["suggested_session"]]
            self.assertTrue(all(x["both_rule_match"] == r["both_rule_match"] for x in block_rows))

    def test_both_rule_pairs_form_their_own_priority_blocks_not_merged_with_domain_bulk(self):
        """The one real domain (in this synthetic fixture) holding both-
        rule pairs must show a SEPARATE, small block0 (only the both-rule
        rows) distinct from its later, much larger standard block(s) --
        never merged into one continuous chunk."""
        content = {"domains": {}}
        counter = 0
        for domain, count in v2c.EXPECTED_DOMAIN_COUNTS.items():
            pairs = []
            for i in range(count):
                counter += 1
                both_rule = domain == "benchmark_v1_vs_final_test" and i < 5
                phash = 5 if both_rule else 20
                dhash = 4 if both_rule else 20
                pairs.append({"pair_id": f"pid-{domain}-{i:05d}", "identity_a": f"a{counter}",
                             "identity_b": f"b{counter}", "phash_distance": phash, "dhash_distance": dhash})
            content["domains"][domain] = pairs
        with mock.patch.object(v2c, "EXPECTED_BOTH_RULE_COUNT_IN_SCOPE", 5):
            rows = v2c.derive_v2_queue_rows(content)
        domain_rows = [r for r in rows if r["domain"] == "benchmark_v1_vs_final_test"]
        priority_rows = [r for r in domain_rows if r["both_rule_match"]]
        standard_rows = [r for r in domain_rows if not r["both_rule_match"]]
        self.assertEqual(len(priority_rows), 5)
        priority_blocks = {r["session_block"] for r in priority_rows}
        standard_blocks = {r["session_block"] for r in standard_rows}
        self.assertEqual(priority_blocks, {0})
        self.assertTrue(min(standard_blocks) > max(priority_blocks))

    def test_identity_order_hash_changes_on_reorder(self):
        rows = [
            {"queue_index": 0, "domain": "d", "workstream": "w", "priority_tier": 1, "both_rule_match": True,
             "pair_id": "p1", "identity_a": "a", "identity_b": "b", "phash_distance": 1, "dhash_distance": 1,
             "session_block": 0, "suggested_session": "d__block0"},
            {"queue_index": 1, "domain": "d", "workstream": "w", "priority_tier": 2, "both_rule_match": False,
             "pair_id": "p2", "identity_a": "c", "identity_b": "d", "phash_distance": 2, "dhash_distance": 2,
             "session_block": 0, "suggested_session": "d__block0"},
        ]
        h1 = v2c.compute_v2_queue_identity_order_sha256(rows)
        reordered = list(reversed(rows))
        h2 = v2c.compute_v2_queue_identity_order_sha256(reordered)
        self.assertNotEqual(h1, h2)


class TestOutsideScopeIdentity(unittest.TestCase):
    def test_only_non_scope_domains_are_counted(self):
        content = {"domains": {
            "within_final_test": [{"pair_id": "in-scope-1"}],  # an in-scope domain name
            "some_other_domain": [{"pair_id": "z1"}, {"pair_id": "a1"}],
        }}
        ids = v2c.derive_outside_scope_pair_ids(content)
        self.assertEqual(ids, ["a1", "z1"])  # sorted, and the in-scope domain excluded

    def test_hash_changes_on_membership_change(self):
        h1 = v2c.compute_outside_scope_identity_order_sha256(["a", "b", "c"])
        h2 = v2c.compute_outside_scope_identity_order_sha256(["a", "b", "d"])
        self.assertNotEqual(h1, h2)

    def test_hash_is_order_independent_of_input_but_sort_is_caller_responsibility(self):
        # compute_outside_scope_identity_order_sha256 hashes exactly what
        # it is given -- derive_outside_scope_pair_ids is what guarantees
        # a canonical (sorted) order upstream.
        self.assertNotEqual(v2c.compute_outside_scope_identity_order_sha256(["a", "b"]),
                            v2c.compute_outside_scope_identity_order_sha256(["b", "a"]))

    def test_derive_is_deterministic(self):
        content = {"domains": {"some_other_domain": [{"pair_id": "z1"}, {"pair_id": "a1"}]}}
        self.assertEqual(v2c.derive_outside_scope_pair_ids(content), v2c.derive_outside_scope_pair_ids(content))


class TestSessionBlocks(unittest.TestCase):
    def test_large_domain_splits_deterministically(self):
        rows = [{"domain": "d", "pair_id": f"p{i}"} for i in range(650)]
        v2c.assign_session_blocks(rows)
        blocks = {r["session_block"] for r in rows}
        self.assertEqual(blocks, {0, 1, 2})
        self.assertEqual(sum(1 for r in rows if r["session_block"] == 0), v2c.MAX_PAIRS_PER_SESSION)
        self.assertEqual(sum(1 for r in rows if r["session_block"] == 2), 650 - 2 * v2c.MAX_PAIRS_PER_SESSION)

    def test_each_session_single_domain(self):
        rows = ([{"domain": "d1", "pair_id": f"a{i}"} for i in range(5)]
               + [{"domain": "d2", "pair_id": f"b{i}"} for i in range(5)])
        v2c.assign_session_blocks(rows)
        for r in rows:
            self.assertTrue(r["suggested_session"].startswith(r["domain"]))


class TestIdentityResolution(unittest.TestCase):
    def test_parse_identity_rejects_wrong_shape(self):
        with self.assertRaises(v2c.IdentityResolutionError):
            v2c.parse_identity("only:three:parts")

    def test_parse_identity_rejects_non_string(self):
        with self.assertRaises(v2c.IdentityResolutionError):
            v2c.parse_identity(12345)

    def test_domain_part_for_final_test_identity(self):
        identity = "northeast_final_test_v1:some-slug:final_test:123"
        self.assertEqual(v2c.domain_part_for_identity(identity), "final_test")

    def test_domain_part_for_expansion_train_identity(self):
        identity = "northeast_expansion_v1:some-slug:train:123"
        self.assertEqual(v2c.domain_part_for_identity(identity), "expansion_train")

    def test_domain_part_for_direct_layout_identity(self):
        identity = "calibration_v2:some-slug:known:123"
        self.assertEqual(v2c.domain_part_for_identity(identity), "calibration_v2")

    def test_domain_part_for_unrecognized_dataset_rejected(self):
        with self.assertRaises(v2c.IdentityResolutionError):
            v2c.domain_part_for_identity("not_a_real_dataset:slug:split:1")


class TestGitCommitHexValidation(unittest.TestCase):
    def test_valid_40_char_hex(self):
        self.assertTrue(v2c.is_git_commit_hex("a" * 40))

    def test_rejects_sha256_length(self):
        self.assertFalse(v2c.is_git_commit_hex("a" * 64))

    def test_rejects_uppercase(self):
        self.assertFalse(v2c.is_git_commit_hex("A" * 40))

    def test_rejects_non_string(self):
        self.assertFalse(v2c.is_git_commit_hex(None))


if __name__ == "__main__":
    unittest.main()
