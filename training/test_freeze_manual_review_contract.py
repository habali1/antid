#!/usr/bin/env python3
"""test_freeze_manual_review_contract.py — offline tests for
manual_review_contract.validate_contract_structure() (the shared total
schema validator) and freeze_manual_review_contract.py, plus a repo-wide
guard that none of the Phase 5F1 preparation modules ever write under
data/.
"""
from __future__ import annotations

import ast
import copy
import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import manual_review_contract as mrc  # noqa: E402
import freeze_manual_review_contract as fmrc  # noqa: E402


def _valid_contract() -> dict:
    queue_artifact = {"path": mrc.QUEUE_ARTIFACT_REL_PATH, "byte_sha256": "a" * 64,
                      "row_count": mrc.N_QUEUE_ROWS, "identity_order_sha256": "b" * 64}
    summary_artifact = {"path": mrc.SUMMARY_ARTIFACT_REL_PATH, "byte_sha256": "c" * 64,
                        "content_sha256": "d" * 64}
    implementation_sources = {name: {"path": path, "sha256": "e" * 64}
                              for name, path in fmrc.IMPLEMENTATION_SOURCE_PATHS.items()}
    content = mrc.build_contract_content(
        queue_artifact=queue_artifact, summary_artifact=summary_artifact,
        implementation_sources=implementation_sources)
    content_sha256 = mrc.compute_content_sha256(content)
    return {
        "schema_version": mrc.SCHEMA_VERSION,
        "status": mrc.CONTRACT_STATUS_FROZEN,
        "content": content,
        "content_sha256": content_sha256,
        "generation": {"generator": "test"},
    }


class TestValidateContractStructureAcceptsValid(unittest.TestCase):
    def test_valid_contract_has_no_problems(self):
        self.assertEqual(mrc.validate_contract_structure(_valid_contract()), [])


class TestValidateContractStructureNeverRaises(unittest.TestCase):
    """The validator must return controlled problems, never throw, on
    arbitrary JSON types at any level."""

    def test_non_dict_top_level(self):
        for bad in ("a string", 42, None, [1, 2, 3], True, 3.14):
            with self.subTest(bad=bad):
                problems = mrc.validate_contract_structure(bad)
                self.assertTrue(len(problems) > 0)

    def test_content_is_not_a_dict(self):
        c = _valid_contract()
        c["content"] = "not a dict"
        problems = mrc.validate_contract_structure(c)
        self.assertTrue(any("content is not a JSON object" in p for p in problems))

    def test_deeply_malformed_nested_values(self):
        c = _valid_contract()
        c["content"]["source_manifests"] = {"weird": [1, {"x": None}, float("nan")]}
        try:
            problems = mrc.validate_contract_structure(c)
        except Exception as exc:  # noqa: BLE001
            self.fail(f"validate_contract_structure raised {type(exc).__name__}: {exc}")
        self.assertTrue(len(problems) > 0)


class TestValidateContractStructureRejections(unittest.TestCase):
    def test_missing_top_level_key_rejected(self):
        c = _valid_contract()
        del c["generation"]
        problems = mrc.validate_contract_structure(c)
        self.assertTrue(any("missing key" in p for p in problems))

    def test_extra_top_level_key_rejected(self):
        c = _valid_contract()
        c["smuggled"] = "x"
        problems = mrc.validate_contract_structure(c)
        self.assertTrue(any("unexpected key" in p for p in problems))

    def test_bool_as_schema_version_rejected(self):
        c = _valid_contract()
        c["schema_version"] = True
        problems = mrc.validate_contract_structure(c)
        self.assertTrue(any("schema_version" in p for p in problems))

    def test_float_schema_version_rejected(self):
        c = _valid_contract()
        c["schema_version"] = 1.0
        problems = mrc.validate_contract_structure(c)
        self.assertTrue(any("schema_version" in p for p in problems))

    def test_wrong_status_rejected(self):
        c = _valid_contract()
        c["status"] = "not_frozen"
        problems = mrc.validate_contract_structure(c)
        self.assertTrue(any("status" in p for p in problems))

    def test_malformed_content_sha256_rejected(self):
        c = _valid_contract()
        c["content_sha256"] = "not-a-hash"
        problems = mrc.validate_contract_structure(c)
        self.assertTrue(any("content_sha256" in p for p in problems))

    def test_content_sha256_mismatch_after_content_mutation_rejected(self):
        c = _valid_contract()
        c["content"]["seed"] = 999999  # mutate content but keep the OLD content_sha256
        problems = mrc.validate_contract_structure(c)
        self.assertTrue(any("content_sha256 mismatch" in p for p in problems))

    def test_bool_seed_rejected(self):
        c = _valid_contract()
        c["content"]["seed"] = True
        c["content_sha256"] = mrc.compute_content_sha256(c["content"])
        problems = mrc.validate_contract_structure(c)
        self.assertTrue(any("seed" in p for p in problems))

    def test_wrong_approved_species_slugs_rejected(self):
        c = _valid_contract()
        c["content"]["approved_species_slugs"] = ["not-the-real-slugs"]
        c["content_sha256"] = mrc.compute_content_sha256(c["content"])
        problems = mrc.validate_contract_structure(c)
        self.assertTrue(any("approved_species_slugs" in p for p in problems))

    def test_extra_slug_appended_rejected(self):
        c = _valid_contract()
        c["content"]["approved_species_slugs"] = list(mrc.APPROVED_SPECIES_SLUGS) + ["extra-slug"]
        c["content_sha256"] = mrc.compute_content_sha256(c["content"])
        problems = mrc.validate_contract_structure(c)
        self.assertTrue(any("approved_species_slugs" in p for p in problems))

    def test_wrong_source_manifest_hash_rejected(self):
        c = _valid_contract()
        c["content"]["source_manifests"][mrc.FINAL_TEST_DATASET]["byte_sha256"] = "0" * 64
        c["content_sha256"] = mrc.compute_content_sha256(c["content"])
        problems = mrc.validate_contract_structure(c)
        self.assertTrue(any("source_manifests" in p for p in problems))

    def test_extra_key_in_source_manifest_entry_rejected(self):
        c = _valid_contract()
        c["content"]["source_manifests"][mrc.FINAL_TEST_DATASET]["smuggled"] = "x"
        c["content_sha256"] = mrc.compute_content_sha256(c["content"])
        problems = mrc.validate_contract_structure(c)
        self.assertTrue(any("source_manifests" in p for p in problems))

    def test_missing_queue_identity_order_hash_rejected(self):
        c = _valid_contract()
        del c["content"]["queue_artifact"]["identity_order_sha256"]
        c["content_sha256"] = mrc.compute_content_sha256(c["content"])
        problems = mrc.validate_contract_structure(c)
        self.assertTrue(any("queue_artifact" in p for p in problems))

    def test_malformed_queue_byte_hash_rejected(self):
        c = _valid_contract()
        c["content"]["queue_artifact"]["byte_sha256"] = "not-hex"
        c["content_sha256"] = mrc.compute_content_sha256(c["content"])
        problems = mrc.validate_contract_structure(c)
        self.assertTrue(any("queue_artifact.byte_sha256" in p for p in problems))

    def test_wrong_queue_row_count_rejected(self):
        c = _valid_contract()
        c["content"]["queue_artifact"]["row_count"] = 1
        c["content_sha256"] = mrc.compute_content_sha256(c["content"])
        problems = mrc.validate_contract_structure(c)
        self.assertTrue(any("queue_artifact.row_count" in p for p in problems))

    def test_missing_summary_content_sha256_rejected(self):
        c = _valid_contract()
        del c["content"]["summary_artifact"]["content_sha256"]
        c["content_sha256"] = mrc.compute_content_sha256(c["content"])
        problems = mrc.validate_contract_structure(c)
        self.assertTrue(any("summary_artifact" in p for p in problems))

    def test_free_form_review_state_in_list_rejected(self):
        c = _valid_contract()
        c["content"]["review_states"] = ["looks_fine"]
        c["content_sha256"] = mrc.compute_content_sha256(c["content"])
        problems = mrc.validate_contract_structure(c)
        self.assertTrue(any("review_states" in p for p in problems))

    def test_altered_remediation_rule_rejected(self):
        c = _valid_contract()
        c["content"]["remediation_rule"] = "a different rule entirely"
        c["content_sha256"] = mrc.compute_content_sha256(c["content"])
        problems = mrc.validate_contract_structure(c)
        self.assertTrue(any("remediation_rule" in p for p in problems))

    def test_truncated_comparison_domains_rejected(self):
        c = _valid_contract()
        c["content"]["comparison_domains"] = c["content"]["comparison_domains"][:10]
        c["content_sha256"] = mrc.compute_content_sha256(c["content"])
        problems = mrc.validate_contract_structure(c)
        self.assertTrue(any("comparison_domains" in p for p in problems))

    def test_reordered_a_b_in_one_domain_rejected(self):
        c = _valid_contract()
        domains = copy.deepcopy(c["content"]["comparison_domains"])
        domains[0] = dict(domains[0])
        domains[0]["parts"] = list(reversed(domains[0]["parts"])) if len(domains[0]["parts"]) == 2 else domains[0]["parts"]
        c["content"]["comparison_domains"] = domains
        c["content_sha256"] = mrc.compute_content_sha256(c["content"])
        problems = mrc.validate_contract_structure(c)
        self.assertTrue(any("comparison_domains" in p for p in problems))

    def test_wrong_stop_domains_rejected(self):
        c = _valid_contract()
        c["content"]["stop_before_inference_domains"] = ["within_final_test"]
        c["content_sha256"] = mrc.compute_content_sha256(c["content"])
        problems = mrc.validate_contract_structure(c)
        self.assertTrue(any("stop_before_inference_domains" in p for p in problems))

    def test_malformed_implementation_source_hash_rejected(self):
        c = _valid_contract()
        first_name = next(iter(c["content"]["implementation_sources"]))
        c["content"]["implementation_sources"][first_name]["sha256"] = "not-hex"
        c["content_sha256"] = mrc.compute_content_sha256(c["content"])
        problems = mrc.validate_contract_structure(c)
        self.assertTrue(any("implementation_sources" in p for p in problems))

    def test_wrong_approved_output_paths_rejected(self):
        c = _valid_contract()
        c["content"]["approved_output_paths"] = {"wrong": "path"}
        c["content_sha256"] = mrc.compute_content_sha256(c["content"])
        problems = mrc.validate_contract_structure(c)
        self.assertTrue(any("approved_output_paths" in p for p in problems))


class TestBuildContractAgainstRealRepo(unittest.TestCase):
    """Integration check against the real, already-generated queue/summary
    (generated separately by generate_manual_review_queue.py --write) --
    skipped if they don't exist yet in this checkout."""

    def setUp(self):
        if not (fmrc.REPO / mrc.QUEUE_ARTIFACT_REL_PATH).exists():
            self.skipTest("real manual_review_queue.csv not present in this checkout")
        if not (fmrc.REPO / mrc.SUMMARY_ARTIFACT_REL_PATH).exists():
            self.skipTest("real manual_review_queue_summary.json not present in this checkout")

    def test_build_contract_is_deterministic_and_self_valid(self):
        a = fmrc.build_contract()
        b = fmrc.build_contract()
        self.assertEqual(a, b)
        self.assertEqual(mrc.validate_contract_structure(a), [])

    def test_content_sha256_matches_recomputation(self):
        contract = fmrc.build_contract()
        recomputed = mrc.compute_content_sha256(contract["content"])
        self.assertEqual(contract["content_sha256"], recomputed)

    def test_implementation_sources_cover_all_eight_modules_including_self(self):
        contract = fmrc.build_contract()
        self.assertEqual(set(contract["content"]["implementation_sources"]), set(fmrc.IMPLEMENTATION_SOURCE_PATHS))
        self.assertIn("freeze_manual_review_contract", contract["content"]["implementation_sources"])

    def test_serialize_round_trips_through_check_comparison(self):
        contract = fmrc.build_contract()
        text = fmrc.serialize(contract)
        self.assertEqual(json.loads(text), contract)


class TestNoDatasetMutation(unittest.TestCase):
    """Static guard: none of the Phase 5F1 preparation modules contain a
    literal 'data/' write-path construction or call os.remove/shutil on a
    data/ path."""

    MODULES = (
        "manual_review_contract", "append_only_ledger", "generate_manual_review_queue",
        "perceptual_hash", "manual_review_tool", "scan_perceptual_duplicates", "adjudicate_pairs",
        "freeze_manual_review_contract",
    )

    def test_no_module_calls_a_write_function_with_a_data_path_literal(self):
        write_calls = {"write_text", "write_bytes", "unlink", "remove", "rmtree", "replace"}
        for modname in self.MODULES:
            source = (HERE / f"{modname}.py").read_text(encoding="utf-8")
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    if node.func.attr in write_calls:
                        for arg in ast.walk(node):
                            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                                self.assertNotIn("data/", arg.value,
                                                f"{modname}.py: {node.func.attr}() call references a "
                                                f"data/ path literal: {arg.value!r}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
