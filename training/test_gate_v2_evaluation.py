#!/usr/bin/env python3
"""test_gate_v2_evaluation.py — synthetic/offline tests for the Gate v2
unknown_test_v2 evaluation contract and evaluator (Phase 5D1).

No test in this file opens a real unknown_test_v2 (or calibration_v2, or
northeast_final_test_v1) image, nor does any test create the real attempt
marker or evaluation output against the real repository files. Every
end-to-end test builds its own tiny synthetic repo (including a tiny,
from-scratch ONNX model) under a temporary directory.

Run directly: python test_gate_v2_evaluation.py
"""
from __future__ import annotations

import hashlib
import inspect
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

import gate_v2_contract as gc  # noqa: E402
import gate_v2_evaluation_contract as ec  # noqa: E402
import freeze_gate_v2_evaluation_contract as fec  # noqa: E402
import eval_unknown_test_v2 as ev  # noqa: E402
import score_calibration_v2 as sc  # noqa: E402
import select_gate_v2_threshold as sel  # noqa: E402

from test_gate_v2_calibration import (  # noqa: E402
    build_tiny_fixture, _manifest_row, _manifest_fields, _write_csv, _write_json,
    _rewrite_calibration_csv, _build_tiny_onnx_model,
)

REAL_EVAL_CONTRACT_PATH = HERE / "gate_v2_evaluation_contract.json"


# =========================================================================
# 1. Pure contract-schema tests (built via the real freeze script against
#    the REAL repo's committed sources -- read-only, no fixture needed)
# =========================================================================
class TestEvaluationContractSchema(unittest.TestCase):
    def test_build_contract_self_validates(self):
        contract = fec.build_contract()
        problems = ec.validate_evaluation_contract(contract)
        self.assertEqual(problems, [])

    def test_frozen_threshold_matches_the_real_selection_artifact(self):
        contract = fec.build_contract()
        self.assertEqual(contract["content"]["threshold"]["threshold_integer"], 61)
        self.assertEqual(contract["content"]["threshold"]["threshold"], 0.61)
        self.assertEqual(contract["content"]["threshold"]["comparison"], "strict_less_than")
        self.assertEqual(contract["content"]["threshold"]["equal_threshold_action"], "normal_results")
        self.assertEqual(contract["content"]["threshold"]["required_status"], "candidate_selected")

    def test_frozen_hashes_match_the_exact_specified_values(self):
        self.assertEqual(ec.FROZEN_SELECTION_CONTRACT_CONTENT_SHA256,
                         "991f7a0b8e83654e45575566eb0648ddcd69866e29c94b8a2030cc6f4bc19f77")
        self.assertEqual(ec.FROZEN_SCORES_BYTE_SHA256,
                         "4fec18a938ef22e06d6073016d1692512e8fb1e4e41ec645d9aeb33afbebe46b")
        self.assertEqual(ec.FROZEN_SCORES_CONTENT_SHA256,
                         "35c7bf36042470dde8fc922106c6528fcb6a06e3bac454c4d196880796e778f4")
        self.assertEqual(ec.FROZEN_SELECTION_BYTE_SHA256,
                         "00e56d2e64941086ee1d1c663890cb95d3daa4dcce1e729ef972879376f1489b")
        self.assertEqual(ec.FROZEN_SELECTION_CONTENT_SHA256,
                         "82bc754dbf46643862504916f4c273922e9ac3cefc45bfa28e5eb4a15dfdc5b5")

    def test_no_unknown_test_v2_image_opened_while_building_contract(self):
        source = inspect.getsource(fec.compute_unknown_test_v2_identity_order_sha256)
        self.assertNotIn("Image", source)
        self.assertNotIn(".jpg", source)
        self.assertNotIn(".png", source)

    def _mutate(self, mutator):
        contract = fec.build_contract()
        content = dict(contract["content"])
        mutator(content)
        contract["content"] = content
        contract["content_sha256"] = ec.compute_content_sha256(content)
        return contract

    def test_wrong_threshold_integer_rejected(self):
        def mutate(content):
            content["threshold"] = dict(content["threshold"])
            content["threshold"]["threshold_integer"] = 60
            content["threshold"]["threshold"] = 0.60
        contract = self._mutate(mutate)
        self.assertEqual(contract["content_sha256"], ec.compute_content_sha256(contract["content"]))
        problems = ec.validate_evaluation_contract(contract)
        self.assertTrue(any("threshold must be exactly" in p for p in problems))

    def test_equal_threshold_action_flipped_to_reject_rejected(self):
        def mutate(content):
            content["threshold"] = dict(content["threshold"])
            content["threshold"]["equal_threshold_action"] = "reject_on_equal"
        contract = self._mutate(mutate)
        problems = ec.validate_evaluation_contract(contract)
        self.assertTrue(any("threshold must be exactly" in p for p in problems))

    def test_scores_binding_hash_swap_rejected(self):
        def mutate(content):
            content["scores_binding"] = {"byte_sha256": "0" * 64, "content_sha256": "1" * 64}
        contract = self._mutate(mutate)
        problems = ec.validate_evaluation_contract(contract)
        self.assertTrue(any("scores_binding must be exactly" in p for p in problems))

    def test_selection_binding_hash_swap_rejected(self):
        def mutate(content):
            content["selection_binding"] = {"byte_sha256": "0" * 64, "content_sha256": "1" * 64}
        contract = self._mutate(mutate)
        problems = ec.validate_evaluation_contract(contract)
        self.assertTrue(any("selection_binding must be exactly" in p for p in problems))

    def test_selection_contract_hash_swap_rejected(self):
        def mutate(content):
            content["selection_contract"] = {"content_sha256": "0" * 64}
        contract = self._mutate(mutate)
        problems = ec.validate_evaluation_contract(contract)
        self.assertTrue(any("selection_contract must be exactly" in p for p in problems))

    def test_candidate_artifact_hash_swap_rejected(self):
        def mutate(content):
            content["bindings"] = dict(content["bindings"])
            content["bindings"]["backbone_onnx"] = dict(content["bindings"]["backbone_onnx"])
            content["bindings"]["backbone_onnx"]["sha256"] = "0" * 64
        contract = self._mutate(mutate)
        problems = ec.validate_evaluation_contract(contract)
        self.assertTrue(any("binding 'backbone_onnx' sha256 must be exactly" in p for p in problems))

    def test_unknown_test_v2_hash_swap_rejected(self):
        def mutate(content):
            content["bindings"] = dict(content["bindings"])
            content["bindings"]["unknown_test_v2_csv"] = dict(content["bindings"]["unknown_test_v2_csv"])
            content["bindings"]["unknown_test_v2_csv"]["sha256"] = "0" * 64
        contract = self._mutate(mutate)
        problems = ec.validate_evaluation_contract(contract)
        self.assertTrue(any("binding 'unknown_test_v2_csv' sha256 must be exactly" in p for p in problems))

    def test_dataset_quota_altered_rejected(self):
        def mutate(content):
            content["dataset_quota"] = dict(content["dataset_quota"])
            content["dataset_quota"]["total"] = 4
        contract = self._mutate(mutate)
        problems = ec.validate_evaluation_contract(contract)
        self.assertTrue(any("dataset_quota does not exactly match" in p for p in problems))

    def test_approved_output_path_redirected_rejected(self):
        def mutate(content):
            content["approved_outputs"] = dict(content["approved_outputs"])
            content["approved_outputs"]["unknown_test_v2_eval"] = "data/arbitrary-new-file.json"
        contract = self._mutate(mutate)
        problems = ec.validate_evaluation_contract(contract)
        self.assertTrue(any("approved_outputs does not exactly match" in p for p in problems))

    def test_approved_output_under_forbidden_prefix_rejected(self):
        def mutate(content):
            content["approved_outputs"] = dict(content["approved_outputs"])
            content["approved_outputs"]["unknown_test_v2_eval"] = "training/artifacts/leaked.json"
        contract = self._mutate(mutate)
        problems = ec.validate_evaluation_contract(contract)
        self.assertTrue(any("forbidden prefix" in p for p in problems))

    def test_runtime_altered_rejected(self):
        def mutate(content):
            content["runtime"] = dict(content["runtime"])
            content["runtime"]["rounding"] = "rounded_to_2dp"
        contract = self._mutate(mutate)
        problems = ec.validate_evaluation_contract(contract)
        self.assertTrue(any("runtime does not exactly match" in p for p in problems))

    def test_validation_criteria_altered_rejected(self):
        def mutate(content):
            content["validation_criteria"] = dict(content["validation_criteria"])
            content["validation_criteria"]["usefulness_floor_numerator"] = 1
        contract = self._mutate(mutate)
        problems = ec.validate_evaluation_contract(contract)
        self.assertTrue(any("validation_criteria does not exactly match" in p for p in problems))

    def test_single_use_rule_altered_rejected(self):
        def mutate(content):
            content["single_use_rule"] = dict(content["single_use_rule"])
            content["single_use_rule"]["unknown_test_v2"] = "may be evaluated repeatedly"
        contract = self._mutate(mutate)
        problems = ec.validate_evaluation_contract(contract)
        self.assertTrue(any("single_use_rule does not exactly match" in p for p in problems))

    def test_closed_sources_emptied_rejected(self):
        def mutate(content):
            content["closed_sources"] = []
        contract = self._mutate(mutate)
        problems = ec.validate_evaluation_contract(contract)
        self.assertTrue(any("closed_sources" in p for p in problems))

    def test_gate_framing_altered_rejected(self):
        def mutate(content):
            content["gate_framing"] = "unknown_species_detector"
        contract = self._mutate(mutate)
        problems = ec.validate_evaluation_contract(contract)
        self.assertTrue(any("gate_framing" in p for p in problems))

    def test_unexpected_top_level_key_rejected(self):
        contract = fec.build_contract()
        contract["unexpected_field"] = "bad"
        problems = ec.validate_evaluation_contract(contract)
        self.assertTrue(any("unexpected top-level key" in p for p in problems))

    def test_unexpected_content_key_rejected(self):
        def mutate(content):
            content["sneaky_field"] = "bad"
        contract = self._mutate(mutate)
        problems = ec.validate_evaluation_contract(contract)
        self.assertTrue(any("content has unexpected key" in p for p in problems))

    def test_missing_content_key_rejected(self):
        def mutate(content):
            del content["single_use_rule"]
        contract = self._mutate(mutate)
        problems = ec.validate_evaluation_contract(contract)
        self.assertTrue(any("content missing key" in p for p in problems))

    def test_bool_schema_version_rejected(self):
        contract = fec.build_contract()
        contract["schema_version"] = True
        problems = ec.validate_evaluation_contract(contract)
        self.assertTrue(any("schema_version" in p for p in problems))

    def test_stale_content_sha256_rejected(self):
        contract = fec.build_contract()
        contract["content"] = dict(contract["content"])
        contract["content"]["policy_name"] = "tampered"
        # deliberately do NOT rehash
        problems = ec.validate_evaluation_contract(contract)
        self.assertTrue(any("content_sha256" in p for p in problems))

    def test_generation_with_unexpected_key_rejected(self):
        contract = fec.build_contract()
        contract["generation"] = dict(contract["generation"])
        contract["generation"]["sneaky"] = "smuggled"
        problems = ec.validate_evaluation_contract(contract)
        self.assertTrue(any("generation has unexpected key" in p for p in problems))

    def test_malformed_top_level_returns_problems_not_exception(self):
        self.assertEqual(ec.validate_evaluation_contract(None), ["contract is not a JSON object"])
        self.assertTrue(ec.validate_evaluation_contract({}))
        self.assertTrue(ec.validate_evaluation_contract("not a dict"))


# =========================================================================
# 2. Pure arithmetic / validation-decision tests (no I/O at all)
# =========================================================================
def _known_record(slug_idx, true_idx, top1_idx, max_cosine, n_species=3):
    slugs = [f"species-{i}" for i in range(n_species)]
    second_idx = (top1_idx + 1) % n_species
    third_idx = (top1_idx + 2) % n_species
    photo_id = f"{slug_idx}-{true_idx}-{top1_idx}-{max_cosine}"
    return {
        "category": "known_holdout", "slug": slugs[slug_idx], "species": slugs[slug_idx].title(),
        "taxon_id": str(100 + slug_idx), "photo_id": photo_id, "observation_uuid": f"u-{photo_id}",
        "image_sha256": hashlib.sha256(photo_id.encode()).hexdigest(),
        "true_class_index": true_idx, "top1_index": top1_idx, "top1_slug": slugs[top1_idx],
        "top3_indices": [top1_idx, second_idx, third_idx],
        "top3_slugs": [slugs[top1_idx], slugs[second_idx], slugs[third_idx]],
        "top3_similarities": [max_cosine, max_cosine - 0.1, max_cosine - 0.2],
        "max_cosine": max_cosine, "top1_correct": top1_idx == true_idx,
    }


def _ood_record(category, max_cosine, tag, n_species=3):
    slugs = [f"species-{i}" for i in range(n_species)]
    photo_id = f"{category}-{tag}"
    return {
        "category": category, "slug": f"{category}-sp", "species": f"{category}-sp".title(),
        "taxon_id": "900", "photo_id": photo_id, "observation_uuid": f"u-{photo_id}",
        "image_sha256": hashlib.sha256(photo_id.encode()).hexdigest(),
        "true_class_index": None, "top1_index": 0, "top1_slug": slugs[0],
        "top3_indices": [0, 1, 2], "top3_slugs": slugs,
        "top3_similarities": [max_cosine, max_cosine - 0.1, max_cosine - 0.2],
        "max_cosine": max_cosine, "top1_correct": None,
    }


class TestValidationArithmetic(unittest.TestCase):
    def test_boundary_below_threshold_rejected(self):
        # 0.609999 < 0.61 -- strictly rejected.
        counts = sel._classify_known([_known_record(0, 0, 0, 0.609999)], ec.FROZEN_THRESHOLD)
        self.assertEqual(counts["rejected"], 1)
        self.assertEqual(counts["accepted"], 0)

    def test_boundary_exactly_at_threshold_accepted(self):
        # Equality at exactly 0.61 is ACCEPTED, never rejected.
        counts = sel._classify_known([_known_record(0, 0, 0, 0.610000)], ec.FROZEN_THRESHOLD)
        self.assertEqual(counts["accepted"], 1)
        self.assertEqual(counts["rejected"], 0)

    def test_boundary_just_above_threshold_accepted(self):
        counts = sel._classify_known([_known_record(0, 0, 0, 0.610001)], ec.FROZEN_THRESHOLD)
        self.assertEqual(counts["accepted"], 1)
        self.assertEqual(counts["rejected"], 0)

    def test_coverage_exactly_at_65pp_boundary_passes(self):
        self.assertTrue(gc.coverage_feasible(65, 100, 65, 100))
        self.assertFalse(gc.coverage_feasible(64, 100, 65, 100))

    def test_usefulness_exactly_at_5pp_boundary_passes(self):
        self.assertTrue(gc.usefulness_ge_floor(60, 100, 55, 100, 5, 100))
        self.assertFalse(gc.usefulness_ge_floor(59, 100, 55, 100, 5, 100))

    def test_health_check_requires_strictly_greater_not_equal(self):
        # Equal rates must FAIL the health check -- it is a strict >, not >=.
        self.assertFalse(ec._rejection_rate_strictly_greater(5, 10, 5, 10))
        self.assertTrue(ec._rejection_rate_strictly_greater(6, 10, 5, 10))
        self.assertFalse(ec._rejection_rate_strictly_greater(4, 10, 5, 10))

    def test_health_check_undefined_when_a_total_is_zero_fails_closed(self):
        self.assertFalse(ec._rejection_rate_strictly_greater(0, 0, 0, 5))
        self.assertFalse(ec._rejection_rate_strictly_greater(0, 5, 0, 0))

    def test_all_three_criteria_pass_yields_validation_passed(self):
        known = ([_known_record(0, 0, 0, 0.9)] * 4 +
                [_known_record(1, 1, 2, 0.2)] * 2)
        result = ec.compute_validation(known)
        self.assertTrue(result["criteria"]["coverage_ok"])
        self.assertTrue(result["criteria"]["usefulness_ok"])
        self.assertTrue(result["criteria"]["health_check_ok"])
        self.assertEqual(result["status"], ec.EVAL_STATUS_PASSED)

    def test_coverage_failure_yields_validation_failed(self):
        # Only 1/6 accepted -- far below the 65% coverage floor.
        known = ([_known_record(0, 0, 0, 0.9)] * 1 +
                [_known_record(1, 1, 2, 0.2)] * 5)
        result = ec.compute_validation(known)
        self.assertFalse(result["criteria"]["coverage_ok"])
        self.assertEqual(result["status"], ec.EVAL_STATUS_FAILED)

    def test_usefulness_failure_yields_validation_failed(self):
        # High coverage but accepted accuracy barely improves on baseline.
        known = ([_known_record(0, 0, 0, 0.9)] * 3 +
                [_known_record(0, 1, 0, 0.9)] * 3)  # all accepted, half wrong
        result = ec.compute_validation(known)
        self.assertTrue(result["criteria"]["coverage_ok"])
        self.assertFalse(result["criteria"]["usefulness_ok"])
        self.assertEqual(result["status"], ec.EVAL_STATUS_FAILED)

    def test_health_check_direction_is_implied_by_real_usefulness_improvement(self):
        # Mathematical property, not an implementation detail: whenever the
        # accepted-known accuracy genuinely improves over baseline (with
        # both correct and incorrect totals positive), the incorrect
        # rejection rate is ALGEBRAICALLY greater than the correct
        # rejection rate -- so a real improvement can never fail the health
        # check. Confirmed here directly against _compute_validation rather
        # than asserted as a standalone axiom.
        known = ([_known_record(0, 0, 0, 0.9)] * 4 + [_known_record(1, 1, 2, 0.2)] * 2)
        result = ec.compute_validation(known)
        self.assertTrue(result["criteria"]["usefulness_ok"])
        self.assertTrue(result["criteria"]["health_check_ok"])

    def test_ood_records_cannot_change_validation_status(self):
        known = ([_known_record(0, 0, 0, 0.9)] * 4 + [_known_record(1, 1, 2, 0.2)] * 2)
        no_ood = ec.compute_validation(known)
        with_permissive_ood = ec.compute_validation(
            known + [_ood_record("out_of_scope_ant", 0.99, i) for i in range(20)]
        )
        self.assertEqual(no_ood["status"], with_permissive_ood["status"])
        self.assertEqual(no_ood["criteria"], with_permissive_ood["criteria"])

    def test_ood_far_and_auc_reported_but_diagnostic_only(self):
        known = ([_known_record(0, 0, 0, 0.9)] * 4 + [_known_record(1, 1, 2, 0.2)] * 2)
        records = known + [_ood_record("out_of_scope_ant", 0.99, 0), _ood_record("non_ant_insect", 0.1, 0),
                          _ood_record("unrelated", 0.1, 0)]
        result = ec.compute_validation(records)
        far = result["metrics"]["diagnostic_ood_far"]
        self.assertEqual(far["out_of_scope_ant"]["false_acceptance_rate"], 1.0)
        self.assertEqual(far["non_ant_insect"]["false_acceptance_rate"], 0.0)
        auc = result["metrics"]["diagnostic_auc_known_vs_ood"]
        self.assertIn("out_of_scope_ant", auc)


# =========================================================================
# 3. validate_eval_content / validate_eval_records (patched tiny quota,
#    no fixture files needed)
# =========================================================================
class TestEvalContentValidator(unittest.TestCase):
    def setUp(self):
        self.quota = {"total": 6, "species_count": 3, "known_holdout": 3,
                     "known_holdout_per_species": 1, "out_of_scope_ant": 1,
                     "non_ant_insect": 1, "unrelated": 1}
        patcher = mock.patch.object(ec, "FROZEN_DATASET_QUOTA", self.quota)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _valid_records(self):
        return [_known_record(i, i, i, 0.9) for i in range(3)] + [
            _ood_record("out_of_scope_ant", 0.1, 0), _ood_record("non_ant_insect", 0.1, 0),
            _ood_record("unrelated", 0.1, 0),
        ]

    def test_valid_records_have_no_problems(self):
        problems, ok, triples = ec.validate_eval_records(self._valid_records(), self.quota)
        self.assertEqual(problems, [])
        self.assertTrue(ok)
        self.assertEqual(len(triples), 6)

    def test_records_not_a_list_reported_not_raised(self):
        problems, ok, triples = ec.validate_eval_records("not a list", self.quota)
        self.assertTrue(problems)
        self.assertFalse(ok)

    def test_wrong_total_count_rejected(self):
        # _valid_records() ends with one "unrelated" OOD row; dropping it
        # leaves that category one row short of its frozen quota.
        records = self._valid_records()[:-1]
        problems, _, _ = ec.validate_eval_records(records, self.quota)
        self.assertTrue(any("unrelated record count 0 != contract 1" in p for p in problems))

    def test_duplicate_photo_id_rejected(self):
        records = self._valid_records()
        records[1] = dict(records[1])
        records[1]["photo_id"] = records[0]["photo_id"]
        problems, _, _ = ec.validate_eval_records(records, self.quota)
        self.assertTrue(any("duplicate photo_id" in p for p in problems))

    def test_missing_field_rejected(self):
        records = self._valid_records()
        del records[0]["top1_slug"]
        problems, _, _ = ec.validate_eval_records(records, self.quota)
        self.assertTrue(any("missing key" in p for p in problems))

    def test_skewed_per_species_with_correct_total_rejected(self):
        records = [_known_record(0, 0, 0, 0.9) for _ in range(3)] + [
            _ood_record("out_of_scope_ant", 0.1, 0), _ood_record("non_ant_insect", 0.1, 0),
            _ood_record("unrelated", 0.1, 0),
        ]
        problems, _, _ = ec.validate_eval_records(records, self.quota)
        self.assertTrue(any("distinct slug" in p or "per-species row count" in p for p in problems))

    def test_reordered_rows_change_identity_order_hash(self):
        records = self._valid_records()
        triples_a = [(r["photo_id"], r["observation_uuid"], r["image_sha256"]) for r in records]
        swapped = list(records)
        swapped[0], swapped[1] = swapped[1], swapped[0]
        triples_b = [(r["photo_id"], r["observation_uuid"], r["image_sha256"]) for r in swapped]
        self.assertNotEqual(gc.compute_identity_order_sha256(triples_a), gc.compute_identity_order_sha256(triples_b))

    def test_category_as_list_rejected_without_raising(self):
        records = self._valid_records()
        records[0] = dict(records[0])
        records[0]["category"] = []
        try:
            problems, _, _ = ec.validate_eval_records(records, self.quota)
        except Exception as exc:  # pragma: no cover
            self.fail(f"validate_eval_records raised {type(exc).__name__}: {exc}")
        self.assertTrue(problems)

    def test_top3_indices_containing_list_rejected_without_raising(self):
        records = self._valid_records()
        records[0] = dict(records[0])
        records[0]["top3_indices"] = [[], 1, 2]
        try:
            problems, _, _ = ec.validate_eval_records(records, self.quota)
        except Exception as exc:  # pragma: no cover
            self.fail(f"validate_eval_records raised {type(exc).__name__}: {exc}")
        self.assertTrue(problems)


# =========================================================================
# 4. CLI surface: no threshold/operator/out override exists
# =========================================================================
class TestCliSurface(unittest.TestCase):
    def test_no_forbidden_cli_flags_exist(self):
        source = inspect.getsource(ev.main)
        for forbidden in ("--threshold", "--operator", "--out", "--geo", "--inference-policy",
                         "--sweep", "--search"):
            self.assertNotIn(forbidden, source)

    def test_preflight_and_evaluate_are_mutually_exclusive_and_required(self):
        source = inspect.getsource(ev.main)
        self.assertIn("mutually_exclusive_group(required=True)", source)
        self.assertIn("--preflight", source)
        self.assertIn("--evaluate", source)

    def test_evaluate_requires_artifacts_dir(self):
        source = inspect.getsource(ev.main)
        self.assertIn("--evaluate requires --artifacts-dir", source)


# =========================================================================
# 5. Preflight isolation: no image, no PIL, no ORT session, no writes
# =========================================================================
class TestPreflightIsolation(unittest.TestCase):
    def test_run_preflight_source_never_imports_pil_or_constructs_a_session(self):
        source = inspect.getsource(ev.run_preflight)
        # Strip the docstring (mentions PIL/ORT only to document they are
        # never used) before checking the CODE body.
        body = source.split('"""', 2)[-1] if source.count('"""') >= 2 else source
        self.assertNotIn("PIL", body)
        self.assertNotIn("Image.open", body)
        self.assertNotIn("InferenceSession", body)
        self.assertNotIn(".write(", body)

    def test_run_evaluate_source_order_all_image_free_init_before_marker(self):
        """Static proof (not merely a passing test run) that EVERY
        image-free, fallible initialization step in run_evaluate's source
        text appears before _create_attempt_marker(, and that the actual
        image-touching calls appear after it. A future edit that
        reintroduces a lazy import (like the numpy-inside-the-loop bug this
        corrects) inside the loop would move it after the marker-creation
        token and fail this test."""
        source = inspect.getsource(ev.run_evaluate)
        marker_call = source.index("_create_attempt_marker(")
        self.assertGreater(marker_call, 0)

        before_tokens = ["import numpy as np", "import PIL", "import inference as api_inference",
                        "import onnxruntime as ort", "load_candidate_session_and_prototypes(",
                        "session.get_providers()"]
        for token in before_tokens:
            idx = source.index(token)
            self.assertLess(idx, marker_call,
                            f"{token!r} must appear before _create_attempt_marker( in run_evaluate")

        after_tokens = ["resolve_one_image(", ".read_bytes()", "Image.open("]
        for token in after_tokens:
            idx = source.index(token)
            self.assertGreater(idx, marker_call,
                               f"{token!r} must appear after _create_attempt_marker( in run_evaluate")


# =========================================================================
# 6. End-to-end one-shot mechanism (tiny synthetic repo + tiny ONNX model)
# =========================================================================
def _cal_score_record_for_row(row, slug_to_idx, top1_idx, max_cosine, n_species):
    slugs = [f"species-{i}" for i in range(n_species)]
    cat = row["category"]
    true_idx = slug_to_idx.get(row["slug"]) if cat == gc.KNOWN_CATEGORY else None
    second_idx = (top1_idx + 1) % n_species
    third_idx = (top1_idx + 2) % n_species
    return {
        "category": cat, "slug": row["slug"], "species": row["species"], "taxon_id": row["taxon_id"],
        "photo_id": row["photo_id"], "observation_uuid": row["observation_uuid"], "image_sha256": row["sha256"],
        "true_class_index": true_idx, "top1_index": top1_idx, "top1_slug": slugs[top1_idx],
        "top3_indices": [top1_idx, second_idx, third_idx],
        "top3_slugs": [slugs[top1_idx], slugs[second_idx], slugs[third_idx]],
        "top3_similarities": [max_cosine, max_cosine - 0.1, max_cosine - 0.2],
        "max_cosine": max_cosine, "top1_correct": (top1_idx == true_idx) if true_idx is not None else None,
    }


def build_tiny_eval_fixture(testcase: unittest.TestCase, tmp: Path) -> dict:
    """Builds a small, internally-consistent repo under `tmp`:

    - a tiny calibration_v2 selection contract + HAND-CRAFTED (not real-ONNX)
      scores.json/selection.json, deterministically engineered to yield
      status=candidate_selected -- calibration scoring math is already
      covered end-to-end by test_gate_v2_calibration.py's own
      TestEndToEndScoring, so it is not re-proven here;
    - a REAL tiny from-scratch ONNX model + REAL tiny synthetic images for
      unknown_test_v2 (the part THIS evaluator actually scores);
    - a tiny evaluation contract bound to all of the above, with every
      scale-dependent gc.FROZEN_*/ec.FROZEN_* constant patched for the
      lifetime of `testcase`.
    """
    n_species = 3
    fx = build_tiny_fixture(testcase, tmp, n_species=n_species, per_species_cal=2, ood_cal=1,
                            non_ant_cal=1, unrelated_cal=1, copy_real_implementation_sources=True)
    repo = fx["repo"]
    selection_contract = fx["contract"]

    for name in ("gate_v2_evaluation_contract", "eval_unknown_test_v2"):
        rel = ec.IMPLEMENTATION_SOURCE_PATHS[name]
        dest = repo / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / rel, dest)

    artifacts_dir = repo / "training/artifacts/fixture_candidate"
    onnx_path = artifacts_dir / "backbone.onnx"
    _build_tiny_onnx_model(onnx_path, n_species)
    selection_contract["content"]["bindings"] = dict(selection_contract["content"]["bindings"])
    selection_contract["content"]["bindings"]["backbone_onnx"] = dict(
        selection_contract["content"]["bindings"]["backbone_onnx"])
    selection_contract["content"]["bindings"]["backbone_onnx"]["sha256"] = gc.sha256_file(onnx_path)

    slug_to_idx = {f"species-{i}": i for i in range(n_species)}

    # ---- real tiny images for unknown_test_v2 (built now, BEFORE the
    # selection contract's content_sha256 is finalized and embedded into
    # the calibration scores/selection bindings below) -----------------
    from PIL import Image

    ut_rows_raw = []
    photo_id = 9000
    for i, slug in enumerate([f"species-{i}" for i in range(n_species)]):
        ut_rows_raw.append(_manifest_row(gc.KNOWN_CATEGORY, slug, 100 + i, photo_id, "placeholder"))
        photo_id += 1
    for cat, taxon in (("out_of_scope_ant", 900), ("non_ant_insect", 901), ("unrelated", 902)):
        ut_rows_raw.append(_manifest_row(cat, f"{cat}-sp", taxon, photo_id, "placeholder"))
        photo_id += 1

    ut_rows = []
    for i, row in enumerate(ut_rows_raw):
        img = Image.new("RGB", (220, 220), ((i * 31) % 256, (i * 53) % 256, (i * 77) % 256))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        data = buf.getvalue()
        slug_dir = repo / "data/unknown_test_v2" / row["slug"]
        slug_dir.mkdir(parents=True, exist_ok=True)
        (slug_dir / f"{row['photo_id']}.png").write_bytes(data)
        row = dict(row)
        row["sha256"] = gc.sha256_bytes(data)
        row["byte_size"] = str(len(data))
        row["width"] = "220"
        row["height"] = "220"
        ut_rows.append(row)

    ut_csv_path = repo / "data/unknown_test_v2/unknown_test_v2.csv"
    _write_csv(ut_csv_path, _manifest_fields(), ut_rows)
    ut_csv_hash = gc.sha256_file(ut_csv_path)
    ut_json_path = repo / "data/unknown_test_v2/unknown_test_v2.json"
    _write_json(ut_json_path, {"manifest": {"sha256": ut_csv_hash, "rows": len(ut_rows)}})
    ut_json_hash = gc.sha256_file(ut_json_path)
    ut_identity_order_sha256 = gc.compute_identity_order_sha256(
        [(r["photo_id"], r["observation_uuid"], r["sha256"]) for r in ut_rows]
    )

    # bind the SAME candidate artifacts + unknown_test_v2 CSV/JSON into the
    # selection contract too, so score_calibration_v2's reused
    # verify_taxonomy_and_prototypes/verify_known_holdout_slugs_map (which
    # take the selection contract, not the eval contract) resolve correctly.
    selection_contract["content"]["bindings"]["unknown_test_v2_csv"] = {
        "path": "data/unknown_test_v2/unknown_test_v2.csv", "sha256": ut_csv_hash}
    selection_contract["content"]["bindings"]["unknown_test_v2_json"] = {
        "path": "data/unknown_test_v2/unknown_test_v2.json", "sha256": ut_json_hash}
    selection_contract["content"]["dataset_quotas"] = dict(selection_contract["content"]["dataset_quotas"])
    selection_contract["content"]["dataset_quotas"]["unknown_test_v2"] = {
        "total": len(ut_rows), "species_count": n_species, "known_holdout": n_species,
        "known_holdout_per_species": 1, "out_of_scope_ant": 1, "non_ant_insect": 1, "unrelated": 1,
        "purpose": gc.FROZEN_UNKNOWN_TEST_V2_PURPOSE,
    }
    selection_contract["content_sha256"] = gc.compute_content_sha256(selection_contract["content"])
    _write_json(fx["contract_path"], selection_contract)

    gc.FROZEN_DATASET_QUOTAS["unknown_test_v2"] = {
        "total": len(ut_rows), "species_count": n_species, "known_holdout": n_species,
        "known_holdout_per_species": 1, "out_of_scope_ant": 1, "non_ant_insect": 1, "unrelated": 1,
    }
    gc.FROZEN_BINDING_PATHS["unknown_test_v2_csv"] = "data/unknown_test_v2/unknown_test_v2.csv"
    gc.FROZEN_BINDING_PATHS["unknown_test_v2_json"] = "data/unknown_test_v2/unknown_test_v2.json"

    # ---- hand-crafted, deterministic calibration_v2 scores/selection -----
    cal_rows = fx["cal_rows"]
    known_rows = [r for r in cal_rows if r["category"] == gc.KNOWN_CATEGORY]
    ood_rows = [r for r in cal_rows if r["category"] != gc.KNOWN_CATEGORY]
    cal_records = []
    for i, row in enumerate(known_rows):
        idx = slug_to_idx[row["slug"]]
        if i < 4:
            cal_records.append(_cal_score_record_for_row(row, slug_to_idx, idx, 0.9, n_species))
        else:
            wrong = (idx + 1) % n_species
            cal_records.append(_cal_score_record_for_row(row, slug_to_idx, wrong, 0.2, n_species))
    for row in ood_rows:
        cal_records.append(_cal_score_record_for_row(row, slug_to_idx, 0, 0.1, n_species))
    cal_records.sort(key=lambda r: [row["photo_id"] for row in cal_rows].index(r["photo_id"]))

    identity_order_sha256 = gc.compute_identity_order_sha256(
        [(r["photo_id"], r["observation_uuid"], r["sha256"]) for r in cal_rows]
    )
    impl_hashes = {name: gc.canonical_lf_sha256_file(repo / path)
                   for name, path in gc.IMPLEMENTATION_SOURCE_PATHS.items()}
    fake_git_head = "a" * 40
    scores_content = {
        "dataset": "calibration_v2", "row_order": "calibration_v2.csv row order, preserved exactly",
        "n_rows": len(cal_records),
        "bindings": {
            "contract_content_sha256": selection_contract["content_sha256"],
            "calibration_v2_csv_sha256": selection_contract["content"]["bindings"]["calibration_v2_csv"]["sha256"],
            gc.CONTENT_KEY_IDENTITY_ORDER_SHA256: identity_order_sha256,
        },
        "runtime": {
            "onnxruntime_version": "0.0.0-fixture", "registered_providers": ["CPUExecutionProvider"],
            "pillow_version": "0.0.0-fixture", "preprocessing_contract": {"image_size": 380},
        },
        "provenance": {"git_head": fake_git_head, "implementation_source_hashes": impl_hashes},
        "records": cal_records,
    }
    scores_doc = {"schema_version": gc.SCORE_SCHEMA_VERSION, "content": scores_content,
                 "content_sha256": gc.compute_content_sha256(scores_content)}
    scores_path = repo / "data/calibration_v2/calibration_v2_scores.json"
    _write_json(scores_path, scores_doc)

    problems = gc.validate_score_content(scores_doc, selection_contract)
    if problems:
        raise AssertionError(f"fixture scores failed self-validation: {problems}")

    result = sel.select_threshold(scores_doc, selection_contract)
    if result["status"] != gc.STATUS_CANDIDATE_SELECTED:
        raise AssertionError(f"fixture calibration data did not yield candidate_selected: {result}")
    selection_content = {
        "dataset": gc.DATASET_NAME, "status": result["status"],
        "bindings": {"contract_content_sha256": selection_contract["content_sha256"],
                    "scores_content_sha256": scores_doc["content_sha256"]},
        "provenance": {"git_head": fake_git_head, "implementation_source_hashes": impl_hashes},
        "result": {k: v for k, v in result.items() if k != "diagnostics"},
        "diagnostics": result["diagnostics"],
    }
    selection_doc = {"schema_version": gc.SELECTION_SCHEMA_VERSION, "content": selection_content,
                     "content_sha256": gc.compute_content_sha256(selection_content)}
    selection_path = repo / "data/calibration_v2/calibration_v2_selection.json"
    _write_json(selection_path, selection_doc)

    fixture_candidate_paths = {
        "candidate_run_manifest": "training/artifacts/fixture_candidate/run_manifest.json",
        "backbone_onnx": "training/artifacts/fixture_candidate/backbone.onnx",
        "prototypes_npy": "training/artifacts/fixture_candidate/prototypes.npy",
        "taxonomy_json": "training/artifacts/fixture_candidate/taxonomy.json",
    }
    fixture_candidate_hashes = {
        "candidate_run_manifest": gc.sha256_file(artifacts_dir / "run_manifest.json"),
        "backbone_onnx": gc.sha256_file(onnx_path),
        "prototypes_npy": gc.sha256_file(artifacts_dir / "prototypes.npy"),
        "taxonomy_json": gc.sha256_file(artifacts_dir / "taxonomy.json"),
    }
    fixture_ut_paths = {"unknown_test_v2_csv": "data/unknown_test_v2/unknown_test_v2.csv",
                        "unknown_test_v2_json": "data/unknown_test_v2/unknown_test_v2.json"}
    fixture_ut_hashes = {"unknown_test_v2_csv": ut_csv_hash, "unknown_test_v2_json": ut_json_hash}
    fixture_binding_paths = {**fixture_candidate_paths, **fixture_ut_paths}
    fixture_binding_hashes = {**fixture_candidate_hashes, **fixture_ut_hashes}
    fixture_quota = dict(gc.FROZEN_DATASET_QUOTAS["unknown_test_v2"])

    ec_patcher = mock.patch.multiple(
        ec,
        FROZEN_SELECTION_CONTRACT_CONTENT_SHA256=selection_contract["content_sha256"],
        FROZEN_SCORES_BYTE_SHA256=gc.sha256_file(scores_path),
        FROZEN_SCORES_CONTENT_SHA256=scores_doc["content_sha256"],
        FROZEN_SELECTION_BYTE_SHA256=gc.sha256_file(selection_path),
        FROZEN_SELECTION_CONTENT_SHA256=selection_doc["content_sha256"],
        FROZEN_THRESHOLD_INTEGER=result["candidate"]["threshold_integer"],
        FROZEN_THRESHOLD=result["candidate"]["threshold"],
        FROZEN_CANDIDATE_ARTIFACT_PATHS=fixture_candidate_paths,
        FROZEN_CANDIDATE_ARTIFACT_HASHES=fixture_candidate_hashes,
        FROZEN_UNKNOWN_TEST_V2_BINDING_PATHS=fixture_ut_paths,
        FROZEN_UNKNOWN_TEST_V2_HASHES=fixture_ut_hashes,
        FROZEN_BINDING_PATHS=fixture_binding_paths,
        FROZEN_BINDING_HASHES=fixture_binding_hashes,
        FROZEN_DATASET_QUOTA=fixture_quota,
    )
    ec_patcher.start()
    testcase.addCleanup(ec_patcher.stop)

    eval_impl_hashes = {name: gc.canonical_lf_sha256_file(repo / path)
                        for name, path in ec.IMPLEMENTATION_SOURCE_PATHS.items()}
    eval_content = {
        "policy_name": ec.FROZEN_POLICY_NAME,
        "notes": "fixture",
        "selection_contract": {"content_sha256": ec.FROZEN_SELECTION_CONTRACT_CONTENT_SHA256},
        "scores_binding": {"byte_sha256": ec.FROZEN_SCORES_BYTE_SHA256,
                          "content_sha256": ec.FROZEN_SCORES_CONTENT_SHA256},
        "selection_binding": {"byte_sha256": ec.FROZEN_SELECTION_BYTE_SHA256,
                             "content_sha256": ec.FROZEN_SELECTION_CONTENT_SHA256},
        "threshold": {
            "threshold_integer": ec.FROZEN_THRESHOLD_INTEGER, "threshold": ec.FROZEN_THRESHOLD,
            "source": ec.FROZEN_THRESHOLD_SOURCE, "comparison": ec.FROZEN_DECISION_SHAPE["comparison"],
            "equal_threshold_action": ec.FROZEN_DECISION_SHAPE["equal_threshold_action"],
            "required_status": ec.FROZEN_THRESHOLD_STATUS_REQUIRED,
        },
        "bindings": {name: {"path": fixture_binding_paths[name], "sha256": fixture_binding_hashes[name]}
                    for name in fixture_binding_paths},
        "implementation_sources": {name: {"path": ec.IMPLEMENTATION_SOURCE_PATHS[name],
                                         "sha256": eval_impl_hashes[name]}
                                  for name in ec.IMPLEMENTATION_SOURCE_KEYS},
        "approved_outputs": dict(ec.FROZEN_APPROVED_OUTPUTS),
        "dataset_quota": fixture_quota,
        "runtime": dict(ec.FROZEN_RUNTIME),
        "decision_shape": dict(ec.FROZEN_DECISION_SHAPE),
        "validation_criteria": dict(ec.FROZEN_VALIDATION_CRITERIA),
        "single_use_rule": dict(ec.FROZEN_SINGLE_USE_RULE),
        "closed_sources": list(ec.FROZEN_CLOSED_SOURCES),
        "gate_framing": ec.FROZEN_GATE_FRAMING,
        ec.CONTENT_KEY_UNKNOWN_TEST_V2_IDENTITY_ORDER_SHA256: ut_identity_order_sha256,
    }
    eval_contract = {
        "schema_version": ec.EVAL_CONTRACT_SCHEMA_VERSION, "status": ec.EVAL_CONTRACT_STATUS_FROZEN,
        "content": eval_content, "content_sha256": ec.compute_content_sha256(eval_content),
        "generation": {"note": "fixture", "generator": "test"},
    }
    eval_contract_path = repo / "training/gate_v2_evaluation_contract.json"
    _write_json(eval_contract_path, eval_contract)

    problems = ec.validate_evaluation_contract(eval_contract)
    if problems:
        raise AssertionError(f"fixture eval contract failed self-validation: {problems}")

    return {
        "repo": repo, "eval_contract": eval_contract, "eval_contract_path": eval_contract_path,
        "selection_contract_path": fx["contract_path"], "scores_path": scores_path,
        "selection_path": selection_path, "ut_csv_path": ut_csv_path, "ut_json_path": ut_json_path,
        "artifacts_dir": artifacts_dir, "ut_rows": ut_rows,
    }


def _eval_args(fx, **overrides):
    # Mirrors the real CLI surface exactly: only --repo and --artifacts-dir
    # are ever settable. Every other path is derived from `repo` alone by
    # eval_unknown_test_v2.py's contract_path_for() and friends -- there is
    # no way to point the evaluator at an input file living somewhere else.
    import argparse
    d = dict(repo=fx["repo"], artifacts_dir=fx["artifacts_dir"], preflight=True, evaluate=False)
    d.update(overrides)
    return argparse.Namespace(**d)


class TestPublishEvalOutput(unittest.TestCase):
    """Exercises eval_unknown_test_v2.publish_eval_output() directly -- the
    actual publication boundary -- rather than only the marker-based
    second-invocation test, which stops earlier (at marker creation) and
    never reaches this code path."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.out_path = Path(self._tmpdir.name) / "unknown_test_v2_eval.json"

    def _tmp_glob(self):
        return list(self.out_path.parent.glob(f"{self.out_path.name}.tmp*"))

    def test_successful_publication_writes_complete_bytes_and_leaves_no_temp_file(self):
        text = '{"hello": "world"}\n'
        ev.publish_eval_output(self.out_path, text)
        self.assertEqual(self.out_path.read_text(encoding="utf-8"), text)
        self.assertEqual(self._tmp_glob(), [])

    def test_publication_refuses_to_overwrite_and_preserves_sentinel_bytes(self):
        sentinel = b"SENTINEL-DO-NOT-TOUCH"
        self.out_path.write_bytes(sentinel)
        with self.assertRaises(ev.EvaluationRunError):
            ev.publish_eval_output(self.out_path, '{"new": "content"}\n')
        self.assertEqual(self.out_path.read_bytes(), sentinel)
        self.assertEqual(self._tmp_glob(), [])

    def test_publication_does_not_use_os_replace(self):
        source = inspect.getsource(ev.publish_eval_output)
        # Strip the docstring (which discusses os.replace() only to explain
        # why it was replaced) before checking the CODE body.
        body = source.split('"""', 2)[-1] if source.count('"""') >= 2 else source
        self.assertNotIn("os.replace(", body)
        self.assertIn("os.link(", body)


class TestOneShotMechanism(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.fx = build_tiny_eval_fixture(self, Path(self._tmpdir.name))
        # The fixture lives under a plain tempdir, not a git checkout --
        # require_clean_tracked_tree/get_git_head are patched at the shared
        # gc level (ec.require_clean_tracked_tree/get_git_head both
        # delegate to these) so the one-shot mechanism itself is what's
        # under test, not git plumbing.
        tree_patcher = mock.patch.object(gc, "require_clean_tracked_tree")
        head_patcher = mock.patch.object(gc, "get_git_head", return_value="a" * 40)
        tree_patcher.start()
        head_patcher.start()
        self.addCleanup(tree_patcher.stop)
        self.addCleanup(head_patcher.stop)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_preflight_succeeds_and_touches_no_image(self):
        args = _eval_args(self.fx)
        with mock.patch("PIL.Image.open", side_effect=AssertionError("must not open an image")):
            result = ev.run_preflight(args)
        self.assertTrue(result["ok"])
        marker_path = Path(result["marker_path"])
        eval_out_path = Path(result["eval_out_path"])
        self.assertFalse(marker_path.exists())
        self.assertFalse(eval_out_path.exists())

    def test_preflight_creates_no_onnx_session(self):
        args = _eval_args(self.fx)
        with mock.patch("onnxruntime.InferenceSession",
                        side_effect=AssertionError("must not construct a session")):
            ev.run_preflight(args)

    def test_marker_created_before_first_image_access_and_evaluate_succeeds(self):
        args = _eval_args(self.fx, evaluate=True, preflight=False)
        opened_images = []
        marker_path = self.fx["repo"] / "data/unknown_test_v2/unknown_test_v2_evaluation_attempt.json"
        real_open = __import__("PIL.Image", fromlist=["Image"]).open

        def _tracking_open(*a, **k):
            # The whole point of this test: assert INSIDE the callback,
            # at the moment of the very first image open, that the marker
            # already exists -- checking only after run_evaluate returns
            # would not prove the ORDER of operations.
            opened_images.append(True)
            if len(opened_images) == 1:
                self.assertTrue(marker_path.exists(),
                                "attempt marker must already exist before the first image is opened")
            return real_open(*a, **k)

        self.assertFalse(marker_path.exists())
        with mock.patch("PIL.Image.open", side_effect=_tracking_open):
            output = ev.run_evaluate(args)
        self.assertTrue(marker_path.exists())
        self.assertTrue(opened_images)
        self.assertIn(output["content"]["validation"]["status"],
                      (ec.EVAL_STATUS_PASSED, ec.EVAL_STATUS_FAILED))
        eval_out_path = self.fx["repo"] / "data/unknown_test_v2/unknown_test_v2_eval.json"
        self.assertTrue(eval_out_path.exists())

    def test_session_construction_failure_leaves_no_marker_and_no_result(self):
        args = _eval_args(self.fx, evaluate=True, preflight=False)
        marker_path = self.fx["repo"] / "data/unknown_test_v2/unknown_test_v2_evaluation_attempt.json"
        eval_out_path = self.fx["repo"] / "data/unknown_test_v2/unknown_test_v2_eval.json"
        with mock.patch("onnxruntime.InferenceSession",
                        side_effect=RuntimeError("simulated session-construction failure")):
            with self.assertRaises(RuntimeError):
                ev.run_evaluate(args)
        self.assertFalse(marker_path.exists())
        self.assertFalse(eval_out_path.exists())

    def test_prototype_or_taxonomy_load_failure_leaves_no_marker_and_no_result(self):
        args = _eval_args(self.fx, evaluate=True, preflight=False)
        marker_path = self.fx["repo"] / "data/unknown_test_v2/unknown_test_v2_evaluation_attempt.json"
        eval_out_path = self.fx["repo"] / "data/unknown_test_v2/unknown_test_v2_eval.json"
        with mock.patch.object(sc, "load_candidate_session_and_prototypes",
                               side_effect=sc.ScoringError("simulated taxonomy/prototype load failure")):
            with self.assertRaises(sc.ScoringError):
                ev.run_evaluate(args)
        self.assertFalse(marker_path.exists())
        self.assertFalse(eval_out_path.exists())

    def test_second_invocation_is_rejected(self):
        args = _eval_args(self.fx, evaluate=True, preflight=False)
        ev.run_evaluate(args)
        with self.assertRaises((ev.EvaluationRunError, ec.EvaluationError, gc.ContractError)):
            ev.run_evaluate(args)

    def test_marker_exists_but_no_eval_output_blocks_a_retry(self):
        args = _eval_args(self.fx)
        pre = ev.run_preflight(args)
        marker_path = Path(pre["marker_path"])
        marker_path.parent.mkdir(parents=True, exist_ok=True)
        marker_path.write_text('{"schema_version": 1, "content": {}, "content_sha256": "x"}', encoding="utf-8")
        eval_args = _eval_args(self.fx, evaluate=True, preflight=False)
        with self.assertRaises((ev.EvaluationRunError, ec.EvaluationError, gc.ContractError)):
            ev.run_evaluate(eval_args)

    def test_simulated_post_marker_failure_leaves_marker_and_no_result(self):
        args = _eval_args(self.fx, evaluate=True, preflight=False)
        marker_path = self.fx["repo"] / "data/unknown_test_v2/unknown_test_v2_evaluation_attempt.json"
        eval_out_path = self.fx["repo"] / "data/unknown_test_v2/unknown_test_v2_eval.json"
        with mock.patch.object(sc, "decode_and_validate_image",
                               side_effect=sc.ScoringError("simulated post-marker failure")):
            with self.assertRaises(sc.ScoringError):
                ev.run_evaluate(args)
        self.assertTrue(marker_path.exists())
        self.assertFalse(eval_out_path.exists())
        leftovers = list(eval_out_path.parent.glob(f"{eval_out_path.name}.tmp*"))
        self.assertEqual(leftovers, [])

    def test_marker_is_never_overwritten_by_a_second_run_attempt(self):
        args = _eval_args(self.fx, evaluate=True, preflight=False)
        marker_path = self.fx["repo"] / "data/unknown_test_v2/unknown_test_v2_evaluation_attempt.json"
        with mock.patch.object(sc, "decode_and_validate_image",
                               side_effect=sc.ScoringError("simulated failure")):
            with self.assertRaises(sc.ScoringError):
                ev.run_evaluate(args)
        before = marker_path.read_bytes()
        with self.assertRaises((ev.EvaluationRunError, ec.EvaluationError, gc.ContractError)):
            ev.run_evaluate(args)
        self.assertEqual(marker_path.read_bytes(), before)

    def test_eval_output_refuses_overwrite_and_is_atomic(self):
        args = _eval_args(self.fx, evaluate=True, preflight=False)
        output = ev.run_evaluate(args)
        eval_out_path = self.fx["repo"] / "data/unknown_test_v2/unknown_test_v2_eval.json"
        before = eval_out_path.read_bytes()
        with self.assertRaises((ev.EvaluationRunError, ec.EvaluationError, gc.ContractError)):
            ev.run_evaluate(args)
        self.assertEqual(eval_out_path.read_bytes(), before)

    def test_evaluate_output_reloads_and_self_validates(self):
        args = _eval_args(self.fx, evaluate=True, preflight=False)
        output = ev.run_evaluate(args)
        contract = ec.load_and_verify_evaluation_contract(self.fx["eval_contract_path"])
        reloaded = ec.load_and_verify_eval_file(
            self.fx["repo"] / "data/unknown_test_v2/unknown_test_v2_eval.json", contract, self.fx["repo"])
        self.assertEqual(reloaded["content_sha256"], output["content_sha256"])

    def test_load_and_verify_eval_file_rejects_marker_byte_hash_mismatch(self):
        args = _eval_args(self.fx, evaluate=True, preflight=False)
        ev.run_evaluate(args)
        contract = ec.load_and_verify_evaluation_contract(self.fx["eval_contract_path"])
        marker_path = self.fx["repo"] / "data/unknown_test_v2/unknown_test_v2_evaluation_attempt.json"
        # Append trailing whitespace: still syntactically the same JSON
        # object once parsed, but a DIFFERENT byte sha256 -- a merely
        # well-formed marker must not be enough.
        marker_path.write_bytes(marker_path.read_bytes() + b" ")
        eval_out_path = self.fx["repo"] / "data/unknown_test_v2/unknown_test_v2_eval.json"
        with self.assertRaises(ec.EvaluationError):
            ec.load_and_verify_eval_file(eval_out_path, contract, self.fx["repo"])

    def test_redirected_unknown_test_v2_csv_copy_does_not_affect_image_resolution(self):
        # A byte-identical copy of unknown_test_v2.csv placed beside a
        # DIFFERENT (here: empty) image tree must never cause the evaluator
        # to resolve images from that alternate directory -- there is no
        # CLI flag to point it there, and the path is derived from --repo
        # alone.
        decoy_dir = Path(self._tmpdir.name).parent / f"{Path(self._tmpdir.name).name}-decoy"
        decoy_dir.mkdir(parents=True, exist_ok=True)
        self.addCleanup(shutil.rmtree, decoy_dir, True)
        real_csv = ev.unknown_test_csv_path_for(self.fx["repo"])
        shutil.copyfile(real_csv, decoy_dir / "unknown_test_v2.csv")
        # decoy_dir intentionally has NO species image subfolders.
        self.assertEqual(
            [p.name for p in decoy_dir.iterdir()], ["unknown_test_v2.csv"],
        )

        args = _eval_args(self.fx, evaluate=True, preflight=False)
        output = ev.run_evaluate(args)  # must succeed, reading only fx["repo"]'s own images
        self.assertEqual(len(output["content"]["records"]), len(self.fx["ut_rows"]))

    def test_no_access_to_inference_policy_or_geo_index(self):
        source = inspect.getsource(ev.run_evaluate) + inspect.getsource(ec.compute_validation)
        self.assertNotIn("inference_policy", source)
        self.assertNotIn("geo_index", source)
        self.assertNotIn("AntIdentifier(", source)


# =========================================================================
# 7. Output validation is DERIVED from records, not merely internally
#    consistent -- mutate a REAL evaluation output's validation block,
#    rehash content_sha256 fresh, and require rejection.
# =========================================================================
class TestOutputValidationIsRecomputed(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.fx = build_tiny_eval_fixture(self, Path(self._tmpdir.name))
        tree_patcher = mock.patch.object(gc, "require_clean_tracked_tree")
        head_patcher = mock.patch.object(gc, "get_git_head", return_value="a" * 40)
        tree_patcher.start()
        head_patcher.start()
        self.addCleanup(tree_patcher.stop)
        self.addCleanup(head_patcher.stop)
        args = _eval_args(self.fx, evaluate=True, preflight=False)
        self.output = ev.run_evaluate(args)
        self.contract = ec.load_and_verify_evaluation_contract(self.fx["eval_contract_path"])

    def tearDown(self):
        self._tmpdir.cleanup()

    def _mutate(self, mutator):
        output = json.loads(json.dumps(self.output))  # deep copy
        mutator(output["content"]["validation"])
        output["content_sha256"] = gc.compute_content_sha256(output["content"])
        return output

    def test_real_output_validates_cleanly(self):
        problems = ec.validate_eval_content(self.output, self.contract)
        self.assertEqual(problems, [])

    def test_status_flipped_rejected(self):
        def mutate(v):
            v["status"] = (ec.EVAL_STATUS_FAILED if v["status"] == ec.EVAL_STATUS_PASSED
                           else ec.EVAL_STATUS_PASSED)
        output = self._mutate(mutate)
        problems = ec.validate_eval_content(output, self.contract)
        self.assertTrue(any("validation.status does not match" in p for p in problems))

    def test_criterion_flipped_rejected(self):
        def mutate(v):
            v["criteria"] = dict(v["criteria"])
            v["criteria"]["coverage_ok"] = not v["criteria"]["coverage_ok"]
        output = self._mutate(mutate)
        problems = ec.validate_eval_content(output, self.contract)
        self.assertTrue(any("validation.criteria does not match" in p for p in problems))

    def test_baseline_metric_altered_rejected(self):
        def mutate(v):
            v["metrics"] = dict(v["metrics"])
            v["metrics"]["baseline"] = dict(v["metrics"]["baseline"])
            v["metrics"]["baseline"]["correct_top1"] = v["metrics"]["baseline"]["correct_top1"] + 1
        output = self._mutate(mutate)
        problems = ec.validate_eval_content(output, self.contract)
        self.assertTrue(any("validation.metrics does not match" in p for p in problems))

    def test_improvement_over_baseline_altered_rejected(self):
        def mutate(v):
            v["metrics"] = dict(v["metrics"])
            v["metrics"]["improvement_over_baseline_pp"] = 999.0
        output = self._mutate(mutate)
        problems = ec.validate_eval_content(output, self.contract)
        self.assertTrue(any("validation.metrics does not match" in p for p in problems))

    def test_per_species_entry_altered_rejected(self):
        def mutate(v):
            v["metrics"] = dict(v["metrics"])
            per_species = [dict(e) for e in v["metrics"]["per_species_known"]]
            per_species[0]["n"] = per_species[0]["n"] + 1000
            v["metrics"]["per_species_known"] = per_species
        output = self._mutate(mutate)
        problems = ec.validate_eval_content(output, self.contract)
        self.assertTrue(any("validation.metrics does not match" in p for p in problems))

    def test_diagnostic_ood_far_altered_rejected(self):
        def mutate(v):
            v["metrics"] = dict(v["metrics"])
            far = {k: dict(val) for k, val in v["metrics"]["diagnostic_ood_far"].items()}
            for entry in far.values():
                entry["false_acceptance_rate"] = 0.0
            v["metrics"]["diagnostic_ood_far"] = far
        output = self._mutate(mutate)
        problems = ec.validate_eval_content(output, self.contract)
        self.assertTrue(any("validation.metrics does not match" in p for p in problems))

    def test_diagnostic_auc_altered_rejected(self):
        def mutate(v):
            v["metrics"] = dict(v["metrics"])
            v["metrics"]["diagnostic_auc_known_vs_ood"] = {
                k: 0.5 for k in v["metrics"]["diagnostic_auc_known_vs_ood"]
            }
        output = self._mutate(mutate)
        problems = ec.validate_eval_content(output, self.contract)
        self.assertTrue(any("validation.metrics does not match" in p for p in problems))

    def test_extra_key_in_validation_rejected(self):
        def mutate(v):
            v["sneaky"] = "smuggled"
        output = self._mutate(mutate)
        problems = ec.validate_eval_content(output, self.contract)
        self.assertTrue(any("validation has unexpected key" in p for p in problems))

    def test_missing_key_in_validation_rejected(self):
        def mutate(v):
            del v["criteria"]
        output = self._mutate(mutate)
        problems = ec.validate_eval_content(output, self.contract)
        self.assertTrue(any("validation missing key" in p for p in problems))

    def test_extra_key_in_a_record_rejected(self):
        output = json.loads(json.dumps(self.output))
        output["content"]["records"][0]["sneaky"] = "smuggled"
        output["content_sha256"] = gc.compute_content_sha256(output["content"])
        problems = ec.validate_eval_content(output, self.contract)
        self.assertTrue(any("unexpected key" in p for p in problems))

    def test_extra_key_in_runtime_rejected(self):
        output = json.loads(json.dumps(self.output))
        output["content"]["runtime"] = dict(output["content"]["runtime"])
        output["content"]["runtime"]["sneaky"] = "smuggled"
        output["content_sha256"] = gc.compute_content_sha256(output["content"])
        problems = ec.validate_eval_content(output, self.contract)
        self.assertTrue(any("runtime has unexpected key" in p for p in problems))

    def test_extra_key_in_provenance_rejected(self):
        output = json.loads(json.dumps(self.output))
        output["content"]["provenance"] = dict(output["content"]["provenance"])
        output["content"]["provenance"]["sneaky"] = "smuggled"
        output["content_sha256"] = gc.compute_content_sha256(output["content"])
        problems = ec.validate_eval_content(output, self.contract)
        self.assertTrue(any("provenance has unexpected key" in p for p in problems))

    def test_extra_key_at_evaluation_top_level_rejected(self):
        output = json.loads(json.dumps(self.output))
        output["sneaky"] = "smuggled"
        problems = ec.validate_eval_content(output, self.contract)
        self.assertTrue(any("unexpected top-level key" in p for p in problems))

    def test_marker_validation_rejects_altered_created_at(self):
        marker_path = self.fx["repo"] / "data/unknown_test_v2/unknown_test_v2_evaluation_attempt.json"
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        marker["content"] = dict(marker["content"])
        marker["content"]["created_at_utc"] = "not-a-timestamp"
        marker["content_sha256"] = gc.compute_content_sha256(marker["content"])
        problems = ec.validate_attempt_marker(marker, self.contract)
        self.assertTrue(any("created_at_utc must be a strict UTC timestamp" in p for p in problems))

    def test_marker_validation_rejects_bindings_mismatch(self):
        marker_path = self.fx["repo"] / "data/unknown_test_v2/unknown_test_v2_evaluation_attempt.json"
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        marker["content"] = dict(marker["content"])
        marker["content"]["bindings"] = {}
        marker["content_sha256"] = gc.compute_content_sha256(marker["content"])
        problems = ec.validate_attempt_marker(marker, self.contract)
        self.assertTrue(any("marker.content.bindings does not exactly match" in p for p in problems))

    def test_marker_validation_rejects_implementation_hash_tamper(self):
        marker_path = self.fx["repo"] / "data/unknown_test_v2/unknown_test_v2_evaluation_attempt.json"
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        marker["content"] = dict(marker["content"])
        hashes = dict(marker["content"]["implementation_source_hashes"])
        hashes["gate_v2_contract"] = "0" * 64
        marker["content"]["implementation_source_hashes"] = hashes
        marker["content_sha256"] = gc.compute_content_sha256(marker["content"])
        problems = ec.validate_attempt_marker(marker, self.contract)
        self.assertTrue(any("implementation_source_hashes does not exactly match" in p for p in problems))


if __name__ == "__main__":
    unittest.main(verbosity=2)
