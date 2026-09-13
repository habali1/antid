#!/usr/bin/env python3
"""test_generate_policy_v2.py — synthetic/offline tests for
generate_inference_policy_v2.py (Phase 5E1).

No test in this file writes training/artifacts/inference_policy.json (the
live serving policy) or any file under the live training/artifacts root.
Every generator test builds its own tiny synthetic Gate v2 evidence chain
(reusing test_gate_v2_evaluation.py's fixture machinery, including a real
tiny from-scratch ONNX model) under a temporary directory, plus a tiny
synthetic parity report with FROZEN_PARITY_* constants patched to match.

Run directly: python test_generate_policy_v2.py
"""
from __future__ import annotations

import argparse
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
import eval_unknown_test_v2 as ev2  # noqa: E402
import generate_inference_policy_v2 as g2  # noqa: E402
import policy_schema as schema  # noqa: E402

from test_gate_v2_evaluation import build_tiny_eval_fixture, _eval_args  # noqa: E402
from test_gate_v2_calibration import _write_json  # noqa: E402


def _fake_passing_evaluation() -> dict:
    """A minimal, correctly-shaped stand-in for ec.load_and_verify_eval_file's
    return value, carrying only the fields generate_inference_policy_v2.py
    actually reads. Used to decouple "does the generator correctly consume
    a passing independent-evaluation result" from "does this tiny fixture's
    real (seed-42) ONNX scoring happen to pass" -- the latter is already
    covered by test_gate_v2_evaluation.py's own 87 tests and is NOT
    guaranteed to pass for an arbitrary tiny synthetic model/image set."""
    return {
        "content_sha256": "f" * 64,
        "content": {
            "validation": {
                "status": ec.EVAL_STATUS_PASSED,
                "threshold_integer": ec.FROZEN_THRESHOLD_INTEGER,
                "threshold": ec.FROZEN_THRESHOLD,
                "metrics": {"diagnostic_ood_far": {"out_of_scope_ant": {"false_acceptance_rate": 0.51}}},
            },
        },
    }


def patch_passing_evaluation(testcase: unittest.TestCase) -> None:
    patcher = mock.patch.object(ec, "load_and_verify_eval_file", return_value=_fake_passing_evaluation())
    patcher.start()
    testcase.addCleanup(patcher.stop)


def _validation_evidence_from_preflight(pre: dict) -> dict:
    """Mirrors build_policy()'s own validation_evidence construction
    exactly (duplicated here deliberately, as test-only scaffolding) so a
    fixture's EXPECTED_V2_VALIDATION_EVIDENCE can be computed from a
    successful preflight BEFORE schema.validate() (which now checks this
    dict for exact equality) is ever invoked -- avoiding the chicken-and-
    egg problem of build_policy() calling schema.validate() internally."""
    return {
        "gate_v2_selection_contract_content_sha256": pre["selection_contract"]["content_sha256"],
        "calibration_v2_scores_byte_sha256": gc.sha256_file(pre["scores_path"]),
        "calibration_v2_scores_content_sha256": pre["scores"]["content_sha256"],
        "calibration_v2_selection_byte_sha256": gc.sha256_file(pre["selection_path"]),
        "calibration_v2_selection_content_sha256": pre["selection_doc"]["content_sha256"],
        "gate_v2_evaluation_contract_content_sha256": pre["eval_contract"]["content_sha256"],
        "unknown_test_v2_evaluation_attempt_byte_sha256": gc.sha256_file(pre["marker_path"]),
        "unknown_test_v2_eval_byte_sha256": gc.sha256_file(pre["eval_out_path"]),
        "unknown_test_v2_eval_content_sha256": pre["evaluation"]["content_sha256"],
        "parity_report_byte_sha256": pre["parity_byte_hash"],
        "validation_status": pre["evaluation"]["content"]["validation"]["status"],
        "diagnostic_out_of_scope_ant_far": pre["diagnostic_far"],
    }


def build_tiny_policy_v2_fixture(testcase: unittest.TestCase, tmp: Path, *, run_evaluate: bool = True) -> dict:
    """Builds a tiny synthetic Gate v2 evidence chain (calibration,
    evaluation contract, and -- unless run_evaluate=False -- a real
    one-shot evaluation run against a real tiny ONNX model) PLUS a tiny
    synthetic parity report, with generate_inference_policy_v2's
    FROZEN_PARITY_* constants patched to match for the lifetime of
    `testcase`."""
    fx = build_tiny_eval_fixture(testcase, tmp)
    repo = fx["repo"]

    # generate_inference_policy_v2's own provenance sources (correction 5)
    # are not among build_tiny_eval_fixture's copied implementation
    # sources -- copy the REAL project files into the fixture repo at the
    # same relative paths so compute_source_provenance(repo) can hash them.
    for rel in g2.PROVENANCE_SOURCE_PATHS.values():
        dest = repo / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / rel, dest)

    tree_patcher = mock.patch.object(gc, "require_clean_tracked_tree")
    head_patcher = mock.patch.object(gc, "get_git_head", return_value="a" * 40)
    tree_patcher.start()
    head_patcher.start()
    testcase.addCleanup(tree_patcher.stop)
    testcase.addCleanup(head_patcher.stop)

    patch_passing_evaluation(testcase)

    # build_tiny_eval_fixture already patched ec.FROZEN_THRESHOLD_INTEGER/
    # FROZEN_THRESHOLD to whatever this fixture's REAL (hand-crafted but
    # genuinely computed) calibration selection actually produced -- NOT
    # necessarily 61/0.61. Schema v2's own FROZEN_THRESHOLD_V2 is a fixed
    # production constant (always 0.61) that this generator's decision/rule
    # block is built around, so it must be patched to match the fixture's
    # actual threshold too, or build_policy() would produce a schema-valid
    # policy whose rule.value disagrees with the fixture's own evidence.
    fixture_threshold = ec.FROZEN_THRESHOLD
    schema_patcher = mock.patch.multiple(
        schema,
        FROZEN_THRESHOLD_V2=fixture_threshold,
        SCHEMA_VERSION_TO_THRESHOLD={schema.SCHEMA_VERSION: schema.FROZEN_THRESHOLD,
                                    schema.SCHEMA_VERSION_V2: fixture_threshold},
        SCHEMA_VERSION_TO_PROBES={
            schema.SCHEMA_VERSION: schema.SCHEMA_VERSION_TO_PROBES[schema.SCHEMA_VERSION],
            schema.SCHEMA_VERSION_V2: ((round(fixture_threshold - 0.0001, 4), True),
                                       (fixture_threshold, False),
                                       (round(fixture_threshold + 0.0001, 4), False)),
        },
    )
    schema_patcher.start()
    testcase.addCleanup(schema_patcher.stop)

    if run_evaluate:
        args = _eval_args(fx, evaluate=True, preflight=False)
        ev2.run_evaluate(args)

    candidate_hashes = {
        "backbone.onnx": gc.sha256_file(fx["artifacts_dir"] / "backbone.onnx"),
        "prototypes.npy": gc.sha256_file(fx["artifacts_dir"] / "prototypes.npy"),
        "taxonomy.json": gc.sha256_file(fx["artifacts_dir"] / "taxonomy.json"),
    }
    n_images = 3
    per_image = [{"top1_agree": True, "top3_set_agree": True} for _ in range(n_images)]
    parity = {
        "deterministic_content": {
            "candidate": {"final_artifact_hashes": dict(candidate_hashes)},
            "model_parity": {
                "n_images": n_images, "per_image": per_image,
                "max_cosine_abs_divergence": {"max": 1.23e-06},
            },
        },
        "generation_metadata": {"workspace_git_dirty": True},
    }
    parity_path = repo / "training/reports/northeast_v1_b4_dev_v2_parity.json"
    _write_json(parity_path, parity)
    parity_byte_hash = gc.sha256_file(parity_path)

    const_patcher = mock.patch.multiple(
        g2,
        FROZEN_PARITY_REPORT_BYTE_SHA256=parity_byte_hash,
        FROZEN_PARITY_N_IMAGES=n_images,
        FROZEN_PARITY_MAX_COSINE_DIVERGENCE=1.23e-06,
    )
    const_patcher.start()
    testcase.addCleanup(const_patcher.stop)

    fx["parity_path"] = parity_path
    fx["parity_byte_hash"] = parity_byte_hash
    fx["candidate_hashes"] = candidate_hashes

    if run_evaluate:
        # Everything a successful preflight needs now exists on disk. Run
        # it once here (with patch_passing_evaluation already active) to
        # compute THIS fixture's own validation_evidence values, and patch
        # schema.EXPECTED_V2_VALIDATION_EVIDENCE to match for the lifetime
        # of `testcase` -- schema.validate() now requires exact equality
        # against that constant, which otherwise holds real production
        # hashes this tiny fixture can never match.
        pre_for_evidence = g2.run_preflight(_g2_args(fx))
        expected_evidence = _validation_evidence_from_preflight(pre_for_evidence)
        evidence_patcher = mock.patch.object(schema, "EXPECTED_V2_VALIDATION_EVIDENCE", expected_evidence)
        evidence_patcher.start()
        testcase.addCleanup(evidence_patcher.stop)

    return fx


def _g2_args(fx, **overrides):
    d = dict(repo=fx["repo"], artifacts_dir=fx["artifacts_dir"])
    d.update(overrides)
    return argparse.Namespace(**d)


class TestArtifactsDirRefusal(unittest.TestCase):
    def test_live_root_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "training/artifacts").mkdir(parents=True)
            with self.assertRaises(g2.GeneratorV2Error):
                g2.require_explicit_non_live_artifacts_dir(repo, repo / "training/artifacts")

    def test_candidate_subdirectory_is_allowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            candidate = repo / "training/artifacts/northeast_v1_b4_dev_v2"
            candidate.mkdir(parents=True)
            resolved = g2.require_explicit_non_live_artifacts_dir(repo, candidate)
            self.assertEqual(resolved, candidate.resolve())


class TestPreflightAndBuildPolicy(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.fx = build_tiny_policy_v2_fixture(self, Path(self._tmpdir.name))
        # This fixture's REAL (seed-42, tiny) unknown_test_v2 scoring does
        # not happen to pass the precommitted criteria -- that outcome is
        # exercised directly by test_refuses_when_independent_validation_failed
        # below (no patch there). Every other test in this class is testing
        # generate_inference_policy_v2's OWN wiring/logic on the assumption
        # that independent validation passed, so it patches the shared
        # loader's return value rather than re-proving evaluation-scoring
        # behavior test_gate_v2_evaluation.py already covers.
        patch_passing_evaluation(self)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_preflight_succeeds_and_writes_nothing(self):
        args = _g2_args(self.fx)
        pre = g2.run_preflight(args)
        self.assertTrue(pre["ok"])
        self.assertEqual(pre["onnxruntime_providers"], ["CPUExecutionProvider"])
        self.assertFalse((self.fx["artifacts_dir"] / "inference_policy.json").exists())

    def test_build_policy_is_schema_v2_and_self_validates(self):
        args = _g2_args(self.fx)
        pre = g2.run_preflight(args)
        policy, content_sha256 = g2.build_policy(pre)
        self.assertEqual(policy["policy_schema_version"], schema.SCHEMA_VERSION_V2)
        self.assertEqual(policy["content"]["rule"]["value"], schema.FROZEN_THRESHOLD_V2)
        self.assertEqual(schema.validate(policy), [])
        self.assertEqual(policy["content_sha256"], content_sha256)

    def test_refuses_when_independent_validation_failed(self):
        failing = _fake_passing_evaluation()
        failing["content"]["validation"]["status"] = ec.EVAL_STATUS_FAILED
        with mock.patch.object(ec, "load_and_verify_eval_file", return_value=failing):
            with self.assertRaises(g2.GeneratorV2Error) as ctx:
                g2.run_preflight(_g2_args(self.fx))
        self.assertIn("independent evaluation status must be exactly", str(ctx.exception))

    def test_refuses_when_eval_not_yet_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx = build_tiny_policy_v2_fixture(self, Path(tmp), run_evaluate=False)
            args = _g2_args(fx)
            with self.assertRaises(g2.GeneratorV2Error):
                g2.run_preflight(args)

    def test_refuses_when_parity_byte_hash_mismatched(self):
        self.fx["parity_path"].write_text(
            self.fx["parity_path"].read_text(encoding="utf-8") + " ", encoding="utf-8")
        with self.assertRaises(g2.GeneratorV2Error):
            g2.run_preflight(_g2_args(self.fx))

    def test_refuses_when_top1_disagreement_nonzero(self):
        parity = json.loads(self.fx["parity_path"].read_text(encoding="utf-8"))
        parity["deterministic_content"]["model_parity"]["per_image"][0]["top1_agree"] = False
        _write_json(self.fx["parity_path"], parity)
        new_hash = gc.sha256_file(self.fx["parity_path"])
        with mock.patch.object(g2, "FROZEN_PARITY_REPORT_BYTE_SHA256", new_hash):
            with self.assertRaises(g2.GeneratorV2Error):
                g2.run_preflight(_g2_args(self.fx))

    def test_refuses_when_top3_set_disagreement_nonzero(self):
        parity = json.loads(self.fx["parity_path"].read_text(encoding="utf-8"))
        parity["deterministic_content"]["model_parity"]["per_image"][0]["top3_set_agree"] = False
        _write_json(self.fx["parity_path"], parity)
        new_hash = gc.sha256_file(self.fx["parity_path"])
        with mock.patch.object(g2, "FROZEN_PARITY_REPORT_BYTE_SHA256", new_hash):
            with self.assertRaises(g2.GeneratorV2Error):
                g2.run_preflight(_g2_args(self.fx))

    def test_refuses_when_max_divergence_differs(self):
        parity = json.loads(self.fx["parity_path"].read_text(encoding="utf-8"))
        parity["deterministic_content"]["model_parity"]["max_cosine_abs_divergence"]["max"] = 9.99e-06
        _write_json(self.fx["parity_path"], parity)
        new_hash = gc.sha256_file(self.fx["parity_path"])
        with mock.patch.object(g2, "FROZEN_PARITY_REPORT_BYTE_SHA256", new_hash):
            with self.assertRaises(g2.GeneratorV2Error):
                g2.run_preflight(_g2_args(self.fx))

    def test_refuses_when_parity_candidate_hash_disagrees(self):
        parity = json.loads(self.fx["parity_path"].read_text(encoding="utf-8"))
        parity["deterministic_content"]["candidate"]["final_artifact_hashes"]["backbone.onnx"] = "0" * 64
        _write_json(self.fx["parity_path"], parity)
        new_hash = gc.sha256_file(self.fx["parity_path"])
        with mock.patch.object(g2, "FROZEN_PARITY_REPORT_BYTE_SHA256", new_hash):
            with self.assertRaises(g2.GeneratorV2Error):
                g2.run_preflight(_g2_args(self.fx))

    def test_records_workspace_git_dirty_faithfully(self):
        pre = g2.run_preflight(_g2_args(self.fx))
        self.assertIs(pre["workspace_git_dirty"], True)
        policy, _ = g2.build_policy(pre)
        self.assertIs(policy["content"]["parity_report"]["generation_metadata_workspace_git_dirty"], True)

    def test_refuses_when_cpu_provider_not_exclusive(self):
        class FakeSession:
            def get_providers(self):
                return ["AzureExecutionProvider"]

        with mock.patch("onnxruntime.InferenceSession", return_value=FakeSession()):
            with self.assertRaises(g2.GeneratorV2Error):
                g2.run_preflight(_g2_args(self.fx))

    def test_refuses_when_target_is_live_artifacts_root(self):
        args = _g2_args(self.fx, artifacts_dir=self.fx["repo"] / "training/artifacts")
        with self.assertRaises(g2.GeneratorV2Error):
            g2.run_preflight(args)

    def test_refuses_when_scores_binding_mismatches_evaluation_contract(self):
        # Append trailing whitespace to the ACTUAL calibration_v2_scores.json
        # bytes on disk: JSON content (and its embedded content_sha256, so
        # the shared gc.load_and_verify_score_file validator still accepts
        # it) is unchanged, but the file's BYTE hash now disagrees with the
        # evaluation contract's own frozen scores_binding.byte_sha256 --
        # exactly the drift this cross-check exists to catch.
        scores_path = self.fx["repo"] / "data/calibration_v2/calibration_v2_scores.json"
        scores_path.write_text(scores_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
        with self.assertRaises(g2.GeneratorV2Error) as ctx:
            g2.run_preflight(_g2_args(self.fx))
        self.assertIn("scores_binding", str(ctx.exception))

    def test_refuses_when_selection_binding_mismatches_evaluation_contract(self):
        # Same technique as the scores-binding test above: change the
        # file's BYTE hash (trailing whitespace) while its own embedded
        # content_sha256 stays self-consistent, so only the cross-check
        # against evaluation_contract.content.selection_binding catches it.
        selection_path = self.fx["repo"] / "data/calibration_v2/calibration_v2_selection.json"
        selection_path.write_text(selection_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
        with self.assertRaises(g2.GeneratorV2Error) as ctx:
            g2.run_preflight(_g2_args(self.fx))
        self.assertIn("selection_binding", str(ctx.exception))

    def test_malformed_selection_json_is_controlled_failure_not_traceback(self):
        selection_path = self.fx["repo"] / "data/calibration_v2/calibration_v2_selection.json"
        selection_path.write_text("{not valid json", encoding="utf-8")
        with self.assertRaises(g2.GeneratorV2Error) as ctx:
            g2.run_preflight(_g2_args(self.fx))
        self.assertIn("not valid JSON", str(ctx.exception))

    def test_malformed_parity_json_is_controlled_failure_not_traceback(self):
        self.fx["parity_path"].write_text("{not valid json", encoding="utf-8")
        new_hash = gc.sha256_file(self.fx["parity_path"])
        with mock.patch.object(g2, "FROZEN_PARITY_REPORT_BYTE_SHA256", new_hash):
            with self.assertRaises(g2.GeneratorV2Error) as ctx:
                g2.run_preflight(_g2_args(self.fx))
        self.assertIn("not valid JSON", str(ctx.exception))

    def test_evaluation_contract_missing_scores_binding_is_controlled_failure(self):
        with mock.patch.object(ec, "load_and_verify_evaluation_contract") as mocked:
            mocked.return_value = {"content": {}, "content_sha256": "0" * 64}
            with self.assertRaises(g2.GeneratorV2Error) as ctx:
                g2.run_preflight(_g2_args(self.fx))
        self.assertIn("scores_binding", str(ctx.exception))

    # ---- every required calibration_v2_selection.json field, missing,
    # each a controlled GeneratorV2Error (never a traceback). Mutating the
    # file changes its byte hash, so the evaluation contract's OWN
    # selection_binding (and its content_sha256) is re-bound to the
    # mutated file too -- otherwise every mutation would be caught by the
    # EARLIER byte-identity check instead of exercising the field-presence
    # checks these tests target. -----------------------------------------
    def _mutate_selection_doc_and_rebind(self, mutate):
        selection_path = self.fx["repo"] / "data/calibration_v2/calibration_v2_selection.json"
        selection_doc = json.loads(selection_path.read_text(encoding="utf-8"))
        mutate(selection_doc)
        _write_json(selection_path, selection_doc)
        new_byte_hash = gc.sha256_file(selection_path)
        new_content_hash = selection_doc.get("content_sha256")

        contract_path = self.fx["repo"] / "training/gate_v2_evaluation_contract.json"
        contract_doc = json.loads(contract_path.read_text(encoding="utf-8"))
        contract_doc["content"] = dict(contract_doc["content"])
        contract_doc["content"]["selection_binding"] = {
            "byte_sha256": new_byte_hash, "content_sha256": new_content_hash,
        }
        contract_doc["content_sha256"] = ec.compute_content_sha256(contract_doc["content"])
        _write_json(contract_path, contract_doc)

        return mock.patch.multiple(
            ec, FROZEN_SELECTION_BYTE_SHA256=new_byte_hash, FROZEN_SELECTION_CONTENT_SHA256=new_content_hash,
        )

    def test_missing_selection_schema_version_is_controlled_failure(self):
        with self._mutate_selection_doc_and_rebind(lambda d: d.pop("schema_version")):
            with self.assertRaises(g2.GeneratorV2Error) as ctx:
                g2.run_preflight(_g2_args(self.fx))
        self.assertIn("schema_version", str(ctx.exception))

    def test_missing_selection_content_sha256_is_controlled_failure(self):
        with self._mutate_selection_doc_and_rebind(lambda d: d.pop("content_sha256")):
            with self.assertRaises(g2.GeneratorV2Error) as ctx:
                g2.run_preflight(_g2_args(self.fx))
        self.assertIn("content_sha256", str(ctx.exception))

    def test_missing_selection_generation_is_controlled_failure(self):
        with self._mutate_selection_doc_and_rebind(lambda d: d.pop("generation")):
            with self.assertRaises(g2.GeneratorV2Error) as ctx:
                g2.run_preflight(_g2_args(self.fx))
        self.assertIn("generation", str(ctx.exception))

    def test_missing_selection_content_status_is_controlled_failure(self):
        with self._mutate_selection_doc_and_rebind(lambda d: d["content"].pop("status")):
            with self.assertRaises(g2.GeneratorV2Error) as ctx:
                g2.run_preflight(_g2_args(self.fx))
        self.assertIn("content.status", str(ctx.exception))

    def test_missing_selection_content_result_is_controlled_failure(self):
        with self._mutate_selection_doc_and_rebind(lambda d: d["content"].pop("result")):
            with self.assertRaises(g2.GeneratorV2Error) as ctx:
                g2.run_preflight(_g2_args(self.fx))
        self.assertIn("content.result", str(ctx.exception))

    def test_missing_selection_content_diagnostics_is_controlled_failure(self):
        with self._mutate_selection_doc_and_rebind(lambda d: d["content"].pop("diagnostics")):
            with self.assertRaises(g2.GeneratorV2Error) as ctx:
                g2.run_preflight(_g2_args(self.fx))
        self.assertIn("content.diagnostics", str(ctx.exception))


class TestPublishAndModes(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.fx = build_tiny_policy_v2_fixture(self, Path(self._tmpdir.name))
        patch_passing_evaluation(self)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_publish_refuses_to_overwrite_and_preserves_sentinel(self):
        dest = self.fx["artifacts_dir"] / "inference_policy.json"
        sentinel = b"SENTINEL-DO-NOT-TOUCH"
        dest.write_bytes(sentinel)
        with self.assertRaises(g2.GeneratorV2Error):
            g2.publish_policy_v2(dest, '{"new": "content"}\n')
        self.assertEqual(dest.read_bytes(), sentinel)
        leftovers = list(dest.parent.glob(f"{dest.name}.tmp*"))
        self.assertEqual(leftovers, [])

    def test_publish_writes_complete_bytes_and_leaves_no_temp_file(self):
        dest = self.fx["artifacts_dir"] / "inference_policy.json"
        text = '{"hello": "world"}\n'
        g2.publish_policy_v2(dest, text)
        self.assertEqual(dest.read_text(encoding="utf-8"), text)
        self.assertEqual(list(dest.parent.glob(f"{dest.name}.tmp*")), [])
        dest.unlink()

    def test_write_then_check_round_trip_never_touches_live_root(self):
        args = _g2_args(self.fx)
        result = g2.cmd_write(args)
        self.assertTrue(result["dest"].exists())
        self.assertEqual(result["dest"], self.fx["artifacts_dir"] / "inference_policy.json")
        self.assertFalse((self.fx["repo"] / "training/artifacts/inference_policy.json").exists())

        checked = g2.cmd_check(args)
        self.assertEqual(checked["content_sha256"], result["content_sha256"])

        with self.assertRaises(g2.GeneratorV2Error):
            g2.cmd_write(args)

    def test_check_fails_closed_when_no_candidate_exists(self):
        with self.assertRaises(g2.GeneratorV2Error):
            g2.cmd_check(_g2_args(self.fx))

    # ---------------------------------------------------- correction 4 ----
    def test_write_prevalidates_via_isolated_loader_and_leaves_nothing_on_failure(self):
        import inference_policy
        dest = self.fx["artifacts_dir"] / "inference_policy.json"
        failing_state = inference_policy.PolicyState(active=False, reason="artifact_hash_mismatch", threshold=None)
        with mock.patch.object(inference_policy, "load_inference_policy", return_value=failing_state):
            with self.assertRaises(g2.GeneratorV2Error) as ctx:
                g2.cmd_write(_g2_args(self.fx))
        self.assertIn("pre-validation", str(ctx.exception))

        # candidate policy remains absent
        self.assertFalse(dest.exists())
        # no leftover temp files under the candidate directory
        self.assertEqual(list(self.fx["artifacts_dir"].glob("*.tmp*")), [])
        # no leftover isolated pre-validation temp directories anywhere
        leftover_tmp_dirs = list(Path(tempfile.gettempdir()).glob("policy_v2_preverify_*"))
        self.assertEqual(leftover_tmp_dirs, [])

        # a later corrected write (mock no longer active) remains possible
        result = g2.cmd_write(_g2_args(self.fx))
        self.assertTrue(result["dest"].exists())

    # ---------------------------------------------------- correction 5 ----
    def test_write_requires_clean_tracked_tree(self):
        with mock.patch.object(gc, "require_clean_tracked_tree",
                               side_effect=gc.ContractError("tracked working tree is not clean")):
            with self.assertRaises(gc.ContractError):
                g2.cmd_write(_g2_args(self.fx))
        self.assertFalse((self.fx["artifacts_dir"] / "inference_policy.json").exists())

    def test_check_does_not_require_a_clean_tree(self):
        args = _g2_args(self.fx)
        g2.cmd_write(args)
        with mock.patch.object(gc, "require_clean_tracked_tree",
                               side_effect=gc.ContractError("tracked working tree is not clean")):
            # --check must still succeed: only --write gates on a clean tree.
            checked = g2.cmd_check(args)
        self.assertTrue(checked["dest"].exists())

    def test_written_policy_carries_diagnostic_provenance(self):
        args = _g2_args(self.fx)
        result = g2.cmd_write(args)
        stored = json.loads(result["dest"].read_text(encoding="utf-8"))
        provenance = stored["content"]["provenance"]
        self.assertEqual(provenance["git_head"], "a" * 40)
        self.assertEqual(set(provenance["source_hashes"]), set(g2.PROVENANCE_SOURCE_PATHS))
        self.assertIn("diagnostic", provenance["note"].lower())
        self.assertIn("api/inference.py", provenance["note"])

    def test_write_refuses_when_schema_copies_diverge(self):
        training_copy = self.fx["repo"] / "training/policy_schema.py"
        training_copy.write_text(training_copy.read_text(encoding="utf-8") + "\n# drift\n", encoding="utf-8")
        with self.assertRaises(g2.GeneratorV2Error) as ctx:
            g2.cmd_write(_g2_args(self.fx))
        self.assertIn("byte-identical", str(ctx.exception))
        self.assertFalse((self.fx["artifacts_dir"] / "inference_policy.json").exists())

    def test_write_refuses_on_crlf_vs_lf_only_schema_copy_divergence(self):
        # The two copies differ ONLY in line endings -- canonical-LF
        # normalization would treat these as equal (and did, before this
        # correction); the required sync gate uses raw read_bytes()
        # equality specifically so this divergence is still caught.
        training_copy = self.fx["repo"] / "training/policy_schema.py"
        api_copy = self.fx["repo"] / "api/policy_schema.py"
        self.assertEqual(training_copy.read_bytes(), api_copy.read_bytes())
        original = training_copy.read_bytes()
        crlf_version = original.replace(b"\n", b"\r\n")
        self.assertNotEqual(crlf_version, original)
        training_copy.write_bytes(crlf_version)
        # canonical-LF hashes are still equal -- confirms this is genuinely
        # a line-ending-only divergence, not a content divergence.
        self.assertEqual(gc.canonical_lf_sha256_file(training_copy), gc.canonical_lf_sha256_file(api_copy))
        with self.assertRaises(g2.GeneratorV2Error) as ctx:
            g2.cmd_write(_g2_args(self.fx))
        self.assertIn("byte-identical", str(ctx.exception))
        self.assertFalse((self.fx["artifacts_dir"] / "inference_policy.json").exists())

    # ---------------------------------------------------- correction 3 ----
    def _write_then_mutate_stored_policy(self, mutate) -> Path:
        args = _g2_args(self.fx)
        result = g2.cmd_write(args)
        dest = result["dest"]
        stored = json.loads(dest.read_text(encoding="utf-8"))
        mutate(stored["content"])
        stored["content_sha256"] = schema.compute_content_sha256(stored["policy_schema_version"], stored["content"])
        dest.write_text(json.dumps(stored, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        return dest

    def test_check_rejects_gate_framing_mutation(self):
        self._write_then_mutate_stored_policy(lambda c: c.__setitem__("gate_framing", "tampered framing"))
        with self.assertRaises(g2.GeneratorV2Error) as ctx:
            g2.cmd_check(_g2_args(self.fx))
        self.assertIn("does not exactly match", str(ctx.exception))

    def test_check_rejects_parity_fact_mutation(self):
        def mutate(c):
            c["parity_report"] = dict(c["parity_report"])
            c["parity_report"]["n_images"] = c["parity_report"]["n_images"] + 1
        self._write_then_mutate_stored_policy(mutate)
        with self.assertRaises(g2.GeneratorV2Error) as ctx:
            g2.cmd_check(_g2_args(self.fx))
        self.assertIn("does not exactly match", str(ctx.exception))

    def test_check_rejects_not_validated_for_mutation(self):
        def mutate(c):
            c["not_validated_for"] = list(c["not_validated_for"]) + ["a smuggled-in claim"]
        self._write_then_mutate_stored_policy(mutate)
        with self.assertRaises(g2.GeneratorV2Error) as ctx:
            g2.cmd_check(_g2_args(self.fx))
        self.assertIn("does not exactly match", str(ctx.exception))

    def test_check_rejects_single_evidence_hash_mutation(self):
        def mutate(c):
            c["validation_evidence"] = dict(c["validation_evidence"])
            original = c["validation_evidence"]["parity_report_byte_sha256"]
            c["validation_evidence"]["parity_report_byte_sha256"] = (
                ("f" if original[0] != "f" else "e") + original[1:]
            )
        self._write_then_mutate_stored_policy(mutate)
        with self.assertRaises(g2.GeneratorV2Error) as ctx:
            g2.cmd_check(_g2_args(self.fx))
        self.assertIn("does not exactly match", str(ctx.exception))

    def test_check_accepts_the_untampered_freshly_written_policy(self):
        args = _g2_args(self.fx)
        result = g2.cmd_write(args)
        checked = g2.cmd_check(args)
        self.assertEqual(checked["content_sha256"], result["content_sha256"])

    def test_check_permits_only_generated_at_to_differ(self):
        args = _g2_args(self.fx)
        result = g2.cmd_write(args)
        dest = result["dest"]
        stored = json.loads(dest.read_text(encoding="utf-8"))
        stored["generation"]["generated_at"] = "2099-01-01T00:00:00Z"
        dest.write_text(json.dumps(stored, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        # content_sha256 is untouched (generation is excluded from it) --
        # --check must still accept this, since only generated_at differs.
        checked = g2.cmd_check(args)
        self.assertEqual(checked["content_sha256"], result["content_sha256"])

    def test_check_rejects_extra_generation_key_even_without_content_change(self):
        args = _g2_args(self.fx)
        result = g2.cmd_write(args)
        dest = result["dest"]
        stored = json.loads(dest.read_text(encoding="utf-8"))
        stored["generation"]["hostname"] = "smuggled-host"
        # content_sha256 deliberately left UNCHANGED -- generation is
        # excluded from it, so this mutation alone never touches the hash.
        dest.write_text(json.dumps(stored, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        with self.assertRaises(g2.GeneratorV2Error) as ctx:
            g2.cmd_check(args)
        self.assertIn("generation", str(ctx.exception))

    def test_check_rejects_changed_generator_version_even_without_content_change(self):
        args = _g2_args(self.fx)
        result = g2.cmd_write(args)
        dest = result["dest"]
        stored = json.loads(dest.read_text(encoding="utf-8"))
        stored["generation"]["generator_version"] = "9.9.9-tampered"
        dest.write_text(json.dumps(stored, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        with self.assertRaises(g2.GeneratorV2Error) as ctx:
            g2.cmd_check(args)
        self.assertIn("generator_version", str(ctx.exception))

    def test_check_rejects_extra_top_level_key_even_without_content_change(self):
        args = _g2_args(self.fx)
        result = g2.cmd_write(args)
        dest = result["dest"]
        stored = json.loads(dest.read_text(encoding="utf-8"))
        stored["smuggled"] = "extra"
        dest.write_text(json.dumps(stored, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        with self.assertRaises(g2.GeneratorV2Error) as ctx:
            g2.cmd_check(args)
        self.assertIn("top-level key set", str(ctx.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
