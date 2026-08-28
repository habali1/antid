#!/usr/bin/env python3
"""Stdlib regression suite for policy_schema.py / policy_evidence.py /
inference_policy_generator.py. Run directly: python test_policy_generator.py

Unit tests hit policy_schema directly (fast, no I/O). Integration tests
clone a scratch fixture (hardlinked binaries, copied JSON) from the real
current evidence and corrupt exactly one thing per scenario, then invoke
the real generator via subprocess against the scratch copies -- the real
frozen/committed evidence is never touched.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import policy_schema as schema  # noqa: E402

REPO_DATA = HERE.parent / "data"
# Outside the repo by default so a test run never leaves untracked clutter
# under version control; override with POLICY_TEST_SCRATCH if needed.
SCRATCH = Path(os.environ.get("POLICY_TEST_SCRATCH",
                               str(Path(tempfile.gettempdir()) / "antid_policy_test_scratch")))
PY = sys.executable
GEN = str(HERE / "inference_policy_generator.py")
VALID_DECISION = {
    "low_confidence_if": {"signal": "raw_pre_geo_max_cosine", "comparison": "strict_less_than", "threshold": 0.6},
    "normal_results_if": {"comparison": "greater_than_or_equal", "threshold": 0.6},
    "equal_threshold_action": "normal_results", "non_finite_action": "request_error",
}
VALID_RULE = {
    "operator_verbatim": "max_sim < value",
    "operator_normalized": {"comparison": "strict_less_than"},
    "signal": "raw max cosine similarity before geo re-ranking",
    "value": 0.6,
}


# ============================================================= unit tests
class TestDecisionSemantics(unittest.TestCase):
    def test_boundary_values(self):
        self.assertTrue(schema.should_abstain(0.5999, VALID_DECISION))
        self.assertFalse(schema.should_abstain(0.6000, VALID_DECISION))
        self.assertFalse(schema.should_abstain(0.6001, VALID_DECISION))

    def test_float32_boundary(self):
        import numpy as np
        widened = float(np.float32(0.6))
        below = float(np.nextafter(np.float32(0.6), np.float32(0)))
        self.assertNotEqual(widened, 0.6)
        self.assertFalse(schema.should_abstain(widened, VALID_DECISION))
        self.assertTrue(schema.should_abstain(below, VALID_DECISION))

    def test_non_finite_raises(self):
        with self.assertRaises(ValueError):
            schema.should_abstain(float("nan"), VALID_DECISION)
        with self.assertRaises(ValueError):
            schema.should_abstain(float("inf"), VALID_DECISION)

    def test_encodings_agree_on_valid(self):
        self.assertEqual(schema.decision_encodings_agree(VALID_DECISION), [])

    def test_encodings_disagree_on_mismatched_threshold(self):
        bad = json.loads(json.dumps(VALID_DECISION))
        bad["normal_results_if"]["threshold"] = 0.55
        errs = schema.decision_encodings_agree(bad)
        self.assertTrue(any("threshold" in e for e in errs))

    def test_encodings_disagree_on_inverted_action(self):
        bad = json.loads(json.dumps(VALID_DECISION))
        bad["equal_threshold_action"] = "low_confidence"
        errs = schema.decision_encodings_agree(bad)
        self.assertTrue(any("equal_threshold_action" in e for e in errs))


class TestValidateNeverRaises(unittest.TestCase):
    def test_malformed_inputs_return_errors_not_raise(self):
        for bad in (None, 42, "a string", [1, 2, 3], float("nan"), {}, {"content": "not an object"}):
            errs = schema.validate(bad)
            self.assertIsInstance(errs, list)
            self.assertTrue(errs)

    def test_nan_in_decision_threshold_is_an_error(self):
        content = {"decision": {"low_confidence_if": {"comparison": "strict_less_than", "threshold": float("nan")},
                                "normal_results_if": {"comparison": "greater_than_or_equal", "threshold": float("nan")},
                                "equal_threshold_action": "normal_results", "non_finite_action": "request_error"},
                   "artifact_hashes": {}, "provider_policy": None, "preprocessing": {}}
        policy = {"policy_schema_version": 1, "content": content, "content_sha256": "x" * 64,
                  "generation": {"generated_at": "x", "generator_version": "x"}}
        errs = schema.validate(policy)
        self.assertTrue(any("finite" in e for e in errs))

    @staticmethod
    def _valid_policy():
        content = {
            "rule": json.loads(json.dumps(VALID_RULE)),
            "decision": json.loads(json.dumps(VALID_DECISION)),
            "artifact_hashes": {"backbone.onnx": "a" * 64, "prototypes.npy": "b" * 64, "taxonomy.json": "c" * 64},
            "provider_policy": {"providers": ["CPUExecutionProvider"], "exclusive": True},
            "preprocessing": {"contract": {
                "rgb_conversion": "x", "resize": "x", "interpolation": "x", "scale_divisor": 255.0,
                "normalize_mean": [0.485, 0.456, 0.406], "normalize_std": [0.229, 0.224, 0.225],
                "dtype": "float32", "channel_layout": "x"}},
        }
        content_sha256 = schema.compute_content_sha256(1, content)
        return {"policy_schema_version": 1, "content": content, "content_sha256": content_sha256,
                "generation": {"generated_at": "2026-01-01T00:00:00Z", "generator_version": "test"}}

    def test_valid_policy_round_trips(self):
        self.assertEqual(schema.validate(self._valid_policy()), [])

    def test_content_hash_tampering_detected(self):
        policy = self._valid_policy()
        policy["content_sha256"] = "0" * 64
        errs = schema.validate(policy)
        self.assertTrue(any("content_sha256 mismatch" in e for e in errs))

    def test_rule_and_decision_signal_encodings_are_enforced(self):
        mutations = (
            ("operator_verbatim", lambda c: c["rule"].__setitem__("operator_verbatim", "max_sim <= value")),
            ("operator_normalized", lambda c: c["rule"].__setitem__(
                "operator_normalized", {"comparison": "less_than_or_equal"})),
            ("rule.signal", lambda c: c["rule"].__setitem__("signal", "post-geo score")),
            ("rule.value", lambda c: c["rule"].__setitem__("value", 0.55)),
            ("frozen threshold", lambda c: (
                c["rule"].__setitem__("value", 0.59995),
                c["decision"]["low_confidence_if"].__setitem__("threshold", 0.59995),
                c["decision"]["normal_results_if"].__setitem__("threshold", 0.59995),
            )),
            ("rule.value", lambda c: c["rule"].__setitem__("value", True)),
            ("low_confidence_if.signal", lambda c: c["decision"]["low_confidence_if"].__setitem__(
                "signal", "rounded_or_post_geo")),
        )
        for expected_error, mutate_content in mutations:
            with self.subTest(expected_error=expected_error):
                policy = self._valid_policy()
                mutate_content(policy["content"])
                policy["content_sha256"] = schema.compute_content_sha256(
                    policy["policy_schema_version"], policy["content"])
                errors = schema.validate(policy)
                self.assertTrue(any(expected_error in e for e in errors), errors)

    def test_training_and_api_schema_copies_are_byte_identical(self):
        api_copy = HERE.parent / "api" / "policy_schema.py"
        self.assertEqual((HERE / "policy_schema.py").read_bytes(), api_copy.read_bytes())

    def test_schema_version_type_strictness(self):
        # 1. start from a known-valid policy
        base = self._valid_policy()
        self.assertEqual(schema.validate(base), [])

        # 2/5. zeroed content_sha256 with the exact valid int version -> hash mismatch
        zeroed = json.loads(json.dumps(base))
        zeroed["content_sha256"] = "0" * 64
        errs = schema.validate(zeroed)
        self.assertTrue(any("content_sha256 mismatch" in e for e in errs), errs)

        # 3. policy_schema_version=True must fail even though True == 1
        bool_version = json.loads(json.dumps(base))
        bool_version["policy_schema_version"] = True
        errs = schema.validate(bool_version)
        self.assertTrue(any("policy_schema_version must be the exact int" in e for e in errs), errs)
        # and it must fail even paired with a bogus hash -- never silently accepted
        # because the version check short-circuited past the hash check
        bool_version["content_sha256"] = "0" * 64
        self.assertTrue(schema.validate(bool_version))

        # 4. policy_schema_version=1.0 must fail even though 1.0 == 1
        float_version = json.loads(json.dumps(base))
        float_version["policy_schema_version"] = 1.0
        errs = schema.validate(float_version)
        self.assertTrue(any("policy_schema_version must be the exact int" in e for e in errs), errs)

        # string/null versions must also fail
        for bad in ("1", None):
            p = json.loads(json.dumps(base))
            p["policy_schema_version"] = bad
            self.assertTrue(schema.validate(p))

        # 6. the untouched valid policy still passes
        self.assertEqual(schema.validate(base), [])


# ====================================================== integration tests
def build_base() -> Path:
    base = SCRATCH / "base"
    if base.exists():
        shutil.rmtree(base)
    (base / "data" / "calibration_v1").mkdir(parents=True)
    (base / "data" / "unknown_test_v1").mkdir(parents=True)
    (base / "artifacts").mkdir(parents=True)
    shutil.copy(REPO_DATA / "calibration_v1" / "calibration_v1.json", base / "data" / "calibration_v1")
    shutil.copy(REPO_DATA / "calibration_v1" / "calibration_v1_scores.json", base / "data" / "calibration_v1")
    shutil.copy(REPO_DATA / "unknown_test_v1" / "unknown_test_v1_eval.json", base / "data" / "unknown_test_v1")
    for name in ("backbone.onnx", "prototypes.npy", "model.pth"):
        os.link(HERE / "artifacts" / name, base / "artifacts" / name)
    for name in ("taxonomy.json", "parity_report.json", "parity_diagnostic.json", "parity_flag_ablation.json"):
        shutil.copy(HERE / "artifacts" / name, base / "artifacts" / name)
    return base


def clone(base: Path, name: str) -> Path:
    dst = SCRATCH / name
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(base, dst)
    return dst


def mutate(path: Path, fn) -> None:
    d = json.loads(path.read_text())
    fn(d)
    path.write_text(json.dumps(d))


def run_gen(scenario: Path, script: str = GEN):
    env = dict(os.environ)
    env["POLICY_GEN_DATA_DIR"] = str(scenario / "data")
    env["POLICY_GEN_ARTIFACTS_DIR"] = str(scenario / "artifacts")
    return subprocess.run([PY, script], env=env, capture_output=True, text=True, timeout=180)


@unittest.skipUnless((HERE / "artifacts" / "backbone.onnx").exists(), "real artifacts not present")
class TestGeneratorIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        SCRATCH.mkdir(parents=True, exist_ok=True)
        cls.base = build_base()

    def assert_fails(self, scenario: Path, needle: str = None):
        r = run_gen(scenario)
        self.assertNotEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("FAIL:", r.stderr)
        if needle:
            self.assertIn(needle, r.stderr)
        self.assertFalse((scenario / "artifacts" / "inference_policy.json").exists())

    def test_happy_path_and_atomic_readback(self):
        s = clone(self.base, "happy")
        r = run_gen(s)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        out = s / "artifacts" / "inference_policy.json"
        self.assertTrue(out.exists())
        policy = json.loads(out.read_text())
        self.assertEqual(schema.validate(policy), [])
        self.assertEqual(len(list((s / "artifacts").glob(".inference_policy.json.tmp*"))), 0)

    def test_two_run_canonical_determinism(self):
        s = clone(self.base, "determinism")
        r1 = run_gen(s)
        h1 = json.loads((s / "artifacts" / "inference_policy.json").read_text())["content_sha256"]
        r2 = run_gen(s)
        h2 = json.loads((s / "artifacts" / "inference_policy.json").read_text())["content_sha256"]
        self.assertEqual(r1.returncode, 0)
        self.assertEqual(r2.returncode, 0)
        self.assertEqual(h1, h2)

    def test_missing_evidence_file(self):
        s = clone(self.base, "missing_file")
        (s / "data" / "unknown_test_v1" / "unknown_test_v1_eval.json").unlink()
        self.assert_fails(s, "missing")

    def test_missing_mandatory_eval_hash(self):
        s = clone(self.base, "missing_eval_hash")
        mutate(s / "data" / "calibration_v1" / "calibration_v1_scores.json", lambda d: d["hashes"].pop("config_yaml"))
        self.assert_fails(s, "missing required hash")

    def test_calibration_hash_mismatch(self):
        s = clone(self.base, "hash_mismatch")
        mutate(s / "data" / "calibration_v1" / "calibration_v1_scores.json",
               lambda d: d["hashes"].__setitem__("taxonomy_json", "0" * 64))
        self.assert_fails(s, "disagrees with current")

    def test_threshold_lineage_mismatch(self):
        s = clone(self.base, "threshold_lineage")
        mutate(s / "data" / "unknown_test_v1" / "unknown_test_v1_eval.json",
               lambda d: d.__setitem__("frozen_threshold", 0.55))
        self.assert_fails(s, "frozen_threshold")

    def test_threshold_source_mismatch(self):
        s = clone(self.base, "threshold_source")
        mutate(s / "data" / "unknown_test_v1" / "unknown_test_v1_eval.json",
               lambda d: d.__setitem__("threshold_source", "some other file"))
        self.assert_fails(s, "threshold_source")

    def test_status_inversion(self):
        s = clone(self.base, "status_inversion")
        mutate(s / "data" / "calibration_v1" / "calibration_v1.json",
               lambda d: d["phase_c_validation"].__setitem__("status", "NOT INDEPENDENTLY VALIDATED"))
        self.assert_fails(s, "phase_c_validation.status")

    def test_bad_operator(self):
        s = clone(self.base, "bad_operator")
        mutate(s / "data" / "calibration_v1" / "calibration_v1.json",
               lambda d: d["frozen_candidate_abstention_threshold"]["machine_readable_rule"].__setitem__(
                   "operator", "max_sim <= value"))
        self.assert_fails(s, "unsupported operator")

    def test_invalid_threshold_value(self):
        s = clone(self.base, "bad_value")
        mutate(s / "data" / "calibration_v1" / "calibration_v1.json",
               lambda d: d["frozen_candidate_abstention_threshold"]["machine_readable_rule"].__setitem__("value", 1.6))
        self.assert_fails(s, "invalid threshold value")

    def test_cross_eval_checkpoint_mismatch(self):
        s = clone(self.base, "ckpt_mismatch")
        mutate(s / "data" / "unknown_test_v1" / "unknown_test_v1_eval.json",
               lambda d: d["hashes"].__setitem__("checkpoint", "9" * 64))
        self.assert_fails(s, "disagrees with current")

    def test_missing_live_artifact(self):
        s = clone(self.base, "missing_artifact")
        (s / "artifacts" / "prototypes.npy").unlink()
        self.assert_fails(s, "required artifact/source missing")

    def test_onnx_hash_disagreement_in_one_report(self):
        s = clone(self.base, "onnx_disagree")
        mutate(s / "artifacts" / "parity_report.json", lambda d: d["artifact_hashes"].__setitem__("backbone_onnx", "1" * 64))
        self.assert_fails(s, "disagrees with the current artifact")

    def test_onnx_binding_unsatisfiable_when_absent_everywhere(self):
        s = clone(self.base, "onnx_unsat")
        mutate(s / "artifacts" / "parity_report.json", lambda d: d["artifact_hashes"].pop("backbone_onnx"))
        mutate(s / "artifacts" / "parity_diagnostic.json", lambda d: d["artifact_hashes"].pop("backbone.onnx"))
        mutate(s / "artifacts" / "parity_flag_ablation.json", lambda d: d["artifact_hashes"].pop("backbone.onnx"))
        self.assert_fails(s, "unsatisfiable")

    def test_onnx_binding_disqualified_by_missing_config(self):
        s = clone(self.base, "onnx_missing_config")
        for name in ("parity_report.json", "parity_diagnostic.json", "parity_flag_ablation.json"):
            mutate(s / "artifacts" / name, lambda d: d["artifact_hashes"].pop("config_yaml"))
        self.assert_fails(s, "unsatisfiable")

    def test_alias_conflict(self):
        s = clone(self.base, "alias_conflict")
        mutate(s / "artifacts" / "parity_report.json", lambda d: d["artifact_hashes"].__setitem__("model.pth", "f" * 64))
        self.assert_fails(s, "conflicting alias hashes")

    def test_altered_parity_input_report_hash(self):
        s = clone(self.base, "altered_input_hash")
        mutate(s / "artifacts" / "parity_flag_ablation.json",
               lambda d: d["input_report_hashes_before"].__setitem__("parity_report_json", "d" * 64))
        self.assert_fails(s, "does not match the current live hash")

    def test_duplicate_sample_rows(self):
        s = clone(self.base, "dup_rows")
        mutate(s / "artifacts" / "parity_flag_ablation.json",
               lambda d: d["integrity_gate"].__setitem__("duplicate_row_keys", 3))
        self.assert_fails(s, "duplicate rows/hashes")

    def test_nonzero_gate_disagreement(self):
        s = clone(self.base, "gate_disagree")
        mutate(s / "artifacts" / "parity_flag_ablation.json",
               lambda d: d["part_b_pairwise_distributions"]
                          ["calibration_mirror_cuda_batch32__vs__production_onnx_cpu_batch1"]
                          .__setitem__("gate_disagreement_count", 2))
        self.assert_fails(s, "gate_disagreement_count is nonzero")

    def test_preprocessing_drift(self):
        # training/ and api/ must be siblings, matching the real repo layout
        # the generator assumes (sys.path.insert(0, HERE.parent / "api")).
        s = clone(self.base, "preproc_drift")
        root = SCRATCH / "preproc_drift_root"
        if root.exists():
            shutil.rmtree(root)
        scratch_training = root / "training"
        scratch_api = root / "api"
        scratch_training.mkdir(parents=True)
        scratch_api.mkdir(parents=True)
        for name in ("inference_policy_generator.py", "policy_schema.py", "policy_evidence.py",
                     "data.py", "config.yaml", "requirements.txt", "model.py"):
            shutil.copy(HERE / name, scratch_training / name)
        corrupted = (HERE.parent / "api" / "inference.py").read_text().replace(
            "NORMALIZE_MEAN = (0.485, 0.456, 0.406)",
            "NORMALIZE_MEAN = (0.400, 0.400, 0.400)")
        self.assertNotEqual(corrupted, (HERE.parent / "api" / "inference.py").read_text())
        (scratch_api / "inference.py").write_text(corrupted)
        for name in ("inference_policy.py", "policy_schema.py"):
            shutil.copy(HERE.parent / "api" / name, scratch_api / name)
        r = run_gen(s, script=str(scratch_training / "inference_policy_generator.py"))
        self.assertNotEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("nonzero divergence", r.stderr)
        self.assertFalse((s / "artifacts" / "inference_policy.json").exists())
        real_untouched = (HERE.parent / "api" / "inference.py").read_text()
        self.assertNotIn("0.400, 0.400, 0.400", real_untouched)


if __name__ == "__main__":
    unittest.main(verbosity=2)
