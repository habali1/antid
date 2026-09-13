#!/usr/bin/env python3
"""test_policy_schema_v2.py — offline/synthetic tests for schema-v2
(Gate v2) additions to policy_schema.py, and for backward compatibility
with schema v1.

Pure dict-level tests only: policy_schema.py never touches a file, so none
of these tests need a fixture repo. See test_generate_policy_v2.py for the
generator-level (file/evidence-reading) tests, and
api/test_inference_policy.py for loader-level tests.

Run directly: python test_policy_schema_v2.py
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import policy_schema as schema  # noqa: E402
import inference_policy_generator as v1_generator  # noqa: E402


ARTIFACT_HASHES = {
    "backbone.onnx": "a" * 64, "prototypes.npy": "b" * 64, "taxonomy.json": "c" * 64,
}
PREPROCESSING_CONTRACT = {
    "rgb_conversion": "img.convert('RGB')", "resize": "squish to fixed 380x380 (both dims set, no crop)",
    "interpolation": "Pillow bilinear", "scale_divisor": 255.0,
    "normalize_mean": [0.485, 0.456, 0.406], "normalize_std": [0.229, 0.224, 0.225],
    "dtype": "float32", "channel_layout": "RGB -> CHW, batched to NCHW",
}


def valid_v1_policy() -> dict:
    content = {
        "rule": {"operator_verbatim": schema.ALLOWED_OPERATOR,
                "operator_normalized": dict(schema.ALLOWED_OPERATOR_NORMALIZED),
                "signal": schema.ALLOWED_RULE_SIGNAL, "value": schema.FROZEN_THRESHOLD},
        "decision": {
            "low_confidence_if": {"signal": schema.ALLOWED_DECISION_SIGNAL, "comparison": "strict_less_than",
                                  "threshold": schema.FROZEN_THRESHOLD},
            "normal_results_if": {"comparison": "greater_than_or_equal", "threshold": schema.FROZEN_THRESHOLD},
            "equal_threshold_action": "normal_results", "non_finite_action": "request_error",
        },
        "artifact_hashes": dict(ARTIFACT_HASHES),
        "provider_policy": {"providers": ["CPUExecutionProvider"], "exclusive": True},
        "preprocessing": {"contract": dict(PREPROCESSING_CONTRACT)},
    }
    return {
        "policy_schema_version": schema.SCHEMA_VERSION, "content": content,
        "content_sha256": schema.compute_content_sha256(schema.SCHEMA_VERSION, content),
        "generation": {"generated_at": "2026-01-01T00:00:00Z", "generator_version": "test"},
    }


def valid_v2_policy() -> dict:
    content = {
        "rule": {"operator_verbatim": schema.ALLOWED_OPERATOR,
                "operator_normalized": dict(schema.ALLOWED_OPERATOR_NORMALIZED),
                "signal": schema.ALLOWED_RULE_SIGNAL, "value": schema.FROZEN_THRESHOLD_V2},
        "decision": {
            "low_confidence_if": {"signal": schema.ALLOWED_DECISION_SIGNAL, "comparison": "strict_less_than",
                                  "threshold": schema.FROZEN_THRESHOLD_V2},
            "normal_results_if": {"comparison": "greater_than_or_equal", "threshold": schema.FROZEN_THRESHOLD_V2},
            "equal_threshold_action": "normal_results", "non_finite_action": "request_error",
        },
        "artifact_hashes": dict(ARTIFACT_HASHES),
        "provider_policy": {"providers": ["CPUExecutionProvider"], "exclusive": True},
        "preprocessing": {"contract": dict(PREPROCESSING_CONTRACT)},
        "validation_evidence": dict(schema.EXPECTED_V2_VALIDATION_EVIDENCE),
    }
    return {
        "policy_schema_version": schema.SCHEMA_VERSION_V2, "content": content,
        "content_sha256": schema.compute_content_sha256(schema.SCHEMA_VERSION_V2, content),
        "generation": {"generated_at": "2026-01-01T00:00:00Z", "generator_version": schema.V2_GENERATOR_VERSION},
    }


class TestSchemaV1StillWorks(unittest.TestCase):
    """Backward compatibility: v1 is untouched by the v2 additions."""

    def test_valid_v1_policy_passes(self):
        self.assertEqual(schema.validate(valid_v1_policy()), [])

    def test_v1_boundary_probes_unchanged(self):
        policy = valid_v1_policy()
        self.assertTrue(schema.should_abstain(0.5999, policy["content"]["decision"]))
        self.assertFalse(schema.should_abstain(0.6000, policy["content"]["decision"]))
        self.assertFalse(schema.should_abstain(0.6001, policy["content"]["decision"]))

    def test_v1_frozen_threshold_and_schema_version_constants_unchanged(self):
        # The literal names/values a v1-only reader would depend on.
        self.assertEqual(schema.SCHEMA_VERSION, 1)
        self.assertEqual(schema.FROZEN_THRESHOLD, 0.6)


class TestSchemaV2Valid(unittest.TestCase):
    def test_valid_v2_policy_passes(self):
        self.assertEqual(schema.validate(valid_v2_policy()), [])

    def test_v2_frozen_threshold_is_061(self):
        self.assertEqual(schema.FROZEN_THRESHOLD_V2, 0.61)

    def test_v2_boundary_probes(self):
        policy = valid_v2_policy()
        self.assertTrue(schema.should_abstain(0.6099, policy["content"]["decision"]))
        self.assertFalse(schema.should_abstain(0.6100, policy["content"]["decision"]))
        self.assertFalse(schema.should_abstain(0.6101, policy["content"]["decision"]))


class TestCrossVersionRuleMismatch(unittest.TestCase):
    """V1 with 0.61 and V2 with 0.60 must both fail -- neither the schema
    nor the loader may silently pair a schema_version with the WRONG
    version's threshold."""

    def test_v1_schema_with_061_rejected(self):
        policy = valid_v1_policy()
        policy["content"] = json.loads(json.dumps(policy["content"]))
        policy["content"]["rule"]["value"] = 0.61
        policy["content"]["decision"]["low_confidence_if"]["threshold"] = 0.61
        policy["content"]["decision"]["normal_results_if"]["threshold"] = 0.61
        policy["content_sha256"] = schema.compute_content_sha256(1, policy["content"])
        errors = schema.validate(policy)
        self.assertTrue(any("rule.value must be the frozen threshold for schema v1" in e for e in errors), errors)

    def test_v2_schema_with_060_rejected(self):
        policy = valid_v2_policy()
        policy["content"] = json.loads(json.dumps(policy["content"]))
        policy["content"]["rule"]["value"] = 0.60
        policy["content"]["decision"]["low_confidence_if"]["threshold"] = 0.60
        policy["content"]["decision"]["normal_results_if"]["threshold"] = 0.60
        policy["content_sha256"] = schema.compute_content_sha256(2, policy["content"])
        errors = schema.validate(policy)
        self.assertTrue(any("rule.value must be the frozen threshold for schema v2" in e for e in errors), errors)


class TestSchemaVersionTypeStrictness(unittest.TestCase):
    """bool/float/string/null/unsupported-int schema versions must all fail
    closed, for BOTH schema v1 and v2 content shapes."""

    def _assert_rejected(self, bad_version):
        for base in (valid_v1_policy, valid_v2_policy):
            with self.subTest(base=base.__name__, version=bad_version):
                policy = base()
                policy["policy_schema_version"] = bad_version
                errors = schema.validate(policy)
                self.assertTrue(errors)
                self.assertTrue(any("policy_schema_version must be exactly one of" in e for e in errors), errors)

    def test_bool_true_rejected(self):
        self._assert_rejected(True)

    def test_bool_false_rejected(self):
        self._assert_rejected(False)

    def test_float_rejected(self):
        self._assert_rejected(1.0)
        self._assert_rejected(2.0)

    def test_string_rejected(self):
        self._assert_rejected("2")

    def test_null_rejected(self):
        self._assert_rejected(None)

    def test_unsupported_integer_rejected(self):
        self._assert_rejected(3)
        self._assert_rejected(0)
        self._assert_rejected(-1)


class TestV2ValidationEvidenceMutations(unittest.TestCase):
    """Every mutation here recomputes content_sha256 fresh AFTER mutating --
    a stale hash is never why these are rejected."""

    def _mutate(self, mutator):
        policy = valid_v2_policy()
        content = json.loads(json.dumps(policy["content"]))
        mutator(content)
        policy["content"] = content
        policy["content_sha256"] = schema.compute_content_sha256(2, content)
        return policy

    def test_missing_validation_evidence_block_rejected(self):
        def mutate(c):
            del c["validation_evidence"]
        errors = schema.validate(self._mutate(mutate))
        self.assertTrue(any("validation_evidence" in e for e in errors), errors)

    def test_validation_evidence_not_an_object_rejected(self):
        def mutate(c):
            c["validation_evidence"] = "not an object"
        errors = schema.validate(self._mutate(mutate))
        self.assertTrue(any("validation_evidence is not a JSON object" in e for e in errors), errors)

    def test_validation_evidence_missing_one_key_rejected(self):
        def mutate(c):
            del c["validation_evidence"]["parity_report_byte_sha256"]
        errors = schema.validate(self._mutate(mutate))
        self.assertTrue(any("validation_evidence missing key" in e for e in errors), errors)

    def test_validation_evidence_extra_key_rejected(self):
        def mutate(c):
            c["validation_evidence"]["sneaky"] = "smuggled"
        errors = schema.validate(self._mutate(mutate))
        self.assertTrue(any("validation_evidence has unexpected key" in e for e in errors), errors)

    def test_validation_evidence_malformed_hash_rejected(self):
        def mutate(c):
            c["validation_evidence"]["gate_v2_selection_contract_content_sha256"] = "not-a-hash"
        errors = schema.validate(self._mutate(mutate))
        self.assertTrue(any("gate_v2_selection_contract_content_sha256" in e for e in errors), errors)

    def test_validation_status_altered_rejected(self):
        def mutate(c):
            c["validation_evidence"]["validation_status"] = "validation_failed"
        errors = schema.validate(self._mutate(mutate))
        self.assertTrue(any("validation_status must be exactly" in e for e in errors), errors)

    def test_diagnostic_far_non_numeric_rejected(self):
        def mutate(c):
            c["validation_evidence"]["diagnostic_out_of_scope_ant_far"] = "51%"
        errors = schema.validate(self._mutate(mutate))
        self.assertTrue(any("diagnostic_out_of_scope_ant_far" in e for e in errors), errors)

    def test_v1_policy_never_requires_validation_evidence(self):
        # v1 has no validation_evidence key at all, and must still pass.
        policy = valid_v1_policy()
        self.assertNotIn("validation_evidence", policy["content"])
        self.assertEqual(schema.validate(policy), [])

    def test_each_hash_field_rejects_a_different_well_formed_hash(self):
        # Every hash key, replaced with a DIFFERENT but still well-formed
        # 64-lowercase-hex value -- format alone must not be sufficient;
        # exact equality with the frozen value is required.
        for key in schema.REQUIRED_V2_VALIDATION_EVIDENCE_HASH_KEYS:
            with self.subTest(key=key):
                def mutate(c, key=key):
                    original = c["validation_evidence"][key]
                    swapped = ("f" if original[0] != "f" else "e") + original[1:]
                    c["validation_evidence"][key] = swapped
                policy = self._mutate(mutate)
                errors = schema.validate(policy)
                self.assertTrue(any(f"validation_evidence.{key} must be exactly" in e for e in errors), errors)

    def test_far_changed_to_another_finite_in_range_number_rejected(self):
        def mutate(c):
            c["validation_evidence"]["diagnostic_out_of_scope_ant_far"] = 0.52
        errors = schema.validate(self._mutate(mutate))
        self.assertTrue(any("diagnostic_out_of_scope_ant_far must be exactly" in e for e in errors), errors)


class TestSchemaV2Envelope(unittest.TestCase):
    """Schema v2's top-level/generation envelope is checked EXACTLY.
    v1 is untouched by every check here -- see test_v1_policy_with_extra_
    top_level_field_is_not_rejected_by_this_block below."""

    def _mutate(self, mutator):
        policy = valid_v2_policy()
        mutator(policy)
        policy["content_sha256"] = schema.compute_content_sha256(2, policy["content"])
        return policy

    def test_extra_top_level_field_rejected(self):
        def mutate(p):
            p["smuggled"] = "extra"
        errors = schema.validate(self._mutate(mutate))
        self.assertTrue(any("unexpected top-level key" in e for e in errors), errors)

    def test_missing_top_level_field_rejected(self):
        def mutate(p):
            del p["generation"]
        errors = schema.validate(self._mutate(mutate))
        self.assertTrue(any("missing top-level key" in e for e in errors), errors)

    def test_extra_generation_field_rejected(self):
        def mutate(p):
            p["generation"] = dict(p["generation"])
            p["generation"]["hostname"] = "smuggled-host"
        errors = schema.validate(self._mutate(mutate))
        self.assertTrue(any("generation has unexpected key" in e for e in errors), errors)

    def test_missing_generation_field_rejected(self):
        def mutate(p):
            p["generation"] = {"generated_at": p["generation"]["generated_at"]}
        errors = schema.validate(self._mutate(mutate))
        self.assertTrue(any("generation missing key" in e for e in errors), errors)

    def test_changed_generator_version_rejected(self):
        def mutate(p):
            p["generation"] = dict(p["generation"])
            p["generation"]["generator_version"] = "9.9.9-not-frozen"
        errors = schema.validate(self._mutate(mutate))
        self.assertTrue(any("generator_version must be exactly" in e for e in errors), errors)

    def test_malformed_timestamp_rejected(self):
        for bad in ("2026-01-01 00:00:00Z", "2026-01-01T00:00:00", "2026-01-01T00:00:00+00:00",
                   "not-a-timestamp", "2026-01-01T00:00:00.000Z"):
            with self.subTest(bad=bad):
                def mutate(p, bad=bad):
                    p["generation"] = dict(p["generation"])
                    p["generation"]["generated_at"] = bad
                errors = schema.validate(self._mutate(mutate))
                self.assertTrue(any("generated_at must match the strict UTC form" in e for e in errors), errors)

    def test_valid_strict_timestamp_accepted(self):
        policy = valid_v2_policy()
        policy["generation"]["generated_at"] = "2030-12-31T23:59:59Z"
        self.assertEqual(schema.validate(policy), [])

    def test_v1_policy_with_extra_top_level_field_is_not_rejected_by_this_block(self):
        # v1's envelope is deliberately unconstrained by the v2-only checks
        # above -- an extra top-level field on a v1 policy is not rejected
        # by THIS mechanism (v1 compatibility must remain unchanged).
        policy = valid_v1_policy()
        policy["smuggled"] = "extra"
        errors = schema.validate(policy)
        self.assertFalse(any("unexpected top-level key" in e for e in errors), errors)

    def test_v1_policy_generator_version_is_unconstrained(self):
        policy = valid_v1_policy()
        policy["generation"]["generator_version"] = "anything-goes-for-v1"
        self.assertEqual(schema.validate(policy), [])


class TestV1GeneratorStillPinnedToV1(unittest.TestCase):
    """Adding schema v2 must not cause the OLD V1 generator to accidentally
    emit schema v2 -- it reads the SAME unrenamed SCHEMA_VERSION/
    FROZEN_THRESHOLD constants, which still mean exactly v1/0.60."""

    def test_v1_generator_module_constants_are_schema_v1(self):
        self.assertEqual(v1_generator.schema.SCHEMA_VERSION, 1)
        self.assertEqual(v1_generator.schema.SCHEMA_VERSION, schema.SCHEMA_VERSION)
        self.assertNotEqual(v1_generator.schema.SCHEMA_VERSION, schema.SCHEMA_VERSION_V2)
        self.assertEqual(v1_generator.schema.FROZEN_THRESHOLD, 0.6)

    def test_v1_generator_source_never_references_schema_v2_names(self):
        import inspect
        source = inspect.getsource(v1_generator)
        for forbidden in ("SCHEMA_VERSION_V2", "FROZEN_THRESHOLD_V2", "validation_evidence"):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
