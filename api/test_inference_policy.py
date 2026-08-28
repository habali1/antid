#!/usr/bin/env python3
"""Dependency-free regression tests for the optional API policy loader."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

import inference_policy
import policy_schema as schema


ARTIFACT_BYTES = {
    "backbone.onnx": b"test onnx bytes\x00",
    "prototypes.npy": b"test prototype bytes\x00",
    "taxonomy.json": b'{"0":{"species_name":"Test ant","slug":"test-ant"}}',
}
PREPROCESSING_CONTRACT = {
    "rgb_conversion": "img.convert('RGB')",
    "resize": "squish to fixed 380x380 (both dims set, no crop)",
    "interpolation": "Pillow bilinear",
    "scale_divisor": 255.0,
    "normalize_mean": [0.485, 0.456, 0.406],
    "normalize_std": [0.229, 0.224, 0.225],
    "dtype": "float32",
    "channel_layout": "RGB -> CHW, batched to NCHW",
}


class TestInferencePolicyLoader(unittest.TestCase):
    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.artifacts = Path(self._temp.name)
        for name, data in ARTIFACT_BYTES.items():
            (self.artifacts / name).write_bytes(data)
        self.write_policy(self.valid_policy())

    def tearDown(self):
        self._temp.cleanup()

    def valid_policy(self) -> dict:
        hashes = {
            name: hashlib.sha256(data).hexdigest()
            for name, data in ARTIFACT_BYTES.items()
        }
        content = {
            "rule": {
                "operator_verbatim": "max_sim < value",
                "operator_normalized": {"comparison": "strict_less_than"},
                "signal": "raw max cosine similarity before geo re-ranking",
                "value": 0.6,
            },
            "decision": {
                "low_confidence_if": {
                    "signal": "raw_pre_geo_max_cosine",
                    "comparison": "strict_less_than",
                    "threshold": 0.6,
                },
                "normal_results_if": {
                    "comparison": "greater_than_or_equal",
                    "threshold": 0.6,
                },
                "equal_threshold_action": "normal_results",
                "non_finite_action": "request_error",
            },
            "artifact_hashes": hashes,
            "provider_policy": {
                "providers": ["CPUExecutionProvider"],
                "exclusive": True,
            },
            "preprocessing": {
                "contract": json.loads(json.dumps(PREPROCESSING_CONTRACT)),
            },
        }
        return {
            "policy_schema_version": 1,
            "content": content,
            "content_sha256": schema.compute_content_sha256(1, content),
            "generation": {
                "generated_at": "2026-01-01T00:00:00Z",
                "generator_version": "test",
            },
        }

    def write_policy(self, policy: dict, *, recompute_hash: bool = False) -> None:
        if recompute_hash:
            policy["content_sha256"] = schema.compute_content_sha256(
                policy["policy_schema_version"], policy["content"])
        (self.artifacts / inference_policy.POLICY_FILENAME).write_text(
            json.dumps(policy), encoding="utf-8")

    def load(self, *, providers=None, contract=None) -> inference_policy.PolicyState:
        return inference_policy.load_inference_policy(
            self.artifacts,
            ["CPUExecutionProvider"] if providers is None else providers,
            PREPROCESSING_CONTRACT if contract is None else contract,
        )

    def test_valid_policy_and_strict_unrounded_boundaries(self):
        state = self.load()
        self.assertTrue(state.active)
        self.assertEqual(state.reason, "active")
        self.assertEqual(state.threshold, 0.6)
        self.assertTrue(state.classify(0.5999))
        self.assertFalse(state.classify(0.6000))
        self.assertFalse(state.classify(0.6001))
        self.assertTrue(state.classify(0.599999999999))

        for value in (float("nan"), float("inf"), float("-inf"), True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                state.classify(value)

    def test_missing_policy_is_inactive_not_false(self):
        (self.artifacts / inference_policy.POLICY_FILENAME).unlink()
        state = self.load()
        self.assertFalse(state.active)
        self.assertEqual(state.reason, "policy_missing")
        self.assertIsNone(state.threshold)
        self.assertIsNone(state.classify(0.2))
        with self.assertRaises(ValueError):
            state.classify(math.nan)

    def test_invalid_json_and_invalid_schema_are_distinct(self):
        path = self.artifacts / inference_policy.POLICY_FILENAME
        path.write_text("{not json", encoding="utf-8")
        self.assertEqual(self.load().reason, "invalid_json")

        policy = self.valid_policy()
        policy["policy_schema_version"] = True
        self.write_policy(policy)
        self.assertEqual(self.load().reason, "invalid_schema")

    def test_content_hash_tampering_disables_gate(self):
        policy = self.valid_policy()
        policy["content_sha256"] = "0" * 64
        self.write_policy(policy)
        self.assertEqual(self.load().reason, "content_hash_mismatch")

    def test_each_live_artifact_hash_is_checked_but_geo_is_not_bound(self):
        for name in ARTIFACT_BYTES:
            with self.subTest(name=name):
                original = (self.artifacts / name).read_bytes()
                (self.artifacts / name).write_bytes(original + b"tampered")
                self.assertEqual(self.load().reason, "artifact_hash_mismatch")
                (self.artifacts / name).write_bytes(original)

        (self.artifacts / "geo_index.json").write_text("not policy-bound", encoding="utf-8")
        self.assertTrue(self.load().active)

    def test_provider_mismatch_in_policy_or_runtime_disables_gate(self):
        self.assertEqual(
            self.load(providers=["AzureExecutionProvider", "CPUExecutionProvider"]).reason,
            "provider_mismatch",
        )

        policy = self.valid_policy()
        policy["content"]["provider_policy"]["providers"] = ["AzureExecutionProvider"]
        self.write_policy(policy, recompute_hash=True)
        self.assertEqual(self.load().reason, "provider_mismatch")

    def test_preprocessing_mismatch_disables_gate(self):
        changed = dict(PREPROCESSING_CONTRACT)
        changed["scale_divisor"] = 1.0
        self.assertEqual(self.load(contract=changed).reason, "preprocessing_mismatch")

        policy = self.valid_policy()
        policy["content"]["preprocessing"]["contract"]["interpolation"] = "nearest"
        self.write_policy(policy, recompute_hash=True)
        self.assertEqual(self.load().reason, "preprocessing_mismatch")

    def test_unsupported_operator_and_signal_encodings_disable_gate(self):
        mutations = (
            lambda c: c["rule"].__setitem__("operator_verbatim", "max_sim <= value"),
            lambda c: c["rule"].__setitem__(
                "operator_normalized", {"comparison": "less_than_or_equal"}),
            lambda c: c["rule"].__setitem__("signal", "post-geo score"),
            lambda c: c["decision"]["low_confidence_if"].__setitem__(
                "signal", "rounded_score"),
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                policy = self.valid_policy()
                mutate(policy["content"])
                self.write_policy(policy, recompute_hash=True)
                self.assertEqual(self.load().reason, "unsupported_rule")

    def test_rule_value_must_match_both_decision_thresholds(self):
        policy = self.valid_policy()
        policy["content"]["rule"]["value"] = 0.55
        self.write_policy(policy, recompute_hash=True)
        self.assertEqual(self.load().reason, "unsupported_rule")

    def test_rehashed_policy_cannot_change_the_frozen_threshold(self):
        policy = self.valid_policy()
        policy["content"]["rule"]["value"] = 0.59995
        policy["content"]["decision"]["low_confidence_if"]["threshold"] = 0.59995
        policy["content"]["decision"]["normal_results_if"]["threshold"] = 0.59995
        self.write_policy(policy, recompute_hash=True)
        state = self.load()
        self.assertFalse(state.active)
        self.assertEqual(state.reason, "unsupported_rule")

    def test_policy_read_io_error_is_inactive(self):
        path = self.artifacts / inference_policy.POLICY_FILENAME
        path.unlink()
        path.mkdir()
        self.assertEqual(self.load().reason, "io_error")


if __name__ == "__main__":
    unittest.main(verbosity=2)
