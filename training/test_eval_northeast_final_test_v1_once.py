"""Offline tests; no final-test image is opened or inference run."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("final_once", HERE / "eval_northeast_final_test_v1_once.py")
import sys
sys.path.insert(0, str(HERE))
final = importlib.util.module_from_spec(spec)
spec.loader.exec_module(final)


class TestReportingRule(unittest.TestCase):
    def test_rule_binds_incomplete_independence_and_all_rows(self):
        rule = json.loads((HERE / "final_test_v1_reporting_rule.json").read_text())
        self.assertEqual(rule["independence_label"], final.EXPECTED_DECISION_STATUS)
        self.assertEqual(rule["v2_perceptual_contract_result"], "not_passed_not_finalized")
        self.assertEqual(rule["metric_denominator"], "all_450_frozen_images_no_exclusions")
        self.assertEqual(rule["total_images"], 450)
        self.assertEqual(rule["unusable_final_test_images_included"], 3)
        self.assertNotIn("--threshold", (HERE / "eval_northeast_final_test_v1_once.py").read_text())

    def test_frozen_hashes(self):
        repo = HERE.parent
        for rel, expected in [(final.CSV_PATH, final.EXPECTED_CSV_SHA256),
                              (final.JSON_PATH, final.EXPECTED_JSON_SHA256),
                              (final.V3_PATH, final.EXPECTED_V3_SHA256),
                              (final.POLICY_PATH, final.EXPECTED_POLICY_SHA256)]:
            self.assertEqual(final.sha256(repo / rel), expected)


class TestMetrics(unittest.TestCase):
    @staticmethod
    def records():
        records = []
        for s in range(15):
            for i in range(30):
                records.append({"true_slug": f"species-{s}", "top1_correct": i < 20,
                                "top3_correct": i < 25, "low_confidence": i >= 15})
        return records

    def test_all_frozen_rows_are_denominator(self):
        metrics = final.compute_metrics(self.records())
        self.assertEqual(metrics["total"], 450)
        self.assertEqual(metrics["top1_correct"], 300)
        self.assertEqual(metrics["top3_correct"], 375)
        self.assertEqual(metrics["accepted"], 225)
        self.assertEqual(metrics["correct_rejected"], 75)
        self.assertEqual(metrics["incorrect_rejected"], 150)
        self.assertEqual(len(metrics["per_species"]), 15)
        self.assertEqual(metrics["macro_top1"], 2 / 3)

    def test_missing_row_fails_closed(self):
        with self.assertRaisesRegex(final.FinalTestError, "all 450"):
            final.compute_metrics(self.records()[:-1])

    def test_nonfinites_cannot_hash_into_report(self):
        with self.assertRaises(ValueError):
            final.canonical_hash({"metric": float("nan")})


class TestExclusivePublish(unittest.TestCase):
    def test_does_not_overwrite_existing_result(self):
        with tempfile.TemporaryDirectory() as temp:
            dest = Path(temp) / "out.json"
            dest.write_bytes(b"first")
            with self.assertRaises(FileExistsError):
                final.publish_exclusive(dest, b"second")
            self.assertEqual(dest.read_bytes(), b"first")
            self.assertEqual(list(Path(temp).iterdir()), [dest])

    def test_publishes_complete_bytes_once(self):
        with tempfile.TemporaryDirectory() as temp:
            dest = Path(temp) / "out.json"
            final.publish_exclusive(dest, b"content\n")
            self.assertEqual(dest.read_bytes(), b"content\n")
            self.assertEqual(list(Path(temp).iterdir()), [dest])


class TestResultValidation(unittest.TestCase):
    def test_complete_report_and_rehashed_mutations(self):
        with tempfile.TemporaryDirectory() as temp:
            marker = Path(temp) / "marker.json"
            marker.write_text("frozen marker\n")
            taxonomy = {i: {"slug": f"species-{i}"} for i in range(65)}
            rows, records = [], []
            for s in range(15):
                for i in range(30):
                    identity = (str(s * 30 + i), f"observation-{s}-{i}", "a" * 64)
                    rows.append({"photo_id": identity[0], "observation_uuid": identity[1],
                                 "sha256": identity[2], "slug": f"species-{s}"})
                    correct = i < 20
                    order = [s, s + 15, s + 30] if correct else [s + 15, s if i < 25 else s + 30,
                                                                  s + 30 if i < 25 else s + 45]
                    similarities = [.8, .7, .6] if i < 15 else [.5, .4, .3]
                    records.append({"photo_id": identity[0], "observation_uuid": identity[1],
                                    "image_sha256": identity[2], "true_slug": f"species-{s}",
                                    "true_class_index": s, "top3_indices": order,
                                    "top3_slugs": [taxonomy[k]["slug"] for k in order],
                                    "top3_similarities": similarities, "max_cosine": similarities[0],
                                    "low_confidence": i >= 15, "top1_correct": correct,
                                    "top3_correct": i < 25})
            pre = {"rows": rows, "marker": marker, "taxonomy": taxonomy,
                   "slug_to_idx": {f"species-{i}": i for i in range(65)},
                   "row_identity_order_sha256": final.gc.compute_identity_order_sha256(
                       [(r["photo_id"], r["observation_uuid"], r["sha256"]) for r in rows])}
            pre.update({k: k for k in ("head", "rule_sha256", "v3_sha256", "csv_sha256",
                                       "json_sha256", "policy_sha256")})
            content = {"dataset": "northeast_final_test_v1",
                       "status": "descriptive_result_not_perceptual_independence_pass",
                       "perceptual_independence": final.EXPECTED_DECISION_STATUS,
                       "v2_perceptual_contract_result": "not_passed_not_finalized",
                       "manual_quality": final.EXPECTED_MANUAL, "threshold": .61,
                       "provider": ["CPUExecutionProvider"], "geo_applied": False,
                       "bindings": {k: pre[k] for k in ("head", "rule_sha256", "v3_sha256",
                                                        "csv_sha256", "json_sha256", "policy_sha256",
                                                        "row_identity_order_sha256")},
                       "attempt_marker_sha256": final.sha256(marker), "records": records,
                       "metrics": final.compute_metrics(records)}
            result = {"schema_version": 1, "content": content,
                      "content_sha256": final.canonical_hash(content)}
            final.validate_result(pre, result)
            for key, bad in [("perceptual_independence", "passed"),
                             ("v2_perceptual_contract_result", "passed"),
                             ("threshold", .60)]:
                altered = json.loads(json.dumps(result))
                altered["content"][key] = bad
                altered["content_sha256"] = final.canonical_hash(altered["content"])
                with self.assertRaises(final.FinalTestError):
                    final.validate_result(pre, altered)
            altered = json.loads(json.dumps(result))
            altered["content"]["records"][0]["low_confidence"] = True
            altered["content"]["metrics"] = final.compute_metrics(altered["content"]["records"])
            altered["content_sha256"] = final.canonical_hash(altered["content"])
            with self.assertRaisesRegex(final.FinalTestError, "record decision"):
                final.validate_result(pre, altered)


if __name__ == "__main__":
    unittest.main()
