#!/usr/bin/env python3
"""test_manual_review_tool.py — offline/synthetic tests for
manual_review_tool.py, against a full synthetic contract+queue+summary
fixture (never the real 600-row queue, never a real image)."""
from __future__ import annotations

import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import manual_review_contract as mrc  # noqa: E402
import manual_review_tool as mrt  # noqa: E402
import append_only_ledger as aol  # noqa: E402
from test_manual_review_fixtures import build_tiny_review_fixture  # noqa: E402


def _decision_fields(row: dict, contract: dict, **overrides) -> dict:
    fields = {
        "queue_index": row["queue_index"], "dataset": row["dataset"], "split": row["split"],
        "slug": row["slug"], "observation_uuid": row["observation_uuid"], "photo_id": row["photo_id"],
        "sha256": row["sha256"],
        "contract_content_sha256": contract["content_sha256"],
        "queue_byte_sha256": contract["content"]["queue_artifact"]["byte_sha256"],
        "queue_identity_order_sha256": contract["content"]["queue_artifact"]["identity_order_sha256"],
        "review_state": "usable", "label_plausibility": "plausible", "duplicate_suspicion": "none",
        "notes": "", "reviewer_id": "reviewer-1", "session_id": "session-1",
        "reviewed_at_utc": "2026-01-01T00:00:00Z",
    }
    fields.update(overrides)
    return fields


class _FixtureTestCase(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.repo = Path(self._tmpdir.name)
        self.fx = build_tiny_review_fixture(self, self.repo)
        self.verified = mrc.load_and_verify_contract(self.repo)
        self.rows = self.verified["queue_rows"]

    def tearDown(self):
        self._tmpdir.cleanup()


class TestLoadVerified(_FixtureTestCase):
    def test_loads_the_fixture_contract(self):
        verified = mrt.load_verified(self.repo)
        self.assertEqual(verified["contract"]["content_sha256"], self.fx["contract"]["content_sha256"])

    def test_missing_contract_rejected(self):
        (self.repo / mrc.CONTRACT_REL_PATH).unlink()
        with self.assertRaises(mrc.ContractError):
            mrt.load_verified(self.repo)

    def test_substituted_queue_rejected(self):
        # Replace the queue CSV with a DIFFERENT (but same-shaped) file --
        # the contract's bound byte hash must reject it.
        queue_path = self.repo / mrc.QUEUE_ARTIFACT_REL_PATH
        original = queue_path.read_bytes()
        mutated = original.replace(b"usable" if b"usable" in original else b"a", b"a", 1) + b" "
        queue_path.write_bytes(mutated)
        with self.assertRaises(mrc.ContractError) as ctx:
            mrt.load_verified(self.repo)
        self.assertIn("not the approved queue", str(ctx.exception))

    def test_substituted_contract_with_different_queue_hash_rejected(self):
        import json
        contract_path = self.repo / mrc.CONTRACT_REL_PATH
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        contract["content"] = dict(contract["content"])
        contract["content"]["queue_artifact"] = dict(contract["content"]["queue_artifact"])
        contract["content"]["queue_artifact"]["byte_sha256"] = "f" * 64
        contract["content_sha256"] = mrc.compute_content_sha256(contract["content"])
        contract_path.write_text(__import__("json").dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        with self.assertRaises(mrc.ContractError):
            mrt.load_verified(self.repo)


class TestOtherEvidenceManifestBindingFailsBeforeImageAccess(_FixtureTestCase):
    """Item 5: a changed/replaced other-evidence manifest (or its JSON
    provenance sidecar) must be rejected BEFORE any image is opened, for
    every one of the 5 other-evidence datasets -- not only expansion/
    final_test."""

    OTHER_EVIDENCE_NAMES = ("benchmark_v1", "calibration_v1", "unknown_test_v1",
                            "calibration_v2", "unknown_test_v2")

    def test_every_dataset_csv_byte_change_rejected(self):
        for name in self.OTHER_EVIDENCE_NAMES:
            with self.subTest(dataset=name):
                repo = Path(tempfile.mkdtemp())
                self.addCleanup(lambda r=repo: __import__("shutil").rmtree(r, ignore_errors=True))
                fx = build_tiny_review_fixture(self, repo)
                csv_path = repo / mrc.OTHER_EVIDENCE_MANIFESTS[name]["path"]
                original = csv_path.read_bytes()
                csv_path.write_bytes(original + b"\n")  # any byte change -- hash no longer matches

                real_open = Path.open
                opened_images = []

                def spy(self, *a, **kw):
                    s = str(self)
                    if s.lower().endswith((".jpg", ".jpeg", ".png")):
                        opened_images.append(s)
                    return real_open(self, *a, **kw)

                with mock.patch.object(Path, "open", spy):
                    with self.assertRaises(mrc.ContractError) as ctx:
                        mrt.load_verified(repo)
                self.assertIn("this is not the approved manifest", str(ctx.exception))
                self.assertEqual(opened_images, [])

    def test_every_dataset_provenance_json_byte_change_rejected(self):
        for name in self.OTHER_EVIDENCE_NAMES:
            with self.subTest(dataset=name):
                repo = Path(tempfile.mkdtemp())
                self.addCleanup(lambda r=repo: __import__("shutil").rmtree(r, ignore_errors=True))
                fx = build_tiny_review_fixture(self, repo)
                json_path = repo / mrc.OTHER_EVIDENCE_MANIFESTS[name]["provenance_json_path"]
                original = json_path.read_bytes()
                json_path.write_bytes(original + b" ")

                real_open = Path.open
                opened_images = []

                def spy(self, *a, **kw):
                    s = str(self)
                    if s.lower().endswith((".jpg", ".jpeg", ".png")):
                        opened_images.append(s)
                    return real_open(self, *a, **kw)

                with mock.patch.object(Path, "open", spy):
                    with self.assertRaises(mrc.ContractError) as ctx:
                        mrt.load_verified(repo)
                self.assertIn("approved provenance file", str(ctx.exception))
                self.assertEqual(opened_images, [])


class TestValidateDecisionFields(_FixtureTestCase):
    def test_valid_decision_has_no_problems(self):
        row = self.rows[0]
        fields = _decision_fields(row, self.verified["contract"])
        self.assertEqual(mrt.validate_decision_fields(fields, row, self.verified["contract"]), [])

    def test_mismatched_sha256_rejected(self):
        row = self.rows[0]
        fields = _decision_fields(row, self.verified["contract"], sha256="0" * 64)
        problems = mrt.validate_decision_fields(fields, row, self.verified["contract"])
        self.assertTrue(any("sha256" in p for p in problems))

    def test_mismatched_contract_content_sha256_rejected(self):
        row = self.rows[0]
        fields = _decision_fields(row, self.verified["contract"], contract_content_sha256="0" * 64)
        problems = mrt.validate_decision_fields(fields, row, self.verified["contract"])
        self.assertTrue(any("contract_content_sha256" in p for p in problems))

    def test_free_form_review_state_rejected(self):
        row = self.rows[0]
        fields = _decision_fields(row, self.verified["contract"], review_state="looks_fine")
        problems = mrt.validate_decision_fields(fields, row, self.verified["contract"])
        self.assertTrue(any("review_state" in p for p in problems))

    def test_missing_field_rejected(self):
        row = self.rows[0]
        fields = _decision_fields(row, self.verified["contract"])
        del fields["notes"]
        problems = mrt.validate_decision_fields(fields, row, self.verified["contract"])
        self.assertTrue(any("missing field" in p for p in problems))

    def test_extra_field_rejected(self):
        row = self.rows[0]
        fields = _decision_fields(row, self.verified["contract"])
        fields["smuggled"] = "x"
        problems = mrt.validate_decision_fields(fields, row, self.verified["contract"])
        self.assertTrue(any("unexpected field" in p for p in problems))

    def test_every_review_state_accepted(self):
        row = self.rows[0]
        for state in mrc.REVIEW_STATES:
            with self.subTest(state=state):
                fields = _decision_fields(row, self.verified["contract"], review_state=state)
                self.assertEqual(mrt.validate_decision_fields(fields, row, self.verified["contract"]), [])


class TestRecordDecisionAndResume(_FixtureTestCase):
    def _first_unreviewed(self):
        records = mrt.load_verified_ledger(self.repo, self.verified)
        return mrt.first_unreviewed_queue_index_from(self.verified, records)

    def test_first_unreviewed_is_zero_initially(self):
        self.assertEqual(self._first_unreviewed(), 0)

    def test_record_then_resume_advances(self):
        row = self.rows[0]
        fields = _decision_fields(row, self.verified["contract"])
        mrt.record_decision(self.repo, self.verified, fields)
        self.assertEqual(self._first_unreviewed(), 1)

    def test_out_of_order_review_still_resumes_at_first_gap(self):
        mrt.record_decision(self.repo, self.verified, _decision_fields(self.rows[0], self.verified["contract"]))
        mrt.record_decision(self.repo, self.verified, _decision_fields(self.rows[2], self.verified["contract"]))
        self.assertEqual(self._first_unreviewed(), 1)

    def test_duplicate_decision_for_same_queue_index_rejected(self):
        row = self.rows[0]
        mrt.record_decision(self.repo, self.verified, _decision_fields(row, self.verified["contract"]))
        with self.assertRaises(mrt.ReviewToolError):
            mrt.record_decision(self.repo, self.verified, _decision_fields(row, self.verified["contract"]))
        self.assertEqual(len(aol.read_ledger(self.fx["ledger_path"])), 1)

    def test_invalid_decision_never_appends(self):
        row = self.rows[0]
        bad = _decision_fields(row, self.verified["contract"], review_state="nonsense")
        with self.assertRaises(mrt.ReviewToolError):
            mrt.record_decision(self.repo, self.verified, bad)
        self.assertEqual(aol.read_ledger(self.fx["ledger_path"]), [])

    def test_unknown_queue_index_rejected(self):
        fields = _decision_fields(self.rows[0], self.verified["contract"], queue_index=99999)
        with self.assertRaises(mrt.ReviewToolError):
            mrt.record_decision(self.repo, self.verified, fields)

    def test_max_decisions_per_session_enforced(self):
        with mock.patch.object(mrc, "MAX_DECISIONS_PER_SESSION", 2):
            mrt.record_decision(self.repo, self.verified,
                               _decision_fields(self.rows[0], self.verified["contract"], session_id="s1"))
            mrt.record_decision(self.repo, self.verified,
                               _decision_fields(self.rows[1], self.verified["contract"], session_id="s1"))
            with self.assertRaises(mrt.ReviewToolError) as ctx:
                mrt.record_decision(self.repo, self.verified,
                                   _decision_fields(self.rows[2], self.verified["contract"], session_id="s1"))
            self.assertIn("maximum", str(ctx.exception))

    def test_ledger_hash_chain_verifiable_after_several_decisions(self):
        for i in range(3):
            mrt.record_decision(self.repo, self.verified, _decision_fields(self.rows[i], self.verified["contract"]))
        records = aol.read_ledger(self.fx["ledger_path"])
        self.assertEqual(aol.verify_chain(records, mrc.REVIEW_RECORD_REQUIRED_FIELDS, mrc.review_record_identity), [])

    def test_review_progress_reports_correctly(self):
        mrt.record_decision(self.repo, self.verified, _decision_fields(self.rows[0], self.verified["contract"]))
        mrt.record_decision(self.repo, self.verified, _decision_fields(self.rows[1], self.verified["contract"]))
        records = mrt.load_verified_ledger(self.repo, self.verified)
        progress = mrt.review_progress_from(self.verified, records)
        self.assertEqual(progress["reviewed_count"], 2)
        self.assertEqual(progress["first_unreviewed_queue_index"], 2)
        self.assertEqual(progress["chain_problems"], [])


class TestBuildDecisionFieldsAutoBinding(_FixtureTestCase):
    def test_binds_contract_and_queue_hashes_automatically(self):
        row = self.rows[0]
        fields = mrt.build_decision_fields(
            row, self.verified["contract"], review_state="usable", label_plausibility="plausible",
            duplicate_suspicion="none", notes="", reviewer_id="r1", session_id="s1",
            reviewed_at_utc="2026-01-01T00:00:00Z")
        self.assertEqual(fields["contract_content_sha256"], self.verified["contract"]["content_sha256"])
        self.assertEqual(mrt.validate_decision_fields(fields, row, self.verified["contract"]), [])


class TestCliModes(_FixtureTestCase):
    def _args(self, **overrides):
        import argparse
        d = dict(repo=self.repo)
        d.update(overrides)
        return argparse.Namespace(**d)

    def test_cmd_preflight_reports_progress(self):
        result = mrt.cmd_preflight(self._args())
        self.assertTrue(result["ok"])
        self.assertEqual(result["reviewed_count"], 0)
        self.assertEqual(result["first_unreviewed_queue_index"], 0)

    def test_cmd_next_reports_first_row_and_never_opens_it(self):
        real_open = Path.open
        opened = []

        def spy(self, *a, **kw):
            opened.append(str(self))
            return real_open(self, *a, **kw)

        with mock.patch.object(Path, "open", spy):
            result = mrt.cmd_next(self._args())
        self.assertFalse(result["done"])
        self.assertEqual(result["queue_index"], 0)
        self.assertIn("image_path", result)
        self.assertTrue(result["image_path"].endswith(".jpg"))
        self.assertEqual([p for p in opened if p == result["image_path"]], [])

    def test_cmd_next_reports_done_when_fully_reviewed(self):
        for row in self.rows:
            mrt.record_decision(self.repo, self.verified, _decision_fields(row, self.verified["contract"]))
        result = mrt.cmd_next(self._args())
        self.assertTrue(result["done"])

    def test_cmd_record_appends_and_returns_hash(self):
        row = self.rows[0]
        args = self._args(queue_index=row["queue_index"], review_state="usable",
                          label_plausibility="plausible", duplicate_suspicion="none",
                          notes="", reviewer_id="r1", session_id="s1",
                          reviewed_at_utc="2026-01-01T00:00:00Z")
        result = mrt.cmd_record(args)
        self.assertTrue(result["recorded"])
        self.assertEqual(result["queue_index"], row["queue_index"])

    def test_cmd_status_verifies_chain(self):
        result = mrt.cmd_status(self._args())
        self.assertEqual(result["chain_problems"], [])
        self.assertEqual(result["reviewed_count"], 0)


class TestManualReviewLedgerSemanticVerification(_FixtureTestCase):
    """Item 1/2/6: every working mode must semantically re-verify existing
    ledger records, not merely trust generic hash-chain verification."""

    def _args(self, **overrides):
        import argparse
        d = dict(repo=self.repo)
        d.update(overrides)
        return argparse.Namespace(**d)

    def _tamper_and_rehash_single_record(self, **field_overrides) -> bytes:
        """Records one valid decision, then rewrites the ledger with an
        identity/binding field altered (queue_index unchanged) and the
        record_hash recomputed so generic verify_chain still passes --
        simulates a record edited outside this tool. Returns the ledger's
        raw bytes AFTER tampering, for later byte-identity comparison."""
        row = self.rows[0]
        mrt.record_decision(self.repo, self.verified, _decision_fields(row, self.verified["contract"]))
        ledger_path = self.fx["ledger_path"]
        record = json.loads(ledger_path.read_text(encoding="utf-8").strip())
        record.update(field_overrides)
        payload = {k: v for k, v in record.items() if k != "record_hash"}
        record["record_hash"] = aol.sha256_hex(aol.canonical_bytes(payload))
        ledger_path.write_text(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n",
                               encoding="utf-8")
        return ledger_path.read_bytes()

    def test_mutate_and_rehash_slug_rejected_by_every_mode_and_ledger_unchanged(self):
        tampered_bytes = self._tamper_and_rehash_single_record(slug="not-the-real-slug")

        with self.assertRaises(mrt.ReviewToolError):
            mrt.cmd_preflight(self._args())
        with self.assertRaises(mrt.ReviewToolError):
            mrt.cmd_next(self._args())
        with self.assertRaises(mrt.ReviewToolError):
            mrt.cmd_status(self._args())

        # A NEW, otherwise-valid decision for a DIFFERENT row must also be
        # rejected -- no further decisions may be appended while the
        # existing ledger is semantically tainted.
        other_row = self.rows[1]
        fields = _decision_fields(other_row, self.verified["contract"])
        with self.assertRaises(mrt.ReviewToolError):
            mrt.record_decision(self.repo, self.verified, fields)

        # No item was silently skipped and nothing was rewritten/repaired:
        # the ledger is still exactly the tampered bytes.
        self.assertEqual(self.fx["ledger_path"].read_bytes(), tampered_bytes)

    def test_mutate_and_rehash_sha256_rejected(self):
        self._tamper_and_rehash_single_record(sha256="f" * 64)
        with self.assertRaises(mrt.ReviewToolError):
            mrt.cmd_preflight(self._args())

    def test_mutate_and_rehash_contract_binding_rejected(self):
        self._tamper_and_rehash_single_record(contract_content_sha256="0" * 64)
        with self.assertRaises(mrt.ReviewToolError):
            mrt.cmd_preflight(self._args())

    def test_cmd_preflight_never_returns_ok_true_with_a_tainted_ledger(self):
        self._tamper_and_rehash_single_record(observation_uuid="not-the-real-uuid")
        try:
            result = mrt.cmd_preflight(self._args())
        except mrt.ReviewToolError:
            return  # correctly refused outright -- satisfies the requirement
        self.fail(f"cmd_preflight must raise or never report ok=True on a tainted ledger, got {result!r}")

    def test_genuinely_valid_incomplete_ledger_still_resumes_normally(self):
        # Sanity/regression companion to the tamper tests above: an
        # UNTAMPERED partial ledger must keep working exactly as before.
        mrt.record_decision(self.repo, self.verified, _decision_fields(self.rows[0], self.verified["contract"]))
        result = mrt.cmd_preflight(self._args())
        self.assertTrue(result["ok"])
        self.assertEqual(result["chain_problems"], [])
        self.assertEqual(result["reviewed_count"], 1)
        self.assertEqual(result["first_unreviewed_queue_index"], 1)
        next_result = mrt.cmd_next(self._args())
        self.assertEqual(next_result["queue_index"], 1)

    def test_interrupted_tail_still_recoverable_by_next_authorized_record(self):
        # Item 5: the deliberate interrupted-append recovery behavior must
        # survive the new semantic checks -- a genuinely interrupted
        # (non-newline-terminated) final line is still recoverable by the
        # next authorized --record, not treated as tainted.
        row0 = self.rows[0]
        mrt.record_decision(self.repo, self.verified, _decision_fields(row0, self.verified["contract"]))
        ledger_path = self.fx["ledger_path"]
        with ledger_path.open("a", encoding="utf-8") as fh:
            fh.write('{"queue_index": 1, "dataset": "trunc')  # interrupted mid-write, no newline

        # Read-only modes must not mutate the ledger while recovering it.
        before = ledger_path.read_bytes()
        result = mrt.cmd_preflight(self._args())
        self.assertTrue(result["ok"])
        self.assertEqual(result["reviewed_count"], 1)
        self.assertEqual(ledger_path.read_bytes(), before, "a read-only mode must never mutate the ledger")

        # And a genuinely new, valid decision still appends successfully.
        row1 = self.rows[1]
        record = mrt.record_decision(self.repo, self.verified, _decision_fields(row1, self.verified["contract"]))
        self.assertEqual(record["queue_index"], 1)
        records = aol.read_ledger(ledger_path)
        self.assertEqual([r["queue_index"] for r in records], [0, 1])
        self.assertEqual(aol.verify_chain(records, mrc.REVIEW_RECORD_REQUIRED_FIELDS, mrc.review_record_identity), [])


class TestNeverExposesModelPredictions(unittest.TestCase):
    def test_source_never_imports_model_or_inference_modules(self):
        import ast
        tree = ast.parse(inspect.getsource(mrt))
        imported_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_names.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_names.add(node.module)
        forbidden = {"onnxruntime", "inference_policy", "inference", "api.inference"}
        self.assertEqual(imported_names & forbidden, set(),
                         f"manual_review_tool.py must never import any of {forbidden}")

    def test_source_never_opens_a_jpg_literal(self):
        source = inspect.getsource(mrt)
        # image_path_for_row constructs a path but must never call .open()/
        # Image.open on it anywhere in this module.
        self.assertNotIn("Image.open", source)
        self.assertNotIn(".read_bytes()", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
