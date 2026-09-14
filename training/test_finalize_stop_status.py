#!/usr/bin/env python3
"""test_finalize_stop_status.py — offline/synthetic tests for
finalize_stop_status.py, the post-adjudication stop-decision gate.

Reuses the real-image tiny fixture (never a real dataset photograph) and a
real cmd_scan() run, then drives adjudicate_pairs.py to completion before
exercising finalize_stop_status.py's --preflight/--finalize/--check modes.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import manual_review_contract as mrc  # noqa: E402
import scan_perceptual_duplicates as spd  # noqa: E402
import adjudicate_pairs as adj  # noqa: E402
import finalize_stop_status as fin  # noqa: E402
from test_manual_review_fixtures import build_tiny_review_fixture  # noqa: E402


class _FinalizeFixtureTestCase(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.repo = Path(self._tmpdir.name)
        self.fx = build_tiny_review_fixture(self, self.repo, slugs=("fx-species-a",), n_final_test=2,
                                            per_slug=1, per_bucket=2, real_images=True)
        self.verified = mrc.load_and_verify_contract(self.repo)
        self.scan_result = spd.cmd_scan(self._args())

    def tearDown(self):
        self._tmpdir.cleanup()

    def _args(self, **overrides):
        d = dict(repo=self.repo)
        d.update(overrides)
        return argparse.Namespace(**d)

    def _all_candidate_ids(self) -> list[str]:
        cp_path = self.repo / mrc.APPROVED_OUTPUT_PATHS["candidate_pairs_report"]
        cp_report = json.loads(cp_path.read_text(encoding="utf-8"))
        ids = []
        for domain_name, pairs in cp_report["content"]["domains"].items():
            for pair in pairs:
                ids.append(pair["pair_id"])
        return ids

    def _adjudicate_all(self, *, same_source_pair_ids: set[str] = frozenset()) -> None:
        cp_path = self.repo / mrc.APPROVED_OUTPUT_PATHS["candidate_pairs_report"]
        cp_bytes = cp_path.read_bytes()
        cp_report = json.loads(cp_bytes.decode("utf-8"))
        report_sha256 = mrc.sha256_bytes(cp_bytes)
        for domain_name, pairs in cp_report["content"]["domains"].items():
            for pair in pairs:
                label = "same_source_image" if pair["pair_id"] in same_source_pair_ids else "different_image"
                fields = adj.build_adjudication_fields(
                    {**pair, "domain": domain_name}, report_sha256, cp_report["content_sha256"],
                    label=label, reviewer_id="r1", session_id="s1", reviewed_at_utc="2026-01-01T00:00:00Z")
                adj.record_adjudication(self.repo, fields)


class TestFinalizePreflight(_FinalizeFixtureTestCase):
    def test_preflight_reports_zero_progress_before_any_adjudication(self):
        result = fin.cmd_preflight(self._args())
        self.assertTrue(result["ok"])
        self.assertEqual(result["adjudicated_count"], 0)
        self.assertFalse(result["complete"])
        self.assertEqual(result["remaining_count"], result["total_candidates"])
        self.assertGreater(result["total_candidates"], 0)

    def test_preflight_never_opens_an_image(self):
        real_open = Path.open
        opened = []

        def spy(self, *a, **kw):
            s = str(self)
            if s.lower().endswith((".jpg", ".jpeg", ".png")):
                opened.append(s)
            return real_open(self, *a, **kw)

        from unittest import mock
        with mock.patch.object(Path, "open", spy):
            fin.cmd_preflight(self._args())
        self.assertEqual(opened, [])


class TestFinalizeRefusesIncomplete(_FinalizeFixtureTestCase):
    def test_finalize_refuses_when_not_all_candidates_adjudicated(self):
        all_ids = self._all_candidate_ids()
        self.assertGreater(len(all_ids), 0)
        with self.assertRaises(fin.FinalizationError):
            fin.cmd_finalize(self._args())
        # Refusal must never publish anything.
        dest = self.repo / mrc.APPROVED_OUTPUT_PATHS["post_adjudication_stop_status_report"]
        self.assertFalse(dest.exists())

    def test_partial_adjudication_content_is_never_complete(self):
        # Pure-function level (the real tiny fixture only ever produces one
        # real candidate, which makes "adjudicate one, leave the rest"
        # impossible to construct end-to-end here): a hand-built state with
        # 2 candidates but only 1 ledger record must build content with
        # complete=False and adjudicated_count < total_candidates.
        pair_a = {"pair_id": "a" * 64, "identity_a": "x", "identity_b": "y",
                  "phash_distance": 0, "dhash_distance": 0}
        pair_b = {"pair_id": "b" * 64, "identity_a": "p", "identity_b": "q",
                  "phash_distance": 0, "dhash_distance": 0}
        state = {
            "verified": {"contract": {"content_sha256": "c" * 64}},
            "reports": {
                "candidate_pairs_report": {"content_sha256": "d" * 64,
                                           "content": {"domains": {"within_final_test": [pair_a, pair_b]}}},
                "stop_status_report": {"content_sha256": "e" * 64,
                                       "content": {"domains": {"within_final_test": {"stop_before_inference": False, "reasons": []}}}},
            },
            "ledger_records": [{"pair_id": "a" * 64, "domain": "within_final_test", "label": "different_image"}],
            "ledger_bytes": b"irrelevant",
        }
        content = fin.build_post_adjudication_content(state)
        self.assertEqual(content["total_candidates"], 2)
        self.assertEqual(content["adjudicated_count"], 1)
        self.assertFalse(content["complete"])


class TestFinalizeSuccess(_FinalizeFixtureTestCase):
    def test_finalize_succeeds_and_confirmed_duplicate_triggers_stop(self):
        cp_path = self.repo / mrc.APPROVED_OUTPUT_PATHS["candidate_pairs_report"]
        cp_report = json.loads(cp_path.read_text(encoding="utf-8"))
        forced_pairs = cp_report["content"]["domains"]["benchmark_v1_vs_final_test"]
        self.assertGreaterEqual(len(forced_pairs), 1)
        same_source_ids = {forced_pairs[0]["pair_id"]}

        # Snapshot the five scan reports + ledger bytes before finalizing.
        before = {name: (self.repo / mrc.APPROVED_OUTPUT_PATHS[name]).read_bytes()
                 for name in mrc.SCAN_REPORT_NAMES}

        self._adjudicate_all(same_source_pair_ids=same_source_ids)
        result = fin.cmd_finalize(self._args())
        self.assertTrue(result["ok"])
        self.assertTrue(result["overall_stop_before_inference"])

        # The five earlier scan reports must be byte-for-byte untouched.
        for name in mrc.SCAN_REPORT_NAMES:
            after = (self.repo / mrc.APPROVED_OUTPUT_PATHS[name]).read_bytes()
            self.assertEqual(after, before[name], f"{name} was modified by finalize")

        dest = self.repo / mrc.APPROVED_OUTPUT_PATHS["post_adjudication_stop_status_report"]
        report = json.loads(dest.read_text(encoding="utf-8"))
        entry = report["content"]["domains"]["benchmark_v1_vs_final_test"]
        self.assertTrue(entry["stop_before_inference"])
        self.assertIn(mrc.STOP_REASON_SAME_SOURCE_IMAGE, entry["reasons"])
        self.assertTrue(report["content"]["complete"])
        self.assertEqual(report["content"]["adjudicated_count"], report["content"]["total_candidates"])

    def test_finalize_without_any_confirmed_duplicate_preserves_leakage_only(self):
        # No same_source_image labels at all -- only a metadata-leakage
        # stop (if any) from the scan-time report may survive; no domain
        # may claim a confirmed-duplicate stop.
        self._adjudicate_all(same_source_pair_ids=frozenset())
        result = fin.cmd_finalize(self._args())
        self.assertTrue(result["ok"])
        dest = self.repo / mrc.APPROVED_OUTPUT_PATHS["post_adjudication_stop_status_report"]
        report = json.loads(dest.read_text(encoding="utf-8"))
        for name, entry in report["content"]["domains"].items():
            self.assertNotIn(mrc.STOP_REASON_SAME_SOURCE_IMAGE, entry["reasons"])

    def test_finalize_refuses_to_overwrite_existing_artifact(self):
        self._adjudicate_all(same_source_pair_ids=frozenset())
        fin.cmd_finalize(self._args())
        with self.assertRaises(fin.FinalizationError):
            fin.cmd_finalize(self._args())

    def test_finalize_never_touches_data_directory(self):
        import inspect
        source = inspect.getsource(fin.cmd_finalize) + inspect.getsource(fin.build_post_adjudication_content)
        self.assertNotIn('"data"', source)


class TestFinalizeCheck(_FinalizeFixtureTestCase):
    def setUp(self):
        super().setUp()
        cp_path = self.repo / mrc.APPROVED_OUTPUT_PATHS["candidate_pairs_report"]
        cp_report = json.loads(cp_path.read_text(encoding="utf-8"))
        forced_pairs = cp_report["content"]["domains"]["benchmark_v1_vs_final_test"]
        self.same_source_ids = {forced_pairs[0]["pair_id"]}
        self._adjudicate_all(same_source_pair_ids=self.same_source_ids)
        fin.cmd_finalize(self._args())
        self.dest = self.repo / mrc.APPROVED_OUTPUT_PATHS["post_adjudication_stop_status_report"]

    def test_check_accepts_the_freshly_published_artifact(self):
        result = fin.cmd_check(self._args())
        self.assertTrue(result["ok"])
        self.assertTrue(result["overall_stop_before_inference"])

    def test_check_fails_when_artifact_not_yet_published(self):
        self.dest.unlink()
        with self.assertRaises(fin.FinalizationError):
            fin.cmd_check(self._args())

    def test_check_rejects_mutation_even_after_rehash(self):
        report = json.loads(self.dest.read_text(encoding="utf-8"))
        # Drop the confirmed same_source_image reason while leaving
        # stop_before_inference True -- an internally-inconsistent AND
        # under-reported stop decision.
        entry = report["content"]["domains"]["benchmark_v1_vs_final_test"]
        entry["reasons"] = [r for r in entry["reasons"] if r != mrc.STOP_REASON_SAME_SOURCE_IMAGE]
        report["content_sha256"] = mrc.compute_content_sha256(report["content"])
        self.dest.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        with self.assertRaises(fin.FinalizationError):
            fin.cmd_check(self._args())

    def test_check_rejects_dropped_leakage_reason(self):
        report = json.loads(self.dest.read_text(encoding="utf-8"))
        for name, entry in report["content"]["domains"].items():
            if mrc.STOP_REASON_MATCHING_OBSERVATION_UUID in entry["reasons"]:
                entry["reasons"] = [r for r in entry["reasons"] if r != mrc.STOP_REASON_MATCHING_OBSERVATION_UUID]
                entry["stop_before_inference"] = bool(entry["reasons"])
                report["content_sha256"] = mrc.compute_content_sha256(report["content"])
                self.dest.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                with self.assertRaises(fin.FinalizationError):
                    fin.cmd_check(self._args())
                return
        self.skipTest("fixture produced no metadata-leakage stop to test dropping")

    def test_check_rejects_incomplete_flag(self):
        report = json.loads(self.dest.read_text(encoding="utf-8"))
        report["content"]["complete"] = False
        report["content_sha256"] = mrc.compute_content_sha256(report["content"])
        self.dest.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        with self.assertRaises(fin.FinalizationError):
            fin.cmd_check(self._args())


class TestFinalizeLedgerSemantics(_FinalizeFixtureTestCase):
    """Item 3: finalize_stop_status must semantically re-verify the
    adjudication ledger against the exact candidate map from the fully
    verified report -- not merely trust generic hash-chain verification.
    Every test here injects a defect via append_only_ledger directly
    (bypassing adjudicate_pairs.record_adjudication's own field checks,
    simulating an externally-tampered-but-hash-chain-valid ledger) and
    proves --finalize (and --preflight, where relevant) still reject it."""

    def setUp(self):
        super().setUp()
        cp_path = self.repo / mrc.APPROVED_OUTPUT_PATHS["candidate_pairs_report"]
        self.cp_bytes = cp_path.read_bytes()
        self.cp_report = json.loads(self.cp_bytes.decode("utf-8"))
        self.report_sha256 = mrc.sha256_bytes(self.cp_bytes)
        self.domain_name, pairs = next((n, p) for n, p in self.cp_report["content"]["domains"].items() if p)
        self.pair = pairs[0]

    def _append_raw_record(self, **field_overrides) -> None:
        """Appends directly via append_only_ledger, bypassing
        adjudicate_pairs.record_adjudication's own field validation --
        simulates a ledger record that is hash-chain valid but
        semantically wrong (as if the file were edited/tampered outside
        this tool, or a future bug wrote a bad record)."""
        import append_only_ledger as aol
        fields = adj.build_adjudication_fields(
            {**self.pair, "domain": self.domain_name}, self.report_sha256, self.cp_report["content_sha256"],
            label="different_image", reviewer_id="r1", session_id="s1", reviewed_at_utc="2026-01-01T00:00:00Z")
        fields.update(field_overrides)
        aol.append_record(adj.ledger_path_for(self.repo), fields,
                          required_fields=mrc.ADJUDICATION_RECORD_REQUIRED_FIELDS,
                          identity_fn=mrc.adjudication_record_identity)

    def test_correct_hash_chain_but_wrong_domain_rejected(self):
        self._append_raw_record(domain="within_calibration_v1")
        with self.assertRaises(fin.FinalizationError):
            fin.cmd_finalize(self._args())

    def test_correct_hash_chain_but_changed_identity_rejected(self):
        self._append_raw_record(identity_a="not-the-real-identity")
        with self.assertRaises(fin.FinalizationError):
            fin.cmd_finalize(self._args())

    def test_correct_hash_chain_but_changed_distance_rejected(self):
        self._append_raw_record(dhash_distance=min(self.pair["dhash_distance"] + 1, mrc.DHASH_BITS))
        with self.assertRaises(fin.FinalizationError):
            fin.cmd_finalize(self._args())

    def test_stale_report_bindings_rejected(self):
        self._append_raw_record(candidate_pairs_report_sha256="0" * 64)
        with self.assertRaises(fin.FinalizationError):
            fin.cmd_finalize(self._args())
        # And the content-sha256 binding, independently.
        ledger_path = adj.ledger_path_for(self.repo)
        ledger_path.unlink()
        self._append_raw_record(scan_report_content_sha256="0" * 64)
        with self.assertRaises(fin.FinalizationError):
            fin.cmd_finalize(self._args())

    def test_all_real_candidates_plus_one_invented_record_rejected(self):
        self._adjudicate_all()  # every real candidate, properly
        self._append_raw_record(pair_id="0" * 64, identity_a="ghost-a", identity_b="ghost-b")
        with self.assertRaises(fin.FinalizationError) as ctx:
            fin.cmd_finalize(self._args())
        self.assertIn("unknown/extra", str(ctx.exception))

    def test_missing_candidate_record_rejected(self):
        # No adjudication at all -- the one real candidate is missing.
        with self.assertRaises(fin.FinalizationError):
            fin.cmd_finalize(self._args())
        result = fin.cmd_preflight(self._args())
        self.assertFalse(result["complete"])

    def test_trailing_incomplete_fragment_rejected_even_with_all_real_records_present(self):
        self._adjudicate_all()  # every real candidate, properly, ledger ends cleanly
        ledger_path = adj.ledger_path_for(self.repo)
        with ledger_path.open("a", encoding="utf-8") as fh:
            fh.write('{"pair_id": "aaaa')  # interrupted mid-write, no trailing newline
        with self.assertRaises(fin.FinalizationError) as ctx:
            fin.cmd_finalize(self._args())
        self.assertIn("newline", str(ctx.exception))
        # Even though every REAL record precedes the fragment and
        # read_ledger() would tolerate/drop it as an interrupted write,
        # finalization must still refuse -- no ignored partial tail is
        # acceptable before final evidence publication.
        dest = self.repo / mrc.APPROVED_OUTPUT_PATHS["post_adjudication_stop_status_report"]
        self.assertFalse(dest.exists())


class TestDerivePostAdjudicationDomainsPure(unittest.TestCase):
    """Pure-function tests (no fixture, no filesystem) for the core
    decision logic: item 4 requires preserving metadata-leakage stops and
    activating a confirmed-duplicate stop only in frozen critical
    domains."""

    def _reports(self, *, cp_domains: dict, leakage_reasons_by_domain: dict) -> dict:
        ss_domains = {name: {"stop_before_inference": bool(reasons), "reasons": reasons}
                     for name, reasons in leakage_reasons_by_domain.items()}
        return {
            "candidate_pairs_report": {"content": {"domains": cp_domains}},
            "stop_status_report": {"content": {"domains": ss_domains}},
        }

    def test_metadata_leakage_stop_is_preserved_verbatim(self):
        stop_domain = next(iter(mrc.STOP_BEFORE_INFERENCE_DOMAINS))
        reports = self._reports(cp_domains={stop_domain: []},
                                leakage_reasons_by_domain={stop_domain: [mrc.STOP_REASON_MATCHING_OBSERVATION_UUID]})
        domains = fin.derive_post_adjudication_domains(reports, ledger_records=[])
        self.assertTrue(domains[stop_domain]["stop_before_inference"])
        self.assertIn(mrc.STOP_REASON_MATCHING_OBSERVATION_UUID, domains[stop_domain]["reasons"])

    def test_confirmed_same_source_activates_stop_only_in_critical_domain(self):
        stop_domain = next(iter(mrc.STOP_BEFORE_INFERENCE_DOMAINS))
        pair = {"pair_id": "a" * 64, "identity_a": "x", "identity_b": "y"}
        reports = self._reports(cp_domains={stop_domain: [pair]}, leakage_reasons_by_domain={})
        ledger = [{"pair_id": "a" * 64, "domain": stop_domain, "label": "same_source_image"}]
        domains = fin.derive_post_adjudication_domains(reports, ledger)
        self.assertTrue(domains[stop_domain]["stop_before_inference"])
        self.assertEqual(domains[stop_domain]["reasons"], [mrc.STOP_REASON_SAME_SOURCE_IMAGE])

    def test_confirmed_same_source_in_non_critical_domain_never_stops(self):
        non_critical = next(d["name"] for d in mrc.COMPARISON_DOMAINS
                            if d["name"] not in mrc.STOP_BEFORE_INFERENCE_DOMAINS)
        pair = {"pair_id": "a" * 64, "identity_a": "x", "identity_b": "y"}
        reports = self._reports(cp_domains={non_critical: [pair]}, leakage_reasons_by_domain={})
        ledger = [{"pair_id": "a" * 64, "domain": non_critical, "label": "same_source_image"}]
        domains = fin.derive_post_adjudication_domains(reports, ledger)
        self.assertFalse(domains[non_critical]["stop_before_inference"])
        self.assertEqual(domains[non_critical]["reasons"], [])

    def test_non_same_source_label_never_stops(self):
        stop_domain = next(iter(mrc.STOP_BEFORE_INFERENCE_DOMAINS))
        pair = {"pair_id": "a" * 64, "identity_a": "x", "identity_b": "y"}
        reports = self._reports(cp_domains={stop_domain: [pair]}, leakage_reasons_by_domain={})
        ledger = [{"pair_id": "a" * 64, "domain": stop_domain, "label": "different_image"}]
        domains = fin.derive_post_adjudication_domains(reports, ledger)
        self.assertFalse(domains[stop_domain]["stop_before_inference"])

    def test_both_reasons_can_coexist(self):
        stop_domain = next(iter(mrc.STOP_BEFORE_INFERENCE_DOMAINS))
        pair = {"pair_id": "a" * 64, "identity_a": "x", "identity_b": "y"}
        reports = self._reports(cp_domains={stop_domain: [pair]},
                                leakage_reasons_by_domain={stop_domain: [mrc.STOP_REASON_MATCHING_OBSERVATION_UUID]})
        ledger = [{"pair_id": "a" * 64, "domain": stop_domain, "label": "same_source_image"}]
        domains = fin.derive_post_adjudication_domains(reports, ledger)
        self.assertEqual(set(domains[stop_domain]["reasons"]),
                         {mrc.STOP_REASON_SAME_SOURCE_IMAGE, mrc.STOP_REASON_MATCHING_OBSERVATION_UUID})


if __name__ == "__main__":
    unittest.main(verbosity=2)
