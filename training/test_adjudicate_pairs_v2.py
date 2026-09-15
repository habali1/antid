#!/usr/bin/env python3
"""test_adjudicate_pairs_v2.py — offline/synthetic tests for
adjudicate_pairs_v2.py. Never opens an image; never touches the real
3,972-row queue or the real ledger path (a small synthetic domain
scope, patched onto pair_adjudication_v2_contract.py's module
constants, is used for the full-I/O integration tests; the eligibility
and lock unit tests need no files at all)."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pair_adjudication_v2_contract as v2c  # noqa: E402
import adjudicate_pairs_v2 as adj2  # noqa: E402
import append_only_ledger as aol  # noqa: E402


# ------------------------------------------------------- pure eligibility --
def _row(pair_id, domain, workstream, queue_index):
    return {"pair_id": pair_id, "domain": domain, "workstream": workstream, "queue_index": queue_index,
            "channel_role": v2c.WORKSTREAM_ROLE[workstream], "priority_tier": 2, "both_rule_match": False,
            "identity_a": f"a-{pair_id}", "identity_b": f"b-{pair_id}", "phash_distance": 20, "dhash_distance": 20,
            "session_block": 0, "suggested_session": f"{domain}__block0"}


A, B, C = v2c.WORKSTREAM_FINAL_TEST_INDEPENDENCE, v2c.WORKSTREAM_FINAL_TEST_INTERNAL_REPETITION, v2c.WORKSTREAM_GATE_EVIDENCE_INDEPENDENCE


def _rec(pair_id, label):
    return {"pair_id": pair_id, "label": label}


class TestEligibility(unittest.TestCase):
    def setUp(self):
        self.rows = [_row("a1", "dom_a", A, 0), _row("a2", "dom_a", A, 1),
                    _row("b1", "dom_b", B, 2), _row("b2", "dom_b", B, 3),
                    _row("c1", "dom_c", C, 4), _row("c2", "dom_c", C, 5)]

    def test_first_eligible_row_is_lowest_queue_index_when_nothing_adjudicated(self):
        self.assertEqual(adj2.first_eligible_row(self.rows, [])["pair_id"], "a1")

    def test_skips_already_adjudicated(self):
        records = [_rec("a1", "different_image")]
        self.assertEqual(adj2.first_eligible_row(self.rows, records)["pair_id"], "a2")

    def test_blocking_channel_stops_on_same_source_image(self):
        records = [_rec("a1", "same_source_image")]
        state = adj2.workstream_stop_state(self.rows, records)
        self.assertTrue(state[A]["stopped"])
        self.assertEqual(state[A]["stop_reason"], "same_source_image")
        # a2 (still in workstream A) is never offered again
        eligible = adj2.first_eligible_row(self.rows, records)
        self.assertNotEqual(eligible["pair_id"], "a2")

    def test_blocking_channel_stops_on_uncertain(self):
        records = [_rec("c1", "uncertain")]
        state = adj2.workstream_stop_state(self.rows, records)
        self.assertTrue(state[C]["stopped"])
        self.assertEqual(state[C]["stop_reason"], "uncertain")

    def test_workstreams_stop_independently(self):
        records = [_rec("a1", "same_source_image")]
        state = adj2.workstream_stop_state(self.rows, records)
        self.assertTrue(state[A]["stopped"])
        self.assertFalse(state[B]["stopped"])
        self.assertFalse(state[C]["stopped"])

    def test_diagnostic_workstream_never_stops(self):
        records = [_rec("b1", "same_source_image")]
        state = adj2.workstream_stop_state(self.rows, records)
        self.assertFalse(state[B]["stopped"])
        eligible = adj2.first_eligible_row(self.rows, records)
        # b2 must still be reachable (diagnostic channel keeps going)
        self.assertIn(eligible["pair_id"], ("a1", "b2"))

    def test_early_stop_skipped_pair_ids(self):
        records = [_rec("a1", "same_source_image")]
        skipped = adj2.early_stop_skipped_pair_ids(self.rows, records)
        self.assertEqual(skipped, {"a2"})

    def test_done_when_every_row_terminal(self):
        records = [_rec("a1", "same_source_image"), _rec("b1", "different_image"), _rec("b2", "different_image"),
                  _rec("c1", "different_image"), _rec("c2", "different_image")]
        self.assertIsNone(adj2.first_eligible_row(self.rows, records))


# --------------------------------------------------------------------- lock --
class TestExclusiveLock(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_lock_acquired_and_released(self):
        lock_path = adj2.lock_path_for(self.tmp)
        self.assertFalse(lock_path.exists())
        with adj2.exclusive_lock(self.tmp):
            self.assertTrue(lock_path.exists())
        self.assertFalse(lock_path.exists())

    def test_pre_existing_lock_is_never_silently_removed(self):
        lock_path = adj2.lock_path_for(self.tmp)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text("stale-token-from-a-crashed-process")
        with self.assertRaises(adj2.AdjudicationV2Error):
            with adj2.exclusive_lock(self.tmp):
                pass  # never reached
        # the stale lock is STILL there -- never deleted just because acquisition failed
        self.assertTrue(lock_path.exists())
        self.assertEqual(lock_path.read_text(), "stale-token-from-a-crashed-process")

    def test_nested_acquisition_refused(self):
        with adj2.exclusive_lock(self.tmp):
            with self.assertRaises(adj2.AdjudicationV2Error):
                with adj2.exclusive_lock(self.tmp):
                    pass

    def test_only_own_token_is_released(self):
        """If something external replaces the lock file's content after
        this invocation created it (structurally shouldn't happen, but
        checked as defense in depth), release must refuse to delete it."""
        lock_path = adj2.lock_path_for(self.tmp)
        with self.assertRaises(adj2.AdjudicationV2Error):
            with adj2.exclusive_lock(self.tmp):
                lock_path.write_text("someone-else-overwrote-this")
        self.assertTrue(lock_path.exists())


# -------------------------------------------------------- full-I/O fixture --
def run_git(args, cwd):
    result = subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"git {args} failed: {result.stderr}")
    return result.stdout.strip()


SYN_DOMAINS = {"syn_blocking_a": (A, 2), "syn_diag": (B, 2), "syn_blocking_c": (C, 1)}


class SyntheticRepoTestCase(unittest.TestCase):
    """Builds a tiny, fully self-consistent v2 contract+queue+scan-report
    fixture in a REAL throwaway git repo (generator_git_commit needs a
    real commit, mirroring test_correct_manual_review.py's established
    pattern), with pair_adjudication_v2_contract.py's module constants
    patched to this synthetic 5-row scope for the test's duration."""

    def setUp(self):
        self.repo = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.repo, ignore_errors=True)
        (self.repo / "training").mkdir(parents=True)
        run_git(["init"], self.repo)
        run_git(["config", "user.email", "test@example.com"], self.repo)
        run_git(["config", "user.name", "Test"], self.repo)
        # ALL SIX approved implementation sources must be tracked (not
        # just adjudicate_pairs_v2.py) -- verify_implementation_source_
        # provenance checks every one of them against a real commit.
        for rel_path in v2c.APPROVED_IMPLEMENTATION_SOURCE_PATHS.values():
            dest = self.repo / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(HERE / Path(rel_path).name, dest)

        # --- synthetic candidate-pairs content + other 4 minimal reports ---
        candidate_content = {"domains": {}}
        for domain, (ws, count) in SYN_DOMAINS.items():
            candidate_content["domains"][domain] = [
                {"pair_id": f"{domain}-p{i}", "identity_a": f"{domain}-a{i}", "identity_b": f"{domain}-b{i}",
                 "phash_distance": 20, "dhash_distance": 20}
                for i in range(count)
            ]
        report_contents = {
            "perceptual_hashes_report": {"note": "synthetic"},
            "metadata_leakage_report": {"findings": []},
            "candidate_pairs_report": candidate_content,
            "domain_summary_report": {"domains": list(SYN_DOMAINS)},
            "stop_status_report": {"overall_stop_before_inference": False},
        }
        scan_report_hashes = {}
        for name, rel_path in v2c.SCAN_REPORT_REL_PATHS.items():
            full = {"schema_version": 1, "content": report_contents[name],
                    "content_sha256": v2c.compute_content_sha256(report_contents[name])}
            data = (json.dumps(full, indent=2, sort_keys=True) + "\n").encode("utf-8")
            path = self.repo / rel_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            scan_report_hashes[name] = {"byte_sha256": v2c.sha256_bytes(data), "content_sha256": full["content_sha256"]}

        self.patches = [
            mock.patch.object(v2c, "SCAN_REPORT_HASHES", scan_report_hashes),
            mock.patch.object(v2c, "EXPECTED_DOMAIN_COUNTS", {d: c for d, (ws, c) in SYN_DOMAINS.items()}),
            mock.patch.object(v2c, "DOMAIN_WORKSTREAM", {d: ws for d, (ws, c) in SYN_DOMAINS.items()}),
            mock.patch.object(v2c, "DOMAIN_VISIT_ORDER", tuple(SYN_DOMAINS)),
            mock.patch.object(v2c, "EXPECTED_SCOPED_TOTAL", sum(c for _, c in SYN_DOMAINS.values())),
            mock.patch.object(v2c, "EXPECTED_OUTSIDE_SCOPE_TOTAL", 0),
            mock.patch.object(v2c, "EXPECTED_FULL_SCAN_TOTAL", sum(c for _, c in SYN_DOMAINS.values())),
            mock.patch.object(v2c, "EXPECTED_BOTH_RULE_COUNT_IN_SCOPE", 0),
            mock.patch.object(v2c, "EXPECTED_EXACT_ZERO_COUNT_IN_SCOPE", 0),
        ]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)

        rows = v2c.derive_v2_queue_rows(candidate_content)
        queue_payload = {"schema_version": v2c.SCHEMA_VERSION, "row_count": len(rows),
                         "identity_order_sha256": v2c.compute_v2_queue_identity_order_sha256(rows), "rows": rows}
        queue_bytes = (json.dumps(queue_payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("utf-8")
        queue_path = self.repo / v2c.QUEUE_ARTIFACT_REL_PATH
        queue_path.parent.mkdir(parents=True, exist_ok=True)
        queue_path.write_bytes(queue_bytes)

        queue_artifact = {"path": v2c.QUEUE_ARTIFACT_REL_PATH, "byte_sha256": v2c.sha256_bytes(queue_bytes),
                          "row_count": len(rows), "identity_order_sha256": queue_payload["identity_order_sha256"]}

        outside_scope_ids = v2c.derive_outside_scope_pair_ids(candidate_content)
        assert outside_scope_ids == [], outside_scope_ids  # synthetic scope has no outside-scope domains
        outside_scope_identity_order_sha256 = v2c.compute_outside_scope_identity_order_sha256(outside_scope_ids)

        # The queue and reports must be committed too -- freeze_pair_
        # adjudication_v2_contract.py's real lifecycle commits source,
        # queue, AND contract together isn't required here (the queue is
        # a separate generated artifact this fixture writes directly),
        # but the SOURCE files' provenance must be real, so commit
        # everything now and record that one commit for every source.
        run_git(["add", "-A"], self.repo)
        run_git(["commit", "-m", "synthetic fixture"], self.repo)
        git_head = run_git(["rev-parse", "HEAD"], self.repo)
        implementation_sources = {
            name: {"path": rel_path, "sha256": v2c.canonical_lf_sha256_file(self.repo / rel_path),
                  "generator_git_commit": git_head}
            for name, rel_path in v2c.APPROVED_IMPLEMENTATION_SOURCE_PATHS.items()
        }
        content = v2c.build_contract_content(queue_artifact=queue_artifact, implementation_sources=implementation_sources,
                                             outside_scope_identity_order_sha256=outside_scope_identity_order_sha256)
        content_sha256 = v2c.compute_content_sha256(content)
        contract = {"schema_version": v2c.SCHEMA_VERSION, "status": v2c.CONTRACT_STATUS_FROZEN, "content": content,
                   "content_sha256": content_sha256,
                   "generation": {"note": v2c.CONTRACT_GENERATION_NOTE,
                                  "generator": v2c.CONTRACT_GENERATOR_PATH}}
        problems = v2c.validate_contract_structure(contract)
        assert not problems, problems
        contract_bytes = (json.dumps(contract, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
        contract_path = self.repo / v2c.CONTRACT_REL_PATH
        contract_path.parent.mkdir(parents=True, exist_ok=True)
        contract_path.write_bytes(contract_bytes)

        self.rows = rows

    def _record_next_eligible(self, label, *, session_id=None, reviewed_at_utc="2026-01-01T00:00:00Z"):
        """Records a decision for the CURRENT next eligible row, using its
        own suggested_session unless a (deliberately wrong) override is
        given. Returns (eligible_row, record)."""
        state = adj2.load_verified_v2_state(self.repo)
        records = adj2.load_verified_ledger_v2(self.repo, state)
        eligible = adj2.first_eligible_row(state["queue_rows"], records)
        sid = eligible["suggested_session"] if session_id is None else session_id
        record = adj2.record_adjudication(self.repo, pair_id=eligible["pair_id"], label=label, reviewer_id="r1",
                                          session_id=sid, reviewed_at_utc=reviewed_at_utc)
        return eligible, record


class TestRecordAdjudicationEndToEnd(SyntheticRepoTestCase):
    def test_state_loads_and_eligibility_matches_derivation(self):
        state = adj2.load_verified_v2_state(self.repo)
        self.assertEqual(len(state["queue_rows"]), sum(c for _, c in SYN_DOMAINS.values()))
        eligible = adj2.first_eligible_row(state["queue_rows"], [])
        self.assertEqual(eligible["queue_index"], 0)

    def test_record_appends_and_post_verifies(self):
        eligible, record = self._record_next_eligible("different_image")
        self.assertEqual(record["pair_id"], eligible["pair_id"])
        ledger_path = adj2.ledger_path_for(self.repo)
        self.assertTrue(ledger_path.exists())
        records = aol.read_ledger(ledger_path)
        self.assertEqual(len(records), 1)
        # lock must not be left behind
        self.assertFalse(adj2.lock_path_for(self.repo).exists())
        # generator provenance is bound to the CONTRACT's own recorded
        # entry, not independently computed
        state = adj2.load_verified_v2_state(self.repo)
        adj_entry = state["contract"]["content"]["implementation_sources"]["adjudicate_pairs_v2"]
        self.assertEqual(records[0]["generator_git_commit"], adj_entry["generator_git_commit"])
        self.assertEqual(records[0]["generator_source_sha256"], adj_entry["sha256"])

    def test_recording_wrong_pair_id_is_rejected(self):
        state = adj2.load_verified_v2_state(self.repo)
        eligible = adj2.first_eligible_row(state["queue_rows"], [])
        not_eligible = next(r for r in state["queue_rows"] if r["pair_id"] != eligible["pair_id"])
        with self.assertRaises(adj2.AdjudicationV2Error):
            adj2.record_adjudication(self.repo, pair_id=not_eligible["pair_id"], label="different_image",
                                     reviewer_id="r1", session_id=not_eligible["suggested_session"],
                                     reviewed_at_utc="2026-01-01T00:00:00Z")
        self.assertFalse(adj2.ledger_path_for(self.repo).exists())

    def test_recording_wrong_session_id_is_rejected(self):
        """Direct-CLI-level regression: a caller passing a plausible but
        WRONG session_id (not the eligible row's own suggested_session)
        must be rejected, never silently accepted or auto-corrected."""
        state = adj2.load_verified_v2_state(self.repo)
        eligible = adj2.first_eligible_row(state["queue_rows"], [])
        wrong_session_id = eligible["suggested_session"] + "-WRONG"
        with self.assertRaises(adj2.AdjudicationV2Error) as ctx:
            adj2.record_adjudication(self.repo, pair_id=eligible["pair_id"], label="different_image",
                                     reviewer_id="r1", session_id=wrong_session_id,
                                     reviewed_at_utc="2026-01-01T00:00:00Z")
        self.assertIn("session_id", str(ctx.exception))
        self.assertFalse(adj2.ledger_path_for(self.repo).exists())

    def test_duplicate_record_rejected(self):
        eligible, _ = self._record_next_eligible("different_image")
        with self.assertRaises(adj2.AdjudicationV2Error):
            adj2.record_adjudication(self.repo, pair_id=eligible["pair_id"], label="different_image",
                                     reviewer_id="r1", session_id=eligible["suggested_session"],
                                     reviewed_at_utc="2026-01-01T00:00:01Z")

    def test_blocking_channel_stops_after_same_source_image(self):
        state = adj2.load_verified_v2_state(self.repo)
        a_row = next(r for r in state["queue_rows"] if r["workstream"] == A)
        # record every A-workstream row in order until we can hit a_row
        eligible = adj2.first_eligible_row(state["queue_rows"], [])
        while eligible["pair_id"] != a_row["pair_id"]:
            eligible, _ = self._record_next_eligible("different_image")
        self._record_next_eligible("same_source_image")
        records = adj2.load_verified_ledger_v2(self.repo, state)
        stop_state = adj2.workstream_stop_state(state["queue_rows"], records)
        self.assertTrue(stop_state[A]["stopped"])
        # workstream C (independent) must still be eligible/unaffected
        self.assertFalse(stop_state[C]["stopped"])

    def test_ledger_tamper_detected_on_reload(self):
        state = adj2.load_verified_v2_state(self.repo)
        self._record_next_eligible("different_image")
        ledger_path = adj2.ledger_path_for(self.repo)
        tampered = ledger_path.read_text().replace("different_image", "same_source_image")
        ledger_path.write_text(tampered)
        with self.assertRaises(adj2.AdjudicationV2Error):
            adj2.load_verified_ledger_v2(self.repo, state)

    def test_forged_generator_git_commit_rejected_even_with_recomputed_hash_chain(self):
        """A stored ledger record with a well-formed 40-hex commit that
        is simply NOT the contract-bound one (a forged/incorrect
        provenance value) must be rejected -- even after the record_hash
        chain is recomputed to match the tampered payload, proving the
        rejection comes from the provenance BINDING check, not merely
        from chain verification."""
        state = adj2.load_verified_v2_state(self.repo)
        self._record_next_eligible("different_image")
        ledger_path = adj2.ledger_path_for(self.repo)
        records = aol.read_ledger(ledger_path)
        forged_commit = "1234567890abcdef1234567890abcdef12345678"
        self.assertNotEqual(forged_commit, records[0]["generator_git_commit"])
        records[0]["generator_git_commit"] = forged_commit
        rewritten = aol.canonical_bytes({k: v for k, v in records[0].items() if k != "record_hash"})
        records[0]["record_hash"] = aol.sha256_hex(rewritten)
        ledger_path.write_bytes((aol.canonical_bytes(records[0]) + b"\n"))
        with self.assertRaises(adj2.AdjudicationV2Error) as ctx:
            adj2.load_verified_ledger_v2(self.repo, state)
        self.assertIn("generator_git_commit", str(ctx.exception))

    def test_forged_generator_source_sha256_rejected_even_with_recomputed_hash_chain(self):
        state = adj2.load_verified_v2_state(self.repo)
        self._record_next_eligible("different_image")
        ledger_path = adj2.ledger_path_for(self.repo)
        records = aol.read_ledger(ledger_path)
        forged_sha = "0" * 64
        self.assertNotEqual(forged_sha, records[0]["generator_source_sha256"])
        records[0]["generator_source_sha256"] = forged_sha
        rewritten = aol.canonical_bytes({k: v for k, v in records[0].items() if k != "record_hash"})
        records[0]["record_hash"] = aol.sha256_hex(rewritten)
        ledger_path.write_bytes((aol.canonical_bytes(records[0]) + b"\n"))
        with self.assertRaises(adj2.AdjudicationV2Error) as ctx:
            adj2.load_verified_ledger_v2(self.repo, state)
        self.assertIn("generator_source_sha256", str(ctx.exception))

    def test_stored_record_with_wrong_session_id_rejected(self):
        state = adj2.load_verified_v2_state(self.repo)
        self._record_next_eligible("different_image")
        ledger_path = adj2.ledger_path_for(self.repo)
        records = aol.read_ledger(ledger_path)
        records[0]["session_id"] = records[0]["session_id"] + "-TAMPERED"
        rewritten = aol.canonical_bytes({k: v for k, v in records[0].items() if k != "record_hash"})
        records[0]["record_hash"] = aol.sha256_hex(rewritten)
        ledger_path.write_bytes((aol.canonical_bytes(records[0]) + b"\n"))
        with self.assertRaises(adj2.AdjudicationV2Error) as ctx:
            adj2.load_verified_ledger_v2(self.repo, state)
        self.assertIn("session_id", str(ctx.exception))

    def test_jointly_tampered_queue_and_contract_rejected(self):
        """Rewrites BOTH queue.json and contract.json so every internal
        hash agrees with the other (a naive attacker's jointly-consistent
        tamper), but the queue no longer matches what the verified
        candidate_pairs_report actually derives -- load_verified_v2_state
        must still reject it via rederivation, not merely via the queue's
        own self-consistency."""
        queue_path = self.repo / v2c.QUEUE_ARTIFACT_REL_PATH
        queue = json.loads(queue_path.read_text())
        # swap two rows' pair_id/identities -- an internally-consistent
        # but semantically wrong reordering of who-is-who
        queue["rows"][0]["pair_id"], queue["rows"][1]["pair_id"] = (
            queue["rows"][1]["pair_id"], queue["rows"][0]["pair_id"])
        new_identity_hash = v2c.compute_v2_queue_identity_order_sha256(queue["rows"])
        queue["identity_order_sha256"] = new_identity_hash
        new_queue_bytes = (json.dumps(queue, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("utf-8")
        queue_path.write_bytes(new_queue_bytes)

        contract_path = self.repo / v2c.CONTRACT_REL_PATH
        contract = json.loads(contract_path.read_text())
        contract["content"]["queue_artifact"]["byte_sha256"] = v2c.sha256_bytes(new_queue_bytes)
        contract["content"]["queue_artifact"]["identity_order_sha256"] = new_identity_hash
        contract["content_sha256"] = v2c.compute_content_sha256(contract["content"])
        contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

        with self.assertRaises(adj2.AdjudicationV2Error) as ctx:
            adj2.load_verified_v2_state(self.repo)
        self.assertIn("rederived", str(ctx.exception))

    def test_altered_outside_scope_identity_set_rejected(self):
        """The contract's outside_scope_identity_order_sha256 no longer
        matching what the verified candidate report actually excludes
        must be rejected, even though the field is individually a
        well-formed hex hash."""
        contract_path = self.repo / v2c.CONTRACT_REL_PATH
        contract = json.loads(contract_path.read_text())
        contract["content"]["outside_scope_identity_order_sha256"] = "f" * 64
        contract["content_sha256"] = v2c.compute_content_sha256(contract["content"])
        contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
        with self.assertRaises(adj2.AdjudicationV2Error) as ctx:
            adj2.load_verified_v2_state(self.repo)
        self.assertIn("outside_scope_identity_order_sha256", str(ctx.exception))

    def test_source_absent_from_claimed_commit_rejected(self):
        """A contract recording a real, resolvable commit for a source is
        still invalid if that PATH never actually existed at that commit
        -- simulated via git's well-known empty-tree object, committed as
        a parentless commit (not reachable from HEAD, and containing no
        files at all)."""
        # 4b825dc642cb6eb9a060e54bf8d69288fbee4904 is git's canonical
        # empty-tree object hash -- identical in every git repository.
        empty_commit = run_git(["commit-tree", "4b825dc642cb6eb9a060e54bf8d69288fbee4904", "-m", "empty"], self.repo)
        contract_path = self.repo / v2c.CONTRACT_REL_PATH
        contract = json.loads(contract_path.read_text())
        contract["content"]["implementation_sources"]["adjudicate_pairs_v2"]["generator_git_commit"] = empty_commit
        contract["content_sha256"] = v2c.compute_content_sha256(contract["content"])
        contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
        with self.assertRaises(adj2.AdjudicationV2Error) as ctx:
            adj2.load_verified_v2_state(self.repo)
        # either "not an ancestor" (the empty commit is unreachable from
        # HEAD) or "does not exist at commit" (the path isn't in it) --
        # both are valid rejections of this invalid provenance.
        msg = str(ctx.exception)
        self.assertTrue("ancestor" in msg or "does not exist at commit" in msg, msg)


if __name__ == "__main__":
    unittest.main()
