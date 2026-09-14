#!/usr/bin/env python3
"""test_correct_manual_review.py — offline/synthetic tests for
correct_manual_review.py, the additive, single-target, provenance-bound
correction mechanism for the completed manual-review ledger.

Uses the same tiny synthetic contract+queue+summary fixture as
test_manual_review_tool.py (never the real 600-row queue, never a real
image). Every module constant that freezes the real Phase 5F2 authority
(original-ledger byte hash/record count/session counts, the approved
queue_index/row-identity/judgment/reason, the expected effective counts)
is patched to match THIS fixture's own actual, computed values -- proving
the mechanism's logic, not the real production numbers, which are
exercised separately by the real, read-only --preflight run reported
alongside these tests.

Generator provenance (item 4) needs a REAL git repository: these fixtures
build one (git init + a real commit containing a copy of the actual
training/correct_manual_review.py at the same relative path), mirroring
test_generate_policy_v2.py's established pattern for
generate_inference_policy_v2.py's own immutable-generation-commit checks.
"""
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
REPO = HERE.parent
sys.path.insert(0, str(HERE))

import manual_review_contract as mrc  # noqa: E402
import manual_review_tool as mrt  # noqa: E402
import correct_manual_review as cmr  # noqa: E402
import append_only_ledger as aol  # noqa: E402
from test_manual_review_fixtures import build_tiny_review_fixture  # noqa: E402


def run_git(repo: Path, *args, check: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(["git", *args], cwd=str(repo), capture_output=True, text=True)
    if check and result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed in {repo}: {result.stdout}{result.stderr}")
    return result


def git_commit_all(repo: Path, message: str) -> str:
    run_git(repo, "add", "-A")
    result = run_git(repo, "commit", "-q", "-m", message, check=False)
    if result.returncode != 0 and "nothing to commit" not in (result.stdout + result.stderr):
        raise RuntimeError(f"git commit failed in {repo}: {result.stdout}{result.stderr}")
    return run_git(repo, "rev-parse", "HEAD").stdout.strip()


def _decision_fields(row: dict, contract: dict, **overrides) -> dict:
    fields = {
        "queue_index": row["queue_index"], "dataset": row["dataset"], "split": row["split"],
        "slug": row["slug"], "observation_uuid": row["observation_uuid"], "photo_id": row["photo_id"],
        "sha256": row["sha256"],
        "contract_content_sha256": contract["content_sha256"],
        "queue_byte_sha256": contract["content"]["queue_artifact"]["byte_sha256"],
        "queue_identity_order_sha256": contract["content"]["queue_artifact"]["identity_order_sha256"],
        "review_state": "unusable_no_visible_ant", "label_plausibility": "plausible",
        "duplicate_suspicion": "none", "notes": "", "reviewer_id": "reviewer-1", "session_id": "session-a",
        "reviewed_at_utc": "2026-01-01T00:00:00Z",
    }
    fields.update(overrides)
    return fields


def build_tiny_correction_fixture(testcase: unittest.TestCase, tmp: Path) -> dict:
    """Builds a tiny synthetic review ledger (real fixture, real git repo)
    and patches every correct_manual_review.py APPROVED_* constant to
    match its actual computed values, for the lifetime of `testcase`.
    Returns a dict with repo/verified/rows/target_row/original_record/
    corrected_judgment/generator_commit/generator_source_sha256."""
    repo = tmp
    fx = build_tiny_review_fixture(testcase, repo)
    verified = mrc.load_and_verify_contract(repo)
    rows = verified["queue_rows"]

    # Record every row's original decision, split across two sessions.
    half = len(rows) // 2
    for i, row in enumerate(rows):
        session_id = "session-a" if i < half else "session-b"
        mrt.record_decision(repo, verified, _decision_fields(row, verified["contract"], session_id=session_id))

    ledger_path = repo / mrc.APPROVED_OUTPUT_PATHS["manual_review_ledger"]
    ledger_records = aol.read_ledger(ledger_path)
    target_row = rows[0]
    original_record = next(r for r in ledger_records if r["queue_index"] == target_row["queue_index"])

    corrected_judgment = {
        "review_state": "poor_quality_usable", "label_plausibility": "uncertain",
        "duplicate_suspicion": "none", "notes": "corrected in the fixture",
    }
    reason = "fixture correction reason"

    # Real git repo + a real commit containing THIS repo's actual
    # correct_manual_review.py at the same relative path.
    dest = repo / cmr.GENERATOR_SOURCE_REL_PATH
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(REPO / cmr.GENERATOR_SOURCE_REL_PATH, dest)
    run_git(repo, "init", "-q")
    run_git(repo, "config", "user.email", "fixture@example.com")
    run_git(repo, "config", "user.name", "Fixture")
    generator_commit = git_commit_all(repo, "fixture initial state")
    generator_source_sha256 = mrc.canonical_lf_sha256_file(dest)

    # Compute the effective counts THIS fixture would have after applying
    # the approved correction, from the real recorded ledger.
    corrections = [{"queue_index": target_row["queue_index"],
                    "corrected_review_state": corrected_judgment["review_state"],
                    "corrected_label_plausibility": corrected_judgment["label_plausibility"],
                    "corrected_duplicate_suspicion": corrected_judgment["duplicate_suspicion"],
                    "corrected_notes": corrected_judgment["notes"], "reason": reason}]
    effective = cmr.effective_records(ledger_records, corrections)
    counts = cmr.effective_counts(effective, verified)

    session_counts = {}
    for r in ledger_records:
        session_counts[r["session_id"]] = session_counts.get(r["session_id"], 0) + 1

    patcher = mock.patch.multiple(
        cmr,
        APPROVED_ORIGINAL_LEDGER_BYTE_SHA256=mrc.sha256_file(ledger_path),
        APPROVED_ORIGINAL_LEDGER_RECORD_COUNT=len(ledger_records),
        APPROVED_ORIGINAL_LEDGER_SESSION_COUNTS=session_counts,
        APPROVED_CORRECTION_QUEUE_INDEX=target_row["queue_index"],
        APPROVED_CORRECTION_ROW_IDENTITY={k: target_row[k] for k in
                                          ("dataset", "split", "slug", "observation_uuid", "photo_id", "sha256")},
        APPROVED_CORRECTION_ORIGINAL_RECORD_HASH=original_record["record_hash"],
        APPROVED_CORRECTION_ORIGINAL_JUDGMENT={k: original_record[k] for k in
                                               ("review_state", "label_plausibility", "duplicate_suspicion", "notes")},
        APPROVED_CORRECTION_CORRECTED_JUDGMENT=dict(corrected_judgment),
        APPROVED_CORRECTION_REASON=reason,
        APPROVED_EFFECTIVE_COUNTS=counts,
    )
    patcher.start()
    testcase.addCleanup(patcher.stop)

    return {
        "repo": repo, "verified": verified, "rows": rows, "target_row": target_row,
        "original_record": original_record, "corrected_judgment": corrected_judgment, "reason": reason,
        "ledger_path": ledger_path, "generator_commit": generator_commit,
        "generator_source_sha256": generator_source_sha256, "expected_effective_counts": counts,
    }


class _CorrectionFixtureTestCase(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.fx = build_tiny_correction_fixture(self, Path(self._tmpdir.name))
        self.repo = self.fx["repo"]
        self.original_ledger_bytes_before = self.fx["ledger_path"].read_bytes()

    def tearDown(self):
        self._tmpdir.cleanup()

    def _assert_original_ledger_unchanged(self):
        self.assertEqual(self.fx["ledger_path"].read_bytes(), self.original_ledger_bytes_before,
                         "the original manual-review ledger must never be modified")

    def _args(self, **overrides):
        import argparse
        d = dict(repo=self.repo, reviewer_id="reviewer-1", session_id="correction-session-1",
                 corrected_at_utc="2026-02-01T00:00:00Z")
        d.update(overrides)
        return argparse.Namespace(**d)

    def _write_approved_correction(self, **overrides):
        args = self._args(**overrides)
        return cmr.cmd_write(args)


class TestNeverTouchesOriginalLedger(_CorrectionFixtureTestCase):
    def test_preflight_never_mutates_original_ledger(self):
        cmr.cmd_preflight(self._args())
        self._assert_original_ledger_unchanged()

    def test_write_never_mutates_original_ledger(self):
        self._write_approved_correction()
        self._assert_original_ledger_unchanged()

    def test_check_never_mutates_original_ledger(self):
        self._write_approved_correction()
        cmr.cmd_check(self._args())
        self._assert_original_ledger_unchanged()

    def test_source_never_uses_ordinary_append_record_for_publication(self):
        import inspect
        source = inspect.getsource(cmr)
        # The real write path must never call append_only_ledger.
        # append_record (plain append mode, not exclusive against a
        # concurrent writer) -- only the local exclusive stage-then-hard-
        # link publisher, and never against the original review ledger.
        self.assertNotIn("aol.append_record(", source)
        self.assertIn("os.link(str(tmp), str(dest))", source)
        self.assertIn("_publish_single_correction_record", source)


class TestFrozenOriginalLedgerAuthority(_CorrectionFixtureTestCase):
    """Item 1: load_verified_original() must reject any original ledger
    that doesn't match the frozen byte hash/record count/session counts,
    even before the first correction exists."""

    def test_preflight_succeeds_against_the_approved_original_ledger(self):
        result = cmr.cmd_preflight(self._args())
        self.assertTrue(result["ok"])
        self.assertTrue(result["approved_correction_pending"])

    def test_semantically_valid_but_byte_different_ledger_rejected(self):
        # Mutate the first record's notes AND correctly recompute its
        # record_hash (and every following record's prev_record_hash/
        # record_hash, since the chain is positional) -- fully chain-valid
        # (mrt.load_verified_ledger would accept it on its own terms), but
        # BYTE-different from the frozen approved hash. The byte-hash gate
        # must still reject it, before any correction is even considered.
        records = aol.read_ledger(self.fx["ledger_path"])
        records[0] = dict(records[0])
        records[0]["notes"] = records[0]["notes"] + " a genuine but unapproved edit"
        prev_hash = aol.GENESIS_HASH
        rewritten = []
        for rec in records:
            payload = {k: v for k, v in rec.items() if k not in ("prev_record_hash", "record_hash")}
            payload["prev_record_hash"] = prev_hash
            record_hash = aol.sha256_hex(aol.canonical_bytes(payload))
            payload["record_hash"] = record_hash
            rewritten.append(payload)
            prev_hash = record_hash
        self.fx["ledger_path"].write_text(
            "\n".join(json.dumps(r, sort_keys=True, separators=(",", ":")) for r in rewritten) + "\n",
            encoding="utf-8")
        # Confirm this rewritten ledger really is chain-valid on its own.
        self.assertEqual(aol.verify_chain(aol.read_ledger(self.fx["ledger_path"]),
                                          mrc.REVIEW_RECORD_REQUIRED_FIELDS, mrc.review_record_identity), [])
        with self.assertRaises(cmr.CorrectionError) as ctx:
            cmr.cmd_preflight(self._args())
        self.assertIn("does not match the frozen approved authority", str(ctx.exception))

    def test_rechained_ledger_rejected_even_if_internally_consistent(self):
        # Rewrite the WHOLE ledger from scratch, same records but in
        # REVERSED order -- a fresh, fully self-consistent chain (every
        # prev_record_hash/record_hash recomputed correctly), same set of
        # records, but different bytes and a different chain structure
        # than the approved original. Must still be rejected.
        records = list(reversed(aol.read_ledger(self.fx["ledger_path"])))
        self.fx["ledger_path"].unlink()
        for rec in records:
            payload = {k: v for k, v in rec.items() if k not in ("prev_record_hash", "record_hash")}
            aol.append_record(self.fx["ledger_path"], payload,
                              required_fields=mrc.REVIEW_RECORD_REQUIRED_FIELDS,
                              identity_fn=mrc.review_record_identity)
        self.assertEqual(aol.verify_chain(aol.read_ledger(self.fx["ledger_path"]),
                                          mrc.REVIEW_RECORD_REQUIRED_FIELDS, mrc.review_record_identity), [])
        with self.assertRaises(cmr.CorrectionError):
            cmr.cmd_preflight(self._args())

    def test_wrong_record_count_rejected(self):
        with mock.patch.object(cmr, "APPROVED_ORIGINAL_LEDGER_RECORD_COUNT", 999999):
            with self.assertRaises(cmr.CorrectionError):
                cmr.cmd_preflight(self._args())

    def test_wrong_session_counts_rejected(self):
        with mock.patch.object(cmr, "APPROVED_ORIGINAL_LEDGER_SESSION_COUNTS", {"bogus-session": 600}):
            with self.assertRaises(cmr.CorrectionError):
                cmr.cmd_preflight(self._args())


class TestSingleApprovedTargetOnly(_CorrectionFixtureTestCase):
    """Item 2: no CLI or Python-API path may target a different
    queue_index or judgment combination."""

    def test_write_cli_has_no_target_or_judgment_flags(self):
        import inspect
        source = inspect.getsource(cmr.main)
        for forbidden in ("--queue-index", "--corrected-review-state", "--corrected-label-plausibility",
                         "--corrected-duplicate-suspicion", "--corrected-notes", "--reason"):
            self.assertNotIn(forbidden, source)

    def test_write_always_targets_the_approved_queue_index(self):
        result = self._write_approved_correction()
        self.assertEqual(result["queue_index"], cmr.APPROVED_CORRECTION_QUEUE_INDEX)

    def test_direct_helper_rejects_a_different_queue_index(self):
        state = cmr.load_verified_original(self.repo)
        other_row = self.fx["rows"][1]
        self.assertNotEqual(other_row["queue_index"], cmr.APPROVED_CORRECTION_QUEUE_INDEX)
        fields = cmr.build_approved_correction_fields(
            state["original_ledger_byte_sha256"], state["verified"]["contract"],
            self.fx["generator_commit"], self.fx["generator_source_sha256"],
            reviewer_id="r1", session_id="s1", corrected_at_utc="2026-02-01T00:00:00Z")
        fields["queue_index"] = other_row["queue_index"]  # tamper: target a different row
        with self.assertRaises(cmr.CorrectionError):
            cmr.record_correction(self.repo, fields)

    def test_direct_helper_rejects_a_different_valid_judgment_combination(self):
        state = cmr.load_verified_original(self.repo)
        fields = cmr.build_approved_correction_fields(
            state["original_ledger_byte_sha256"], state["verified"]["contract"],
            self.fx["generator_commit"], self.fx["generator_source_sha256"],
            reviewer_id="r1", session_id="s1", corrected_at_utc="2026-02-01T00:00:00Z")
        fields["corrected_review_state"] = "usable"  # a perfectly VALID enum value, just not approved
        with self.assertRaises(cmr.CorrectionError):
            cmr.record_correction(self.repo, fields)

    def test_build_approved_correction_fields_has_no_target_parameters(self):
        import inspect
        sig = inspect.signature(cmr.build_approved_correction_fields)
        for forbidden in ("queue_index", "corrected_review_state", "corrected_label_plausibility",
                         "corrected_duplicate_suspicion", "corrected_notes", "reason"):
            self.assertNotIn(forbidden, sig.parameters)


class TestLifecycle(_CorrectionFixtureTestCase):
    """Item 3: preflight/write/check are distinct."""

    def test_check_rejects_absent_correction_ledger(self):
        with self.assertRaises(cmr.CorrectionError):
            cmr.cmd_check(self._args())

    def test_preflight_reports_pending_when_absent(self):
        result = cmr.cmd_preflight(self._args())
        self.assertTrue(result["approved_correction_pending"])
        self.assertEqual(result["total_corrections"], 0)

    def test_write_then_check_succeeds(self):
        self._write_approved_correction()
        result = cmr.cmd_check(self._args())
        self.assertTrue(result["ok"])
        self.assertFalse(result["approved_correction_pending"])
        self.assertEqual(result["effective_counts"], self.fx["expected_effective_counts"])

    def test_write_refuses_if_artifact_already_exists(self):
        self._write_approved_correction()
        with self.assertRaises(cmr.CorrectionError):
            self._write_approved_correction()
        records = aol.read_ledger(cmr.corrections_ledger_path_for(self.repo))
        self.assertEqual(len(records), 1)

    def test_check_rejects_extra_correction_even_after_rehash(self):
        self._write_approved_correction()
        records = aol.read_ledger(cmr.corrections_ledger_path_for(self.repo))
        extra = dict(records[0])
        extra["queue_index"] = self.fx["rows"][1]["queue_index"]  # a second, different-target record
        cmr.corrections_ledger_path_for(self.repo).unlink()
        for rec in records + [extra]:
            payload = {k: v for k, v in rec.items() if k not in ("prev_record_hash", "record_hash")}
            try:
                aol.append_record(cmr.corrections_ledger_path_for(self.repo), payload,
                                  required_fields=cmr.CORRECTION_RECORD_REQUIRED_FIELDS,
                                  identity_fn=cmr.correction_record_identity)
            except aol.LedgerError:
                pass  # the second record is expected to be structurally odd; still on disk either way
        with self.assertRaises(cmr.CorrectionError):
            cmr.cmd_check(self._args())

    def test_check_rejects_substituted_correction_even_after_rehash(self):
        self._write_approved_correction()
        path = cmr.corrections_ledger_path_for(self.repo)
        record = json.loads(path.read_text(encoding="utf-8").strip())
        record["corrected_notes"] = "substituted"
        payload = {k: v for k, v in record.items() if k != "record_hash"}
        record["record_hash"] = aol.sha256_hex(aol.canonical_bytes(payload))
        path.write_text(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
        with self.assertRaises(cmr.CorrectionError):
            cmr.cmd_check(self._args())


class TestGeneratorProvenance(_CorrectionFixtureTestCase):
    """Item 4: each correction record binds the generator's git commit and
    source hash; --write requires the source committed and the tree
    clean; --check validates the recorded commit as a real ancestor and
    re-verifies the source hash both at that commit and currently."""

    def test_write_requires_committed_source(self):
        # Modify the generator source copy in the fixture repo WITHOUT
        # committing -- --write must refuse.
        gen_path = self.repo / cmr.GENERATOR_SOURCE_REL_PATH
        gen_path.write_text(gen_path.read_text(encoding="utf-8") + "\n# uncommitted local edit\n",
                            encoding="utf-8")
        with self.assertRaises(cmr.CorrectionError):
            self._write_approved_correction()

    def test_write_requires_clean_tracked_tree(self):
        # Dirty an unrelated TRACKED file.
        contract_path = self.repo / mrc.CONTRACT_REL_PATH
        contract_path.write_text(contract_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        with self.assertRaises(cmr.CorrectionError):
            self._write_approved_correction()

    def test_written_record_binds_generator_commit_and_source_hash(self):
        self._write_approved_correction()
        records = aol.read_ledger(cmr.corrections_ledger_path_for(self.repo))
        rec = records[0]
        self.assertEqual(rec["generator_git_commit"], self.fx["generator_commit"])
        self.assertEqual(rec["generator_source_sha256"], self.fx["generator_source_sha256"])
        self.assertEqual(rec["generator_source_path"], cmr.GENERATOR_SOURCE_REL_PATH)
        self.assertEqual(rec["generator_protocol_version"], cmr.GENERATOR_PROTOCOL_VERSION)

    def test_unrelated_descendant_commit_remains_valid(self):
        self._write_approved_correction()
        # A later, unrelated commit (touching a different file) must NOT
        # invalidate the existing correction's evidence.
        unrelated = self.repo / "unrelated_file.txt"
        unrelated.write_text("unrelated change\n", encoding="utf-8")
        git_commit_all(self.repo, "unrelated later commit")
        result = cmr.cmd_check(self._args())
        self.assertTrue(result["ok"])

    def test_generator_source_mutation_after_commit_invalidates_evidence(self):
        self._write_approved_correction()
        # The generator source itself changes in a LATER commit -- the
        # recorded evidence must become stale (working tree no longer
        # matches the recorded hash).
        gen_path = self.repo / cmr.GENERATOR_SOURCE_REL_PATH
        gen_path.write_text(gen_path.read_text(encoding="utf-8") + "\n# a later, real edit\n",
                            encoding="utf-8")
        git_commit_all(self.repo, "edit the generator source later")
        with self.assertRaises(cmr.CorrectionError):
            cmr.cmd_check(self._args())

    def test_mutate_and_rehash_generator_commit_rejected(self):
        self._write_approved_correction()
        path = cmr.corrections_ledger_path_for(self.repo)
        record = json.loads(path.read_text(encoding="utf-8").strip())
        record["generator_git_commit"] = "0" * 40
        payload = {k: v for k, v in record.items() if k != "record_hash"}
        record["record_hash"] = aol.sha256_hex(aol.canonical_bytes(payload))
        path.write_text(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
        with self.assertRaises(cmr.CorrectionError):
            cmr.cmd_check(self._args())

    def test_mutate_and_rehash_generator_source_sha256_rejected(self):
        self._write_approved_correction()
        path = cmr.corrections_ledger_path_for(self.repo)
        record = json.loads(path.read_text(encoding="utf-8").strip())
        record["generator_source_sha256"] = "f" * 64
        payload = {k: v for k, v in record.items() if k != "record_hash"}
        record["record_hash"] = aol.sha256_hex(aol.canonical_bytes(payload))
        path.write_text(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
        with self.assertRaises(cmr.CorrectionError):
            cmr.cmd_check(self._args())


class TestExactEffectiveCounts(_CorrectionFixtureTestCase):
    """Item 5: effective counts must match the frozen expected counts
    exactly once the approved correction is applied."""

    def test_effective_counts_match_expected_after_write(self):
        self._write_approved_correction()
        result = cmr.cmd_preflight(self._args())
        self.assertEqual(result["effective_counts"], self.fx["expected_effective_counts"])

    def test_wrong_expected_counts_constant_rejected(self):
        self._write_approved_correction()
        bogus = {**self.fx["expected_effective_counts"], "final_test_unusable_total": 999}
        with mock.patch.object(cmr, "APPROVED_EFFECTIVE_COUNTS", bogus):
            with self.assertRaises(cmr.CorrectionError):
                cmr.cmd_preflight(self._args())

    def test_effective_records_row_count_unchanged(self):
        self._write_approved_correction()
        state = cmr.load_verified_original(self.repo)
        corrections = cmr.load_verified_corrections(self.repo, state)
        effective = cmr.effective_records(state["original_records"], corrections)
        self.assertEqual(len(effective), len(self.fx["rows"]))


class TestMutateAndRehashRejection(_CorrectionFixtureTestCase):
    def _tamper_and_rehash(self, **field_overrides) -> bytes:
        self._write_approved_correction()
        path = cmr.corrections_ledger_path_for(self.repo)
        record = json.loads(path.read_text(encoding="utf-8").strip())
        record.update(field_overrides)
        payload = {k: v for k, v in record.items() if k != "record_hash"}
        record["record_hash"] = aol.sha256_hex(aol.canonical_bytes(payload))
        path.write_text(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
        return path.read_bytes()

    def test_mutate_and_rehash_row_identity_rejected(self):
        tampered_bytes = self._tamper_and_rehash(sha256="f" * 64)
        with self.assertRaises(cmr.CorrectionError):
            cmr.cmd_preflight(self._args())
        with self.assertRaises(cmr.CorrectionError):
            cmr.cmd_check(self._args())
        self.assertEqual(cmr.corrections_ledger_path_for(self.repo).read_bytes(), tampered_bytes)
        self._assert_original_ledger_unchanged()

    def test_mutate_and_rehash_reason_rejected(self):
        self._tamper_and_rehash(reason="a different, unapproved reason")
        with self.assertRaises(cmr.CorrectionError):
            cmr.cmd_preflight(self._args())

    def test_mutate_and_rehash_contract_binding_rejected(self):
        self._tamper_and_rehash(contract_content_sha256="0" * 64)
        with self.assertRaises(cmr.CorrectionError):
            cmr.cmd_preflight(self._args())


class TestExclusiveAtomicPublication(_CorrectionFixtureTestCase):
    """Item 1-6 of the publication-race correction: --write must publish
    via a structurally exclusive, atomic stage-then-hard-link pattern,
    never append_only_ledger.append_record's plain-append TOCTOU-prone
    path."""

    def _dest(self) -> Path:
        return cmr.corrections_ledger_path_for(self.repo)

    def _leftover_temp_files(self) -> list:
        dest = self._dest()
        return list(dest.parent.glob(f"{dest.name}.tmp*"))

    def test_successful_publication_is_one_valid_record_no_temp_left(self):
        self._write_approved_correction()
        dest = self._dest()
        raw = dest.read_bytes()
        self.assertTrue(raw.endswith(b"\n"))
        self.assertEqual(raw.count(b"\n"), 1)
        records = aol.read_ledger(dest)
        self.assertEqual(len(records), 1)
        self.assertEqual(aol.verify_chain(records, cmr.CORRECTION_RECORD_REQUIRED_FIELDS,
                                          cmr.correction_record_identity), [])
        self.assertEqual(self._leftover_temp_files(), [])
        self._assert_original_ledger_unchanged()

    def test_preexisting_sentinel_destination_never_changed(self):
        dest = self._dest()
        dest.parent.mkdir(parents=True, exist_ok=True)
        sentinel = b'{"not": "a real correction record"}\n'
        dest.write_bytes(sentinel)
        with self.assertRaises(cmr.CorrectionError):
            self._write_approved_correction()
        self.assertEqual(dest.read_bytes(), sentinel)
        self.assertEqual(self._leftover_temp_files(), [])
        self._assert_original_ledger_unchanged()

    def test_race_destination_created_between_validation_and_publish_loses_cleanly(self):
        dest = self._dest()
        sentinel = b'{"winner": "a concurrent writer published first"}\n'
        real_link = cmr.os.link

        def racing_link(src, dst):
            # Simulate a concurrent writer publishing between our own
            # dest.exists() check and this os.link call.
            Path(dst).write_bytes(sentinel)
            return real_link(src, dst)

        with mock.patch.object(cmr.os, "link", side_effect=racing_link):
            with self.assertRaises(cmr.CorrectionError):
                self._write_approved_correction()
        self.assertEqual(dest.read_bytes(), sentinel)
        self.assertEqual(self._leftover_temp_files(), [])
        self._assert_original_ledger_unchanged()

    def test_two_concurrent_writers_cannot_both_succeed(self):
        import threading
        results, errors = [], []
        barrier = threading.Barrier(2)
        real_link = cmr.os.link

        def synced_link(src, dst):
            barrier.wait(timeout=5)
            return real_link(src, dst)

        def worker():
            try:
                results.append(self._write_approved_correction())
            except cmr.CorrectionError as exc:
                errors.append(exc)

        with mock.patch.object(cmr.os, "link", side_effect=synced_link):
            t1 = threading.Thread(target=worker)
            t2 = threading.Thread(target=worker)
            t1.start()
            t2.start()
            t1.join()
            t2.join()

        self.assertEqual(len(results), 1, f"exactly one writer must succeed, got {len(results)}")
        self.assertEqual(len(errors), 1, f"exactly one writer must fail, got {len(errors)}")
        dest = self._dest()
        records = aol.read_ledger(dest)
        self.assertEqual(len(records), 1)
        self.assertEqual(self._leftover_temp_files(), [])
        self._assert_original_ledger_unchanged()

    def test_publication_failure_during_prevalidation_leaves_nothing_behind(self):
        # aol.verify_chain is shared with manual_review_tool.py (same
        # module object) -- a blanket mock would also break the ORIGINAL
        # ledger's own chain check earlier in the flow, so only fail it
        # for the corrections-ledger's own required-field set.
        dest = self._dest()
        real_verify_chain = cmr.aol.verify_chain

        def selective_verify_chain(records, required_fields, identity_fn):
            if required_fields == cmr.CORRECTION_RECORD_REQUIRED_FIELDS:
                return ["simulated prevalidation failure"]
            return real_verify_chain(records, required_fields, identity_fn)

        with mock.patch.object(cmr.aol, "verify_chain", side_effect=selective_verify_chain):
            with self.assertRaises(cmr.CorrectionError):
                self._write_approved_correction()
        self.assertFalse(dest.exists())
        self.assertEqual(self._leftover_temp_files(), [])
        self._assert_original_ledger_unchanged()

    def test_publication_failure_on_unexpected_link_error_leaves_nothing_behind(self):
        dest = self._dest()
        with mock.patch.object(cmr.os, "link", side_effect=OSError("simulated disk error")):
            with self.assertRaises(OSError):
                self._write_approved_correction()
        self.assertFalse(dest.exists())
        self.assertEqual(self._leftover_temp_files(), [])
        self._assert_original_ledger_unchanged()

    def test_fsync_failure_after_temp_file_created_leaves_no_destination_and_no_temp(self):
        # The temp file IS created and written to (open + write succeed);
        # only the fsync call fails -- the whole temp-file lifecycle,
        # including open/write/flush/fsync, must be inside cleanup
        # protection, not merely the prevalidation/link steps after it.
        dest = self._dest()
        with mock.patch.object(cmr.os, "fsync", side_effect=OSError("simulated fsync failure")):
            with self.assertRaises(OSError):
                self._write_approved_correction()
        self.assertFalse(dest.exists())
        self.assertEqual(self._leftover_temp_files(), [])
        self._assert_original_ledger_unchanged()

    def test_helper_never_reads_or_writes_the_original_ledger(self):
        import inspect
        source = inspect.getsource(cmr._publish_single_correction_record)
        self.assertNotIn("manual_review_ledger", source)
        self.assertNotIn("mrt.", source)


class TestPostPublicationVerification(_CorrectionFixtureTestCase):
    """Item 2/3 of the final correction: record_correction must reload the
    PUBLISHED artifact from disk through the full real verification path
    before reporting success, and fail closed (without deleting or
    rewriting it) if that re-verification ever disagrees."""

    def _dest(self) -> Path:
        return cmr.corrections_ledger_path_for(self.repo)

    def test_post_publication_verification_is_actually_executed(self):
        # load_verified_corrections must be called MORE than once during a
        # successful --write: once to check existing corrections before
        # publishing, and again afterward to reload and re-verify the
        # PUBLISHED artifact from disk.
        real_load_verified_corrections = cmr.load_verified_corrections
        calls = []

        def counting(repo, state):
            calls.append(state)
            return real_load_verified_corrections(repo, state)

        with mock.patch.object(cmr, "load_verified_corrections", side_effect=counting):
            self._write_approved_correction()
        self.assertGreaterEqual(len(calls), 2, "post-publication reload/re-verification must run")
        # The post-publication call must be against a FRESH state object
        # (reloaded from disk after publish), not the same pre-publish
        # state instance reused.
        self.assertIsNot(calls[0], calls[-1])

    def test_forced_post_publication_count_failure_prevents_success_but_keeps_artifact(self):
        # validate_effective_counts is only ever called (in the --write
        # path) as part of the POST-publication re-verification -- forcing
        # it to report a mismatch proves that failure blocks success.
        with mock.patch.object(cmr, "validate_effective_counts",
                               return_value=["simulated post-publication count mismatch"]):
            with self.assertRaises(cmr.CorrectionError) as ctx:
                self._write_approved_correction()
        self.assertIn("post-publication verification failed", str(ctx.exception))
        self.assertIn("NOT deleted or rewritten", str(ctx.exception))
        # The artifact was published and must be left exactly as it is --
        # not deleted, not rewritten -- for manual review.
        dest = self._dest()
        self.assertTrue(dest.exists())
        records = aol.read_ledger(dest)
        self.assertEqual(len(records), 1)
        self._assert_original_ledger_unchanged()

    def test_forced_post_publication_reload_failure_prevents_success_but_keeps_artifact(self):
        # Simulate the reloaded artifact failing full re-verification
        # (e.g. an unknown/extra correction somehow present) -- the
        # published file must still not be touched.
        real_load_verified_corrections = cmr.load_verified_corrections
        call_count = {"n": 0}

        def flaky(repo, state):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return real_load_verified_corrections(repo, state)
            raise cmr.CorrectionError("simulated post-publication reload failure")

        with mock.patch.object(cmr, "load_verified_corrections", side_effect=flaky):
            with self.assertRaises(cmr.CorrectionError) as ctx:
                self._write_approved_correction()
        self.assertIn("post-publication verification failed", str(ctx.exception))
        dest = self._dest()
        self.assertTrue(dest.exists())
        self._assert_original_ledger_unchanged()

    def test_forced_post_publication_byte_mismatch_prevents_success(self):
        # Corrupt the artifact AFTER a real, self-consistent publish
        # succeeds (simulating bytes changing between publish and the
        # post-publication re-read) -- read_ledger's own interrupted-
        # write tolerance silently drops the trailing garbage (no
        # newline), so chain/semantic re-verification alone would NOT
        # catch this; only the explicit final byte-identity check does.
        real_publish = cmr._publish_single_correction_record

        def publish_then_corrupt(dest, fields):
            result = real_publish(dest, fields)
            with open(dest, "ab") as fh:
                fh.write(b"corruption-with-no-trailing-newline")
            return result

        with mock.patch.object(cmr, "_publish_single_correction_record", side_effect=publish_then_corrupt):
            with self.assertRaises(cmr.CorrectionError) as ctx:
                self._write_approved_correction()
        self.assertIn("post-publication verification failed", str(ctx.exception))
        self.assertIn("NOT deleted or rewritten", str(ctx.exception))
        # The (corrupted) artifact is left exactly as it is -- not
        # deleted, not rewritten.
        self.assertTrue(self._dest().exists())
        self._assert_original_ledger_unchanged()


if __name__ == "__main__":
    unittest.main(verbosity=2)
