#!/usr/bin/env python3
"""test_freeze_pair_adjudication_v2_contract.py — offline/synthetic
tests for freeze_pair_adjudication_v2_contract.py's two-commit lifecycle:
a source-preparation commit (all six approved implementation sources
tracked, tree clean) must exist BEFORE --write will record any
provenance, and --write refuses outright if a source is merely present
in the working tree but not yet committed. --check re-verifies that
recorded provenance is still currently valid (ancestor commit + source
exists there with the recorded hash + the working tree still matches),
not merely that content_sha256 self-consistency holds.

Uses a small synthetic domain scope (patched onto pair_adjudication_v2_
contract.py's module constants) in a REAL throwaway git repo -- never
the real 3,972-row queue, never a real image."""
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
import freeze_pair_adjudication_v2_contract as fz  # noqa: E402

A = v2c.WORKSTREAM_FINAL_TEST_INDEPENDENCE
B = v2c.WORKSTREAM_FINAL_TEST_INTERNAL_REPETITION
C = v2c.WORKSTREAM_GATE_EVIDENCE_INDEPENDENCE
SYN_DOMAINS = {"syn_blocking_a": (A, 2), "syn_diag": (B, 2), "syn_blocking_c": (C, 1)}


def run_git(args, cwd):
    result = subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"git {args} failed: {result.stderr}")
    return result.stdout.strip()


class FreezeFixtureTestCase(unittest.TestCase):
    """Builds a throwaway repo with the five committed scan reports
    (synthetic, small) and a real committed v2 queue, but leaves the SIX
    approved implementation sources for each test to place/commit (or
    not) as that test requires -- this is what lets tests exercise both
    the "not yet committed" refusal and the real success path."""

    def setUp(self):
        self.repo = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.repo, ignore_errors=True)
        (self.repo / "training").mkdir(parents=True)
        run_git(["init"], self.repo)
        run_git(["config", "user.email", "test@example.com"], self.repo)
        run_git(["config", "user.name", "Test"], self.repo)

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
            mock.patch.object(fz, "REPO", self.repo),
            mock.patch.object(fz, "CONTRACT_PATH", self.repo / v2c.CONTRACT_REL_PATH),
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

    def _copy_all_sources(self):
        for rel_path in v2c.APPROVED_IMPLEMENTATION_SOURCE_PATHS.values():
            dest = self.repo / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(HERE / Path(rel_path).name, dest)

    def _commit_all(self, message="commit"):
        run_git(["add", "-A"], self.repo)
        run_git(["commit", "-m", message], self.repo)


class TestWriteRefusesUncommittedSources(FreezeFixtureTestCase):
    def test_write_refuses_when_sources_not_committed_at_all(self):
        # nothing committed yet -- not even an initial commit exists
        with self.assertRaises(fz.FreezeError):
            fz.cmd_write()
        self.assertFalse((self.repo / v2c.CONTRACT_REL_PATH).exists())

    def test_write_refuses_when_a_source_is_present_but_uncommitted(self):
        self._copy_all_sources()
        # commit everything EXCEPT leave one source file modified afterward
        self._commit_all("initial")
        (self.repo / "training" / "finalize_stop_status_v2.py").write_text(
            (self.repo / "training" / "finalize_stop_status_v2.py").read_text() + "\n# uncommitted edit\n")
        with self.assertRaises(fz.FreezeError) as ctx:
            fz.cmd_write()
        self.assertIn("not clean", str(ctx.exception))
        self.assertFalse((self.repo / v2c.CONTRACT_REL_PATH).exists())

    def test_write_refuses_when_a_source_file_is_missing_entirely(self):
        # copy only 5 of 6 sources, commit, then try to write
        for name, rel_path in v2c.APPROVED_IMPLEMENTATION_SOURCE_PATHS.items():
            if name == "pair_adjudication_v2_review":
                continue
            dest = self.repo / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(HERE / Path(rel_path).name, dest)
        self._commit_all("partial sources")
        with self.assertRaises(fz.FreezeError) as ctx:
            fz.cmd_write()
        self.assertIn("not tracked at HEAD", str(ctx.exception))


class TestWriteThenCheckLifecycle(FreezeFixtureTestCase):
    def setUp(self):
        super().setUp()
        self._copy_all_sources()
        self._commit_all("source preparation")

    def test_write_succeeds_after_real_source_commit(self):
        result = fz.cmd_write()
        self.assertEqual(result, 0)
        self.assertTrue((self.repo / v2c.CONTRACT_REL_PATH).exists())

    def test_write_records_the_real_commit_for_every_source(self):
        fz.cmd_write()
        contract = json.loads((self.repo / v2c.CONTRACT_REL_PATH).read_text())
        head = run_git(["rev-parse", "HEAD"], self.repo)
        for name in v2c.APPROVED_IMPLEMENTATION_SOURCE_PATHS:
            entry = contract["content"]["implementation_sources"][name]
            self.assertEqual(entry["generator_git_commit"], head)

    def test_check_passes_immediately_after_write(self):
        fz.cmd_write()
        self.assertEqual(fz.cmd_check(), 0)

    def test_write_refuses_to_overwrite_existing_contract(self):
        fz.cmd_write()
        with self.assertRaises(fz.FreezeError):
            fz.cmd_write()

    def test_check_fails_after_a_bound_source_changes_uncommitted(self):
        """The recorded generation commit itself stays immutable, but a
        change to the bound source SINCE that commit (even uncommitted)
        must fail --check -- this is the "bound-source change fails"
        requirement, distinct from Gate v2's diagnostic-only provenance."""
        fz.cmd_write()
        self.assertEqual(fz.cmd_check(), 0)
        source_path = self.repo / "training" / "adjudicate_pairs_v2.py"
        source_path.write_text(source_path.read_text() + "\n# a later edit\n")
        self.assertEqual(fz.cmd_check(), 1)

    def test_check_still_passes_after_an_unrelated_later_commit(self):
        """An unrelated later commit (touching neither the contract's
        bound sources nor the queue) must NOT invalidate an already-
        written contract -- the recorded generation commit only needs to
        be an ancestor of current HEAD, never equal to it."""
        fz.cmd_write()
        (self.repo / "training" / "unrelated_file.txt").write_text("unrelated\n")
        self._commit_all("unrelated later commit")
        self.assertEqual(fz.cmd_check(), 0)

    def test_contract_generation_rejects_jointly_rehashed_noncanonical_queue(self):
        queue_path = self.repo / v2c.QUEUE_ARTIFACT_REL_PATH
        queue = json.loads(queue_path.read_text(encoding="utf-8"))
        queue["rows"][0]["pair_id"] = "substituted-but-self-consistent"
        queue["identity_order_sha256"] = v2c.compute_v2_queue_identity_order_sha256(queue["rows"])
        queue_path.write_text(json.dumps(queue, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
                              encoding="utf-8")
        reports = fz._verify_scan_report_hashes()
        with self.assertRaises(fz.FreezeError) as ctx:
            fz._build_queue_artifact(reports)
        self.assertIn("not the exact canonical queue", str(ctx.exception))

    def test_check_rejects_unvalidated_generation_mutation(self):
        fz.cmd_write()
        contract_path = self.repo / v2c.CONTRACT_REL_PATH
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        contract["generation"]["unexpected_semantic_field"] = "not hash bound"
        contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
                                 encoding="utf-8")
        self.assertEqual(fz.cmd_check(), 1)


if __name__ == "__main__":
    unittest.main()
