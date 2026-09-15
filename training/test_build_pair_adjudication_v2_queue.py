#!/usr/bin/env python3
"""test_build_pair_adjudication_v2_queue.py — offline/synthetic tests for
build_pair_adjudication_v2_queue.py's failure-closed behavior. Never
opens an image."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pair_adjudication_v2_contract as v2c  # noqa: E402
import build_pair_adjudication_v2_queue as bq  # noqa: E402


class TestLoadVerifiedCandidatePairsReport(unittest.TestCase):
    def test_rejects_when_scan_evidence_changed(self):
        def _raise(repo):
            raise v2c.ScanEvidenceError("byte_sha256 mismatch -- committed scan evidence has changed")
        with mock.patch.object(v2c, "load_and_verify_scan_reports", side_effect=_raise):
            with self.assertRaises(bq.QueueBuildError):
                bq._load_verified_candidate_pairs_report()

    def test_passes_through_verified_candidate_pairs_content(self):
        sentinel = {"domains": {"d": []}}
        with mock.patch.object(v2c, "load_and_verify_scan_reports",
                               return_value={"candidate_pairs_report": {"content": sentinel}}):
            self.assertIs(bq._load_verified_candidate_pairs_report(), sentinel)


class TestCmdWriteRefusesOverwrite(unittest.TestCase):
    def test_write_refuses_when_file_already_exists(self):
        with mock.patch.object(bq, "QUEUE_PATH", Path(__file__)):  # any existing file
            with self.assertRaises(bq.QueueBuildError):
                bq.cmd_write()


if __name__ == "__main__":
    unittest.main()
