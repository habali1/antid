#!/usr/bin/env python3
"""test_finalize_stop_status_v2.py — offline/synthetic tests for
finalize_stop_status_v2.py. Never opens an image; exercises the pure
per-workstream status derivation with synthetic queue rows/ledger
records, and the ready_to_finalize gate."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pair_adjudication_v2_contract as v2c  # noqa: E402
import finalize_stop_status_v2 as fsv2  # noqa: E402

A, B, C = v2c.WORKSTREAM_FINAL_TEST_INDEPENDENCE, v2c.WORKSTREAM_FINAL_TEST_INTERNAL_REPETITION, v2c.WORKSTREAM_GATE_EVIDENCE_INDEPENDENCE


def _row(pair_id, workstream):
    return {"pair_id": pair_id, "workstream": workstream, "domain": "d", "queue_index": 0}


def _rec(pair_id, workstream, label):
    return {"pair_id": pair_id, "workstream": workstream, "label": label}


class TestWorkstreamStatus(unittest.TestCase):
    def test_blocking_pending_when_incomplete_and_clean(self):
        rows = [_row("a1", A), _row("a2", A)]
        status = fsv2.derive_workstream_status(rows, [], A)
        self.assertEqual(status["status"], fsv2.STATUS_PENDING)
        self.assertFalse(status["stop"])

    def test_blocking_passed_when_complete_and_clean(self):
        rows = [_row("a1", A), _row("a2", A)]
        records = [_rec("a1", A, "different_image"), _rec("a2", A, "different_photo_same_observation")]
        status = fsv2.derive_workstream_status(rows, records, A)
        self.assertEqual(status["status"], fsv2.STATUS_PASSED)
        self.assertFalse(status["stop"])

    def test_blocking_failed_on_same_source_image_even_if_incomplete(self):
        rows = [_row("a1", A), _row("a2", A), _row("a3", A)]
        records = [_rec("a1", A, "same_source_image")]
        status = fsv2.derive_workstream_status(rows, records, A)
        self.assertEqual(status["status"], fsv2.STATUS_FAILED)
        self.assertTrue(status["stop"])
        self.assertEqual(status["early_stop_unreviewed_count"], 2)

    def test_blocking_inconclusive_on_uncertain(self):
        rows = [_row("c1", C), _row("c2", C)]
        records = [_rec("c1", C, "uncertain")]
        status = fsv2.derive_workstream_status(rows, records, C)
        self.assertEqual(status["status"], fsv2.STATUS_INCONCLUSIVE)
        self.assertTrue(status["stop"])

    def test_uncertain_never_counted_as_passed_or_cleared(self):
        rows = [_row("c1", C)]
        records = [_rec("c1", C, "uncertain")]
        status = fsv2.derive_workstream_status(rows, records, C)
        self.assertNotEqual(status["status"], fsv2.STATUS_PASSED)

    def test_diagnostic_pending_when_incomplete(self):
        rows = [_row("b1", B), _row("b2", B)]
        records = [_rec("b1", B, "different_image")]
        status = fsv2.derive_workstream_status(rows, records, B)
        self.assertEqual(status["status"], fsv2.DIAGNOSTIC_STATUS_PENDING)

    def test_diagnostic_complete_requires_all_reviewed_even_after_same_source(self):
        rows = [_row("b1", B), _row("b2", B)]
        records = [_rec("b1", B, "same_source_image")]
        status = fsv2.derive_workstream_status(rows, records, B)
        self.assertEqual(status["status"], fsv2.DIAGNOSTIC_STATUS_PENDING)
        self.assertTrue(status["effective_sample_size_limitation"])
        records.append(_rec("b2", B, "different_image"))
        status2 = fsv2.derive_workstream_status(rows, records, B)
        self.assertEqual(status2["status"], fsv2.DIAGNOSTIC_STATUS_COMPLETE)

    def test_diagnostic_without_same_source_has_no_limitation_note(self):
        rows = [_row("b1", B)]
        records = [_rec("b1", B, "different_image")]
        status = fsv2.derive_workstream_status(rows, records, B)
        self.assertFalse(status["effective_sample_size_limitation"])
        self.assertIsNone(status["note"])


class TestReadyToFinalize(unittest.TestCase):
    def test_not_ready_while_any_workstream_pending(self):
        workstreams = {A: {"status": fsv2.STATUS_PASSED}, B: {"status": fsv2.DIAGNOSTIC_STATUS_PENDING},
                       C: {"status": fsv2.STATUS_PASSED}}
        self.assertFalse(fsv2._is_ready_to_finalize(workstreams))

    def test_ready_when_a_and_c_stopped_and_b_complete(self):
        workstreams = {A: {"status": fsv2.STATUS_FAILED}, B: {"status": fsv2.DIAGNOSTIC_STATUS_COMPLETE},
                       C: {"status": fsv2.STATUS_INCONCLUSIVE}}
        self.assertTrue(fsv2._is_ready_to_finalize(workstreams))

    def test_ready_when_all_passed_and_complete(self):
        workstreams = {A: {"status": fsv2.STATUS_PASSED}, B: {"status": fsv2.DIAGNOSTIC_STATUS_COMPLETE},
                       C: {"status": fsv2.STATUS_PASSED}}
        self.assertTrue(fsv2._is_ready_to_finalize(workstreams))


if __name__ == "__main__":
    unittest.main()
