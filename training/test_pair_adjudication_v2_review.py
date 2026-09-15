#!/usr/bin/env python3
"""test_pair_adjudication_v2_review.py — offline/synthetic tests for
pair_adjudication_v2_review.py. Proves PIL/Tkinter are never imported by
anything except launch_review_ui() itself (never called here), and
exercises the two-step select+confirm state machine and session-progress
math with synthetic queue rows -- never a real image, never a real
window."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


class TestNoDeferredImportsAtModuleScope(unittest.TestCase):
    def test_importing_review_module_never_pulls_in_tkinter_or_pil(self):
        for name in list(sys.modules):
            if name == "tkinter" or name.startswith("tkinter.") or name == "PIL" or name.startswith("PIL."):
                del sys.modules[name]
        import pair_adjudication_v2_review  # noqa: F401
        self.assertNotIn("tkinter", sys.modules)
        self.assertNotIn("PIL", sys.modules)

    def test_launch_review_ui_is_the_only_function_mentioning_tkinter_source(self):
        import inspect
        import pair_adjudication_v2_review as rv
        for name, fn in vars(rv).items():
            if not inspect.isfunction(fn) or fn.__module__ != rv.__name__:
                continue
            src = inspect.getsource(fn)
            if name == "launch_review_ui":
                continue
            self.assertNotIn("import tkinter", src, f"{name} must not import tkinter")
            self.assertNotIn("from PIL", src, f"{name} must not import PIL")


import pair_adjudication_v2_review as rv  # noqa: E402
import pair_adjudication_v2_contract as v2c  # noqa: E402


def _row(pair_id, queue_index, session, domain="d", workstream=v2c.WORKSTREAM_GATE_EVIDENCE_INDEPENDENCE):
    return {"pair_id": pair_id, "queue_index": queue_index, "suggested_session": session, "domain": domain,
            "workstream": workstream, "session_block": 0}


class TestSessionProgress(unittest.TestCase):
    def test_position_and_last_row_flag(self):
        rows = [_row("p1", 0, "s1"), _row("p2", 1, "s1"), _row("p3", 2, "s2")]
        progress = rv.session_progress(rows, [], rows[1])
        self.assertEqual(progress["position_in_session"], 2)
        self.assertEqual(progress["session_row_count"], 2)
        self.assertTrue(progress["is_last_row_in_session"])

    def test_not_last_row(self):
        rows = [_row("p1", 0, "s1"), _row("p2", 1, "s1")]
        progress = rv.session_progress(rows, [], rows[0])
        self.assertFalse(progress["is_last_row_in_session"])

    def test_overall_counts(self):
        rows = [_row("p1", 0, "s1"), _row("p2", 1, "s1")]
        records = [{"pair_id": "p1"}]
        progress = rv.session_progress(rows, records, rows[1])
        self.assertEqual(progress["overall_adjudicated_count"], 1)
        self.assertEqual(progress["overall_total_count"], 2)


class TestReviewSessionStateMachine(unittest.TestCase):
    """ReviewSession.__init__ calls adj2.load_verified_v2_state and
    load_v1_contract_content -- both are stubbed here so this test never
    touches the filesystem or a real contract."""

    def _make_session(self):
        session = rv.ReviewSession.__new__(rv.ReviewSession)
        session.repo = Path("/nonexistent")
        session.reviewer_id = "r1"
        session.session_id = "s1"
        session.pending_label = None
        session.state = {"queue_rows": [_row("p1", 0, "s1")]}
        session.v1_contract_content = {}
        return session

    def test_confirm_without_select_is_rejected(self):
        session = self._make_session()
        with self.assertRaises(rv.ReviewUIError):
            session.confirm(reviewed_at_utc="2026-01-01T00:00:00Z")

    def test_select_invalid_label_rejected(self):
        session = self._make_session()
        with self.assertRaises(rv.ReviewUIError):
            session.select("not_a_real_label")

    def test_select_then_confirm_calls_record_adjudication_exactly_once(self):
        session = self._make_session()
        session.select("different_image")
        self.assertEqual(session.pending_label, "different_image")
        with mock.patch.object(rv.adj2, "load_verified_ledger_v2", return_value=[]), \
             mock.patch.object(rv.adj2, "record_adjudication", return_value={"pair_id": "p1"}) as rec:
            result = session.confirm(reviewed_at_utc="2026-01-01T00:00:00Z")
        rec.assert_called_once()
        self.assertEqual(rec.call_args.kwargs["pair_id"], "p1")
        self.assertEqual(rec.call_args.kwargs["label"], "different_image")
        self.assertEqual(result["pair_id"], "p1")
        # pending selection is cleared after a successful confirm
        self.assertIsNone(session.pending_label)

    def test_clear_selection(self):
        session = self._make_session()
        session.select("uncertain")
        session.clear_selection()
        self.assertIsNone(session.pending_label)

    def test_current_row_never_crosses_into_next_session(self):
        session = self._make_session()
        session.state = {"queue_rows": [_row("p1", 0, "s1"), _row("p2", 1, "s2")]}
        with mock.patch.object(session, "current_records", return_value=[
                {"pair_id": "p1", "workstream": v2c.WORKSTREAM_GATE_EVIDENCE_INDEPENDENCE,
                 "label": "different_image"}]):
            self.assertIsNone(session.current_row())
            self.assertEqual(session.next_session_id(), "s2")

    def test_blocking_mid_session_never_exposes_another_sessions_row(self):
        session = self._make_session()
        session.state = {"queue_rows": [
            _row("p1", 0, "s1", workstream=v2c.WORKSTREAM_FINAL_TEST_INDEPENDENCE),
            _row("p2", 1, "s1", workstream=v2c.WORKSTREAM_FINAL_TEST_INDEPENDENCE),
            _row("p3", 2, "s2", workstream=v2c.WORKSTREAM_GATE_EVIDENCE_INDEPENDENCE),
        ]}
        records = [{"pair_id": "p1", "workstream": v2c.WORKSTREAM_FINAL_TEST_INDEPENDENCE,
                    "label": "same_source_image"}]
        with mock.patch.object(session, "current_records", return_value=records):
            # p2 is skipped by the channel stop; p3 is globally next but must
            # remain hidden until the reviewer explicitly launches s2.
            self.assertIsNone(session.current_row())
            self.assertEqual(session.next_session_id(), "s2")


class TestReviewSessionLaunchRefusesWrongSession(unittest.TestCase):
    """Exercises the REAL __init__ (not the __new__ bypass above) to
    prove the UI refuses to launch/record for a session different from
    the current frozen block -- before any image path is ever resolved."""

    def test_launch_refuses_mismatched_session_id(self):
        fake_state = {"queue_rows": [_row("p1", 0, "correct-session-block0")]}
        with mock.patch.object(rv.adj2, "load_verified_v2_state", return_value=fake_state), \
             mock.patch.object(rv.adj2, "load_verified_ledger_v2", return_value=[]), \
             mock.patch.object(rv, "load_v1_contract_content", return_value={}):
            with self.assertRaises(rv.ReviewUIError) as ctx:
                rv.ReviewSession(Path("/nonexistent"), reviewer_id="r1", session_id="wrong-session-block0")
        self.assertIn("does not match", str(ctx.exception))

    def test_launch_accepts_matching_session_id(self):
        fake_state = {"queue_rows": [_row("p1", 0, "correct-session-block0")]}
        with mock.patch.object(rv.adj2, "load_verified_v2_state", return_value=fake_state), \
             mock.patch.object(rv.adj2, "load_verified_ledger_v2", return_value=[]), \
             mock.patch.object(rv, "load_v1_contract_content", return_value={}):
            session = rv.ReviewSession(Path("/nonexistent"), reviewer_id="r1", session_id="correct-session-block0")
        self.assertEqual(session.session_id, "correct-session-block0")

    def test_launch_allowed_when_queue_already_exhausted(self):
        """No eligible row at all (queue exhausted) -- nothing to check
        session_id against, so launching (to view the done-state) is not
        refused."""
        fake_state = {"queue_rows": []}
        with mock.patch.object(rv.adj2, "load_verified_v2_state", return_value=fake_state), \
             mock.patch.object(rv.adj2, "load_verified_ledger_v2", return_value=[]), \
             mock.patch.object(rv, "load_v1_contract_content", return_value={}):
            session = rv.ReviewSession(Path("/nonexistent"), reviewer_id="r1", session_id="anything")
            self.assertIsNone(session.current_row())


if __name__ == "__main__":
    unittest.main()
