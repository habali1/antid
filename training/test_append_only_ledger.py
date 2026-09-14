#!/usr/bin/env python3
"""test_append_only_ledger.py — offline tests for append_only_ledger.py."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import append_only_ledger as aol  # noqa: E402

REQUIRED = frozenset({"item_id", "value", "prev_record_hash", "record_hash"})


def identity_fn(rec):
    return rec.get("item_id")


class TestAppendOnlyLedger(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.path = Path(self._tmpdir.name) / "ledger.jsonl"

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_read_ledger_of_nonexistent_file_is_empty(self):
        self.assertEqual(aol.read_ledger(self.path), [])

    def test_append_first_record_chains_to_genesis(self):
        rec = aol.append_record(self.path, {"item_id": "a", "value": 1}, required_fields=REQUIRED, identity_fn=identity_fn)
        self.assertEqual(rec["prev_record_hash"], aol.GENESIS_HASH)
        self.assertTrue(aol.LedgerError)  # module imported fine

    def test_chain_links_sequential_records(self):
        r1 = aol.append_record(self.path, {"item_id": "a", "value": 1}, required_fields=REQUIRED, identity_fn=identity_fn)
        r2 = aol.append_record(self.path, {"item_id": "b", "value": 2}, required_fields=REQUIRED, identity_fn=identity_fn)
        self.assertEqual(r2["prev_record_hash"], r1["record_hash"])

    def test_verify_chain_passes_for_untampered_ledger(self):
        aol.append_record(self.path, {"item_id": "a", "value": 1}, required_fields=REQUIRED, identity_fn=identity_fn)
        aol.append_record(self.path, {"item_id": "b", "value": 2}, required_fields=REQUIRED, identity_fn=identity_fn)
        records = aol.read_ledger(self.path)
        self.assertEqual(aol.verify_chain(records, REQUIRED, identity_fn), [])

    def test_duplicate_identity_rejected(self):
        aol.append_record(self.path, {"item_id": "a", "value": 1}, required_fields=REQUIRED, identity_fn=identity_fn)
        with self.assertRaises(aol.LedgerError):
            aol.append_record(self.path, {"item_id": "a", "value": 999}, required_fields=REQUIRED, identity_fn=identity_fn)
        # the rejected attempt must not have appended anything
        self.assertEqual(len(aol.read_ledger(self.path)), 1)

    def test_missing_required_field_rejected(self):
        with self.assertRaises(aol.LedgerError):
            aol.append_record(self.path, {"item_id": "a"}, required_fields=REQUIRED, identity_fn=identity_fn)

    def test_extra_field_rejected(self):
        with self.assertRaises(aol.LedgerError):
            aol.append_record(self.path, {"item_id": "a", "value": 1, "smuggled": "x"},
                             required_fields=REQUIRED, identity_fn=identity_fn)

    def test_tampering_with_an_earlier_record_is_detected(self):
        aol.append_record(self.path, {"item_id": "a", "value": 1}, required_fields=REQUIRED, identity_fn=identity_fn)
        aol.append_record(self.path, {"item_id": "b", "value": 2}, required_fields=REQUIRED, identity_fn=identity_fn)
        lines = self.path.read_text(encoding="utf-8").splitlines()
        first = json.loads(lines[0])
        first["value"] = 999  # tamper without recomputing record_hash
        lines[0] = json.dumps(first, sort_keys=True, separators=(",", ":"))
        self.path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        records = aol.read_ledger(self.path)
        problems = aol.verify_chain(records, REQUIRED, identity_fn)
        self.assertTrue(any("record_hash" in p and "recomputed" in p for p in problems), problems)

    def test_reordering_records_breaks_the_chain(self):
        aol.append_record(self.path, {"item_id": "a", "value": 1}, required_fields=REQUIRED, identity_fn=identity_fn)
        aol.append_record(self.path, {"item_id": "b", "value": 2}, required_fields=REQUIRED, identity_fn=identity_fn)
        lines = self.path.read_text(encoding="utf-8").splitlines()
        self.path.write_text("\n".join(reversed(lines)) + "\n", encoding="utf-8")

        records = aol.read_ledger(self.path)
        problems = aol.verify_chain(records, REQUIRED, identity_fn)
        self.assertTrue(any("prev_record_hash" in p for p in problems), problems)

    def test_appending_onto_a_broken_chain_is_refused(self):
        aol.append_record(self.path, {"item_id": "a", "value": 1}, required_fields=REQUIRED, identity_fn=identity_fn)
        lines = self.path.read_text(encoding="utf-8").splitlines()
        rec = json.loads(lines[0])
        rec["value"] = 42
        self.path.write_text(json.dumps(rec, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
        with self.assertRaises(aol.LedgerError):
            aol.append_record(self.path, {"item_id": "b", "value": 2}, required_fields=REQUIRED, identity_fn=identity_fn)

    def test_malformed_non_final_json_line_is_controlled_failure(self):
        self.path.write_text("{not valid json\n{\"item_id\": \"a\", \"value\": 1, "
                             "\"prev_record_hash\": \"" + aol.GENESIS_HASH + "\", \"record_hash\": \"x\"}\n",
                             encoding="utf-8")
        with self.assertRaises(aol.LedgerError):
            aol.read_ledger(self.path)

    # ---- item 5: the three raw trailing-newline recovery cases ----------
    def test_case_a_corrupt_final_line_with_trailing_newline_fails_closed(self):
        # The file ends WITH a newline -- so the malformed final line was
        # fully written and terminated. That is real corruption, not an
        # interrupted write: append must fail closed and must NEVER
        # auto-delete the bad line.
        r1 = aol.append_record(self.path, {"item_id": "a", "value": 1}, required_fields=REQUIRED, identity_fn=identity_fn)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write('{"item_id": "b", "value": 2, "prev_record_h}\n')  # malformed but newline-terminated
        before = self.path.read_bytes()

        with self.assertRaises(aol.LedgerError):
            aol.read_ledger(self.path)
        with self.assertRaises(aol.LedgerError):
            aol.append_record(self.path, {"item_id": "c", "value": 3}, required_fields=REQUIRED, identity_fn=identity_fn)

        # Never auto-deleted -- the corrupt bytes are still there untouched.
        self.assertEqual(self.path.read_bytes(), before)

    def test_case_b_incomplete_fragment_without_newline_is_truncated_and_recovered(self):
        # A genuinely interrupted append: a complete, valid first record,
        # then a partial (truncated) second line with NO trailing newline
        # -- must be treated as if the second append never happened, not
        # as ledger corruption, and the fragment is physically removed.
        r1 = aol.append_record(self.path, {"item_id": "a", "value": 1}, required_fields=REQUIRED, identity_fn=identity_fn)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write('{"item_id": "b", "value": 2, "prev_record_h')  # cut off mid-write, no newline
        records = aol.read_ledger(self.path)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["record_hash"], r1["record_hash"])

        # A later append succeeds and does not need any pre-repair step.
        r2 = aol.append_record(self.path, {"item_id": "c", "value": 3}, required_fields=REQUIRED, identity_fn=identity_fn)
        self.assertEqual(r2["prev_record_hash"], r1["record_hash"])

        # Close and reopen: read the file fresh from disk, not the
        # in-memory dict append_record() returned.
        reopened = aol.read_ledger(self.path)
        self.assertEqual(len(reopened), 2)
        self.assertEqual([r["item_id"] for r in reopened], ["a", "c"])
        self.assertEqual(reopened[1]["record_hash"], r2["record_hash"])
        self.assertEqual(aol.verify_chain(reopened, REQUIRED, identity_fn), [])

        # And the raw file on disk actually contains two clean, complete,
        # newline-terminated lines -- not a merged/garbage line.
        raw_lines = self.path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(raw_lines), 2)
        for line in raw_lines:
            json.loads(line)  # each line parses on its own

    def test_case_c_valid_final_record_without_newline_is_preserved_not_truncated(self):
        # Process interruption AFTER the complete JSON bytes landed but
        # BEFORE the trailing newline was written. The record is real and
        # must be kept -- never truncated -- with a newline safely
        # appended so the next append starts its own line.
        r1 = aol.append_record(self.path, {"item_id": "a", "value": 1}, required_fields=REQUIRED, identity_fn=identity_fn)
        r2_payload = {"item_id": "b", "value": 2, "prev_record_hash": r1["record_hash"]}
        r2_hash = aol.sha256_hex(aol.canonical_bytes(r2_payload))
        r2_payload["record_hash"] = r2_hash
        r2_line = json.dumps(r2_payload, sort_keys=True, separators=(",", ":"))
        # Simulate: previous append wrote the complete record bytes but the
        # process died before the trailing "\n" landed.
        with self.path.open("r+b") as fh:
            fh.seek(0, 2)
            fh.write(r2_line.encode("utf-8"))
            fh.flush()
            os.fsync(fh.fileno())
        self.assertFalse(self.path.read_bytes().endswith(b"\n"))

        # Reading it back must find BOTH records -- the valid final one is
        # never dropped just because it lacks a trailing newline.
        records = aol.read_ledger(self.path)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[1]["record_hash"], r2_hash)

        r3 = aol.append_record(self.path, {"item_id": "c", "value": 3}, required_fields=REQUIRED, identity_fn=identity_fn)
        self.assertEqual(r3["prev_record_hash"], r2_hash)

        # Close and reopen: verify the complete persisted chain, all three
        # records, on a fresh read from disk.
        reopened = aol.read_ledger(self.path)
        self.assertEqual([r["item_id"] for r in reopened], ["a", "b", "c"])
        self.assertEqual(aol.verify_chain(reopened, REQUIRED, identity_fn), [])
        raw_lines = self.path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(raw_lines), 3)
        for line in raw_lines:
            json.loads(line)

    def test_unhashable_identity_is_a_controlled_problem_not_typeerror(self):
        # identity_fn returns an unhashable list -- verify_chain must
        # report it as a problem, never raise TypeError from `in`/`add`.
        aol.append_record(self.path, {"item_id": "a", "value": 1}, required_fields=REQUIRED, identity_fn=identity_fn)
        records = aol.read_ledger(self.path)
        problems = aol.verify_chain(records, REQUIRED, lambda rec: [rec.get("item_id")])
        self.assertTrue(any("unhashable" in p or "could not be computed" in p for p in problems), problems)

    def test_non_dict_record_values_never_raise_during_verify(self):
        # Every kind of malformed JSON value inside a record's fields (not
        # just at the top level) must produce a controlled problem, never
        # an uncaught exception, when replayed through verify_chain.
        weird_records = [
            {"item_id": "a", "value": float("nan"), "prev_record_hash": aol.GENESIS_HASH, "record_hash": "bad"},
            {"item_id": None, "value": {"nested": [1, 2, {"x": None}]},
             "prev_record_hash": aol.GENESIS_HASH, "record_hash": "bad"},
            "not-a-dict-at-all",
            42,
            None,
        ]
        try:
            problems = aol.verify_chain(weird_records, REQUIRED, identity_fn)
        except Exception as exc:  # noqa: BLE001
            self.fail(f"verify_chain raised {type(exc).__name__}: {exc}")
        self.assertTrue(len(problems) > 0)

    def test_never_overwrites_an_earlier_line(self):
        aol.append_record(self.path, {"item_id": "a", "value": 1}, required_fields=REQUIRED, identity_fn=identity_fn)
        before = self.path.read_text(encoding="utf-8")
        aol.append_record(self.path, {"item_id": "b", "value": 2}, required_fields=REQUIRED, identity_fn=identity_fn)
        after = self.path.read_text(encoding="utf-8")
        self.assertTrue(after.startswith(before))


if __name__ == "__main__":
    unittest.main(verbosity=2)
