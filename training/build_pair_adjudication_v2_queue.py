#!/usr/bin/env python3
"""build_pair_adjudication_v2_queue.py — builds (and, once, writes) the
frozen 3,972-row Phase 5F4 perceptual-pair adjudication v2 queue.

Reads training/perceptual_duplicate_candidate_pairs.json ONLY -- the
already-committed, immutable Phase 5F1/5F3 scan evidence -- and verifies
its byte AND content sha256 against the frozen values recorded in
pair_adjudication_v2_contract.py before deriving anything from it. This
module never opens an image and never runs inference; it only reorders
and re-labels rows that the frozen scan already produced.

This generator does NOT load pair_adjudication_v2_contract.json (the
contract binds this queue's own hashes -- see
freeze_pair_adjudication_v2_contract.py, which must run AFTER this).

Three modes:
  --preflight  validates everything and prints the derived summary;
               writes ZERO bytes.
  --write      atomically and exclusively publishes the queue JSON --
               stage into a same-directory temp file, fsync, prevalidate
               the staged bytes, publish via exclusive hard-link (refuses
               to overwrite an existing file), always clean up the temp
               file, then reload and re-verify the published file.
  --check      re-derives the queue from the CURRENT candidate report and
               requires the existing queue JSON to match EXACTLY
               byte-for-byte. Writes nothing.

Usage:
    python build_pair_adjudication_v2_queue.py --preflight
    python build_pair_adjudication_v2_queue.py --write
    python build_pair_adjudication_v2_queue.py --check
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

import pair_adjudication_v2_contract as v2c  # noqa: E402

QUEUE_PATH = REPO / v2c.QUEUE_ARTIFACT_REL_PATH


class QueueBuildError(RuntimeError):
    """Always fails closed -- on any problem, nothing is written."""


def _load_verified_candidate_pairs_report() -> dict:
    try:
        reports = v2c.load_and_verify_scan_reports(REPO)
    except v2c.ScanEvidenceError as exc:
        raise QueueBuildError(str(exc)) from exc
    return reports["candidate_pairs_report"]["content"]


def build_queue_rows() -> list[dict]:
    content = _load_verified_candidate_pairs_report()
    return v2c.derive_v2_queue_rows(content)


def serialize(rows: list[dict]) -> str:
    payload = {
        "schema_version": v2c.SCHEMA_VERSION,
        "row_count": len(rows),
        "identity_order_sha256": v2c.compute_v2_queue_identity_order_sha256(rows),
        "rows": rows,
    }
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n"


def _summarize(rows: list[dict]) -> dict:
    domain_counts: dict[str, int] = {}
    workstream_counts: dict[str, int] = {}
    for r in rows:
        domain_counts[r["domain"]] = domain_counts.get(r["domain"], 0) + 1
        workstream_counts[r["workstream"]] = workstream_counts.get(r["workstream"], 0) + 1
    sessions = v2c.session_blocks_summary(rows)
    return {
        "row_count": len(rows),
        "both_rule_count": sum(1 for r in rows if r["both_rule_match"]),
        "domain_counts": domain_counts,
        "workstream_counts": workstream_counts,
        "session_block_count": len(sessions),
        "identity_order_sha256": v2c.compute_v2_queue_identity_order_sha256(rows),
    }


def cmd_preflight() -> int:
    rows = build_queue_rows()
    summary = _summarize(rows)
    print(json.dumps({"ok": True, "mode": "preflight", **summary}, indent=2, sort_keys=True))
    return 0


def cmd_check() -> int:
    rows = build_queue_rows()
    expected_bytes = serialize(rows).encode("utf-8")
    if not QUEUE_PATH.exists():
        print(f"[check] FAILED: {QUEUE_PATH} does not exist")
        return 1
    actual_bytes = QUEUE_PATH.read_bytes()
    if actual_bytes != expected_bytes:
        print(f"[check] FAILED: {QUEUE_PATH} does not match the reconstructed content byte-for-byte")
        return 1
    print(f"[check] PASS: {QUEUE_PATH} matches the reconstructed content byte-for-byte")
    print(json.dumps({"ok": True, "mode": "check", **_summarize(rows)}, indent=2, sort_keys=True))
    return 0


def cmd_write() -> int:
    if QUEUE_PATH.exists():
        raise QueueBuildError(f"refusing to overwrite existing queue: {QUEUE_PATH}")
    rows = build_queue_rows()
    data = serialize(rows).encode("utf-8")
    QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = QUEUE_PATH.with_suffix(QUEUE_PATH.suffix + f".tmp{os.getpid()}")
    try:
        with tmp.open("wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        # prevalidate the staged bytes before publishing
        staged = json.loads(tmp.read_bytes().decode("utf-8"))
        if staged["row_count"] != len(rows):
            raise QueueBuildError("staged queue row_count does not match derived rows -- refusing to publish")
        os.link(str(tmp), str(QUEUE_PATH))
    finally:
        tmp.unlink(missing_ok=True)
    # reload and re-verify the published artifact from disk
    published_bytes = QUEUE_PATH.read_bytes()
    if published_bytes != data:
        raise QueueBuildError(f"published {QUEUE_PATH} does not match staged bytes after publication -- "
                             f"left on disk for manual review, NOT deleted")
    published = json.loads(published_bytes.decode("utf-8"))
    if published["identity_order_sha256"] != v2c.compute_v2_queue_identity_order_sha256(rows):
        raise QueueBuildError(f"published {QUEUE_PATH} identity_order_sha256 mismatch after reload -- "
                             f"left on disk for manual review, NOT deleted")
    print(f"wrote {QUEUE_PATH}")
    print(json.dumps({"ok": True, "mode": "write", "byte_sha256": v2c.sha256_bytes(published_bytes),
                     **_summarize(rows)}, indent=2, sort_keys=True))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = ap.parse_args()

    try:
        if args.write:
            return cmd_write()
        if args.check:
            return cmd_check()
        return cmd_preflight()
    except QueueBuildError as e:
        print(f"[build_pair_adjudication_v2_queue] FAILURE: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
