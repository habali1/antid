#!/usr/bin/env python3
"""finalize_stop_status_v2.py — the Phase 5F4 post-adjudication stop-
decision gate for the v2 (workstream-scoped) adjudication protocol.

Produces
training/perceptual_duplicate_post_adjudication_stop_status_v2.json, a
SEPARATE artifact from both scan_perceptual_duplicates.py's scan-time
stop_status_report and Phase 5F1's own
perceptual_duplicate_post_adjudication_stop_status.json. This artifact
never modifies the five committed scan reports or the v2 adjudication
ledger; it only reads and reports.

Reports the three workstreams' status INDEPENDENTLY -- the two blocking
channels (final_test_independence, gate_evidence_independence) are never
collapsed into one pass/fail field, and the diagnostic channel
(final_test_internal_repetition) is reported separately again with its
own effective-sample-size-limitation framing rather than a pass/fail
verdict. Every one of the 12,567 outside-scope candidates, and every
early-stop-unreviewed row inside a stopped blocking channel, is reported
as not_adjudicated (or not_adjudicated_due_to_early_stop) -- never as
not_duplicate.

CLI modes:
  --preflight  loads/verifies the contract, the v2 queue, all five scan
               reports, and the (possibly still-empty) v2 ledger; reports
               adjudication progress per workstream WITHOUT requiring
               completion. No image access.
  --finalize   refuses to complete unless: final_test_independence is
               exhausted cleanly or stopped by a recorded blocking
               result; gate_evidence_independence is exhausted cleanly or
               stopped by a recorded blocking result; and all 53
               final_test_internal_repetition candidates are adjudicated.
               Publishes exclusively/atomically. NEVER modifies the five
               scan reports or the ledger.
  --check      validates the ALREADY-PUBLISHED artifact: schema, every
               cross-report/ledger binding, and that its content is
               EXACTLY re-derivable from the current state -- without
               rerunning or replacing it.

Never imports onnxruntime, PIL, or any inference/image module.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pair_adjudication_v2_contract as v2c  # noqa: E402
import adjudicate_pairs_v2 as adj2  # noqa: E402

FINALIZATION_ARTIFACT_NAME = "post_adjudication_stop_status_v2_report"
BLOCKING_WORKSTREAMS = (v2c.WORKSTREAM_FINAL_TEST_INDEPENDENCE, v2c.WORKSTREAM_GATE_EVIDENCE_INDEPENDENCE)

STATUS_PASSED, STATUS_FAILED, STATUS_INCONCLUSIVE, STATUS_PENDING = (
    v2c.STATUS_PASSED, v2c.STATUS_FAILED, v2c.STATUS_INCONCLUSIVE, v2c.STATUS_PENDING)
DIAGNOSTIC_STATUS_COMPLETE = "complete"
DIAGNOSTIC_STATUS_PENDING = "pending"


class FinalizationV2Error(RuntimeError):
    """A verification or completeness problem. Always fails closed."""


def _fail(msg: str) -> None:
    raise FinalizationV2Error(msg)


def _report_path(repo: Path) -> Path:
    return Path(repo) / v2c.APPROVED_OUTPUT_PATHS[FINALIZATION_ARTIFACT_NAME]


def verify_everything(repo: Path) -> dict:
    """The one required first step for every mode: full v2 state
    verification (contract + queue + five scan reports, hash-bound) via
    adjudicate_pairs_v2.load_verified_v2_state, THEN the v2 ledger's full
    semantic verification via
    adjudicate_pairs_v2.load_verified_ledger_v2(..., require_canonical_tail=True)
    -- the SAME check --preflight/--next/--record apply, plus the
    additional canonical-newline-terminated-bytes requirement finalization
    alone needs."""
    state = adj2.load_verified_v2_state(repo)
    ledger_path = adj2.ledger_path_for(repo)
    ledger_bytes = ledger_path.read_bytes() if ledger_path.exists() else b""
    try:
        ledger_records = adj2.load_verified_ledger_v2(repo, state, require_canonical_tail=True)
    except adj2.AdjudicationV2Error as exc:
        _fail(f"v2 adjudication ledger failed verification, refusing to finalize: {exc}")
    return {"v2_state": state, "ledger_records": ledger_records, "ledger_bytes": ledger_bytes}


def _rows_for_workstream(queue_rows: list[dict], workstream: str) -> list[dict]:
    return [r for r in queue_rows if r["workstream"] == workstream]


def derive_workstream_status(queue_rows: list[dict], ledger_records: list[dict], workstream: str) -> dict:
    rows = _rows_for_workstream(queue_rows, workstream)
    total = len(rows)
    records_here = [r for r in ledger_records if r.get("workstream") == workstream]
    adjudicated = len(records_here)
    same_source = [r for r in records_here if r.get("label") == "same_source_image"]
    uncertain = [r for r in records_here if r.get("label") == "uncertain"]
    row_pair_ids_here = {r["pair_id"] for r in rows}
    all_early_stop_skipped = adj2.early_stop_skipped_pair_ids(queue_rows, ledger_records)
    early_stop_skipped = all_early_stop_skipped & row_pair_ids_here
    remaining = total - adjudicated - len(early_stop_skipped)

    if v2c.WORKSTREAM_ROLE[workstream] == v2c.CHANNEL_ROLE_BLOCKING:
        if same_source:
            status, stopped = STATUS_FAILED, True
        elif uncertain:
            status, stopped = STATUS_INCONCLUSIVE, True
        elif adjudicated == total:
            status, stopped = STATUS_PASSED, False
        else:
            status, stopped = STATUS_PENDING, False
        return {
            "role": "blocking", "status": status, "stop": stopped,
            "total": total, "adjudicated_count": adjudicated, "remaining_count": remaining,
            "early_stop_unreviewed_count": len(early_stop_skipped),
            "confirmed_same_source_count": len(same_source), "uncertain_count": len(uncertain),
        }

    # diagnostic (final_test_internal_repetition): never stops, always fully reviewed to conclude
    complete = adjudicated == total
    return {
        "role": "diagnostic",
        "status": DIAGNOSTIC_STATUS_COMPLETE if complete else DIAGNOSTIC_STATUS_PENDING,
        "total": total, "adjudicated_count": adjudicated, "remaining_count": remaining,
        "confirmed_same_source_count": len(same_source),
        "effective_sample_size_limitation": len(same_source) > 0,
        "note": ("confirmed same-source pairs within northeast_final_test_v1 are an "
                "effective-sample-size limitation of that dataset -- they do NOT constitute "
                "development-data leakage and do not by themselves stop inference") if same_source else None,
    }


def build_post_adjudication_content(state: dict) -> dict:
    """Pure function over already-verified state -- no filesystem access,
    no image access. Deterministic: the same state always produces the
    same content, which is what makes --check's exact re-derivation
    comparison meaningful."""
    v2_state, ledger_records = state["v2_state"], state["ledger_records"]
    contract, queue_rows = v2_state["contract"], v2_state["queue_rows"]

    workstreams = {ws: derive_workstream_status(queue_rows, ledger_records, ws)
                  for ws in (v2c.WORKSTREAM_FINAL_TEST_INDEPENDENCE, v2c.WORKSTREAM_FINAL_TEST_INTERNAL_REPETITION,
                            v2c.WORKSTREAM_GATE_EVIDENCE_INDEPENDENCE)}

    scoped_count = len(queue_rows)
    full_count = contract["content"]["candidate_report_row_count"]
    outside_scope_count = contract["content"]["outside_scope_candidate_count"]
    outside_scope_identity_order_sha256 = contract["content"]["outside_scope_identity_order_sha256"]
    adjudicated_count = len(ledger_records)
    early_stop_unreviewed_count = len(adj2.early_stop_skipped_pair_ids(queue_rows, ledger_records))
    remaining_count = scoped_count - adjudicated_count - early_stop_unreviewed_count

    return {
        "v2_contract_content_sha256": contract["content_sha256"],
        "v2_queue_byte_sha256": v2c.sha256_bytes(v2_state["queue_bytes"]),
        "v2_queue_identity_order_sha256": contract["content"]["queue_artifact"]["identity_order_sha256"],
        "candidate_pairs_report_content_sha256": v2_state["candidate_pairs_report_content_sha256"],
        "v2_adjudication_ledger_byte_sha256": v2c.sha256_bytes(state["ledger_bytes"]),
        "full_candidate_count": full_count,
        "scoped_candidate_count": scoped_count,
        "outside_scope_candidate_count": outside_scope_count,
        "outside_scope_identity_order_sha256": outside_scope_identity_order_sha256,
        "outside_scope_status": v2c.NOT_ADJUDICATED,
        "adjudicated_count": adjudicated_count,
        "remaining_count": remaining_count,
        "early_stop_unreviewed_count": early_stop_unreviewed_count,
        "workstreams": workstreams,
        "stop_before_final_test_inference": workstreams[v2c.WORKSTREAM_FINAL_TEST_INDEPENDENCE]["stop"],
        "stop_before_gate_promotion": workstreams[v2c.WORKSTREAM_GATE_EVIDENCE_INDEPENDENCE]["stop"],
        "ready_to_finalize": _is_ready_to_finalize(workstreams),
    }


def _is_ready_to_finalize(workstreams: dict) -> bool:
    a = workstreams[v2c.WORKSTREAM_FINAL_TEST_INDEPENDENCE]
    b = workstreams[v2c.WORKSTREAM_FINAL_TEST_INTERNAL_REPETITION]
    c = workstreams[v2c.WORKSTREAM_GATE_EVIDENCE_INDEPENDENCE]
    a_ready = a["status"] in (STATUS_PASSED, STATUS_FAILED, STATUS_INCONCLUSIVE)
    c_ready = c["status"] in (STATUS_PASSED, STATUS_FAILED, STATUS_INCONCLUSIVE)
    b_ready = b["status"] == DIAGNOSTIC_STATUS_COMPLETE
    return a_ready and b_ready and c_ready


# --------------------------------------------------------------- commands --
def cmd_preflight(args) -> dict:
    state = verify_everything(args.repo)
    content = build_post_adjudication_content(state)
    return {
        "ok": True, "adjudicated_count": content["adjudicated_count"],
        "remaining_count": content["remaining_count"],
        "early_stop_unreviewed_count": content["early_stop_unreviewed_count"],
        "workstream_statuses": {k: v["status"] for k, v in content["workstreams"].items()},
        "ready_to_finalize": content["ready_to_finalize"],
    }


def cmd_finalize(args) -> dict:
    state = verify_everything(args.repo)
    content = build_post_adjudication_content(state)
    if not content["ready_to_finalize"]:
        statuses = {k: v["status"] for k, v in content["workstreams"].items()}
        _fail(f"refusing to finalize: not every workstream is exhausted or blocking-stopped -- "
              f"statuses: {statuses}")

    content_sha256 = v2c.compute_content_sha256(content)
    report = {"schema_version": v2c.SCHEMA_VERSION, "content": content, "content_sha256": content_sha256}
    dest = _report_path(args.repo)
    if dest.exists():
        _fail(f"refusing to overwrite existing finalization artifact: {dest}")

    data = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + f".tmp{os.getpid()}")
    try:
        with tmp.open("wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.link(str(tmp), str(dest))
    finally:
        tmp.unlink(missing_ok=True)

    published_bytes = dest.read_bytes()
    if published_bytes != data:
        _fail(f"published {dest} does not match staged bytes -- left on disk for manual review")

    return {"ok": True, "path": str(dest), "content_sha256": content_sha256,
            "stop_before_final_test_inference": content["stop_before_final_test_inference"],
            "stop_before_gate_promotion": content["stop_before_gate_promotion"]}


def cmd_check(args) -> dict:
    state = verify_everything(args.repo)
    dest = _report_path(args.repo)
    if not dest.exists():
        _fail(f"finalization artifact does not exist yet: {dest} -- run --finalize first "
              f"(a separately authorized phase)")
    try:
        report = json.loads(dest.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        _fail(f"{dest} is not valid JSON: {exc}")

    if not isinstance(report, dict) or set(report) != {"schema_version", "content", "content_sha256"}:
        _fail(f"{dest} has an unexpected top-level key set")
    if report.get("schema_version") != v2c.SCHEMA_VERSION:
        _fail(f"{dest} schema_version mismatch")
    try:
        recomputed = v2c.compute_content_sha256(report["content"])
    except (TypeError, ValueError) as exc:
        _fail(f"{dest} content is not canonicalizable: {exc}")
    if recomputed != report.get("content_sha256"):
        _fail(f"{dest} content_sha256 does not match its own content")

    expected_content = build_post_adjudication_content(state)
    if report["content"] != expected_content:
        _fail("finalization artifact does not exactly match the current re-derivation from the scan "
              "reports, v2 queue, and v2 adjudication ledger")

    return {"ok": True, "path": str(dest), "content_sha256": report["content_sha256"],
            "stop_before_final_test_inference": report["content"]["stop_before_final_test_inference"],
            "stop_before_gate_promotion": report["content"]["stop_before_gate_promotion"]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", type=Path, default=HERE.parent)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--finalize", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = ap.parse_args()

    try:
        if args.preflight:
            result = cmd_preflight(args)
        elif args.finalize:
            result = cmd_finalize(args)
        else:
            result = cmd_check(args)
        print(json.dumps(result, indent=2, default=str))
        return 0
    except (FinalizationV2Error, adj2.AdjudicationV2Error) as e:
        print(f"[finalize_stop_status_v2] FAILURE: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
