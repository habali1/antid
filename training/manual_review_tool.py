#!/usr/bin/env python3
"""manual_review_tool.py — pausable, append-only, hash-chained manual review
of the frozen 600-row queue, bound to the frozen contract.

This module NEVER imports onnxruntime, api/inference.py, or api/
inference_policy.py, and never reads a model artifact -- a reviewer must
never see a model prediction while making a quality/plausibility/duplicate-
suspicion judgment.

CLI modes:
  --preflight  loads and verifies contract + summary + queue + (if present)
               ledger; reports progress. No image access.
  --next       resolves and reports the next approved, unreviewed queue
               entry and its EXACT (but not yet opened) image path.
  --record     appends one validated decision for a specific --queue-index,
               binding it to the frozen contract/queue identity.
  --status     re-verifies the full ledger chain and reports progress.

Actual image display/opening is reserved for a later phase -- --next only
ever prints a path, never opens it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import manual_review_contract as mrc  # noqa: E402
import append_only_ledger as aol  # noqa: E402

DEFAULT_LEDGER_REL_PATH = mrc.APPROVED_OUTPUT_PATHS["manual_review_ledger"]


class ReviewToolError(RuntimeError):
    """A validation or ledger problem. Always fails closed."""


def _fail(msg: str) -> None:
    raise ReviewToolError(msg)


def ledger_path_for(repo: Path) -> Path:
    return Path(repo) / DEFAULT_LEDGER_REL_PATH


def image_path_for_row(repo: Path, verified: dict, row: dict) -> Path:
    """The exact (never-opened, in this phase) image path for a queue row.
    The directory layout (a "clean" curation subdirectory for both
    queue-backed datasets) comes ONLY from the verified contract's own
    `domain_part_image_layouts` mapping -- never a hardcoded assumption --
    so a changed/substituted contract's layout rule is what actually
    governs resolution. Extension is resolved deterministically -- real
    queue rows are NOT all `.jpg` (167 real rows are `.jpeg`, 23 are
    `.png`) -- via the shared resolver, which fails closed on zero or
    ambiguous-multiple matches and never opens the file."""
    image_layouts = verified["contract"]["content"]["domain_part_image_layouts"]
    domain_part = mrc.domain_part_for_dataset_and_split(row["dataset"], row["split"])
    return mrc.resolve_image_path(Path(repo) / "data", image_layouts, domain_part, row["slug"], row["photo_id"])


def load_verified(repo: Path) -> dict:
    """The one required first step for every mode: load and verify the
    frozen contract + summary + queue. Raises ContractError (never
    partially-trusts a substituted artifact) before anything else runs."""
    return mrc.load_and_verify_contract(repo)


def validate_decision_fields(fields: dict, row: dict, contract: dict) -> list[str]:
    """Structural + enum validation of a proposed decision, entirely via
    .get() -- never raises. The identity/contract-binding fields are
    checked for exact agreement with the frozen row and contract (a
    decision cannot be silently recorded against a different row or a
    different contract version); the judgment fields (review_state,
    label_plausibility, duplicate_suspicion, notes, reviewer_id,
    session_id, reviewed_at_utc) are checked against the frozen enums."""
    problems: list[str] = []
    actual_fields = set(fields) | {"prev_record_hash", "record_hash"}
    if actual_fields != mrc.REVIEW_RECORD_REQUIRED_FIELDS:
        missing = mrc.REVIEW_RECORD_REQUIRED_FIELDS - actual_fields
        extra = actual_fields - mrc.REVIEW_RECORD_REQUIRED_FIELDS
        if missing:
            problems.append(f"missing field(s): {sorted(missing)}")
        if extra:
            problems.append(f"unexpected field(s): {sorted(extra)}")

    for field in ("dataset", "split", "slug", "observation_uuid", "photo_id", "sha256"):
        if fields.get(field) != row.get(field):
            problems.append(f"{field} {fields.get(field)!r} does not match the frozen queue row "
                            f"({row.get(field)!r})")
    if fields.get("queue_index") != row.get("queue_index"):
        problems.append("queue_index does not match the frozen queue row")
    if fields.get("contract_content_sha256") != contract["content_sha256"]:
        problems.append("contract_content_sha256 does not match the currently verified contract")
    if fields.get("queue_byte_sha256") != contract["content"]["queue_artifact"]["byte_sha256"]:
        problems.append("queue_byte_sha256 does not match the currently verified queue")
    if fields.get("queue_identity_order_sha256") != contract["content"]["queue_artifact"]["identity_order_sha256"]:
        problems.append("queue_identity_order_sha256 does not match the currently verified queue")

    if fields.get("review_state") not in mrc.REVIEW_STATES:
        problems.append(f"review_state must be one of {sorted(mrc.REVIEW_STATES)}, got "
                        f"{fields.get('review_state')!r}")
    if fields.get("label_plausibility") not in mrc.LABEL_PLAUSIBILITY_VALUES:
        problems.append(f"label_plausibility must be one of {sorted(mrc.LABEL_PLAUSIBILITY_VALUES)}, "
                        f"got {fields.get('label_plausibility')!r}")
    if fields.get("duplicate_suspicion") not in mrc.DUPLICATE_SUSPICION_VALUES:
        problems.append(f"duplicate_suspicion must be one of {sorted(mrc.DUPLICATE_SUSPICION_VALUES)}, "
                        f"got {fields.get('duplicate_suspicion')!r}")
    if not isinstance(fields.get("notes"), str):
        problems.append("notes must be a string (may be empty)")
    if not isinstance(fields.get("reviewer_id"), str) or not fields.get("reviewer_id"):
        problems.append("reviewer_id must be a non-empty string")
    if not isinstance(fields.get("session_id"), str) or not fields.get("session_id"):
        problems.append("session_id must be a non-empty string")
    if not mrc.is_strict_utc_timestamp(fields.get("reviewed_at_utc")):
        problems.append("reviewed_at_utc must match the strict UTC form 'YYYY-MM-DDTHH:MM:SSZ'")
    return problems


def _session_decision_count(existing: list[dict], session_id: str) -> int:
    return sum(1 for r in existing if r.get("session_id") == session_id)


def build_decision_fields(row: dict, contract: dict, *, review_state: str, label_plausibility: str,
                          duplicate_suspicion: str, notes: str, reviewer_id: str, session_id: str,
                          reviewed_at_utc: str) -> dict:
    """Auto-fills every contract/queue/row-identity binding field from the
    VERIFIED contract and row -- a caller supplies only the actual
    judgment fields, never the binding fields, so a decision can never be
    (even accidentally) recorded with a mismatched identity."""
    return {
        "queue_index": row["queue_index"], "dataset": row["dataset"], "split": row["split"],
        "slug": row["slug"], "observation_uuid": row["observation_uuid"], "photo_id": row["photo_id"],
        "sha256": row["sha256"],
        "contract_content_sha256": contract["content_sha256"],
        "queue_byte_sha256": contract["content"]["queue_artifact"]["byte_sha256"],
        "queue_identity_order_sha256": contract["content"]["queue_artifact"]["identity_order_sha256"],
        "review_state": review_state, "label_plausibility": label_plausibility,
        "duplicate_suspicion": duplicate_suspicion, "notes": notes,
        "reviewer_id": reviewer_id, "session_id": session_id, "reviewed_at_utc": reviewed_at_utc,
    }


def load_verified_ledger(repo: Path, verified: dict) -> list[dict]:
    """THE one semantic verification path for the manual-review ledger --
    used by EVERY working mode, so none of them can trust an existing
    record that is merely hash-chain-consistent. Beyond generic chain
    verification (aol.verify_chain), this: rejects an unknown/extra
    queue_index (not a row in the frozen queue), rejects a duplicate
    queue_index, and re-runs validate_decision_fields() for EVERY existing
    record against its exact frozen queue row and the current verified
    contract -- so a record whose identity-bound field was altered and the
    (unkeyed) hash chain recomputed can never be silently trusted just
    because the chain still replays. Returns records ONLY once every check
    has passed; raises ReviewToolError otherwise. Read-only: never writes,
    truncates, or otherwise mutates the ledger file -- only aol.
    append_record's own interrupted-tail recovery may do that, and only
    when actually appending."""
    ledger_path = ledger_path_for(repo)
    try:
        records = aol.read_ledger(ledger_path)
    except aol.LedgerError as exc:
        _fail(f"manual-review ledger is corrupt: {exc}")
    chain_problems = aol.verify_chain(records, mrc.REVIEW_RECORD_REQUIRED_FIELDS, mrc.review_record_identity)
    if chain_problems:
        _fail(f"manual-review ledger failed chain verification: {chain_problems}")

    queue_rows_by_index = {r["queue_index"]: r for r in verified["queue_rows"]}
    contract = verified["contract"]
    seen_indices: set = set()
    for rec in records:
        idx = rec.get("queue_index")
        if idx in seen_indices:
            _fail(f"manual-review ledger has a duplicate queue_index {idx!r}")
        seen_indices.add(idx)
        row = queue_rows_by_index.get(idx)
        if row is None:
            _fail(f"manual-review ledger references unknown/extra queue_index {idx!r} -- not a "
                  f"valid frozen queue index")
        problems = validate_decision_fields(rec, row, contract)
        if problems:
            _fail(f"manual-review ledger record for queue_index {idx!r} failed semantic "
                  f"re-validation against the frozen queue row/contract: {problems}")
    return records


def record_decision(repo: Path, verified: dict, fields: dict) -> dict:
    """Validates `fields` against the frozen queue/contract and the
    per-session cap, semantically re-verifies every EXISTING ledger record
    (via load_verified_ledger), then appends exactly one new decision.
    Raises ReviewToolError (never writes) on any problem."""
    contract = verified["contract"]
    queue_rows_by_index = {r["queue_index"]: r for r in verified["queue_rows"]}
    row = queue_rows_by_index.get(fields.get("queue_index"))
    if row is None:
        _fail(f"queue_index {fields.get('queue_index')!r} is not a valid frozen queue index")

    problems = validate_decision_fields(fields, row, contract)
    if problems:
        _fail(f"decision rejected: {problems}")

    existing = load_verified_ledger(repo, verified)

    session_id = fields["session_id"]
    if _session_decision_count(existing, session_id) >= mrc.MAX_DECISIONS_PER_SESSION:
        _fail(f"session {session_id!r} has already recorded the maximum "
              f"{mrc.MAX_DECISIONS_PER_SESSION} decisions")

    try:
        return aol.append_record(ledger_path_for(repo), fields, required_fields=mrc.REVIEW_RECORD_REQUIRED_FIELDS,
                                 identity_fn=mrc.review_record_identity)
    except aol.LedgerError as exc:
        _fail(str(exc))


def first_unreviewed_queue_index_from(verified: dict, records: list[dict]) -> int | None:
    reviewed = {r.get("queue_index") for r in records}
    for row in sorted(verified["queue_rows"], key=lambda r: r["queue_index"]):
        if row["queue_index"] not in reviewed:
            return row["queue_index"]
    return None


def review_progress_from(verified: dict, records: list[dict]) -> dict:
    """Pure: derives progress from an ALREADY-verified records snapshot
    (load_verified_ledger's return) -- never a second raw read. By
    construction, `records` only ever reaches here after passing every
    chain and semantic check, so chain_problems is always empty."""
    reviewed_indices = {r.get("queue_index") for r in records}
    return {
        "total_queue_rows": len(verified["queue_rows"]),
        "reviewed_count": len(reviewed_indices),
        "remaining_count": len(verified["queue_rows"]) - len(reviewed_indices),
        "first_unreviewed_queue_index": first_unreviewed_queue_index_from(verified, records),
        "chain_problems": [],
    }


# --------------------------------------------------------------- CLI --
def cmd_preflight(args) -> dict:
    verified = load_verified(args.repo)
    records = load_verified_ledger(args.repo, verified)
    progress = review_progress_from(verified, records)
    return {"ok": True, "contract_content_sha256": verified["contract"]["content_sha256"], **progress}


def cmd_next(args) -> dict:
    verified = load_verified(args.repo)
    records = load_verified_ledger(args.repo, verified)
    idx = first_unreviewed_queue_index_from(verified, records)
    if idx is None:
        return {"done": True}
    row = next(r for r in verified["queue_rows"] if r["queue_index"] == idx)
    image_path = image_path_for_row(args.repo, verified, row)
    return {
        "done": False, "queue_index": row["queue_index"], "suggested_session": row["suggested_session"],
        "dataset": row["dataset"], "slug": row["slug"], "split": row["split"], "species": row["species"],
        "taxon_id": row["taxon_id"], "observation_uuid": row["observation_uuid"], "photo_id": row["photo_id"],
        "sha256": row["sha256"], "image_path": str(image_path),
    }


def cmd_record(args) -> dict:
    verified = load_verified(args.repo)
    row = next((r for r in verified["queue_rows"] if r["queue_index"] == args.queue_index), None)
    if row is None:
        _fail(f"queue_index {args.queue_index!r} is not a valid frozen queue index")
    fields = build_decision_fields(
        row, verified["contract"], review_state=args.review_state,
        label_plausibility=args.label_plausibility, duplicate_suspicion=args.duplicate_suspicion,
        notes=args.notes or "", reviewer_id=args.reviewer_id, session_id=args.session_id,
        reviewed_at_utc=args.reviewed_at_utc)
    record = record_decision(args.repo, verified, fields)
    return {"recorded": True, "queue_index": record["queue_index"], "record_hash": record["record_hash"]}


def cmd_status(args) -> dict:
    verified = load_verified(args.repo)
    records = load_verified_ledger(args.repo, verified)
    return {"contract_content_sha256": verified["contract"]["content_sha256"],
            **review_progress_from(verified, records)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", type=Path, default=HERE.parent)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--next", action="store_true")
    mode.add_argument("--record", action="store_true")
    mode.add_argument("--status", action="store_true")
    ap.add_argument("--queue-index", type=int)
    ap.add_argument("--review-state", choices=sorted(mrc.REVIEW_STATES))
    ap.add_argument("--label-plausibility", choices=sorted(mrc.LABEL_PLAUSIBILITY_VALUES))
    ap.add_argument("--duplicate-suspicion", choices=sorted(mrc.DUPLICATE_SUSPICION_VALUES))
    ap.add_argument("--notes", default="")
    ap.add_argument("--reviewer-id")
    ap.add_argument("--session-id")
    ap.add_argument("--reviewed-at-utc")
    args = ap.parse_args()

    try:
        if args.preflight:
            result = cmd_preflight(args)
        elif args.next:
            result = cmd_next(args)
        elif args.record:
            required = ("queue_index", "review_state", "label_plausibility", "duplicate_suspicion",
                       "reviewer_id", "session_id", "reviewed_at_utc")
            missing = [f"--{f.replace('_', '-')}" for f in required if getattr(args, f) is None]
            if missing:
                print(f"[manual_review_tool] FAILURE: --record requires: {missing}")
                return 1
            result = cmd_record(args)
        else:
            result = cmd_status(args)
        print(json.dumps(result, indent=2, default=str))
        return 0
    except (ReviewToolError, mrc.ContractError) as e:
        print(f"[manual_review_tool] FAILURE: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
