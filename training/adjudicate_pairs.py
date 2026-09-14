#!/usr/bin/env python3
"""adjudicate_pairs.py — manual, append-only, hash-chained adjudication of
perceptual-duplicate candidate pairs produced by
scan_perceptual_duplicates.py --scan.

Every adjudication record binds the exact candidate row it was made
against (identity_a, identity_b, domain, phash_distance, dhash_distance),
plus the candidate-pairs-report file's own byte hash and the scan report's
content_sha256 -- not merely a pair_id -- so a record cannot be replayed
against a substituted candidate-pairs report that reuses the same pair_id
for a different underlying pair or different measured distances.

CLI modes (every one of them goes through
scan_perceptual_duplicates.load_and_verify_full_scan_state -- the full
shared contract + all-five-reports + population re-derivation check, never
a candidate-report-only envelope check):
  --preflight  loads/verifies the contract and the completed candidate-
               pairs report; reports progress. No image access.
  --next       reports the next unadjudicated candidate pair.
  --record     appends one validated adjudication.
  --status     re-verifies the full ledger chain and reports progress.

Never imports onnxruntime or any inference/model module. --next/--record
are NOT run in Phase 5F1 (there is no completed candidate-pairs report to
adjudicate yet).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import manual_review_contract as mrc  # noqa: E402
import append_only_ledger as aol  # noqa: E402
import scan_perceptual_duplicates as spd  # noqa: E402

DEFAULT_LEDGER_REL_PATH = mrc.APPROVED_OUTPUT_PATHS["pair_adjudication_ledger"]


class AdjudicationError(RuntimeError):
    """A validation or ledger problem. Always fails closed."""


def _fail(msg: str) -> None:
    raise AdjudicationError(msg)


def ledger_path_for(repo: Path) -> Path:
    return Path(repo) / DEFAULT_LEDGER_REL_PATH


def load_verified_state(repo: Path) -> dict:
    """THE required first step for every mode here: the full shared
    scan-state verification (spd.load_and_verify_full_scan_state) --
    contract, all five scan reports, schema/binding, AND genuine
    re-derivation against the real frozen manifests. NEVER a
    candidate-report-only envelope check: a mutate-and-rehash candidate
    report can pass its own local schema/binding checks while disagreeing
    with what the frozen population and reported hashes actually derive
    to -- only the shared loader's population re-derivation catches that.
    Adds the candidate-pairs report's own raw file bytes (needed for the
    byte-hash binding every adjudication record carries)."""
    state = spd.load_and_verify_full_scan_state(repo)
    cp_path = Path(repo) / mrc.APPROVED_OUTPUT_PATHS["candidate_pairs_report"]
    state["candidate_pairs_report_bytes"] = cp_path.read_bytes()
    return state


def candidates_by_pair_id(report: dict) -> dict[str, dict]:
    """Flattens the per-domain candidate-pairs report into pair_id ->
    {identity_a, identity_b, domain, phash_distance, dhash_distance}."""
    flattened: dict[str, dict] = {}
    for domain_name, pairs in report["content"]["domains"].items():
        for pair in pairs:
            flattened[pair["pair_id"]] = {**pair, "domain": domain_name}
    return flattened


def validate_adjudication_fields(fields: dict, candidate_row: dict, report_sha256: str,
                                 scan_report_content_sha256: str) -> list[str]:
    """Structural + enum validation, entirely via .get() -- never raises.
    Verifies identity_a/identity_b/domain/distances match the frozen
    candidate row, not merely that pair_id is present."""
    problems: list[str] = []
    actual_fields = set(fields) | {"prev_record_hash", "record_hash"}
    if actual_fields != mrc.ADJUDICATION_RECORD_REQUIRED_FIELDS:
        missing = mrc.ADJUDICATION_RECORD_REQUIRED_FIELDS - actual_fields
        extra = actual_fields - mrc.ADJUDICATION_RECORD_REQUIRED_FIELDS
        if missing:
            problems.append(f"missing field(s): {sorted(missing)}")
        if extra:
            problems.append(f"unexpected field(s): {sorted(extra)}")

    for field in ("identity_a", "identity_b", "domain", "phash_distance", "dhash_distance"):
        if fields.get(field) != candidate_row.get(field):
            problems.append(f"{field} {fields.get(field)!r} does not match the frozen candidate "
                            f"row ({candidate_row.get(field)!r})")
    if fields.get("candidate_pairs_report_sha256") != report_sha256:
        problems.append("candidate_pairs_report_sha256 does not match the currently verified report")
    if fields.get("scan_report_content_sha256") != scan_report_content_sha256:
        problems.append("scan_report_content_sha256 does not match the currently verified report")

    if fields.get("label") not in mrc.ADJUDICATION_LABELS:
        problems.append(f"label must be one of {sorted(mrc.ADJUDICATION_LABELS)}, got {fields.get('label')!r}")
    if not isinstance(fields.get("notes"), str):
        problems.append("notes must be a string (may be empty)")
    if not isinstance(fields.get("reviewer_id"), str) or not fields.get("reviewer_id"):
        problems.append("reviewer_id must be a non-empty string")
    if not isinstance(fields.get("session_id"), str) or not fields.get("session_id"):
        problems.append("session_id must be a non-empty string")
    if not mrc.is_strict_utc_timestamp(fields.get("reviewed_at_utc")):
        problems.append("reviewed_at_utc must match the strict UTC form 'YYYY-MM-DDTHH:MM:SSZ'")
    return problems


def build_adjudication_fields(candidate_row: dict, report_sha256: str, scan_report_content_sha256: str, *,
                              label: str, reviewer_id: str, session_id: str, reviewed_at_utc: str,
                              notes: str = "") -> dict:
    return {
        "pair_id": candidate_row["pair_id"], "identity_a": candidate_row["identity_a"],
        "identity_b": candidate_row["identity_b"], "domain": candidate_row["domain"],
        "phash_distance": candidate_row["phash_distance"], "dhash_distance": candidate_row["dhash_distance"],
        "candidate_pairs_report_sha256": report_sha256, "scan_report_content_sha256": scan_report_content_sha256,
        "label": label, "reviewer_id": reviewer_id, "session_id": session_id,
        "reviewed_at_utc": reviewed_at_utc, "notes": notes,
    }


def load_verified_ledger(repo: Path, state: dict, *, require_canonical_tail: bool = False) -> list[dict]:
    """THE one semantic verification path for the pair-adjudication
    ledger -- used by EVERY working mode here, AND (via
    require_canonical_tail=True) by finalize_stop_status.py, so there is
    only ever one copy of this logic. Beyond generic chain verification
    (aol.verify_chain): rejects an unknown/extra pair_id (not in the fully
    verified candidate report), rejects a duplicate pair_id, and re-runs
    validate_adjudication_fields() for EVERY existing record against its
    exact candidate row and both report bindings -- so a record whose
    identity-bound field was altered and the (unkeyed) hash chain
    recomputed can never be silently trusted just because the chain still
    replays. Returns records ONLY once every check has passed. Read-only:
    never writes, truncates, or otherwise mutates the ledger file.

    `require_canonical_tail=True` additionally requires the raw ledger
    bytes end with a newline and that read_ledger()'s interrupted-write
    tolerance did not silently drop a trailing physical line -- this is
    the STRONGER guarantee finalize_stop_status.py needs before publishing
    final evidence. The read-only working modes here (--preflight/--next/
    --status) deliberately do NOT require this: a genuinely interrupted
    append must remain recoverable by the next authorized --record, not
    become a hard failure just because someone ran --preflight first."""
    ledger_path = ledger_path_for(repo)
    ledger_bytes = ledger_path.read_bytes() if ledger_path.exists() else b""
    try:
        records = aol.read_ledger(ledger_path)
    except aol.LedgerError as exc:
        _fail(f"pair-adjudication ledger is corrupt: {exc}")
    chain_problems = aol.verify_chain(records, mrc.ADJUDICATION_RECORD_REQUIRED_FIELDS,
                                      mrc.adjudication_record_identity)
    if chain_problems:
        _fail(f"pair-adjudication ledger failed chain verification: {chain_problems}")

    candidates = candidates_by_pair_id(state["reports"]["candidate_pairs_report"])
    report_sha256 = mrc.sha256_bytes(state["candidate_pairs_report_bytes"])
    scan_report_content_sha256 = state["reports"]["candidate_pairs_report"]["content_sha256"]

    seen_ids: set = set()
    for rec in records:
        pair_id = rec.get("pair_id")
        if pair_id in seen_ids:
            _fail(f"pair-adjudication ledger has a duplicate pair_id {pair_id!r}")
        seen_ids.add(pair_id)
        candidate_row = candidates.get(pair_id)
        if candidate_row is None:
            _fail(f"pair-adjudication ledger references unknown/extra pair_id {pair_id!r} -- not "
                  f"in the verified candidate report")
        problems = validate_adjudication_fields(rec, candidate_row, report_sha256, scan_report_content_sha256)
        if problems:
            _fail(f"pair-adjudication ledger record {pair_id!r} failed semantic re-validation "
                  f"against the verified candidate row/report bindings: {problems}")

    if require_canonical_tail:
        if ledger_bytes and not ledger_bytes.endswith(b"\n"):
            _fail("pair-adjudication ledger does not end with a newline -- a trailing incomplete "
                  "fragment is never acceptable before final evidence publication, even one that "
                  "would otherwise be tolerated as an interrupted write")
        raw_lines = [line for line in ledger_bytes.split(b"\n") if line.strip()]
        if len(raw_lines) != len(records):
            _fail(f"pair-adjudication ledger has {len(raw_lines)} physical line(s) but only "
                  f"{len(records)} parsed as records -- a trailing fragment was silently dropped")

    return records


def record_adjudication(repo: Path, fields: dict) -> dict:
    state = load_verified_state(repo)
    report = state["reports"]["candidate_pairs_report"]
    candidates = candidates_by_pair_id(report)
    candidate_row = candidates.get(fields.get("pair_id"))
    if candidate_row is None:
        _fail(f"pair_id {fields.get('pair_id')!r} is not a known candidate pair")

    report_sha256 = mrc.sha256_bytes(state["candidate_pairs_report_bytes"])
    problems = validate_adjudication_fields(fields, candidate_row, report_sha256, report["content_sha256"])
    if problems:
        _fail(f"adjudication rejected: {problems}")

    load_verified_ledger(repo, state)  # semantically re-verify every existing record first

    try:
        return aol.append_record(ledger_path_for(repo), fields, required_fields=mrc.ADJUDICATION_RECORD_REQUIRED_FIELDS,
                                 identity_fn=mrc.adjudication_record_identity)
    except aol.LedgerError as exc:
        _fail(str(exc))


def confirmed_same_source_pairs(repo: Path, state: dict | None = None) -> list[dict]:
    state = state or load_verified_state(repo)
    records = load_verified_ledger(repo, state)
    return [r for r in records if r.get("label") == "same_source_image"]


def first_unadjudicated_pair_id_from(candidates: dict, records: list[dict]) -> str | None:
    adjudicated = {r.get("pair_id") for r in records}
    for pair_id in sorted(candidates):
        if pair_id not in adjudicated:
            return pair_id
    return None


# --------------------------------------------------------------- CLI --
def cmd_preflight(args) -> dict:
    state = load_verified_state(args.repo)
    records = load_verified_ledger(args.repo, state)
    candidates = candidates_by_pair_id(state["reports"]["candidate_pairs_report"])
    return {"ok": True, "total_candidates": len(candidates), "adjudicated_count": len(records),
            "remaining_count": len(candidates) - len(records), "chain_problems": []}


def cmd_next(args) -> dict:
    state = load_verified_state(args.repo)
    records = load_verified_ledger(args.repo, state)
    candidates = candidates_by_pair_id(state["reports"]["candidate_pairs_report"])
    pair_id = first_unadjudicated_pair_id_from(candidates, records)
    if pair_id is None:
        return {"done": True}
    return {"done": False, **candidates[pair_id]}


def cmd_record(args) -> dict:
    state = load_verified_state(args.repo)
    report = state["reports"]["candidate_pairs_report"]
    candidates = candidates_by_pair_id(report)
    candidate_row = candidates.get(args.pair_id)
    if candidate_row is None:
        _fail(f"pair_id {args.pair_id!r} is not a known candidate pair")
    fields = build_adjudication_fields(
        candidate_row, mrc.sha256_bytes(state["candidate_pairs_report_bytes"]), report["content_sha256"],
        label=args.label, reviewer_id=args.reviewer_id, session_id=args.session_id,
        reviewed_at_utc=args.reviewed_at_utc, notes=args.notes or "")
    record = record_adjudication(args.repo, fields)
    return {"recorded": True, "pair_id": record["pair_id"], "record_hash": record["record_hash"]}


def cmd_status(args) -> dict:
    return cmd_preflight(args)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", type=Path, default=HERE.parent)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--next", action="store_true")
    mode.add_argument("--record", action="store_true")
    mode.add_argument("--status", action="store_true")
    ap.add_argument("--pair-id")
    ap.add_argument("--label", choices=sorted(mrc.ADJUDICATION_LABELS))
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
            required = ("pair_id", "label", "reviewer_id", "session_id", "reviewed_at_utc")
            missing = [f"--{f.replace('_', '-')}" for f in required if getattr(args, f) is None]
            if missing:
                print(f"[adjudicate_pairs] FAILURE: --record requires: {missing}")
                return 1
            result = cmd_record(args)
        else:
            result = cmd_status(args)
        print(json.dumps(result, indent=2, default=str))
        return 0
    except (AdjudicationError, mrc.ContractError, spd.ScanError) as e:
        print(f"[adjudicate_pairs] FAILURE: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
