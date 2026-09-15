#!/usr/bin/env python3
"""adjudicate_pairs_v2.py — manual, append-only, hash-chained adjudication
of the frozen Phase 5F4 v2 queue (training/pair_adjudication_v2_queue.json),
built on top of the shared, immutable Phase 5F1/5F3 scan evidence.

Unlike Phase 5F1's adjudicate_pairs.py (which offers every candidate for
adjudication in pair_id order with no workstream concept), this module
enforces v2's three-independent-workstream, early-stop-aware ordering:
only the CURRENT next eligible queue row may ever be recorded -- no
skipping, no duplicate record, no alternate pair, no wrong session/block.
"Eligible" excludes any row belonging to a blocking workstream
(final_test_independence / gate_evidence_independence) that has already
recorded a blocking event (same_source_image or uncertain) elsewhere in
that SAME workstream; the two blocking workstreams stop independently of
each other, and the diagnostic workstream (final_test_internal_repetition)
never stops early -- its 53 rows are always eligible until each is
adjudicated.

NEW versus v1 (required because concurrent-terminal writes to the same
ledger were previously a real risk): every --record acquires a
structurally exclusive per-write lock (atomic O_CREAT|O_EXCL file
creation) before touching the ledger. A pre-existing lock is NEVER
silently removed or overridden -- it is reported for review. Inside the
lock: the ledger is reverified, the record is appended and fsynced, then
reloaded and fully reverified, and only the lock this invocation itself
created is ever released.

CLI modes (every one goes through load_verified_v2_state -- contract +
queue + all five scan reports, hash-bound, never a partial check):
  --preflight  loads/verifies everything; reports progress and the
               currently eligible next pair (if any). No image access.
  --next       reports the next eligible queue row, or {"done": true} if
               every workstream is either exhausted or blocking-stopped.
  --record     appends one validated adjudication for the CURRENT next
               eligible pair only.
  --status     re-verifies the full ledger chain and reports progress,
               including per-workstream status/stop flags.

Never imports onnxruntime, PIL, or any inference/image module. This
module never opens an image.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

import pair_adjudication_v2_contract as v2c  # noqa: E402
import append_only_ledger as aol  # noqa: E402

LEDGER_REL_PATH = v2c.APPROVED_OUTPUT_PATHS["pair_adjudication_v2_ledger"]
LOCK_REL_PATH = v2c.APPROVED_OUTPUT_PATHS["pair_adjudication_v2_lock"]

GENERATOR_SOURCE_REL_PATH = "training/adjudicate_pairs_v2.py"
GENERATOR_PROTOCOL_VERSION = v2c.GENERATOR_PROTOCOL_VERSION

BLOCKING_EVENT_LABELS = frozenset({"same_source_image", "uncertain"})


class AdjudicationV2Error(RuntimeError):
    """A validation, lock, or ledger problem. Always fails closed."""


def _fail(msg: str) -> None:
    raise AdjudicationV2Error(msg)


def ledger_path_for(repo: Path) -> Path:
    return Path(repo) / LEDGER_REL_PATH


def lock_path_for(repo: Path) -> Path:
    return Path(repo) / LOCK_REL_PATH


# ------------------------------------------------------------ locking --
@contextmanager
def exclusive_lock(repo: Path):
    """Acquires a structurally exclusive lock via atomic O_CREAT|O_EXCL
    file creation -- fails closed (AdjudicationV2Error) if the lock file
    already exists, NEVER silently removing or overriding it. On exit,
    releases only the lock THIS invocation created: re-reads the lock
    file's content and only unlinks it if it still holds the exact token
    this invocation wrote, so a concurrently-recovered/replaced lock file
    (which should be structurally impossible given O_CREAT|O_EXCL, but is
    checked anyway as defense in depth) is never deleted out from under
    another owner."""
    path = lock_path_for(repo)
    path.parent.mkdir(parents=True, exist_ok=True)
    token = f"{os.getpid()}:{uuid.uuid4().hex}"
    try:
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        _fail(f"lock file already exists at {path} -- a pre-existing or stale lock is NEVER silently "
              f"removed or overridden; investigate whether another adjudicate_pairs_v2.py invocation is "
              f"genuinely in progress, then remove the lock file manually only once you are certain it is "
              f"stale")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(token)
            fh.flush()
            os.fsync(fh.fileno())
        yield token
    finally:
        try:
            current = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            current = None
        if current == token:
            path.unlink()
        elif current is not None:
            _fail(f"lock file at {path} no longer holds the token this invocation created -- refusing "
                  f"to delete a lock this invocation may not own; manual review required")


# ------------------------------------------------------------ state load --
def load_verified_v2_state(repo: Path) -> dict:
    """THE required first step for every mode here: loads and fully
    verifies the frozen v2 contract, the frozen v2 queue (hash-bound to
    the contract), and the five committed scan reports (hash-bound to the
    contract) -- never a partial check. Returns
    {"contract": ..., "queue_rows": [...], "queue_by_pair_id": {...},
    "queue_bytes": bytes}."""
    repo = Path(repo)
    contract_path = repo / v2c.CONTRACT_REL_PATH
    if not contract_path.exists():
        _fail(f"v2 contract does not exist yet -- run freeze_pair_adjudication_v2_contract.py "
              f"--write first: {contract_path}")
    contract_bytes = contract_path.read_bytes()
    try:
        contract = json.loads(contract_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        _fail(f"{contract_path} is not valid UTF-8 JSON: {exc}")
    problems = v2c.validate_contract_structure(contract)
    if problems:
        _fail(f"{contract_path} failed structural validation: {problems}")

    # Beyond structure: the contract's implementation-source provenance
    # must be CURRENTLY valid (ancestor commit, source exists with the
    # recorded hash at that commit, and the working tree still matches)
    # -- never merely well-formed. This is what item 2 of the correction
    # requires: "load_verified_v2_state() must verify the contract's
    # complete implementation-source provenance, not only its structure."
    provenance_problems = v2c.verify_implementation_source_provenance(repo, contract["content"])
    if provenance_problems:
        _fail(f"{contract_path} implementation-source provenance failed verification: {provenance_problems}")

    try:
        scan_reports = v2c.load_and_verify_scan_reports(repo)
    except v2c.ScanEvidenceError as exc:
        _fail(str(exc))

    queue_artifact = contract["content"]["queue_artifact"]
    queue_path = repo / v2c.QUEUE_ARTIFACT_REL_PATH
    if not queue_path.exists():
        _fail(f"v2 queue does not exist yet -- run build_pair_adjudication_v2_queue.py "
              f"--write first: {queue_path}")
    queue_bytes = queue_path.read_bytes()
    if v2c.sha256_bytes(queue_bytes) != queue_artifact["byte_sha256"]:
        _fail(f"{queue_path} byte sha256 does not match the frozen contract's queue_artifact binding")
    try:
        queue = json.loads(queue_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        _fail(f"{queue_path} is not valid UTF-8 JSON: {exc}")
    if queue.get("row_count") != queue_artifact["row_count"]:
        _fail(f"{queue_path} row_count does not match the frozen contract's queue_artifact binding")
    if queue.get("identity_order_sha256") != queue_artifact["identity_order_sha256"]:
        _fail(f"{queue_path} identity_order_sha256 does not match the frozen contract's queue_artifact binding")
    rows = queue.get("rows")
    if not isinstance(rows, list) or len(rows) != queue_artifact["row_count"]:
        _fail(f"{queue_path} rows do not match the bound row_count")
    for row in rows:
        actual = set(row) if isinstance(row, dict) else set()
        if actual != v2c.QUEUE_ROW_REQUIRED_FIELDS:
            _fail(f"{queue_path} row has an invalid field set: {row!r}")
    recomputed_identity_hash = v2c.compute_v2_queue_identity_order_sha256(rows)
    if recomputed_identity_hash != queue_artifact["identity_order_sha256"]:
        _fail(f"{queue_path} rows do not recompute to the bound identity_order_sha256 -- the queue "
              f"has been reordered or altered since it was frozen")

    # Beyond internal self-consistency (the queue's own bytes/rows hash to
    # what the contract recorded): REDERIVE the canonical queue from the
    # verified committed candidate_pairs_report and require it to match
    # the loaded queue exactly, row for row -- a queue.json and
    # contract.json that were jointly modified to agree with EACH OTHER
    # (recomputed hashes and all) but no longer agree with what the real
    # candidate report actually produces must still be rejected here.
    candidate_pairs_content = scan_reports["candidate_pairs_report"]["content"]
    try:
        rederived_rows = v2c.derive_v2_queue_rows(candidate_pairs_content)
    except ValueError as exc:
        _fail(f"could not rederive the v2 queue from the verified candidate_pairs_report: {exc}")
    if rederived_rows != rows:
        _fail(f"{queue_path} does not match the queue mechanically rederived from the verified committed "
              f"candidate_pairs_report -- the published queue and/or contract have been altered")

    # Same defense for the excluded (outside-scope) population: a jointly
    # modified queue+contract pair could otherwise smuggle a scoped
    # candidate out of the mandatory queue by simply relabeling it
    # "outside scope" while still recomputing every hash consistently.
    outside_scope_ids = v2c.derive_outside_scope_pair_ids(candidate_pairs_content)
    if len(outside_scope_ids) != v2c.EXPECTED_OUTSIDE_SCOPE_TOTAL:
        _fail(f"rederived {len(outside_scope_ids)} outside-scope pair_ids, expected exactly "
              f"{v2c.EXPECTED_OUTSIDE_SCOPE_TOTAL}")
    recomputed_outside_scope_hash = v2c.compute_outside_scope_identity_order_sha256(outside_scope_ids)
    if recomputed_outside_scope_hash != contract["content"]["outside_scope_identity_order_sha256"]:
        _fail("the excluded (outside-scope) candidate population rederived from the verified candidate_"
              "pairs_report does not match the frozen contract's outside_scope_identity_order_sha256 -- "
              "the contract and/or the real candidate population have diverged")

    queue_by_pair_id = {r["pair_id"]: r for r in rows}
    if len(queue_by_pair_id) != len(rows):
        _fail(f"{queue_path} contains a duplicate pair_id")

    return {
        "contract": contract, "contract_bytes": contract_bytes,
        "queue_rows": rows, "queue_by_pair_id": queue_by_pair_id, "queue_bytes": queue_bytes,
        "outside_scope_pair_ids": outside_scope_ids,
        "candidate_pairs_report_byte_sha256": contract["content"]["scan_report_hashes"]["candidate_pairs_report"]["byte_sha256"],
        "candidate_pairs_report_content_sha256": contract["content"]["scan_report_hashes"]["candidate_pairs_report"]["content_sha256"],
    }


# ------------------------------------------------------- eligibility --
def workstream_stop_state(queue_rows: list[dict], records: list[dict]) -> dict[str, dict]:
    """Per workstream: {"stopped": bool, "stop_reason": str | None,
    "stop_pair_id": str | None}. A blocking workstream
    (final_test_independence / gate_evidence_independence) stops the
    moment ANY of its own rows records same_source_image or uncertain --
    independently of the other blocking workstream. The diagnostic
    workstream (final_test_internal_repetition) never stops early."""
    by_pair_id = {r["pair_id"]: r for r in queue_rows}
    state = {ws: {"stopped": False, "stop_reason": None, "stop_pair_id": None}
             for ws in (v2c.WORKSTREAM_FINAL_TEST_INDEPENDENCE, v2c.WORKSTREAM_FINAL_TEST_INTERNAL_REPETITION,
                       v2c.WORKSTREAM_GATE_EVIDENCE_INDEPENDENCE)}
    for rec in records:
        row = by_pair_id.get(rec.get("pair_id"))
        if row is None:
            continue
        ws = row["workstream"]
        if v2c.WORKSTREAM_ROLE[ws] != v2c.CHANNEL_ROLE_BLOCKING:
            continue
        if state[ws]["stopped"]:
            continue
        if rec.get("label") in BLOCKING_EVENT_LABELS:
            state[ws] = {"stopped": True, "stop_reason": rec.get("label"), "stop_pair_id": rec.get("pair_id")}
    return state


def first_eligible_row(queue_rows: list[dict], records: list[dict]) -> dict | None:
    """The single next row that --record is permitted to accept, in
    queue_index order: skips any pair_id already adjudicated, and skips
    (never returns) any row in a blocking workstream that has already
    stopped -- those rows are reported by finalize_stop_status_v2.py as
    not_adjudicated_due_to_early_stop, never offered here."""
    adjudicated = {r.get("pair_id") for r in records}
    stop_state = workstream_stop_state(queue_rows, records)
    for row in sorted(queue_rows, key=lambda r: r["queue_index"]):
        if row["pair_id"] in adjudicated:
            continue
        if stop_state[row["workstream"]]["stopped"]:
            continue
        return row
    return None


def early_stop_skipped_pair_ids(queue_rows: list[dict], records: list[dict]) -> set[str]:
    adjudicated = {r.get("pair_id") for r in records}
    stop_state = workstream_stop_state(queue_rows, records)
    return {row["pair_id"] for row in queue_rows
            if row["pair_id"] not in adjudicated and stop_state[row["workstream"]]["stopped"]}


# --------------------------------------------------------- ledger I/O --
def load_verified_ledger_v2(repo: Path, state: dict, *, require_canonical_tail: bool = False) -> list[dict]:
    """THE one semantic verification path for the v2 ledger -- used by
    EVERY working mode here, and (via require_canonical_tail=True) by
    finalize_stop_status_v2.py. Beyond generic chain verification
    (aol.verify_chain): rejects an unknown/extra pair_id (not in the
    frozen queue), rejects a duplicate pair_id (aol.verify_chain already
    does via identity_fn, this is defense in depth), and re-checks every
    existing record's identity-bound fields against its exact frozen
    queue row and the bound contract/queue/report hashes -- so a record
    whose bound field was altered and the (unkeyed) hash chain
    recomputed can never be silently trusted just because the chain still
    replays. Read-only."""
    ledger_path = ledger_path_for(repo)
    ledger_bytes = ledger_path.read_bytes() if ledger_path.exists() else b""
    try:
        records = aol.read_ledger(ledger_path)
    except aol.LedgerError as exc:
        _fail(f"v2 ledger is corrupt: {exc}")
    chain_problems = aol.verify_chain(records, v2c.ADJUDICATION_RECORD_REQUIRED_FIELDS, v2c.adjudication_record_identity)
    if chain_problems:
        _fail(f"v2 ledger failed chain verification: {chain_problems}")

    queue_by_pair_id = state["queue_by_pair_id"]
    for rec in records:
        row = queue_by_pair_id.get(rec.get("pair_id"))
        if row is None:
            _fail(f"v2 ledger references unknown/extra pair_id {rec.get('pair_id')!r} -- not in the frozen queue")
        problems = validate_record_bindings(rec, row, state)
        if problems:
            _fail(f"v2 ledger record {rec.get('pair_id')!r} failed semantic re-validation "
                  f"against the frozen queue row/contract bindings: {problems}")

    if require_canonical_tail:
        if ledger_bytes and not ledger_bytes.endswith(b"\n"):
            _fail("v2 ledger does not end with a newline -- a trailing incomplete fragment is never "
                  "acceptable before final evidence publication")
        raw_lines = [line for line in ledger_bytes.split(b"\n") if line.strip()]
        if len(raw_lines) != len(records):
            _fail(f"v2 ledger has {len(raw_lines)} physical line(s) but only {len(records)} parsed "
                  f"as records -- a trailing fragment was silently dropped")

    return records


def validate_record_bindings(fields: dict, row: dict, state: dict) -> list[str]:
    """Structural + enum validation via .get() only -- never raises.
    Verifies every identity/domain/distance/session field matches the
    frozen queue row, not merely that pair_id is present, and that every
    hash-binding field matches the currently verified contract/queue/
    candidate-report bindings."""
    problems: list[str] = []
    # pre-append fields lack prev_record_hash/record_hash (added by
    # aol.append_record); a stored ledger record already has them, so the
    # union is a no-op there -- this lets one function validate both.
    actual_fields = set(fields) | {"prev_record_hash", "record_hash"}
    if actual_fields != v2c.ADJUDICATION_RECORD_REQUIRED_FIELDS:
        missing = v2c.ADJUDICATION_RECORD_REQUIRED_FIELDS - actual_fields
        extra = actual_fields - v2c.ADJUDICATION_RECORD_REQUIRED_FIELDS
        if missing:
            problems.append(f"missing field(s): {sorted(missing)}")
        if extra:
            problems.append(f"unexpected field(s): {sorted(extra)}")
        return problems  # further per-field checks assume the key set is right

    for field in ("queue_index", "domain", "workstream", "channel_role", "priority_tier", "both_rule_match",
                 "identity_a", "identity_b", "phash_distance", "dhash_distance",
                 "session_block", "suggested_session"):
        if fields.get(field) != row.get(field):
            problems.append(f"{field} {fields.get(field)!r} does not match the frozen queue row ({row.get(field)!r})")

    contract_content_sha256 = state["contract"]["content_sha256"]
    if fields.get("v2_contract_content_sha256") != contract_content_sha256:
        problems.append("v2_contract_content_sha256 does not match the currently verified contract")
    if fields.get("v2_queue_byte_sha256") != v2c.sha256_bytes(state["queue_bytes"]):
        problems.append("v2_queue_byte_sha256 does not match the currently verified queue")
    if fields.get("v2_queue_identity_order_sha256") != state["contract"]["content"]["queue_artifact"]["identity_order_sha256"]:
        problems.append("v2_queue_identity_order_sha256 does not match the currently verified queue")
    if fields.get("candidate_pairs_report_byte_sha256") != state["candidate_pairs_report_byte_sha256"]:
        problems.append("candidate_pairs_report_byte_sha256 does not match the currently verified report")
    if fields.get("candidate_pairs_report_content_sha256") != state["candidate_pairs_report_content_sha256"]:
        problems.append("candidate_pairs_report_content_sha256 does not match the currently verified report")

    if fields.get("label") not in v2c.ADJUDICATION_LABELS:
        problems.append(f"label must be one of {sorted(v2c.ADJUDICATION_LABELS)}, got {fields.get('label')!r}")
    if not isinstance(fields.get("notes"), str):
        problems.append("notes must be a string (may be empty)")
    if not isinstance(fields.get("reviewer_id"), str) or not fields.get("reviewer_id"):
        problems.append("reviewer_id must be a non-empty string")
    # session_id is NEVER an arbitrary caller-supplied value: it is
    # mechanically bound to the frozen queue row's own suggested_session
    # -- a record whose session_id was ANY other well-formed nonempty
    # string, including a plausible-looking session name, is rejected.
    if fields.get("session_id") != row.get("suggested_session"):
        problems.append(f"session_id {fields.get('session_id')!r} does not match the frozen queue row's "
                        f"suggested_session {row.get('suggested_session')!r} -- a record's session "
                        f"identifier is mechanically bound to the frozen queue block, never an arbitrary "
                        f"caller-supplied value")
    if not v2c.is_strict_utc_timestamp(fields.get("reviewed_at_utc")):
        problems.append("reviewed_at_utc must match the strict UTC form 'YYYY-MM-DDTHH:MM:SSZ'")

    # Generator provenance is bound to the contract's OWN recorded
    # adjudicate_pairs_v2 implementation-source entry -- never an
    # independently computed current-HEAD/current-working-tree value. A
    # well-formed but merely PLAUSIBLE commit/hash (e.g. forged, or from
    # a different real commit) that does not exactly match the contract's
    # binding is rejected, even though it individually passes the hex-
    # format checks.
    adj_entry = (state.get("contract", {}).get("content", {}).get("implementation_sources", {})
                .get("adjudicate_pairs_v2"))
    if not isinstance(adj_entry, dict):
        problems.append("the currently verified contract has no adjudicate_pairs_v2 implementation-source "
                        "entry to bind generator provenance to")
    else:
        if fields.get("generator_source_path") != adj_entry.get("path"):
            problems.append(f"generator_source_path must be exactly {adj_entry.get('path')!r} (the "
                            f"contract-approved path), got {fields.get('generator_source_path')!r}")
        if fields.get("generator_source_sha256") != adj_entry.get("sha256"):
            problems.append("generator_source_sha256 does not match the currently verified contract's "
                            "own adjudicate_pairs_v2 provenance binding")
        if fields.get("generator_git_commit") != adj_entry.get("generator_git_commit"):
            problems.append("generator_git_commit does not match the currently verified contract's own "
                            "adjudicate_pairs_v2 provenance binding")
    if fields.get("generator_protocol_version") != GENERATOR_PROTOCOL_VERSION:
        problems.append(f"generator_protocol_version must be exactly {GENERATOR_PROTOCOL_VERSION!r}")

    return problems


def build_fields(repo: Path, row: dict, state: dict, *, label: str, reviewer_id: str, session_id: str,
                 reviewed_at_utc: str, notes: str = "") -> dict:
    """`session_id` MUST already equal `row["suggested_session"]` --
    callers (record_adjudication) enforce this before calling; it is not
    re-derived here so that an explicit --session-id mismatch surfaces as
    a clear rejection rather than being silently overwritten. Generator
    provenance is read directly from the currently verified contract's
    own adjudicate_pairs_v2 implementation-source entry -- never computed
    live from the current git HEAD or the current working-tree file, so
    a record can never bind to a plausible-but-uncontracted value."""
    adj_entry = state["contract"]["content"]["implementation_sources"]["adjudicate_pairs_v2"]
    return {
        "queue_index": row["queue_index"], "pair_id": row["pair_id"],
        "identity_a": row["identity_a"], "identity_b": row["identity_b"], "domain": row["domain"],
        "workstream": row["workstream"], "channel_role": row["channel_role"],
        "priority_tier": row["priority_tier"], "both_rule_match": row["both_rule_match"],
        "phash_distance": row["phash_distance"], "dhash_distance": row["dhash_distance"],
        "session_block": row["session_block"], "suggested_session": row["suggested_session"],
        "v2_contract_content_sha256": state["contract"]["content_sha256"],
        "v2_queue_byte_sha256": v2c.sha256_bytes(state["queue_bytes"]),
        "v2_queue_identity_order_sha256": state["contract"]["content"]["queue_artifact"]["identity_order_sha256"],
        "candidate_pairs_report_byte_sha256": state["candidate_pairs_report_byte_sha256"],
        "candidate_pairs_report_content_sha256": state["candidate_pairs_report_content_sha256"],
        "label": label, "notes": notes, "reviewer_id": reviewer_id, "session_id": session_id,
        "reviewed_at_utc": reviewed_at_utc,
        "generator_git_commit": adj_entry["generator_git_commit"],
        "generator_source_path": adj_entry["path"],
        "generator_source_sha256": adj_entry["sha256"],
        "generator_protocol_version": GENERATOR_PROTOCOL_VERSION,
    }


def _require_clean_tracked_tree(repo: Path) -> None:
    """Requires the TRACKED working tree to be clean before any real
    --record (mirrors the established gate_v2_contract.
    require_clean_tracked_tree / correct_manual_review pattern).
    Untracked files are explicitly allowed -- only drift in already-
    tracked content (which would include an uncommitted edit to
    adjudicate_pairs_v2.py itself, already caught separately by
    verify_implementation_source_provenance, but checked again here as a
    direct precondition of the write path) is a problem."""
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=str(repo), stderr=subprocess.PIPE,
        ).decode("utf-8")
    except (subprocess.CalledProcessError, OSError) as exc:
        _fail(f"could not check git status in {repo}: {exc}")
    if out.strip():
        _fail(f"tracked working tree is not clean, refusing to record an adjudication:\n{out}")


def record_adjudication(repo: Path, *, pair_id: str, label: str, reviewer_id: str, session_id: str,
                        reviewed_at_utc: str, notes: str = "") -> dict:
    """The ONLY path by which a v2 adjudication is ever appended. Fields
    are derived entirely from the frozen queue row for `pair_id` -- the
    caller supplies only the judgment (label/notes) and WHO/WHEN
    (reviewer_id/session_id/reviewed_at_utc); every identity/domain/
    session/hash-binding field is populated here, never accepted from the
    caller, so a caller cannot smuggle in a mismatched binding.

    `session_id` must equal the current eligible row's own
    `suggested_session` EXACTLY -- it is never an arbitrary caller-chosen
    label; this is enforced here (the single underlying record function
    both the CLI and the review UI go through), not only by a UI-layer
    convenience check.

    Requires `pair_id` to be exactly the current next eligible pair (no
    skipping, no alternate pair). Requires a clean tracked tree (the
    adjudicator source and every other approved implementation source
    must be exactly their committed, contract-bound content -- re-checked
    again, explicitly, right before a real write, on top of the load-time
    provenance check inside load_verified_v2_state). Acquires the
    exclusive per-write lock before touching the ledger; inside the lock,
    reverifies the ledger, appends (fsynced by aol.append_record), then
    reloads and fully reverifies before returning."""
    state = load_verified_v2_state(repo)
    _require_clean_tracked_tree(repo)
    records = load_verified_ledger_v2(repo, state)
    eligible = first_eligible_row(state["queue_rows"], records)
    if eligible is None:
        _fail("no pair is currently eligible for adjudication -- every workstream is either "
              "exhausted or has already recorded a blocking result")
    if eligible["pair_id"] != pair_id:
        _fail(f"pair_id {pair_id!r} is not the current next eligible pair (expected "
              f"{eligible['pair_id']!r}) -- no skipping, duplicate record, or alternate pair is permitted")
    if session_id != eligible["suggested_session"]:
        _fail(f"session_id {session_id!r} does not match the current frozen queue block "
              f"{eligible['suggested_session']!r} -- a record's session identifier is mechanically bound "
              f"to the frozen queue row, never an arbitrary caller-supplied value")

    fields = build_fields(repo, eligible, state, label=label, reviewer_id=reviewer_id, session_id=session_id,
                          reviewed_at_utc=reviewed_at_utc, notes=notes)
    problems = validate_record_bindings(fields, eligible, state)
    if problems:
        _fail(f"adjudication rejected: {problems}")

    with exclusive_lock(repo):
        records_at_lock_time = load_verified_ledger_v2(repo, state)
        eligible_at_lock_time = first_eligible_row(state["queue_rows"], records_at_lock_time)
        if eligible_at_lock_time is None or eligible_at_lock_time["pair_id"] != pair_id:
            _fail(f"the next eligible pair changed since this invocation started (a concurrent writer "
                  f"likely recorded a decision) -- refusing to record; re-run --next")
        try:
            record = aol.append_record(ledger_path_for(repo), fields,
                                       required_fields=v2c.ADJUDICATION_RECORD_REQUIRED_FIELDS,
                                       identity_fn=v2c.adjudication_record_identity)
        except aol.LedgerError as exc:
            _fail(str(exc))
        final_records = load_verified_ledger_v2(repo, state)
        if not any(r.get("pair_id") == pair_id for r in final_records):
            _fail("post-append verification failed: the newly appended record was not found on reload "
                  "-- left on disk for manual review, never silently retried")

    return record


# --------------------------------------------------------------- CLI --
def cmd_preflight(args) -> dict:
    state = load_verified_v2_state(args.repo)
    records = load_verified_ledger_v2(args.repo, state)
    eligible = first_eligible_row(state["queue_rows"], records)
    stop_state = workstream_stop_state(state["queue_rows"], records)
    return {
        "ok": True, "total_queue_rows": len(state["queue_rows"]), "adjudicated_count": len(records),
        "remaining_count": len(state["queue_rows"]) - len(records),
        "early_stop_skipped_count": len(early_stop_skipped_pair_ids(state["queue_rows"], records)),
        "workstream_stop_state": stop_state,
        "next_eligible_pair_id": eligible["pair_id"] if eligible else None,
        "done": eligible is None,
    }


def cmd_next(args) -> dict:
    state = load_verified_v2_state(args.repo)
    records = load_verified_ledger_v2(args.repo, state)
    eligible = first_eligible_row(state["queue_rows"], records)
    if eligible is None:
        return {"done": True}
    return {"done": False, **eligible}


def cmd_record(args) -> dict:
    record = record_adjudication(args.repo, pair_id=args.pair_id, label=args.label, reviewer_id=args.reviewer_id,
                                 session_id=args.session_id, reviewed_at_utc=args.reviewed_at_utc,
                                 notes=args.notes or "")
    return {"recorded": True, "pair_id": record["pair_id"], "record_hash": record["record_hash"]}


def cmd_status(args) -> dict:
    return cmd_preflight(args)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", type=Path, default=REPO)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--next", action="store_true")
    mode.add_argument("--record", action="store_true")
    mode.add_argument("--status", action="store_true")
    ap.add_argument("--pair-id")
    ap.add_argument("--label", choices=sorted(v2c.ADJUDICATION_LABELS))
    ap.add_argument("--notes", default="")
    ap.add_argument("--reviewer-id")
    ap.add_argument("--session-id", help="must exactly equal the current eligible row's suggested_session "
                    "(see --next) -- never an arbitrary label; a mismatch is rejected, not overridden")
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
                print(f"[adjudicate_pairs_v2] FAILURE: --record requires: {missing}")
                return 1
            result = cmd_record(args)
        else:
            result = cmd_status(args)
        print(json.dumps(result, indent=2, default=str))
        return 0
    except AdjudicationV2Error as e:
        print(f"[adjudicate_pairs_v2] FAILURE: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
