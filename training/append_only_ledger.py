#!/usr/bin/env python3
"""append_only_ledger.py — a generic, hash-chained, append-only JSONL ledger.

Used by Phase 5F1's manual-review tool (training/manual_review_tool.py) and
pair-adjudication tool (training/adjudicate_pairs.py). Each line is one JSON
record. Every record's `record_hash` covers every OTHER field of that same
record (canonical JSON, sorted keys) chained to the previous record's
`record_hash` via `prev_record_hash` -- editing, reordering, or deleting any
earlier line changes what the chain recomputes, so tampering is detectable
by a full replay from the genesis hash. Domain-specific validation (which
fields are required, what an "identity" is, business-rule checks) is the
caller's responsibility; this module only enforces the exact-schema/
chain/identity-uniqueness mechanics shared by both callers.

Nothing here opens an image or touches a model.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable

GENESIS_HASH = "0" * 64


class LedgerError(RuntimeError):
    """A ledger integrity or append problem. Always fails closed."""


def canonical_bytes(obj: dict) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_ledger(path: Path) -> list[dict]:
    """Reads every line as one JSON record, in file order. Returns [] if the
    file does not exist yet (a brand-new ledger).

    The raw trailing-newline state on disk decides how a malformed FINAL
    line is treated -- this is the only signal that distinguishes "an
    append was interrupted mid-write" from "the file was corrupted after a
    complete write":
      - File ends with a newline: every nonblank line, including the last,
        was fully written and terminated. A malformed one anywhere,
        including last, is real corruption and raises LedgerError.
      - File does NOT end with a newline: the final nonblank line's bytes
        may be an interrupted write (process killed / disk full after
        partial bytes, or even after the complete JSON bytes but before
        the trailing newline landed). If that final line fails to parse,
        it is treated as "that append never happened" and silently
        dropped -- never corruption. A malformed line anywhere ELSE (not
        final) is still real corruption."""
    path = Path(path)
    if not path.exists():
        return []
    raw = path.read_bytes()
    if not raw:
        return []
    ends_with_newline = raw.endswith(b"\n")
    raw_lines = raw.split(b"\n")
    if ends_with_newline:
        raw_lines.pop()  # the "" produced by the trailing newline is not a line
    non_blank = [(i, line) for i, line in enumerate(raw_lines) if line.strip()]
    records: list[dict] = []
    for pos, (line_no, line_bytes) in enumerate(non_blank):
        is_final_physical_line = line_no == len(raw_lines) - 1
        try:
            parsed = json.loads(line_bytes.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            if is_final_physical_line and not ends_with_newline:
                # Interrupted final append (never newline-terminated) --
                # tolerated as "that write never happened", not corruption.
                break
            raise LedgerError(f"{path} line {line_no + 1} is not valid JSON: {exc}") from exc
        records.append(parsed)
    return records


def _recover_tail(path: Path) -> None:
    """Brings `path` into a state append_record() can safely append onto,
    based on the exact raw trailing-newline state on disk:

      - File ends with a newline: nothing to do here. If the final
        nonblank line is malformed, that is corruption, not a recoverable
        interrupted write -- it is left untouched (never auto-deleted) and
        the subsequent read_ledger() call raises LedgerError, so
        append_record() fails closed.
      - File does NOT end with a newline and the final line fails to
        parse as JSON: that fragment is an interrupted write. It is
        physically truncated -- and ONLY that fragment -- via an atomic,
        fsynced temp-file-then-replace, so a later append can never land
        glued onto leftover partial bytes.
      - File does NOT end with a newline but the final line IS valid JSON
        (the process was interrupted after writing the complete record's
        bytes but before the trailing newline landed): the record is real
        and must be kept. A single newline byte is safely appended
        (opened in append mode, flushed, fsynced) so the next append
        starts its own line.

    No-op if the file doesn't exist or is empty."""
    path = Path(path)
    if not path.exists():
        return
    raw = path.read_bytes()
    if not raw:
        return
    if raw.endswith(b"\n"):
        return  # clean tail -- any corruption here is for read_ledger() to raise, not repair

    lines = raw.split(b"\n")
    last_line = lines[-1]
    if last_line.strip() == b"":
        # Trailing whitespace-only fragment with no terminating newline --
        # nothing meaningful to recover; leave as-is (read_ledger ignores it).
        return
    try:
        json.loads(last_line.decode("utf-8"))
        is_valid_final_record = True
    except (json.JSONDecodeError, UnicodeDecodeError):
        is_valid_final_record = False

    if is_valid_final_record:
        # Interrupted after complete JSON bytes but before the newline --
        # terminate the existing valid record; never truncate it.
        with path.open("ab") as fh:
            fh.write(b"\n")
            fh.flush()
            os.fsync(fh.fileno())
        return

    # Interrupted mid-write -- truncate only the incomplete final fragment.
    good_lines = lines[:-1]
    new_content = b"\n".join(good_lines) + (b"\n" if good_lines else b"")
    tmp = path.with_suffix(path.suffix + f".repair{os.getpid()}")
    with tmp.open("wb") as fh:
        fh.write(new_content)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(str(tmp), str(path))


def _safe_identity(identity_fn: Callable[[dict], Any], rec: dict, index: int, problems: list[str]) -> tuple[bool, Any]:
    """Never lets identity_fn's result (or identity_fn itself) raise past
    this point -- an unhashable or exception-raising identity is reported
    as a controlled problem, never a bare TypeError."""
    try:
        identity = identity_fn(rec)
        hash(identity)
    except Exception as exc:  # noqa: BLE001 -- identity computation must never crash verification
        problems.append(f"record {index} identity could not be computed or is unhashable: {exc}")
        return False, None
    return True, identity


def verify_chain(records: list[dict], required_fields: frozenset[str],
                 identity_fn: Callable[[dict], Any]) -> list[str]:
    """Full replay from the genesis hash. `required_fields` is the EXACT key
    set every record must have (including prev_record_hash/record_hash) --
    missing AND extra fields both fail. Returns a list of problems; empty
    list means the chain is fully intact. Total over arbitrary JSON values
    in `records` -- never raises TypeError/ValueError/KeyError, no matter
    what a record's field values are (lists, dicts, None, NaN, unhashable
    types)."""
    problems: list[str] = []
    prev_hash = GENESIS_HASH
    seen_identities: set = set()
    for i, rec in enumerate(records):
        if not isinstance(rec, dict):
            problems.append(f"record {i} is not a JSON object")
            prev_hash = GENESIS_HASH
            continue
        actual_fields = set(rec)
        if actual_fields != required_fields:
            missing = required_fields - actual_fields
            extra = actual_fields - required_fields
            if missing:
                problems.append(f"record {i} missing field(s): {sorted(missing)}")
            if extra:
                problems.append(f"record {i} has unexpected field(s): {sorted(extra)}")
            continue
        if rec.get("prev_record_hash") != prev_hash:
            problems.append(f"record {i} prev_record_hash {rec.get('prev_record_hash')!r} "
                            f"!= expected {prev_hash!r} -- chain broken")

        payload = {k: v for k, v in rec.items() if k != "record_hash"}
        try:
            recomputed_hash = sha256_hex(canonical_bytes(payload))
        except (TypeError, ValueError) as exc:
            problems.append(f"record {i} could not be canonicalized for hashing: {exc}")
            prev_hash = GENESIS_HASH
            continue
        recorded_hash = rec.get("record_hash")
        if recorded_hash != recomputed_hash:
            problems.append(f"record {i} record_hash {recorded_hash!r} != recomputed "
                            f"{recomputed_hash!r} -- record has been altered since it was written")

        identity_ok, identity = _safe_identity(identity_fn, rec, i, problems)
        if identity_ok:
            if identity in seen_identities:
                problems.append(f"record {i} duplicate identity {identity!r} -- a decision for "
                                f"this item already exists earlier in the ledger")
            seen_identities.add(identity)
        prev_hash = recorded_hash if isinstance(recorded_hash, str) else GENESIS_HASH
    return problems


def append_record(path: Path, fields: dict, *, required_fields: frozenset[str],
                  identity_fn: Callable[[dict], Any]) -> dict:
    """Validates the EXISTING ledger's chain first (refusing to append onto
    a already-broken chain), rejects a duplicate identity, then appends one
    new record with `prev_record_hash` set to the current tip and its own
    freshly computed `record_hash`. Returns the full record as written.
    Never silently rewrites an earlier COMPLETE line -- opens `path` in
    append mode only for the new record, and flushes + fsyncs before
    returning so a crash immediately after append_record() cannot lose the
    write silently. The raw trailing-newline state is recovered first via
    _recover_tail(): an interrupted mid-write fragment is truncated, a
    complete-but-unterminated final record is newline-terminated, and a
    genuinely corrupt (newline-terminated but malformed) final line is left
    untouched so the read below fails closed instead of being repaired."""
    path = Path(path)
    _recover_tail(path)
    existing = read_ledger(path)
    problems = verify_chain(existing, required_fields, identity_fn)
    if problems:
        raise LedgerError(f"{path} existing ledger failed chain verification, refusing to "
                          f"append: {problems}")

    payload = dict(fields)
    prev_hash = existing[-1]["record_hash"] if existing else GENESIS_HASH
    payload["prev_record_hash"] = prev_hash
    if set(payload) | {"record_hash"} != required_fields:
        actual = set(payload) | {"record_hash"}
        missing = required_fields - actual
        extra = actual - required_fields
        raise LedgerError(f"new record fields do not match the required schema -- "
                          f"missing: {sorted(missing)}, extra: {sorted(extra)}")

    new_identity = identity_fn(payload)
    for rec in existing:
        if identity_fn(rec) == new_identity:
            raise LedgerError(f"duplicate decision for identity {new_identity!r} -- rejected "
                              f"(append-only: an existing decision is never overwritten)")

    record_hash = sha256_hex(canonical_bytes(payload))
    payload["record_hash"] = record_hash

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True))
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    return payload
