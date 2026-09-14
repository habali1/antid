#!/usr/bin/env python3
"""correct_manual_review.py — an additive, provenance-bound correction
mechanism for the completed Phase 5F2 manual-review ledger
(training/manual_review_ledger.jsonl).

The original ledger is append-only and closed: this module NEVER edits,
truncates, rechains, or replaces it, and never writes to it. A correction
is a reviewer's post-hoc clarification of one already-recorded decision --
the original record remains exactly as written, forever; the correction is
a SEPARATE record in a SEPARATE append-only ledger
(training/manual_review_corrections.jsonl) that only takes effect when
explicitly folded into an "effective view" computed in memory.

THIS PHASE AUTHORIZES EXACTLY ONE CORRECTION -- queue_index 430, with the
exact original/corrected judgment values frozen below. There is no CLI or
Python-API path to target a different queue_index or a different judgment
combination: every value that matters is a module constant, verified
against the ACTUAL frozen queue row, the ACTUAL original ledger record, and
(for --write) re-asserted rather than accepted as input. A future
correction requires a separately reviewed protocol/source update -- this
version deliberately does not generalize.

Every correction record binds, and this module verifies:
  - the frozen Phase 5F1 contract's content_sha256;
  - the frozen queue's byte_sha256 and identity_order_sha256;
  - the ORIGINAL ledger's current byte_sha256 (recomputed fresh every time,
    and additionally gated against the approved frozen value below BEFORE
    a single correction is even considered -- an original ledger that
    doesn't match is rejected outright, not merely "unbound");
  - the queue_index being corrected and its exact frozen row identity;
  - the ORIGINAL record's exact record_hash and its exact original
    review_state/label_plausibility/duplicate_suspicion/notes;
  - the exact corrected review_state/label_plausibility/
    duplicate_suspicion/notes;
  - the frozen reason;
  - THIS FILE's own provenance: the git commit at which it was committed
    when the correction was written, its canonical-LF sha256 at that
    commit, and a fixed protocol version -- following the same immutable-
    generation-commit pattern established for Gate v2's inference-policy
    generator (generate_inference_policy_v2.py): the recorded commit need
    only be an ANCESTOR of the current HEAD (an unrelated later commit
    does not invalidate existing evidence), but the source must match
    exactly, both at that commit and in the current working tree.

CLI modes (deliberately asymmetric with earlier ledgers -- this is a
write-once artifact, not an appendable one):
  --preflight  loads/verifies the contract, semantically re-verifies the
               original 600-record ledger (via manual_review_tool's
               existing shared path) INCLUDING the frozen byte-hash/
               record-count/session-count authority gate, semantically
               re-verifies the corrections ledger if one exists, and
               reports the approved correction as pending (file absent)
               or applied (file present, exactly one valid record). No
               image access, no writes.
  --write      publishes the ONE approved correction as a complete,
               newline-terminated one-record JSONL artifact via a local
               stage-then-hard-link publisher (never
               append_only_ledger.append_record, which opens its
               destination in plain append mode and is not exclusive
               against a concurrent writer) -- structurally exclusive and
               atomic: the destination is created via os.link, which
               fails at the filesystem level if it already exists,
               instead of relying on a prior existence check alone.
               Refuses if any correction artifact already exists (or is
               created concurrently, mid-publish). Accepts ONLY
               operational metadata (reviewer_id, session_id,
               corrected_at_utc) -- requires the correction source itself
               to be committed and the tracked tree clean.
  --check      requires the correction artifact to exist and contain
               EXACTLY the one approved record, with the resulting
               effective counts matching the frozen expected counts.
               Zero, extra, duplicate, or different corrections fail.

Never imports onnxruntime or any inference/model module. Never opens an
image.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import uuid
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import manual_review_contract as mrc  # noqa: E402
import manual_review_tool as mrt  # noqa: E402
import append_only_ledger as aol  # noqa: E402

# Deliberately NOT added to mrc.APPROVED_OUTPUT_PATHS: that dict is
# embedded verbatim in the already-frozen contract content, and adding a
# key there would change manual_review_contract.json's content_sha256.
CORRECTIONS_LEDGER_REL_PATH = "training/manual_review_corrections.jsonl"
GENERATOR_SOURCE_REL_PATH = "training/correct_manual_review.py"
GENERATOR_PROTOCOL_VERSION = "phase5f2-correction-v1-single-target-430"

# ------------------------------------------------- frozen original-ledger authority
# The Phase 5F2 review's actual, completed state -- gated BEFORE a single
# correction is even considered. An original ledger with a different byte
# hash, record count, or session-count shape is a DIFFERENT ledger, and no
# correction bound to these values may ever be evaluated against it.
APPROVED_ORIGINAL_LEDGER_BYTE_SHA256 = "75a2ef01058d978ddec19daf4e9eb4d599ed497cc03a506ac23fd3ed0e06bec4"
APPROVED_ORIGINAL_LEDGER_RECORD_COUNT = 600
APPROVED_ORIGINAL_LEDGER_SESSION_COUNTS = {
    "phase5f2-review-session-1": 300,
    "phase5f2-review-session-2": 300,
}

# ------------------------------------------------- the one approved correction
APPROVED_CORRECTION_QUEUE_INDEX = 430
APPROVED_CORRECTION_ROW_IDENTITY = {
    "dataset": "northeast_final_test_v1",
    "split": "final_test",
    "slug": "formica-exsectoides",
    "observation_uuid": "7224f1f9-a390-4970-a20b-7315a22c2c6b",
    "photo_id": "576551143",
    "sha256": "c630fb5f3515d3e870ae35a11f6f09af7b8663db36b50fcb4bcd06d9dd743fbd",
}
APPROVED_CORRECTION_ORIGINAL_RECORD_HASH = "d2baded8ab3effa346cecf68fc3bd77e753adfcd86cde2058b295b038833ae57"
APPROVED_CORRECTION_ORIGINAL_JUDGMENT = {
    "review_state": "unusable_no_visible_ant",
    "label_plausibility": "plausible",
    "duplicate_suspicion": "none",
    "notes": "",
}
APPROVED_CORRECTION_CORRECTED_JUDGMENT = {
    "review_state": "poor_quality_usable",
    "label_plausibility": "uncertain",
    "duplicate_suspicion": "none",
    "notes": "Odak karınca yuvasında; karıncalar küçük veya odak dışında.",
}
APPROVED_CORRECTION_REASON = (
    "Reviewer clarification after completing the review: the original "
    "unusable_no_visible_ant judgment was too strict. On reconsideration the "
    "ant is present but small and/or out of focus, not truly invisible -- this "
    "is a poor_quality_usable image, and the plausibility call is uncertain "
    "rather than plausible. The original append-only record is not, and never "
    "will be, altered; this is an additive, separately provenance-bound "
    "clarification of it."
)

APPROVED_EFFECTIVE_COUNTS = {
    "review_state": {"usable": 542, "poor_quality_usable": 53, "unusable_no_visible_ant": 1,
                     "unusable_wrong_organism": 4},
    "label_plausibility": {"implausible": 5, "plausible": 542, "uncertain": 53},
    "duplicate_suspicion": {"none": 600},
    "final_test_poor_quality_usable": 41,
    "final_test_unusable_total": 3,
}

CORRECTION_RECORD_REQUIRED_FIELDS = frozenset({
    "queue_index", "dataset", "split", "slug", "observation_uuid", "photo_id", "sha256",
    "contract_content_sha256", "queue_byte_sha256", "queue_identity_order_sha256",
    "original_ledger_byte_sha256", "original_record_hash",
    "original_review_state", "original_label_plausibility", "original_duplicate_suspicion", "original_notes",
    "corrected_review_state", "corrected_label_plausibility", "corrected_duplicate_suspicion", "corrected_notes",
    "reason", "reviewer_id", "session_id", "corrected_at_utc",
    "generator_git_commit", "generator_source_path", "generator_source_sha256", "generator_protocol_version",
    "prev_record_hash", "record_hash",
})

GIT_COMMIT_HEX_RE = re.compile(r"^[0-9a-f]{40}$")


class CorrectionError(RuntimeError):
    """A validation, provenance, or ledger problem. Always fails closed."""


def _fail(msg: str) -> None:
    raise CorrectionError(msg)


def corrections_ledger_path_for(repo: Path) -> Path:
    return Path(repo) / CORRECTIONS_LEDGER_REL_PATH


def correction_record_identity(record: dict) -> object:
    return record.get("queue_index")


def original_ledger_byte_sha256(repo: Path) -> str:
    """The ORIGINAL ledger's byte hash, recomputed fresh from disk every
    call -- never cached, never trusted from a prior run."""
    path = Path(repo) / mrc.APPROVED_OUTPUT_PATHS["manual_review_ledger"]
    if not path.exists():
        _fail(f"original manual-review ledger does not exist: {path}")
    return mrc.sha256_file(path)


# --------------------------------------------------------- git provenance --
def _git_head(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(repo), stderr=subprocess.PIPE
        ).decode("utf-8").strip()
    except (subprocess.CalledProcessError, OSError) as exc:
        _fail(f"could not determine git HEAD in {repo}: {exc}")


def _git_object_type(repo: Path, ref: str) -> str | None:
    try:
        result = subprocess.run(["git", "cat-file", "-t", ref], cwd=str(repo), capture_output=True, text=True)
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _git_is_ancestor(repo: Path, ancestor: str, descendant: str) -> bool:
    try:
        result = subprocess.run(["git", "merge-base", "--is-ancestor", ancestor, descendant],
                                cwd=str(repo), capture_output=True)
    except OSError:
        return False
    return result.returncode == 0


def _git_blob_canonical_lf_sha256(repo: Path, commit: str, rel_path: str) -> str:
    """Reads `rel_path` as it existed AT `commit` (via `git show`, never
    the working tree) and returns its canonical-LF sha256. Fails closed
    (CorrectionError, never an uncaught CalledProcessError) if git can't
    produce that blob (e.g. the path didn't exist at that commit)."""
    try:
        result = subprocess.run(["git", "show", f"{commit}:{rel_path}"], cwd=str(repo), capture_output=True)
    except OSError as exc:
        _fail(f"could not read {rel_path} at commit {commit}: {exc}")
    if result.returncode != 0:
        _fail(f"git show {commit}:{rel_path} failed: {result.stderr.decode('utf-8', errors='replace').strip()}")
    return mrc.sha256_bytes(mrc.canonical_lf_bytes(result.stdout))


def _require_clean_tracked_tree(repo: Path) -> None:
    """Requires the TRACKED working tree to be clean. Untracked files are
    explicitly allowed (mirrors the established Gate v2 pattern in
    gate_v2_contract.require_clean_tracked_tree) -- only drift in
    already-tracked content is a problem here."""
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=str(repo), stderr=subprocess.PIPE,
        ).decode("utf-8")
    except (subprocess.CalledProcessError, OSError) as exc:
        _fail(f"could not check git status in {repo}: {exc}")
    if out.strip():
        _fail(f"tracked working tree is not clean, refusing to write a correction:\n{out}")


def validate_generator_provenance(repo: Path, fields: dict) -> list[str]:
    """Fail-closed validation of a correction record's generator
    provenance as the IMMUTABLE generation commit -- deliberately never
    required to equal the CURRENT HEAD (an unrelated later commit must not
    invalidate existing evidence). What IS required: a strict 40-hex
    commit hash that resolves to a real commit and is an ancestor of the
    current HEAD; the recorded source hash matches the canonical-LF hash
    of correct_manual_review.py AS IT EXISTED AT THAT COMMIT (read via
    `git show`); and the CURRENT working-tree copy also matches -- any
    change to this generator since, including a fix that touches
    correct_manual_review.py itself, makes the evidence stale. Returns a
    problem list; never raises."""
    problems: list[str] = []
    commit = fields.get("generator_git_commit")
    if not isinstance(commit, str) or not GIT_COMMIT_HEX_RE.match(commit):
        return ["generator_git_commit is not a strict 40-character lowercase hex commit hash"]
    obj_type = _git_object_type(repo, commit)
    if obj_type != "commit":
        return [f"generator_git_commit {commit!r} does not resolve to a commit object in this "
               f"repository (git cat-file -t reported {obj_type!r})"]
    current_head = _git_head(repo)
    if not _git_is_ancestor(repo, commit, current_head):
        problems.append(f"generator_git_commit {commit!r} is not an ancestor of the current HEAD "
                        f"{current_head!r}")

    if fields.get("generator_source_path") != GENERATOR_SOURCE_REL_PATH:
        problems.append(f"generator_source_path must be {GENERATOR_SOURCE_REL_PATH!r}")
        return problems
    if fields.get("generator_protocol_version") != GENERATOR_PROTOCOL_VERSION:
        problems.append(f"generator_protocol_version must be {GENERATOR_PROTOCOL_VERSION!r}")

    recorded_hash = fields.get("generator_source_sha256")
    if not mrc.is_sha256_hex(recorded_hash):
        problems.append("generator_source_sha256 is not a 64-char lowercase hex string")
        return problems
    commit_hash = _git_blob_canonical_lf_sha256(repo, commit, GENERATOR_SOURCE_REL_PATH)
    if commit_hash != recorded_hash:
        problems.append(f"generator_source_sha256 {recorded_hash!r} does not match the canonical-LF "
                        f"hash {commit_hash!r} of {GENERATOR_SOURCE_REL_PATH} as it existed at commit "
                        f"{commit!r}")
    working_tree_path = Path(repo) / GENERATOR_SOURCE_REL_PATH
    if not working_tree_path.exists():
        problems.append(f"generator source file missing from the working tree: {working_tree_path}")
    else:
        working_tree_hash = mrc.canonical_lf_sha256_file(working_tree_path)
        if working_tree_hash != recorded_hash:
            problems.append(f"{GENERATOR_SOURCE_REL_PATH} has changed since commit {commit} -- this "
                            f"correction's generator provenance is stale")
    return problems


# ----------------------------------------------------------- validation --
def validate_correction_fields(fields: dict, *, original_row: dict, original_record: dict,
                               verified: dict, current_original_ledger_byte_sha256: str) -> list[str]:
    """Structural + binding + approved-target validation, entirely via
    .get() -- never raises. `original_row` is the ACTUAL frozen queue row
    for this queue_index; `original_record` is the ACTUAL semantically-
    verified original ledger record for it. Beyond matching those (proving
    the correction is bound to reality), every identity/judgment field is
    ALSO checked against the single APPROVED_CORRECTION_* target -- this
    phase authorizes exactly one correction, and no other combination,
    however internally consistent, is accepted."""
    problems: list[str] = []
    actual_fields = set(fields) | {"prev_record_hash", "record_hash"}
    if actual_fields != CORRECTION_RECORD_REQUIRED_FIELDS:
        missing = CORRECTION_RECORD_REQUIRED_FIELDS - actual_fields
        extra = actual_fields - CORRECTION_RECORD_REQUIRED_FIELDS
        if missing:
            problems.append(f"missing field(s): {sorted(missing)}")
        if extra:
            problems.append(f"unexpected field(s): {sorted(extra)}")

    # Bound to the ACTUAL frozen queue row / original record.
    for field in ("dataset", "split", "slug", "observation_uuid", "photo_id", "sha256"):
        if fields.get(field) != original_row.get(field):
            problems.append(f"{field} {fields.get(field)!r} does not match the frozen queue row "
                            f"({original_row.get(field)!r})")
    if fields.get("queue_index") != original_row.get("queue_index"):
        problems.append("queue_index does not match the frozen queue row")
    if fields.get("original_record_hash") != original_record.get("record_hash"):
        problems.append("original_record_hash does not match the actual original ledger record's "
                        "record_hash")
    for field, orig_key in (("original_review_state", "review_state"),
                            ("original_label_plausibility", "label_plausibility"),
                            ("original_duplicate_suspicion", "duplicate_suspicion"),
                            ("original_notes", "notes")):
        if fields.get(field) != original_record.get(orig_key):
            problems.append(f"{field} {fields.get(field)!r} does not match the actual original "
                            f"record's {orig_key} ({original_record.get(orig_key)!r})")

    # Bound to the single APPROVED correction target -- no other queue_index
    # or judgment combination is accepted this phase, regardless of how
    # internally consistent it is.
    if fields.get("queue_index") != APPROVED_CORRECTION_QUEUE_INDEX:
        problems.append(f"queue_index {fields.get('queue_index')!r} is not the approved correction "
                        f"target {APPROVED_CORRECTION_QUEUE_INDEX!r} -- this phase authorizes exactly "
                        f"one correction")
    for field, expected in APPROVED_CORRECTION_ROW_IDENTITY.items():
        if fields.get(field) != expected:
            problems.append(f"{field} {fields.get(field)!r} != the approved correction's frozen "
                            f"{expected!r}")
    if fields.get("original_record_hash") != APPROVED_CORRECTION_ORIGINAL_RECORD_HASH:
        problems.append(f"original_record_hash {fields.get('original_record_hash')!r} != the "
                        f"approved {APPROVED_CORRECTION_ORIGINAL_RECORD_HASH!r}")
    for field, expected in (("original_review_state", APPROVED_CORRECTION_ORIGINAL_JUDGMENT["review_state"]),
                            ("original_label_plausibility", APPROVED_CORRECTION_ORIGINAL_JUDGMENT["label_plausibility"]),
                            ("original_duplicate_suspicion", APPROVED_CORRECTION_ORIGINAL_JUDGMENT["duplicate_suspicion"]),
                            ("original_notes", APPROVED_CORRECTION_ORIGINAL_JUDGMENT["notes"])):
        if fields.get(field) != expected:
            problems.append(f"{field} {fields.get(field)!r} != the approved original judgment {expected!r}")
    for field, expected in (("corrected_review_state", APPROVED_CORRECTION_CORRECTED_JUDGMENT["review_state"]),
                            ("corrected_label_plausibility", APPROVED_CORRECTION_CORRECTED_JUDGMENT["label_plausibility"]),
                            ("corrected_duplicate_suspicion", APPROVED_CORRECTION_CORRECTED_JUDGMENT["duplicate_suspicion"]),
                            ("corrected_notes", APPROVED_CORRECTION_CORRECTED_JUDGMENT["notes"])):
        if fields.get(field) != expected:
            problems.append(f"{field} {fields.get(field)!r} != the approved corrected judgment {expected!r}")
    if fields.get("reason") != APPROVED_CORRECTION_REASON:
        problems.append("reason does not match the approved, frozen correction reason")

    contract = verified["contract"]
    if fields.get("contract_content_sha256") != contract["content_sha256"]:
        problems.append("contract_content_sha256 does not match the currently verified contract")
    if fields.get("queue_byte_sha256") != contract["content"]["queue_artifact"]["byte_sha256"]:
        problems.append("queue_byte_sha256 does not match the currently verified queue")
    if fields.get("queue_identity_order_sha256") != contract["content"]["queue_artifact"]["identity_order_sha256"]:
        problems.append("queue_identity_order_sha256 does not match the currently verified queue")
    if fields.get("original_ledger_byte_sha256") != current_original_ledger_byte_sha256:
        problems.append("original_ledger_byte_sha256 does not match the ORIGINAL ledger's current "
                        "byte hash -- the original ledger may have changed, or this correction was "
                        "written against a different copy of it")

    if not isinstance(fields.get("reviewer_id"), str) or not fields.get("reviewer_id"):
        problems.append("reviewer_id must be a non-empty string")
    if not isinstance(fields.get("session_id"), str) or not fields.get("session_id"):
        problems.append("session_id must be a non-empty string")
    if not mrc.is_strict_utc_timestamp(fields.get("corrected_at_utc")):
        problems.append("corrected_at_utc must match the strict UTC form 'YYYY-MM-DDTHH:MM:SSZ'")
    return problems


def build_approved_correction_fields(current_original_ledger_byte_sha256: str, contract: dict,
                                     generator_git_commit: str, generator_source_sha256: str, *,
                                     reviewer_id: str, session_id: str, corrected_at_utc: str) -> dict:
    """Builds the ONE approved correction's fields entirely from the
    APPROVED_CORRECTION_* / APPROVED_CORRECTION_REASON constants -- the
    caller supplies ONLY operational metadata (reviewer_id, session_id,
    corrected_at_utc) and the dynamic provenance bindings (contract state,
    original-ledger hash, generator commit/source hash). There is no
    parameter through which a different queue_index or judgment could be
    supplied."""
    fields = {
        "queue_index": APPROVED_CORRECTION_QUEUE_INDEX,
        **APPROVED_CORRECTION_ROW_IDENTITY,
        "contract_content_sha256": contract["content_sha256"],
        "queue_byte_sha256": contract["content"]["queue_artifact"]["byte_sha256"],
        "queue_identity_order_sha256": contract["content"]["queue_artifact"]["identity_order_sha256"],
        "original_ledger_byte_sha256": current_original_ledger_byte_sha256,
        "original_record_hash": APPROVED_CORRECTION_ORIGINAL_RECORD_HASH,
        "original_review_state": APPROVED_CORRECTION_ORIGINAL_JUDGMENT["review_state"],
        "original_label_plausibility": APPROVED_CORRECTION_ORIGINAL_JUDGMENT["label_plausibility"],
        "original_duplicate_suspicion": APPROVED_CORRECTION_ORIGINAL_JUDGMENT["duplicate_suspicion"],
        "original_notes": APPROVED_CORRECTION_ORIGINAL_JUDGMENT["notes"],
        "corrected_review_state": APPROVED_CORRECTION_CORRECTED_JUDGMENT["review_state"],
        "corrected_label_plausibility": APPROVED_CORRECTION_CORRECTED_JUDGMENT["label_plausibility"],
        "corrected_duplicate_suspicion": APPROVED_CORRECTION_CORRECTED_JUDGMENT["duplicate_suspicion"],
        "corrected_notes": APPROVED_CORRECTION_CORRECTED_JUDGMENT["notes"],
        "reason": APPROVED_CORRECTION_REASON,
        "reviewer_id": reviewer_id, "session_id": session_id, "corrected_at_utc": corrected_at_utc,
        "generator_git_commit": generator_git_commit, "generator_source_path": GENERATOR_SOURCE_REL_PATH,
        "generator_source_sha256": generator_source_sha256, "generator_protocol_version": GENERATOR_PROTOCOL_VERSION,
    }
    return fields


def load_verified_original(repo: Path) -> dict:
    """The one required first step: load/verify the frozen contract, then
    semantically re-verify the ORIGINAL 600-record ledger through
    manual_review_tool's EXISTING shared verification path, THEN gate the
    result against the frozen Phase 5F2 authority (byte hash, record
    count, session counts) -- BEFORE a single correction is even
    considered. An original ledger that doesn't match this authority is
    rejected outright."""
    verified = mrt.load_verified(repo)
    original_records = mrt.load_verified_ledger(repo, verified)
    byte_sha256 = original_ledger_byte_sha256(repo)
    if byte_sha256 != APPROVED_ORIGINAL_LEDGER_BYTE_SHA256:
        _fail(f"original manual-review ledger byte sha256 {byte_sha256!r} does not match the frozen "
              f"approved authority {APPROVED_ORIGINAL_LEDGER_BYTE_SHA256!r} -- refusing to evaluate "
              f"any correction against an unexpected original ledger")
    if len(original_records) != APPROVED_ORIGINAL_LEDGER_RECORD_COUNT:
        _fail(f"original ledger has {len(original_records)} record(s), expected exactly "
              f"{APPROVED_ORIGINAL_LEDGER_RECORD_COUNT}")
    session_counts = dict(Counter(r["session_id"] for r in original_records))
    if session_counts != APPROVED_ORIGINAL_LEDGER_SESSION_COUNTS:
        _fail(f"original ledger session counts {session_counts!r} do not match the frozen approved "
              f"authority {APPROVED_ORIGINAL_LEDGER_SESSION_COUNTS!r}")
    return {"verified": verified, "original_records": original_records, "original_ledger_byte_sha256": byte_sha256}


def load_verified_corrections(repo: Path, state: dict) -> list[dict]:
    """THE one semantic verification path for the corrections ledger.
    Beyond generic chain verification: rejects an unknown queue_index,
    rejects a duplicate correction, re-runs validate_correction_fields()
    (which itself enforces the single-approved-target binding) AND
    validate_generator_provenance() for every existing correction.
    Read-only. Returns records only once every check has passed -- by
    construction this is always 0 or exactly 1 record, since anything
    else fails closed."""
    verified, original_records = state["verified"], state["original_records"]
    ledger_path = corrections_ledger_path_for(repo)
    try:
        records = aol.read_ledger(ledger_path)
    except aol.LedgerError as exc:
        _fail(f"corrections ledger is corrupt: {exc}")
    chain_problems = aol.verify_chain(records, CORRECTION_RECORD_REQUIRED_FIELDS, correction_record_identity)
    if chain_problems:
        _fail(f"corrections ledger failed chain verification: {chain_problems}")

    queue_rows_by_index = {r["queue_index"]: r for r in verified["queue_rows"]}
    original_by_index = {r["queue_index"]: r for r in original_records}

    seen_indices: set = set()
    for rec in records:
        idx = rec.get("queue_index")
        if idx in seen_indices:
            _fail(f"corrections ledger has a duplicate correction for queue_index {idx!r}")
        seen_indices.add(idx)
        row = queue_rows_by_index.get(idx)
        original_record = original_by_index.get(idx)
        if row is None or original_record is None:
            _fail(f"corrections ledger references unknown queue_index {idx!r}")
        problems = validate_correction_fields(
            rec, original_row=row, original_record=original_record, verified=verified,
            current_original_ledger_byte_sha256=state["original_ledger_byte_sha256"])
        problems += validate_generator_provenance(repo, rec)
        if problems:
            _fail(f"corrections ledger record for queue_index {idx!r} failed semantic "
                  f"re-validation: {problems}")
    return records


def effective_records(original_records: list[dict], corrections: list[dict]) -> list[dict]:
    """Pure: the 600-record effective view, original records overridden in
    memory by any matching correction's judgment fields. Row count is
    always unchanged."""
    corrections_by_index = {c["queue_index"]: c for c in corrections}
    effective = []
    for rec in sorted(original_records, key=lambda r: r["queue_index"]):
        correction = corrections_by_index.get(rec["queue_index"])
        if correction is None:
            effective.append({**rec, "corrected": False})
        else:
            effective.append({
                **rec,
                "review_state": correction["corrected_review_state"],
                "label_plausibility": correction["corrected_label_plausibility"],
                "duplicate_suspicion": correction["corrected_duplicate_suspicion"],
                "notes": correction["corrected_notes"],
                "corrected": True, "correction_reason": correction["reason"],
            })
    return effective


def effective_counts(effective: list[dict], verified: dict) -> dict:
    queue_rows_by_index = {r["queue_index"]: r for r in verified["queue_rows"]}
    review_state = Counter(r["review_state"] for r in effective)
    label_plausibility = Counter(r["label_plausibility"] for r in effective)
    duplicate_suspicion = Counter(r["duplicate_suspicion"] for r in effective)
    final_test_poor_quality_usable = sum(
        1 for r in effective if r["review_state"] == "poor_quality_usable"
        and queue_rows_by_index[r["queue_index"]]["dataset"] == mrc.FINAL_TEST_DATASET)
    final_test_unusable = sum(
        1 for r in effective if r["review_state"].startswith("unusable")
        and queue_rows_by_index[r["queue_index"]]["dataset"] == mrc.FINAL_TEST_DATASET)
    return {
        "review_state": dict(review_state), "label_plausibility": dict(label_plausibility),
        "duplicate_suspicion": dict(duplicate_suspicion),
        "final_test_poor_quality_usable": final_test_poor_quality_usable,
        "final_test_unusable_total": final_test_unusable,
    }


def validate_effective_counts(counts: dict) -> list[str]:
    """Enforces the exact expected effective counts once the approved
    correction is applied -- defense in depth alongside the byte-hash/
    field-level bindings above."""
    problems: list[str] = []
    for key in ("review_state", "label_plausibility", "duplicate_suspicion"):
        if counts.get(key) != APPROVED_EFFECTIVE_COUNTS[key]:
            problems.append(f"effective {key} counts {counts.get(key)!r} != approved "
                            f"{APPROVED_EFFECTIVE_COUNTS[key]!r}")
    for key in ("final_test_poor_quality_usable", "final_test_unusable_total"):
        if counts.get(key) != APPROVED_EFFECTIVE_COUNTS[key]:
            problems.append(f"effective {key} {counts.get(key)!r} != approved "
                            f"{APPROVED_EFFECTIVE_COUNTS[key]!r}")
    return problems


def _publish_single_correction_record(dest: Path, fields: dict) -> dict:
    """Publishes the ONE approved correction as a complete, newline-
    terminated, one-record JSONL artifact -- structurally exclusive and
    atomic, NOT append_only_ledger.append_record(). append_record() opens
    its destination in plain append ("a") mode: a check-then-append is a
    TOCTOU race, since two concurrent callers can both observe the
    destination absent and both then write to it. Since this protocol
    permits exactly one record for this artifact, ever, this instead:

      1. builds the complete record in memory (prev_record_hash = the
         ledger genesis hash; record_hash computed via append_only_
         ledger's own canonical hashing functions -- the artifact remains
         fully hash-chain-compatible, just never built via a second,
         separately-timed append);
      2. serializes it as exactly one newline-terminated JSONL line;
      3. stages that content into a UNIQUE temp file in the destination's
         own directory (pid + a random uuid4 suffix, so two concurrent
         callers in the same or different processes never collide on the
         temp name), flushed and fsynced;
      4. re-reads the staged bytes and fully re-validates them (an exact
         byte round-trip, exactly one parsed record, and a full
         append_only_ledger.verify_chain replay) BEFORE attempting to
         publish;
      5. publishes via os.link(tmp, dest) -- which fails atomically at
         the filesystem level (FileExistsError) if `dest` already exists,
         the same same-directory hard-link pattern
         scan_perceptual_duplicates.publish_reports_atomically and
         finalize_stop_status.cmd_finalize already use for their own
         exclusive publications -- never os.replace, never a plain
         append, and never trusting a prior .exists() check by itself;
      6. ALWAYS removes the temp file, whether publication succeeded or
         failed;
      7. re-reads the PUBLISHED destination afterward and confirms its
         bytes equal the prevalidated staged bytes exactly.

    On any failure -- including a destination that appears between this
    function's own checks and the os.link call -- a pre-existing
    destination (whether it predates this call or was just created by a
    concurrent writer) is left byte-for-byte untouched, and CorrectionError
    is raised. This helper is kept local to correct_manual_review.py
    (never added to append_only_ledger.py) so the frozen Phase 5F1
    contract's implementation-source hashes remain unchanged."""
    payload = dict(fields)
    payload["prev_record_hash"] = aol.GENESIS_HASH
    payload["record_hash"] = aol.sha256_hex(aol.canonical_bytes(payload))
    data = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.parent / f"{dest.name}.tmp{os.getpid()}.{uuid.uuid4().hex}"
    # The ENTIRE temp-file lifecycle -- open/write/flush/fsync included --
    # is inside this try/finally, so a failure at ANY of those steps (not
    # merely during prevalidation or the link attempt) still guarantees
    # the temp file is removed. tmp.unlink(missing_ok=True) is safe even
    # if the open() itself never got far enough to create the file.
    try:
        with tmp.open("wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())

        staged_bytes = tmp.read_bytes()
        if staged_bytes != data:
            _fail("staged temporary correction file does not match the intended content")
        staged_records = aol.read_ledger(tmp)
        if len(staged_records) != 1:
            _fail(f"staged temporary correction file does not contain exactly one record "
                  f"(found {len(staged_records)})")
        chain_problems = aol.verify_chain(staged_records, CORRECTION_RECORD_REQUIRED_FIELDS,
                                          correction_record_identity)
        if chain_problems:
            _fail(f"staged temporary correction file failed chain verification: {chain_problems}")

        if dest.exists():
            _fail(f"refusing to publish: a correction artifact already exists: {dest}")
        try:
            os.link(str(tmp), str(dest))
        except FileExistsError:
            _fail(f"refusing to publish: a correction artifact was created concurrently: {dest}")
    finally:
        tmp.unlink(missing_ok=True)

    published_bytes = dest.read_bytes()
    if published_bytes != data:
        _fail("published correction artifact does not match the prevalidated staged content")
    return payload


def record_correction(repo: Path, fields: dict) -> dict:
    """Validates `fields` (structural/binding/approved-target AND
    generator provenance) against the freshly re-verified original ledger
    and the existing corrections ledger, publishes exactly one new
    correction record via _publish_single_correction_record -- an
    exclusive, atomic, stage-then-hard-link publish, never an ordinary
    append -- and THEN, before reporting success, reloads the PUBLISHED
    artifact from disk through the full real verification path (never
    trusting the in-memory dict _publish_single_correction_record
    returned): exactly one correction exists, the hash chain is valid,
    every approved-target/original-ledger binding is valid, generator
    provenance is valid, the resulting effective counts exactly match
    APPROVED_EFFECTIVE_COUNTS, and the persisted bytes equal the
    prevalidated staged bytes. A post-publication verification failure
    raises CorrectionError WITHOUT deleting or rewriting the artifact --
    an already-published-but-then-failing-re-verification artifact is
    left exactly as it is, for manual review, never silently repaired.
    Raises CorrectionError (never writes) on any pre-publication problem.
    NEVER touches training/manual_review_ledger.jsonl."""
    state = load_verified_original(repo)
    verified, original_records = state["verified"], state["original_records"]
    original_by_index = {r["queue_index"]: r for r in original_records}
    queue_rows_by_index = {r["queue_index"]: r for r in verified["queue_rows"]}

    idx = fields.get("queue_index")
    row = queue_rows_by_index.get(idx)
    original_record = original_by_index.get(idx)
    if row is None or original_record is None:
        _fail(f"queue_index {idx!r} is not a valid frozen queue index with an original decision")

    problems = validate_correction_fields(
        fields, original_row=row, original_record=original_record, verified=verified,
        current_original_ledger_byte_sha256=state["original_ledger_byte_sha256"])
    problems += validate_generator_provenance(repo, fields)
    if problems:
        _fail(f"correction rejected: {problems}")

    load_verified_corrections(repo, state)  # semantically re-verify every existing correction first

    dest = corrections_ledger_path_for(repo)
    published = _publish_single_correction_record(dest, fields)

    # ---- post-publication verification: reload from disk, trust nothing ----
    post_state = load_verified_original(repo)
    try:
        post_corrections = load_verified_corrections(repo, post_state)
    except CorrectionError as exc:
        _fail(f"post-publication verification failed (artifact left as published, for manual "
              f"review -- NOT deleted or rewritten): {exc}")
    if len(post_corrections) != 1:
        _fail(f"post-publication verification failed: expected exactly 1 correction on disk, "
              f"found {len(post_corrections)} (artifact left as published, for manual review -- "
              f"NOT deleted or rewritten)")
    post_effective = effective_records(post_state["original_records"], post_corrections)
    post_counts = effective_counts(post_effective, post_state["verified"])
    count_problems = validate_effective_counts(post_counts)
    if count_problems:
        _fail(f"post-publication verification failed: effective counts do not match the approved "
              f"expected counts: {count_problems} (artifact left as published, for manual review "
              f"-- NOT deleted or rewritten)")
    expected_bytes = (json.dumps(published, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    if dest.read_bytes() != expected_bytes:
        _fail("post-publication verification failed: the persisted correction artifact's bytes do "
              "not equal the prevalidated staged content (artifact left as published, for manual "
              "review -- NOT deleted or rewritten)")

    return published


# --------------------------------------------------------------- CLI --
def cmd_preflight(args) -> dict:
    state = load_verified_original(args.repo)
    corrections = load_verified_corrections(args.repo, state)
    effective = effective_records(state["original_records"], corrections)
    counts = effective_counts(effective, state["verified"])
    pending = len(corrections) == 0
    if not pending:
        count_problems = validate_effective_counts(counts)
        if count_problems:
            _fail(f"effective counts do not match the approved expected counts: {count_problems}")
    return {
        "ok": True, "contract_content_sha256": state["verified"]["contract"]["content_sha256"],
        "original_ledger_byte_sha256": state["original_ledger_byte_sha256"],
        "total_original_records": len(state["original_records"]), "total_corrections": len(corrections),
        "approved_correction_queue_index": APPROVED_CORRECTION_QUEUE_INDEX,
        "approved_correction_pending": pending, "effective_counts": counts,
    }


def cmd_write(args) -> dict:
    dest = corrections_ledger_path_for(args.repo)
    if dest.exists():
        _fail(f"refusing to write: a correction artifact already exists: {dest}")
    _require_clean_tracked_tree(args.repo)
    head = _git_head(args.repo)
    generator_source_sha256 = _git_blob_canonical_lf_sha256(args.repo, head, GENERATOR_SOURCE_REL_PATH)
    working_tree_path = Path(args.repo) / GENERATOR_SOURCE_REL_PATH
    if not working_tree_path.exists():
        _fail(f"generator source file missing from the working tree: {working_tree_path}")
    working_tree_hash = mrc.canonical_lf_sha256_file(working_tree_path)
    if working_tree_hash != generator_source_sha256:
        _fail(f"{GENERATOR_SOURCE_REL_PATH} in the working tree does not match its committed content "
              f"at HEAD {head!r} -- commit the correction source before running --write")

    state = load_verified_original(args.repo)
    fields = build_approved_correction_fields(
        state["original_ledger_byte_sha256"], state["verified"]["contract"], head, generator_source_sha256,
        reviewer_id=args.reviewer_id, session_id=args.session_id, corrected_at_utc=args.corrected_at_utc)
    record = record_correction(args.repo, fields)
    return {"recorded": True, "queue_index": record["queue_index"], "record_hash": record["record_hash"]}


def cmd_check(args) -> dict:
    dest = corrections_ledger_path_for(args.repo)
    if not dest.exists():
        _fail(f"correction artifact does not exist yet: {dest} -- run --write first (a separately "
              f"authorized step)")
    result = cmd_preflight(args)
    if result["total_corrections"] != 1:
        _fail(f"--check requires exactly the one approved correction to exist; found "
              f"{result['total_corrections']}")
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", type=Path, default=HERE.parent)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    ap.add_argument("--reviewer-id")
    ap.add_argument("--session-id")
    ap.add_argument("--corrected-at-utc")
    args = ap.parse_args()

    try:
        if args.preflight:
            result = cmd_preflight(args)
        elif args.write:
            required = ("reviewer_id", "session_id", "corrected_at_utc")
            missing = [f"--{f.replace('_', '-')}" for f in required if getattr(args, f) is None]
            if missing:
                print(f"[correct_manual_review] FAILURE: --write requires: {missing}")
                return 1
            result = cmd_write(args)
        else:
            result = cmd_check(args)
        print(json.dumps(result, indent=2, default=str))
        return 0
    except (CorrectionError, mrt.ReviewToolError, mrc.ContractError) as e:
        print(f"[correct_manual_review] FAILURE: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
