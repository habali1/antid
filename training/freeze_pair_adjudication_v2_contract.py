#!/usr/bin/env python3
"""freeze_pair_adjudication_v2_contract.py — builds/checks
training/pair_adjudication_v2_contract.json, binding the ALREADY-PUBLISHED
training/pair_adjudication_v2_queue.json (generated first, independently,
by build_pair_adjudication_v2_queue.py --write) by its exact byte hash,
row count, and identity-order hash, plus canonical-LF source hashes and
immutable git-commit provenance for every approved v2 implementation
source file.

TWO-COMMIT LIFECYCLE (required -- see verify_implementation_source_provenance
in pair_adjudication_v2_contract.py for the full runtime check this binds
into): (1) a source-preparation commit adds/updates the six approved
implementation source files (and their tests/docs) for real, with nothing
else pending; only THEN can (2) `--write` run here, recording THAT commit
as `generator_git_commit` for every source -- never the working tree, and
never a commit that does not actually contain the path. `--write` refuses
outright unless the tracked tree is clean and every approved source is
tracked (committed) at HEAD; a source added to the working tree but not
yet committed is exactly the bug this refusal exists to catch.

Two modes:
  --check (default)  Reconstructs the expected content in memory and
                      compares it byte-for-byte against the existing file
                      on disk, THEN separately re-verifies the recorded
                      implementation-source provenance is still valid
                      (ancestor-of-HEAD, source exists with the recorded
                      hash at the recorded commit, AND the current working
                      tree still matches -- a bound-source change fails
                      even though the recorded generation commit itself
                      stays immutable). Writes NOTHING.
  --write             Atomically and exclusively writes the reconstructed
                      contract -- refuses to overwrite an existing file.

Usage:
    python freeze_pair_adjudication_v2_contract.py            # --check (default)
    python freeze_pair_adjudication_v2_contract.py --check
    python freeze_pair_adjudication_v2_contract.py --write
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

import pair_adjudication_v2_contract as v2c  # noqa: E402

CONTRACT_PATH = REPO / v2c.CONTRACT_REL_PATH

# THE single source of truth for the approved implementation-source name/
# path set is pair_adjudication_v2_contract.py -- never a locally
# redefined copy, so this generator and validate_contract_structure's
# schema check can never disagree about what "approved" means.
IMPLEMENTATION_SOURCE_PATHS = v2c.APPROVED_IMPLEMENTATION_SOURCE_PATHS


class FreezeError(RuntimeError):
    """Refuses to write. Always fails closed."""


def _fail(msg: str) -> None:
    raise FreezeError(msg)


def _require_clean_tracked_tree(repo: Path) -> None:
    """Requires the TRACKED working tree to be clean (mirrors the
    established gate_v2_contract.require_clean_tracked_tree /
    correct_manual_review._require_clean_tracked_tree pattern).
    Untracked files are explicitly allowed -- only drift in already-
    tracked content is a problem here. Required before --write: HEAD must
    be the exact, final, committed state this contract's provenance will
    bind to."""
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=str(repo), stderr=subprocess.PIPE,
        ).decode("utf-8")
    except (subprocess.CalledProcessError, OSError) as exc:
        _fail(f"could not check git status in {repo}: {exc}")
    if out.strip():
        _fail(f"tracked working tree is not clean, refusing to write the contract (commit the "
              f"source-preparation changes first):\n{out}")


def _current_git_head() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(REPO), capture_output=True, text=True)
    if result.returncode != 0:
        raise FreezeError(f"could not determine git HEAD: {result.stderr.strip()}")
    return result.stdout.strip()


def _build_queue_artifact(scan_reports: dict) -> dict:
    """Load the published queue and require it to be the exact canonical
    serialization mechanically derived from the already-verified candidate
    report. Binding a merely self-consistent row count/hash is insufficient:
    a dirty or substituted queue generator must not be able to freeze a queue
    that runtime verification would later reject."""
    queue_path = REPO / v2c.QUEUE_ARTIFACT_REL_PATH
    if not queue_path.exists():
        raise FreezeError(f"queue JSON does not exist yet -- run build_pair_adjudication_v2_queue.py "
                          f"--write first: {queue_path}")
    queue_bytes = queue_path.read_bytes()
    try:
        queue = json.loads(queue_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FreezeError(f"{queue_path} is not valid UTF-8 JSON: {exc}") from exc
    candidate_content = scan_reports["candidate_pairs_report"]["content"]
    try:
        expected_rows = v2c.derive_v2_queue_rows(candidate_content)
    except (KeyError, TypeError, ValueError) as exc:
        raise FreezeError(f"could not derive the canonical queue from the verified candidate report: {exc}") from exc
    expected_identity_sha256 = v2c.compute_v2_queue_identity_order_sha256(expected_rows)
    expected_queue = {
        "schema_version": v2c.SCHEMA_VERSION,
        "row_count": len(expected_rows),
        "identity_order_sha256": expected_identity_sha256,
        "rows": expected_rows,
    }
    expected_bytes = (json.dumps(expected_queue, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("utf-8")
    if queue_bytes != expected_bytes or queue != expected_queue:
        raise FreezeError(f"{queue_path} is not the exact canonical queue mechanically derived from the "
                          "verified candidate_pairs_report")

    row_count = queue.get("row_count")
    identity_order_sha256 = queue.get("identity_order_sha256")
    if not v2c.is_strict_int(row_count) or row_count != v2c.EXPECTED_SCOPED_TOTAL:
        raise FreezeError(f"{queue_path} row_count must be exactly {v2c.EXPECTED_SCOPED_TOTAL}, got {row_count!r}")
    if identity_order_sha256 != expected_identity_sha256:
        raise FreezeError(f"{queue_path} identity_order_sha256 is missing or malformed")
    return {
        "path": v2c.QUEUE_ARTIFACT_REL_PATH,
        "byte_sha256": v2c.sha256_bytes(queue_bytes),
        "row_count": row_count,
        "identity_order_sha256": identity_order_sha256,
    }


def _verify_scan_report_hashes() -> dict:
    try:
        return v2c.load_and_verify_scan_reports(REPO)
    except v2c.ScanEvidenceError as exc:
        raise FreezeError(str(exc)) from exc


def _build_outside_scope_identity_order_sha256(scan_reports: dict) -> str:
    candidate_pairs_content = scan_reports["candidate_pairs_report"]["content"]
    outside_scope_ids = v2c.derive_outside_scope_pair_ids(candidate_pairs_content)
    if len(outside_scope_ids) != v2c.EXPECTED_OUTSIDE_SCOPE_TOTAL:
        raise FreezeError(f"derived {len(outside_scope_ids)} outside-scope pair_ids, expected exactly "
                          f"{v2c.EXPECTED_OUTSIDE_SCOPE_TOTAL}")
    return v2c.compute_outside_scope_identity_order_sha256(outside_scope_ids)


def _require_sources_tracked_at_head(head: str) -> None:
    """Contract generation must refuse any source absent from the
    recorded commit -- checked BEFORE anything is bound, against the
    commit that --write is about to record (current HEAD, only reachable
    once the tracked tree is clean)."""
    for name, rel_path in sorted(IMPLEMENTATION_SOURCE_PATHS.items()):
        result = subprocess.run(["git", "show", f"{head}:{rel_path}"], cwd=str(REPO), capture_output=True)
        if result.returncode != 0:
            _fail(f"approved implementation source {rel_path!r} (name {name!r}) is not tracked at HEAD "
                  f"{head!r} -- it must be committed (the source-preparation commit) before the contract "
                  f"can be generated; refusing to record provenance against a commit that does not "
                  f"actually contain this path")


def _build_implementation_sources(head: str) -> dict:
    """For each APPROVED implementation source: the canonical-LF sha256
    of its content AT `head` (read via `git show`, never the working
    tree -- `_require_clean_tracked_tree` already guarantees the working
    tree and HEAD agree byte-for-byte for tracked content at this point,
    but reading from `head` is what makes the recorded provenance
    genuinely a property of a commit, not of this invocation's local
    disk state). `generator_git_commit` is always `head` for every source
    -- one shared, single, real commit, never a per-file value, and never
    the working tree."""
    sources = {}
    for name, rel_path in sorted(IMPLEMENTATION_SOURCE_PATHS.items()):
        committed_sha256 = v2c.git_show_canonical_lf_sha256(REPO, head, rel_path)
        if committed_sha256 is None:
            _fail(f"could not read {rel_path!r} at commit {head!r} -- refusing to record invalid provenance")
        sources[name] = {"path": rel_path, "sha256": committed_sha256, "generator_git_commit": head}
    return sources


def build_contract(*, implementation_sources: dict | None = None) -> dict:
    """When `implementation_sources` is omitted (the --write path): built
    fresh against the CURRENT git HEAD, which must already track every
    approved source (checked by _require_sources_tracked_at_head) -- this
    is the only place a NEW generation commit is ever chosen. When given
    (the --check path): the PUBLISHED contract's own implementation_
    sources dict is reused VERBATIM, never rebuilt against current HEAD
    -- an unrelated later commit must not change what --check reconstructs
    and compares byte-for-byte. Callers of the --check path separately
    re-verify that pinned provenance is still currently valid via
    v2c.verify_implementation_source_provenance."""
    scan_reports = _verify_scan_report_hashes()
    queue_artifact = _build_queue_artifact(scan_reports)
    outside_scope_identity_order_sha256 = _build_outside_scope_identity_order_sha256(scan_reports)
    if implementation_sources is None:
        head = _current_git_head()
        _require_sources_tracked_at_head(head)
        implementation_sources = _build_implementation_sources(head)
    content = v2c.build_contract_content(queue_artifact=queue_artifact, implementation_sources=implementation_sources,
                                         outside_scope_identity_order_sha256=outside_scope_identity_order_sha256)
    content_sha256 = v2c.compute_content_sha256(content)
    contract = {
        "schema_version": v2c.SCHEMA_VERSION,
        "status": v2c.CONTRACT_STATUS_FROZEN,
        "content": content,
        "content_sha256": content_sha256,
        "generation": {
            "note": v2c.CONTRACT_GENERATION_NOTE,
            "generator": v2c.CONTRACT_GENERATOR_PATH,
        },
    }
    problems = v2c.validate_contract_structure(contract)
    if problems:
        raise FreezeError(f"reconstructed contract failed self-validation: {problems}")
    return contract


def serialize(contract: dict) -> str:
    return json.dumps(contract, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def cmd_check() -> int:
    if not CONTRACT_PATH.exists():
        print(f"[check] FAILED: {CONTRACT_PATH} does not exist")
        return 1
    actual_bytes = CONTRACT_PATH.read_bytes()
    try:
        published = json.loads(actual_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        print(f"[check] FAILED: {CONTRACT_PATH} is not valid JSON: {exc}")
        return 1
    published_problems = v2c.validate_contract_structure(published)
    if published_problems:
        print(f"[check] FAILED: {CONTRACT_PATH} failed structural validation: {published_problems}")
        return 1
    published_content = published.get("content") if isinstance(published, dict) else None
    published_sources = published_content.get("implementation_sources") if isinstance(published_content, dict) else None
    if not isinstance(published_sources, dict):
        print(f"[check] FAILED: {CONTRACT_PATH} content.implementation_sources is missing or malformed")
        return 1

    # Rebuild everything EXCEPT implementation_sources fresh against the
    # CURRENT committed state (queue, scan reports, outside-scope
    # identity) -- these are never commit-pinned, so re-deriving them
    # from current state is correct and required (catches a stale/
    # tampered queue or scan report). implementation_sources is reused
    # VERBATIM from the published file (pinned, immutable-generation-
    # commit data) so an unrelated later commit never breaks this
    # byte-for-byte comparison.
    reconstructed = build_contract(implementation_sources=published_sources)
    reconstructed_contract = dict(reconstructed)
    reconstructed_contract["generation"] = published.get("generation", reconstructed["generation"])
    expected_bytes = serialize(reconstructed_contract).encode("utf-8")
    if actual_bytes != expected_bytes:
        print(f"[check] FAILED: {CONTRACT_PATH} does not match the reconstructed content byte-for-byte")
        return 1

    # Beyond byte-identity: separately re-verify the ALREADY-PUBLISHED
    # file's own recorded provenance is still CURRENTLY valid (ancestor
    # check + commit-content check + working-tree check) -- this is what
    # actually exercises "immutable generation commit, but a bound-source
    # change fails", independent of whether HEAD has moved on since.
    provenance_problems = v2c.verify_implementation_source_provenance(REPO, published_content)
    if provenance_problems:
        print(f"[check] FAILED: published contract's implementation-source provenance is no longer valid: "
             f"{provenance_problems}")
        return 1

    print(f"[check] PASS: {CONTRACT_PATH} matches the reconstructed content byte-for-byte, and its "
         f"implementation-source provenance re-verifies against the current repository")
    print(f"content_sha256: {published.get('content_sha256')}")
    return 0


def cmd_write() -> int:
    if CONTRACT_PATH.exists():
        raise FreezeError(f"refusing to overwrite existing contract: {CONTRACT_PATH}")
    _require_clean_tracked_tree(REPO)
    contract = build_contract()
    data = serialize(contract).encode("utf-8")
    CONTRACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CONTRACT_PATH.with_suffix(CONTRACT_PATH.suffix + f".tmp{os.getpid()}")
    try:
        with tmp.open("wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.link(str(tmp), str(CONTRACT_PATH))
    finally:
        tmp.unlink(missing_ok=True)
    published_bytes = CONTRACT_PATH.read_bytes()
    if published_bytes != data:
        raise FreezeError(f"published {CONTRACT_PATH} does not match staged bytes -- left on disk for review")
    print(f"wrote {CONTRACT_PATH}")
    print(f"content_sha256: {contract['content_sha256']}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--write", action="store_true")
    args = ap.parse_args()

    try:
        return cmd_write() if args.write else cmd_check()
    except FreezeError as e:
        print(f"[freeze_pair_adjudication_v2_contract] FAILURE: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
