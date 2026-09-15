#!/usr/bin/env python3
"""pair_adjudication_v2_contract.py — the single shared schema/constants/
arithmetic module for the Phase 5F4 perceptual-pair adjudication v2
protocol.

This is a SEPARATE, additive, versioned system layered on top of the
already-committed, immutable Phase 5F1 scan evidence (the frozen Phase 5F1
contract, the scanner, the five committed scan reports, the perceptual
thresholds, and all 16,539 candidates). It revises FINALIZATION SCOPE
only: which candidates must be adjudicated before which conclusion may be
drawn, and how those conclusions are reported. It does not alter the scan,
the perceptual thresholds, the candidate labels' meanings, or any
candidate record -- every value pulled from the five committed reports is
verified against them fresh, never assumed.

Three independent workstreams, mechanically derived from (and verified
against) the real committed `candidate_pairs_report`:

  A. final_test_independence (BLOCKING) -- 1,681 candidates across the 7
     domains pairing final_test against train/development/the 5
     other-evidence sets. A confirmed same_source_image here means
     final_test_independence=failed and stop_before_final_test_inference
     =true.
  B. final_test_internal_repetition (DIAGNOSTIC) -- 53 candidates,
     within_final_test only. Confirmed same-source pairs here are an
     effective-sample-size limitation, reported as such -- they do NOT
     constitute development-data leakage and do NOT alone set
     stop_before_final_test_inference.
  C. gate_evidence_independence (BLOCKING) -- 2,238 candidates across 5
     domains binding calibration_v2/expansion_development/expansion_train
     against each other and against unknown_test_v2. A confirmed
     same_source_image here means gate_evidence_independence=failed and
     stop_before_gate_promotion=true. The frozen 0.61 threshold is
     unchanged; remediation/deployment is a later, separately reviewed
     decision.

Scoped total 3,972 + outside-scope 12,567 = the full frozen scan's 16,539.
Every outside-scope candidate is reported as not_adjudicated, never
not_duplicate -- silence about a candidate is not evidence of anything.

Nothing in this module opens an image or touches a model.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

import manual_review_contract as mrc

# ---------------------------------------------------------------- generic --
# Pure, generic byte/hash/schema utilities are reused directly from
# manual_review_contract.py (read-only import; that module is never
# edited) rather than duplicated -- the established convention throughout
# this codebase (e.g. gate_v2_evaluation_contract.py reusing
# gate_v2_contract.py).
is_sha256_hex = mrc.is_sha256_hex
is_strict_int = mrc.is_strict_int
sha256_bytes = mrc.sha256_bytes
sha256_file = mrc.sha256_file
canonical_json_bytes = mrc.canonical_json_bytes
compute_content_sha256 = mrc.compute_content_sha256
canonical_lf_bytes = mrc.canonical_lf_bytes
canonical_lf_sha256_file = mrc.canonical_lf_sha256_file
is_strict_utc_timestamp = mrc.is_strict_utc_timestamp

SCHEMA_VERSION = 1
CONTRACT_STATUS_FROZEN = "frozen_before_pair_adjudication_v2"
CONTRACT_REL_PATH = "training/pair_adjudication_v2_contract.json"
GENERATOR_PROTOCOL_VERSION = "pair-adjudication-v2-protocol-v1"

QUEUE_ARTIFACT_REL_PATH = "training/pair_adjudication_v2_queue.json"

APPROVED_OUTPUT_PATHS = {
    "pair_adjudication_v2_ledger": "training/pair_adjudication_v2_ledger.jsonl",
    "pair_adjudication_v2_lock": "training/pair_adjudication_v2.lock",
    "post_adjudication_stop_status_v2_report":
        "training/perceptual_duplicate_post_adjudication_stop_status_v2.json",
}

# ---------------------------------------------------- frozen input bindings --
# The already-committed, immutable Phase 5F1 contract and the five
# committed scan reports this protocol is layered on top of. These are
# LITERAL frozen values (verified against the real files fresh on every
# load via load_and_verify_v2_inputs() below) -- never re-derived, never
# assumed to still be true just because they were true once.
PHASE_5F1_CONTRACT_CONTENT_SHA256 = "f32133d6c532b89890f5ec026e338eb71d029c1a60cedce0ef9bc00ef852e4a9"

SCAN_REPORT_HASHES = {
    "perceptual_hashes_report": {"byte_sha256": "d0fd1f6ec081d3387a01228b409d80cc439ff002774a7efb318e930b7f37b58d",
                                 "content_sha256": "c5e5af6e10349c926e2098b358b8c7a41988d7d530189288f5248d9e60d0ad6b"},
    "metadata_leakage_report": {"byte_sha256": "4c962c36a7f60ed5e7305162ab430f7860262ef972145fd9a3a6dd0b836342cb",
                                "content_sha256": "52c824eac24b20a2deaf3c71dc2353dae5c8513e3017ac908d5d145f82d31066"},
    "candidate_pairs_report": {"byte_sha256": "3ce05cf2cdb91d77d6a5b153e4ee4d1f6be1fbd29de1788e2a1bac3964c9f93b",
                               "content_sha256": "aa565447b7aa5c9ac19f997eaa314d9ebeaa5f562a12ebb35f0baec02332a769"},
    "domain_summary_report": {"byte_sha256": "38075e7c9203750274bc85394a4488062eb50a0ba101eaeb4718756162865082",
                              "content_sha256": "cf673de4d409c070bcf814bcf6ba834378d044997ce2b18bb403a17886e9b27d"},
    "stop_status_report": {"byte_sha256": "91dfc220ce82949990a2f18896b6e6d53bf6d0f3db8a8b2dc0dea49bf4a52deb",
                           "content_sha256": "b6cce48c5156a97bef504b60d670758ef13c42b2edf966c7f9ba667583492947"},
}
assert set(SCAN_REPORT_HASHES) == set(mrc.SCAN_REPORT_NAMES)
SCAN_REPORT_HASH_ENTRY_KEYS = frozenset({"byte_sha256", "content_sha256"})

SCAN_REPORT_REL_PATHS = {
    "perceptual_hashes_report": "training/perceptual_duplicate_hashes.json",
    "metadata_leakage_report": "training/perceptual_duplicate_metadata_leakage.json",
    "candidate_pairs_report": "training/perceptual_duplicate_candidate_pairs.json",
    "domain_summary_report": "training/perceptual_duplicate_domain_summary.json",
    "stop_status_report": "training/perceptual_duplicate_stop_status.json",
}
assert set(SCAN_REPORT_REL_PATHS) == set(SCAN_REPORT_HASHES)


class ScanEvidenceError(RuntimeError):
    """A committed scan report on disk no longer matches its frozen
    approved hash. Always fails closed."""


def load_and_verify_scan_reports(repo: Path) -> dict:
    """Re-verifies every one of the five committed Phase 5F1/5F3 scan
    reports' byte AND content sha256 against SCAN_REPORT_HASHES, fresh,
    from the files on disk -- never trusts that the module constants are
    still an accurate description of the repository. Shared by
    freeze_pair_adjudication_v2_contract.py, build_pair_adjudication_v2_queue.py,
    and adjudicate_pairs_v2.py so there is only one copy of this check.
    Returns {report_name: {"content": ..., "byte_sha256": ..., "raw": bytes}}."""
    repo = Path(repo)
    reports: dict[str, dict] = {}
    for name, rel_path in SCAN_REPORT_REL_PATHS.items():
        path = repo / rel_path
        if not path.exists():
            raise ScanEvidenceError(f"committed scan report does not exist: {path}")
        raw = path.read_bytes()
        byte_sha256 = sha256_bytes(raw)
        expected = SCAN_REPORT_HASHES[name]
        if byte_sha256 != expected["byte_sha256"]:
            raise ScanEvidenceError(f"{path} byte_sha256 {byte_sha256} != frozen approved "
                                    f"{expected['byte_sha256']!r} -- committed scan evidence has changed")
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ScanEvidenceError(f"{path} is not valid UTF-8 JSON: {exc}") from exc
        content = parsed.get("content")
        if not isinstance(content, dict):
            raise ScanEvidenceError(f"{path} has no content object")
        try:
            content_sha256 = compute_content_sha256(content)
        except (TypeError, ValueError) as exc:
            raise ScanEvidenceError(f"{path} content is not canonicalizable: {exc}") from exc
        if content_sha256 != expected["content_sha256"]:
            raise ScanEvidenceError(f"{path} content_sha256 {content_sha256} != frozen approved "
                                    f"{expected['content_sha256']!r} -- committed scan evidence has changed")
        reports[name] = {"content": content, "byte_sha256": byte_sha256, "content_sha256": content_sha256, "raw": raw}
    return reports


# ----------------------------------------- approved implementation sources --
# The exact, frozen name/path set every v2 contract's implementation_sources
# must bind -- no more, no fewer, and never a different path under a
# recycled name. THE single source of truth (freeze_pair_adjudication_v2_
# contract.py imports this rather than keeping its own copy) so a renamed
# or substituted source can never slip past a divergent local list.
APPROVED_IMPLEMENTATION_SOURCE_PATHS = {
    "pair_adjudication_v2_contract": "training/pair_adjudication_v2_contract.py",
    "build_pair_adjudication_v2_queue": "training/build_pair_adjudication_v2_queue.py",
    "freeze_pair_adjudication_v2_contract": "training/freeze_pair_adjudication_v2_contract.py",
    "adjudicate_pairs_v2": "training/adjudicate_pairs_v2.py",
    "pair_adjudication_v2_review": "training/pair_adjudication_v2_review.py",
    "finalize_stop_status_v2": "training/finalize_stop_status_v2.py",
}


# --------------------------------------------- git-backed source provenance --
# Every function below is metadata/git-only (never opens an image, never
# runs inference) and NEVER raises -- git/OS failures are reported as
# `None`/`False`/problem-list entries, the same fail-closed-but-controlled
# discipline as everything else in this module.
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


def _git_head(repo: Path) -> str | None:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(repo), capture_output=True, text=True)
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def git_show_canonical_lf_sha256(repo: Path, commit: str, rel_path: str) -> str | None:
    """The canonical-LF sha256 of rel_path's content AT commit (read via
    `git show`, never the working tree), or None if that commit does not
    track the path (or git/OS itself fails) -- never raises."""
    try:
        result = subprocess.run(["git", "show", f"{commit}:{rel_path}"], cwd=str(repo), capture_output=True)
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return sha256_bytes(canonical_lf_bytes(result.stdout))


def verify_implementation_source_provenance(repo: Path, content: dict) -> list[str]:
    """Runtime (git-access) verification of content["implementation_sources"]
    -- goes BEYOND validate_contract_structure's pure schema check, which
    only confirms shape. For every one of the APPROVED_IMPLEMENTATION_
    SOURCE_PATHS entries: the recorded generator_git_commit must be a
    strict 40-hex string that resolves to a real commit object AND is an
    ancestor of (or equal to) the current HEAD -- an unrelated LATER
    commit never invalidates this (the immutable-generation-commit
    pattern), but a commit that never existed, or one not yet merged into
    history, is rejected. The canonical-LF hash of that exact path's
    content AT that recorded commit (read via `git show`, never the
    working tree) must equal the recorded sha256 -- this is what actually
    proves the source existed, with this content, at that commit (a
    contract naming a commit where the path never existed, or existed
    with different content, fails here). The CURRENT working-tree copy of
    that same path must ALSO hash to the recorded sha256 -- unlike Gate
    v2's diagnostic-only provenance, this binding is AUTHORITATIVE: any
    edit to a bound source since generation, committed or not, makes the
    contract's implementation-source provenance invalid, not merely
    stale. Returns a problem list; empty means every source's provenance
    is currently valid. Never raises."""
    repo = Path(repo)
    problems: list[str] = []
    impl = content.get("implementation_sources") if isinstance(content, dict) else None
    if not isinstance(impl, dict):
        return ["content.implementation_sources is not a JSON object"]
    current_head = _git_head(repo)
    if current_head is None:
        return [f"could not determine current git HEAD in {repo}"]
    for name, rel_path in APPROVED_IMPLEMENTATION_SOURCE_PATHS.items():
        entry = impl.get(name)
        if not isinstance(entry, dict):
            problems.append(f"implementation_sources[{name!r}] is missing")
            continue
        commit = entry.get("generator_git_commit")
        if not is_git_commit_hex(commit):
            problems.append(f"implementation_sources[{name!r}].generator_git_commit is not a strict "
                            f"40-character lowercase hex commit hash")
            continue
        obj_type = _git_object_type(repo, commit)
        if obj_type != "commit":
            problems.append(f"implementation_sources[{name!r}].generator_git_commit {commit!r} does not "
                            f"resolve to a commit object in this repository (git cat-file -t reported "
                            f"{obj_type!r})")
            continue
        if not _git_is_ancestor(repo, commit, current_head):
            problems.append(f"implementation_sources[{name!r}].generator_git_commit {commit!r} is not an "
                            f"ancestor of (or equal to) the current HEAD {current_head!r}")
            continue

        recorded_sha256 = entry.get("sha256")
        if not is_sha256_hex(recorded_sha256):
            problems.append(f"implementation_sources[{name!r}].sha256 is not a 64-char lowercase hex string")
            continue

        committed_sha256 = git_show_canonical_lf_sha256(repo, commit, rel_path)
        if committed_sha256 is None:
            problems.append(f"implementation_sources[{name!r}] path {rel_path!r} does not exist at commit "
                            f"{commit!r} -- invalid provenance (a source can never be recorded against a "
                            f"commit that does not actually contain it)")
            continue
        if committed_sha256 != recorded_sha256:
            problems.append(f"implementation_sources[{name!r}] recorded sha256 {recorded_sha256!r} does not "
                            f"match the canonical-LF hash {committed_sha256!r} of {rel_path!r} as it existed "
                            f"at commit {commit!r}")

        working_tree_path = repo / rel_path
        if not working_tree_path.exists():
            problems.append(f"implementation_sources[{name!r}] path {rel_path!r} does not exist in the "
                            f"current working tree")
            continue
        working_tree_sha256 = canonical_lf_sha256_file(working_tree_path)
        if working_tree_sha256 != recorded_sha256:
            problems.append(f"implementation_sources[{name!r}] current working-tree content of {rel_path!r} "
                            f"no longer matches its recorded/committed provenance -- the bound source has "
                            f"changed since generation")
    return problems


# ----------------------------------------------------------- workstreams --
WORKSTREAM_FINAL_TEST_INDEPENDENCE = "final_test_independence"
WORKSTREAM_FINAL_TEST_INTERNAL_REPETITION = "final_test_internal_repetition"
WORKSTREAM_GATE_EVIDENCE_INDEPENDENCE = "gate_evidence_independence"

CHANNEL_ROLE_BLOCKING = "blocking"
CHANNEL_ROLE_DIAGNOSTIC = "diagnostic"

WORKSTREAM_ROLE = {
    WORKSTREAM_FINAL_TEST_INDEPENDENCE: CHANNEL_ROLE_BLOCKING,
    WORKSTREAM_FINAL_TEST_INTERNAL_REPETITION: CHANNEL_ROLE_DIAGNOSTIC,
    WORKSTREAM_GATE_EVIDENCE_INDEPENDENCE: CHANNEL_ROLE_BLOCKING,
}

CHANNEL_A_DOMAINS = (
    "benchmark_v1_vs_final_test", "calibration_v1_vs_final_test", "calibration_v2_vs_final_test",
    "expansion_development_vs_final_test", "expansion_train_vs_final_test",
    "final_test_vs_unknown_test_v1", "final_test_vs_unknown_test_v2",
)
CHANNEL_B_DOMAINS = ("within_final_test",)
CHANNEL_C_DOMAINS = (
    "calibration_v2_vs_expansion_development", "calibration_v2_vs_expansion_train",
    "expansion_development_vs_unknown_test_v2", "expansion_train_vs_unknown_test_v2",
    "calibration_v2_vs_unknown_test_v2",
)

DOMAIN_WORKSTREAM: dict[str, str] = {}
DOMAIN_WORKSTREAM.update({d: WORKSTREAM_FINAL_TEST_INDEPENDENCE for d in CHANNEL_A_DOMAINS})
DOMAIN_WORKSTREAM.update({d: WORKSTREAM_FINAL_TEST_INTERNAL_REPETITION for d in CHANNEL_B_DOMAINS})
DOMAIN_WORKSTREAM.update({d: WORKSTREAM_GATE_EVIDENCE_INDEPENDENCE for d in CHANNEL_C_DOMAINS})

# Frozen expected per-domain candidate counts, mechanically verified
# against the real committed candidate_pairs_report (never merely copied
# and trusted) both here (module-load-time assertions below) and again at
# every load_and_verify_v2_inputs() call against the live file.
EXPECTED_DOMAIN_COUNTS = {
    "benchmark_v1_vs_final_test": 332, "calibration_v1_vs_final_test": 253,
    "calibration_v2_vs_final_test": 188, "expansion_development_vs_final_test": 104,
    "expansion_train_vs_final_test": 569, "final_test_vs_unknown_test_v1": 133,
    "final_test_vs_unknown_test_v2": 102,
    "within_final_test": 53,
    "calibration_v2_vs_expansion_development": 182, "calibration_v2_vs_expansion_train": 1041,
    "expansion_development_vs_unknown_test_v2": 115, "expansion_train_vs_unknown_test_v2": 662,
    "calibration_v2_vs_unknown_test_v2": 238,
}
assert set(EXPECTED_DOMAIN_COUNTS) == set(DOMAIN_WORKSTREAM)

EXPECTED_SCOPED_TOTAL = 3972
EXPECTED_OUTSIDE_SCOPE_TOTAL = 12567
EXPECTED_FULL_SCAN_TOTAL = 16539
assert sum(EXPECTED_DOMAIN_COUNTS.values()) == EXPECTED_SCOPED_TOTAL
assert EXPECTED_SCOPED_TOTAL + EXPECTED_OUTSIDE_SCOPE_TOTAL == EXPECTED_FULL_SCAN_TOTAL

EXPECTED_BOTH_RULE_COUNT_IN_SCOPE = 18
EXPECTED_EXACT_ZERO_COUNT_IN_SCOPE = 0

# ---- deterministic domain visitation order for queue construction -------
# Documented, fixed, and tested against the real committed candidate
# report. The frozen priority order is LITERAL: (a) ALL 18 both-pHash-
# and-dHash-rule candidates first, globally, ahead of every standard
# candidate; then the standard candidates domain-by-domain in this fixed
# sequence. The 18 both-rule candidates happen to fall in exactly 5 of
# these 13 domains (expansion_train_vs_final_test, final_test_vs_
# unknown_test_v2, calibration_v2_vs_expansion_train, expansion_
# development_vs_unknown_test_v2, expansion_train_vs_unknown_test_v2) --
# derive_v2_queue_rows visits THIS SAME order to pull out each such
# domain's both-rule rows into their own small, single-domain priority
# block (never merged across domains, and never merged with that same
# domain's later standard block) before any standard-phase row is queued.
# This is how "all 18 both-rule pairs first" and "one domain per session"
# are both satisfied: the priority phase spends 5 small blocks, then the
# standard phase proceeds exactly as if those 18 rows had never existed.
DOMAIN_VISIT_ORDER = (
    "calibration_v2_vs_unknown_test_v2",
    "within_final_test",
    "benchmark_v1_vs_final_test",
    "calibration_v1_vs_final_test",
    "calibration_v2_vs_final_test",
    "expansion_development_vs_final_test",
    "expansion_train_vs_final_test",
    "final_test_vs_unknown_test_v1",
    "final_test_vs_unknown_test_v2",
    "calibration_v2_vs_expansion_development",
    "calibration_v2_vs_expansion_train",
    "expansion_development_vs_unknown_test_v2",
    "expansion_train_vs_unknown_test_v2",
)
assert set(DOMAIN_VISIT_ORDER) == set(EXPECTED_DOMAIN_COUNTS)
assert len(DOMAIN_VISIT_ORDER) == len(set(DOMAIN_VISIT_ORDER)) == 13

PRIORITY_TIER_BOTH_RULE = 1
PRIORITY_TIER_STANDARD = 2

MAX_PAIRS_PER_SESSION = 300
RECOMMENDED_SESSION_MINUTES = (60, 90)

# ---------------------------------------------- adjudication label semantics --
# The four labels themselves are REUSED, not redefined, from the
# already-frozen Phase 5F1 contract (mrc.ADJUDICATION_LABELS) -- the same
# frozenset object, so they can never silently drift between v1 and v2.
ADJUDICATION_LABELS = mrc.ADJUDICATION_LABELS

STATUS_PASSED = "passed"
STATUS_FAILED = "failed"
STATUS_INCONCLUSIVE = "inconclusive"
STATUS_PENDING = "pending"

# Frozen, verbatim semantics for each label WITHIN a blocking channel
# (final_test_independence / gate_evidence_independence). Diagnostic
# channel (final_test_internal_repetition) semantics are handled
# separately below -- it is never "failed", only reported.
LABEL_CHANNEL_EFFECT = {
    "same_source_image": "blocking_failure",       # channel -> failed, its stop flag -> true
    "different_photo_same_observation": "adjudicated_not_duplicate",  # adjudicated; not a same-source-image duplicate
    "different_image": "cleared",
    "uncertain": "blocking_inconclusive",          # channel -> inconclusive, never cleared/passed
}
assert set(LABEL_CHANNEL_EFFECT) == set(ADJUDICATION_LABELS)

NOT_ADJUDICATED = "not_adjudicated"
NOT_ADJUDICATED_DUE_TO_EARLY_STOP = "not_adjudicated_due_to_early_stop"

REMEDIATION_RULE = (
    "This protocol revises finalization SCOPE only -- it does not alter the "
    "Phase 5F1 scan, the frozen pHash<=10/dHash<=8 candidate rule, the "
    "frozen 0.61 calibration threshold, or the meaning of any of the four "
    "adjudication labels. A confirmed same_source_image in "
    "final_test_independence or gate_evidence_independence is a BLOCKING "
    "finding for that channel alone -- the two blocking channels are "
    "reported and stopped independently, never collapsed into one pass/"
    "fail field. A confirmed same_source_image in "
    "final_test_internal_repetition is reported as an effective-sample-"
    "size limitation of northeast_final_test_v1 -- it is NOT development-"
    "data leakage and does not by itself set stop_before_final_test_"
    "inference. An 'uncertain' label makes its channel inconclusive and "
    "blocking; it is never treated as cleared or passed, and resolving it "
    "requires a separately reviewed correction/second-review protocol, "
    "never a silent re-adjudication. No image is removed, replaced, or "
    "relabeled. No candidate outside the 3,972-pair mandatory scope is "
    "ever reported as not_duplicate -- an outside-scope or early-stop-"
    "skipped candidate is reported as not_adjudicated, full stop. "
    "Remediation or deployment following a blocking finding is a later, "
    "separately reviewed decision -- this protocol only detects and "
    "reports, it does not remediate."
)

# ------------------------------------------------------------- queue schema --
QUEUE_ROW_REQUIRED_FIELDS = frozenset({
    "queue_index", "domain", "workstream", "channel_role", "priority_tier", "both_rule_match",
    "pair_id", "identity_a", "identity_b", "phash_distance", "dhash_distance",
    "session_block", "suggested_session",
})


def derive_v2_queue_rows(candidate_pairs_content: dict) -> list[dict]:
    """Pure, deterministic, metadata-only: from the committed
    candidate_pairs_report's `content` dict (as loaded from disk, never
    re-derives hashes, never opens an image), builds the exact 3,972-row
    v2 adjudication queue under the frozen LITERAL priority order:

      Phase 1 (priority, tier 1): ALL both-pHash-and-dHash-rule candidates
      first, globally -- visited domain-by-domain in DOMAIN_VISIT_ORDER
      (skipping any domain with none), each domain's both-rule rows
      sorted by pair_id and forming their own small, single-domain
      priority block.
      Phase 2 (standard, tier 2): every remaining candidate, domain-by-
      domain in DOMAIN_VISIT_ORDER, sorted by pair_id -- exactly as if
      the phase-1 rows had never existed.

    Raises ValueError (never silently produces a wrong-shaped queue) if a
    domain is missing or its row count disagrees with
    EXPECTED_DOMAIN_COUNTS."""
    domains = candidate_pairs_content["domains"]
    domain_both: dict[str, list[dict]] = {}
    domain_rest: dict[str, list[dict]] = {}
    for domain_name in DOMAIN_VISIT_ORDER:
        pairs = domains.get(domain_name)
        if not isinstance(pairs, list):
            raise ValueError(f"candidate_pairs_report.content.domains[{domain_name!r}] is missing or not a list")
        if len(pairs) != EXPECTED_DOMAIN_COUNTS[domain_name]:
            raise ValueError(f"domain {domain_name!r} has {len(pairs)} candidates, expected exactly "
                            f"{EXPECTED_DOMAIN_COUNTS[domain_name]}")
        domain_both[domain_name] = sorted((p for p in pairs if p["phash_distance"] <= mrc.PHASH_MAX_HAMMING_DISTANCE
                                          and p["dhash_distance"] <= mrc.DHASH_MAX_HAMMING_DISTANCE),
                                          key=lambda p: p["pair_id"])
        domain_rest[domain_name] = sorted((p for p in pairs if not (p["phash_distance"] <= mrc.PHASH_MAX_HAMMING_DISTANCE
                                          and p["dhash_distance"] <= mrc.DHASH_MAX_HAMMING_DISTANCE)),
                                          key=lambda p: p["pair_id"])

    def _append(rows, queue_index, domain_name, tier, pair_list):
        workstream = DOMAIN_WORKSTREAM[domain_name]
        role = WORKSTREAM_ROLE[workstream]
        for pair in pair_list:
            rows.append({
                "queue_index": queue_index, "domain": domain_name, "workstream": workstream,
                "channel_role": role, "priority_tier": tier,
                "both_rule_match": tier == PRIORITY_TIER_BOTH_RULE,
                "pair_id": pair["pair_id"], "identity_a": pair["identity_a"], "identity_b": pair["identity_b"],
                "phash_distance": pair["phash_distance"], "dhash_distance": pair["dhash_distance"],
            })
            queue_index += 1
        return queue_index

    rows: list[dict] = []
    queue_index = 0
    # Phase 1: every both-rule candidate, globally first, domain order = DOMAIN_VISIT_ORDER.
    for domain_name in DOMAIN_VISIT_ORDER:
        if domain_both[domain_name]:
            queue_index = _append(rows, queue_index, domain_name, PRIORITY_TIER_BOTH_RULE, domain_both[domain_name])
    # Phase 2: every remaining candidate, domain order = DOMAIN_VISIT_ORDER.
    for domain_name in DOMAIN_VISIT_ORDER:
        if domain_rest[domain_name]:
            queue_index = _append(rows, queue_index, domain_name, PRIORITY_TIER_STANDARD, domain_rest[domain_name])

    if len(rows) != EXPECTED_SCOPED_TOTAL:
        raise ValueError(f"derived queue has {len(rows)} rows, expected exactly {EXPECTED_SCOPED_TOTAL}")
    both_rule_total = sum(1 for r in rows if r["both_rule_match"])
    if both_rule_total != EXPECTED_BOTH_RULE_COUNT_IN_SCOPE:
        raise ValueError(f"derived queue has {both_rule_total} both-rule pairs, expected exactly "
                        f"{EXPECTED_BOTH_RULE_COUNT_IN_SCOPE}")
    exact_zero_total = sum(1 for r in rows if r["phash_distance"] == 0 and r["dhash_distance"] == 0)
    if exact_zero_total != EXPECTED_EXACT_ZERO_COUNT_IN_SCOPE:
        raise ValueError(f"derived queue has {exact_zero_total} exact pHash=0/dHash=0 pairs, expected exactly "
                        f"{EXPECTED_EXACT_ZERO_COUNT_IN_SCOPE}")
    seen_pair_ids = set()
    for r in rows:
        if r["pair_id"] in seen_pair_ids:
            raise ValueError(f"pair_id {r['pair_id']!r} appears more than once in the derived queue")
        seen_pair_ids.add(r["pair_id"])
    assign_session_blocks(rows)
    return rows


def assign_session_blocks(rows: list[dict]) -> None:
    """Mutates each row in place, adding `session_block` (0-based index
    within its domain) and `suggested_session` (a stable, human-readable
    identifier `{domain}__block{N}`). Each session/block holds rows from
    EXACTLY one domain, at most MAX_PAIRS_PER_SESSION of them, and from
    exactly one priority tier -- a domain's tier-1 (priority, both-rule)
    rows NEVER share a block with that same domain's tier-2 (standard)
    rows, even though both eventually appear under the same domain name;
    this is what keeps the 5 both-rule priority blocks small and distinct
    from their domain's later bulk block, per the frozen literal priority
    order. Rows are grouped by domain preserving queue order (which
    already puts every tier-1 row before every tier-2 row), then split
    into contiguous same-tier runs; each run is chunked into blocks of at
    most MAX_PAIRS_PER_SESSION, with block numbering continuing across
    runs within the same domain (never restarting at 0 for the second
    run) so no two blocks of one domain collide."""
    by_domain: dict[str, list[dict]] = {}
    for row in rows:
        by_domain.setdefault(row["domain"], []).append(row)
    for domain_name, domain_rows in by_domain.items():
        next_block = 0
        idx = 0
        n = len(domain_rows)
        while idx < n:
            tier = domain_rows[idx].get("priority_tier")
            run: list[dict] = []
            while idx < n and domain_rows[idx].get("priority_tier") == tier:
                run.append(domain_rows[idx])
                idx += 1
            for i, row in enumerate(run):
                block = next_block + i // MAX_PAIRS_PER_SESSION
                row["session_block"] = block
                row["suggested_session"] = f"{domain_name}__block{block}"
            blocks_used = (len(run) + MAX_PAIRS_PER_SESSION - 1) // MAX_PAIRS_PER_SESSION
            next_block += blocks_used


def compute_v2_queue_identity_order_sha256(rows: list[dict]) -> str:
    """Binds exact row order AND content -- a reordering or any field
    change produces a different hash."""
    parts = []
    for r in rows:
        parts.append("\x1f".join([
            str(r["queue_index"]), r["domain"], r["workstream"], str(r["priority_tier"]),
            "1" if r["both_rule_match"] else "0", r["pair_id"], r["identity_a"], r["identity_b"],
            str(r["phash_distance"]), str(r["dhash_distance"]),
            str(r["session_block"]), r["suggested_session"],
        ]))
    return sha256_bytes("\x1e".join(parts).encode("utf-8"))


def derive_outside_scope_pair_ids(candidate_pairs_content: dict) -> list[str]:
    """Pure, deterministic, metadata-only: every pair_id in the committed
    candidate_pairs_report that does NOT belong to one of the 13 in-scope
    domains (EXPECTED_DOMAIN_COUNTS) -- the exact 12,567-candidate
    outside-scope population, sorted lexicographically for a stable,
    order-independent identity. These remain not_adjudicated, never
    not_duplicate -- this function only counts/hashes them, it never
    adjudicates or opens an image."""
    domains = candidate_pairs_content["domains"]
    outside_ids = [p["pair_id"] for domain_name, pairs in domains.items()
                  if domain_name not in EXPECTED_DOMAIN_COUNTS for p in pairs]
    outside_ids.sort()
    return outside_ids


def compute_outside_scope_identity_order_sha256(outside_scope_pair_ids: list[str]) -> str:
    """Binds the exact SET of outside-scope pair_ids (sorted, so this is
    an order-independent identity of the excluded population, not a
    queue-order binding) -- any addition, removal, or substitution among
    the 12,567 excluded candidates produces a different hash."""
    return sha256_bytes("\x1e".join(outside_scope_pair_ids).encode("utf-8"))


def session_blocks_summary(rows: list[dict]) -> list[dict]:
    """Reporting helper: one entry per actual frozen session/block, in
    first-appearance (queue) order -- never hardcode a session count
    elsewhere; always derive it from this."""
    seen: dict[str, dict] = {}
    order: list[str] = []
    for r in rows:
        key = r["suggested_session"]
        if key not in seen:
            seen[key] = {"suggested_session": key, "domain": r["domain"], "session_block": r["session_block"],
                        "workstream": r["workstream"], "row_count": 0,
                        "first_queue_index": r["queue_index"], "last_queue_index": r["queue_index"]}
            order.append(key)
        seen[key]["row_count"] += 1
        seen[key]["last_queue_index"] = r["queue_index"]
    return [seen[k] for k in order]


# ------------------------------------------------------ queue artifact schema --
QUEUE_ARTIFACT_ENTRY_KEYS = frozenset({"path", "byte_sha256", "row_count", "identity_order_sha256"})

# ------------------------------------------------------ adjudication record --
# Bound to: the v2 contract, the v2 queue, the candidate report's own
# byte/content hashes, the exact queue row identity/workstream/priority/
# session, the judgment, WHO/WHEN, and this protocol's own generator
# provenance -- same discipline as correct_manual_review.py's correction
# records, generalized to a many-record append-only ledger.
ADJUDICATION_RECORD_REQUIRED_FIELDS = frozenset({
    "queue_index", "pair_id", "identity_a", "identity_b", "domain", "workstream", "channel_role",
    "priority_tier", "both_rule_match", "phash_distance", "dhash_distance",
    "session_block", "suggested_session",
    "v2_contract_content_sha256", "v2_queue_byte_sha256", "v2_queue_identity_order_sha256",
    "candidate_pairs_report_byte_sha256", "candidate_pairs_report_content_sha256",
    "label", "notes", "reviewer_id", "session_id", "reviewed_at_utc",
    "generator_git_commit", "generator_source_path", "generator_source_sha256", "generator_protocol_version",
    "prev_record_hash", "record_hash",
})


def adjudication_record_identity(record: dict) -> Any:
    return record.get("pair_id")


# ----------------------------------------------- identity -> image path --
# Every v2 candidate identity has the form "dataset:slug:split:photo_id"
# (mrc.row_identity's own format, produced by scan_perceptual_duplicates.py
# for all 8 domain parts). Resolving one to a file path reuses the SAME
# mrc.resolve_image_path logic v1's manual_review_tool.py uses, against
# the frozen v1 contract's own domain_part_image_layouts -- v2 defines no
# layout rules of its own. Metadata-only: never opens the file.
class IdentityResolutionError(RuntimeError):
    """An identity string could not be parsed or resolved to a domain
    part. Always fails closed."""


def parse_identity(identity: str) -> dict:
    if not isinstance(identity, str):
        raise IdentityResolutionError(f"identity must be a string, got {identity!r}")
    parts = identity.split(":")
    if len(parts) != 4:
        raise IdentityResolutionError(f"identity {identity!r} does not have exactly 4 ':'-separated parts")
    dataset, slug, split, photo_id = parts
    return {"dataset": dataset, "slug": slug, "split": split, "photo_id": photo_id}


def domain_part_for_identity(identity: str) -> str:
    fields = parse_identity(identity)
    dataset = fields["dataset"]
    if dataset == mrc.FINAL_TEST_DATASET:
        return mrc.DOMAIN_PART_FINAL_TEST
    if dataset == mrc.EXPANSION_DATASET:
        return mrc.domain_part_for_dataset_and_split(dataset, fields["split"])
    if dataset in mrc.OTHER_EVIDENCE_SETS or dataset == mrc.DOMAIN_PART_BENCHMARK_V1:
        return dataset
    raise IdentityResolutionError(f"identity {identity!r} has an unrecognized dataset {dataset!r}")


def resolve_image_path_for_identity(repo: Path, v1_contract_content: dict, identity: str) -> Path:
    """Never opens the file -- Path construction/`.exists()` only."""
    fields = parse_identity(identity)
    domain_part = domain_part_for_identity(identity)
    image_layouts = v1_contract_content["domain_part_image_layouts"]
    return mrc.resolve_image_path(Path(repo) / "data", image_layouts, domain_part, fields["slug"], fields["photo_id"])


# ---------------------------------------------------------- contract build --
CONTENT_REQUIRED_KEYS = frozenset({
    "phase_5f1_contract_content_sha256", "scan_report_hashes",
    "candidate_report_row_count", "scoped_candidate_count", "outside_scope_candidate_count",
    "outside_scope_identity_order_sha256",
    "domain_counts", "domain_workstreams", "domain_visit_order",
    "workstream_roles", "label_channel_effect", "not_adjudicated_values",
    "priority_tiers", "max_pairs_per_session", "recommended_session_minutes",
    "queue_artifact", "queue_row_required_fields", "adjudication_labels",
    "adjudication_record_required_fields", "approved_output_paths", "remediation_rule",
    "implementation_sources",
})


def build_contract_content(*, queue_artifact: dict, implementation_sources: dict,
                           outside_scope_identity_order_sha256: str) -> dict:
    """Pure function, no filesystem access: `queue_artifact` and
    `implementation_sources` are computed by the CALLER
    (freeze_pair_adjudication_v2_contract.py) from the actual files on
    disk -- this function only assembles them alongside the constants
    above into the one frozen content dict. `outside_scope_identity_order_
    sha256` is the caller's deterministic hash of the 12,567 excluded
    pair_ids (derive_outside_scope_pair_ids +
    compute_outside_scope_identity_order_sha256), re-derived fresh from
    the verified candidate_pairs_report -- never a stored/assumed value."""
    return {
        "phase_5f1_contract_content_sha256": PHASE_5F1_CONTRACT_CONTENT_SHA256,
        "scan_report_hashes": {k: dict(v) for k, v in SCAN_REPORT_HASHES.items()},
        "candidate_report_row_count": EXPECTED_FULL_SCAN_TOTAL,
        "scoped_candidate_count": EXPECTED_SCOPED_TOTAL,
        "outside_scope_candidate_count": EXPECTED_OUTSIDE_SCOPE_TOTAL,
        "outside_scope_identity_order_sha256": outside_scope_identity_order_sha256,
        "domain_counts": dict(EXPECTED_DOMAIN_COUNTS),
        "domain_workstreams": dict(DOMAIN_WORKSTREAM),
        "domain_visit_order": list(DOMAIN_VISIT_ORDER),
        "workstream_roles": dict(WORKSTREAM_ROLE),
        "label_channel_effect": dict(LABEL_CHANNEL_EFFECT),
        "not_adjudicated_values": [NOT_ADJUDICATED, NOT_ADJUDICATED_DUE_TO_EARLY_STOP],
        "priority_tiers": {"both_rule": PRIORITY_TIER_BOTH_RULE, "standard": PRIORITY_TIER_STANDARD},
        "max_pairs_per_session": MAX_PAIRS_PER_SESSION,
        "recommended_session_minutes": list(RECOMMENDED_SESSION_MINUTES),
        "queue_artifact": dict(queue_artifact),
        "queue_row_required_fields": sorted(QUEUE_ROW_REQUIRED_FIELDS),
        "adjudication_labels": sorted(ADJUDICATION_LABELS),
        "adjudication_record_required_fields": sorted(ADJUDICATION_RECORD_REQUIRED_FIELDS),
        "approved_output_paths": dict(APPROVED_OUTPUT_PATHS),
        "remediation_rule": REMEDIATION_RULE,
        "implementation_sources": {k: dict(v) for k, v in implementation_sources.items()},
    }


CONTRACT_REQUIRED_TOP_KEYS = frozenset({"schema_version", "status", "content", "content_sha256", "generation"})
CONTRACT_GENERATION_ALLOWED_KEYS = frozenset({"note", "generator"})
CONTRACT_GENERATION_NOTE = "generation metadata is NOT part of content and NOT covered by content_sha256"
CONTRACT_GENERATOR_PATH = "training/freeze_pair_adjudication_v2_contract.py"
IMPLEMENTATION_SOURCE_ENTRY_KEYS = frozenset({"path", "sha256", "generator_git_commit"})
GIT_COMMIT_HEX_RE = re.compile(r"^[0-9a-f]{40}$")


def is_git_commit_hex(value: Any) -> bool:
    return isinstance(value, str) and bool(GIT_COMMIT_HEX_RE.match(value))


def _check_exact_keys(problems: list[str], label: str, actual: Any, required: frozenset) -> bool:
    if not isinstance(actual, dict):
        problems.append(f"{label} is not a JSON object")
        return False
    actual_keys = set(actual)
    missing = required - actual_keys
    extra = actual_keys - required
    ok = True
    if missing:
        problems.append(f"{label} missing key(s): {sorted(missing)}")
        ok = False
    if extra:
        problems.append(f"{label} has unexpected key(s): {sorted(extra)}")
        ok = False
    return ok


def validate_contract_structure(contract: Any) -> list[str]:
    """Pure, no filesystem access: rejects missing/extra keys at every
    level, bool-as-int, malformed hashes, incorrect counts, and any
    disagreement with the frozen module constants above. NEVER raises --
    arbitrary JSON always produces a controlled problem list."""
    problems: list[str] = []
    if not isinstance(contract, dict):
        return ["contract is not a JSON object"]
    _check_exact_keys(problems, "contract", contract, CONTRACT_REQUIRED_TOP_KEYS)

    sv = contract.get("schema_version")
    if not is_strict_int(sv) or sv != SCHEMA_VERSION:
        problems.append(f"schema_version must be strict int {SCHEMA_VERSION}, got {sv!r}")
    if contract.get("status") != CONTRACT_STATUS_FROZEN:
        problems.append(f"status must be exactly {CONTRACT_STATUS_FROZEN!r}, got {contract.get('status')!r}")

    generation = contract.get("generation")
    if not isinstance(generation, dict):
        problems.append("generation is not a JSON object")
    else:
        _check_exact_keys(problems, "generation", generation, CONTRACT_GENERATION_ALLOWED_KEYS)
        if generation.get("note") != CONTRACT_GENERATION_NOTE:
            problems.append("generation.note does not match the frozen value")
        if generation.get("generator") != CONTRACT_GENERATOR_PATH:
            problems.append("generation.generator does not match the frozen generator path")

    content = contract.get("content")
    if not isinstance(content, dict):
        problems.append("content is not a JSON object")
        return problems

    recorded_hash = contract.get("content_sha256")
    if not is_sha256_hex(recorded_hash):
        problems.append(f"content_sha256 is not a 64-char lowercase hex string: {recorded_hash!r}")
    else:
        try:
            recomputed = compute_content_sha256(content)
        except (TypeError, ValueError) as exc:
            problems.append(f"content is not canonicalizable: {exc}")
        else:
            if recorded_hash != recomputed:
                problems.append(f"content_sha256 mismatch: recorded={recorded_hash} recomputed={recomputed}")

    _check_exact_keys(problems, "content", content, CONTENT_REQUIRED_KEYS)

    if content.get("phase_5f1_contract_content_sha256") != PHASE_5F1_CONTRACT_CONTENT_SHA256:
        problems.append("content.phase_5f1_contract_content_sha256 does not match the frozen approved value")

    scan_hashes = content.get("scan_report_hashes")
    if not isinstance(scan_hashes, dict) or set(scan_hashes) != set(SCAN_REPORT_HASHES):
        problems.append(f"content.scan_report_hashes keys must be exactly {sorted(SCAN_REPORT_HASHES)}")
    else:
        for name, expected in SCAN_REPORT_HASHES.items():
            entry = scan_hashes.get(name)
            if not _check_exact_keys(problems, f"content.scan_report_hashes[{name!r}]", entry,
                                     SCAN_REPORT_HASH_ENTRY_KEYS):
                continue
            for field in ("byte_sha256", "content_sha256"):
                if entry.get(field) != expected[field]:
                    problems.append(f"content.scan_report_hashes[{name!r}].{field} must be {expected[field]!r}")

    for key, expected in (("candidate_report_row_count", EXPECTED_FULL_SCAN_TOTAL),
                          ("scoped_candidate_count", EXPECTED_SCOPED_TOTAL),
                          ("outside_scope_candidate_count", EXPECTED_OUTSIDE_SCOPE_TOTAL)):
        if not is_strict_int(content.get(key)) or content.get(key) != expected:
            problems.append(f"content.{key} must be strict int {expected}, got {content.get(key)!r}")
    if not is_sha256_hex(content.get("outside_scope_identity_order_sha256")):
        problems.append("content.outside_scope_identity_order_sha256 is not a 64-char lowercase hex string")

    if content.get("domain_counts") != dict(EXPECTED_DOMAIN_COUNTS):
        problems.append("content.domain_counts must be exactly the frozen approved domain counts")
    if content.get("domain_workstreams") != dict(DOMAIN_WORKSTREAM):
        problems.append("content.domain_workstreams must be exactly the frozen approved mapping")
    if content.get("domain_visit_order") != list(DOMAIN_VISIT_ORDER):
        problems.append("content.domain_visit_order must be exactly the frozen approved sequence")
    if content.get("workstream_roles") != dict(WORKSTREAM_ROLE):
        problems.append("content.workstream_roles must be exactly the frozen approved mapping")
    if content.get("label_channel_effect") != dict(LABEL_CHANNEL_EFFECT):
        problems.append("content.label_channel_effect must be exactly the frozen approved mapping")
    if content.get("not_adjudicated_values") != [NOT_ADJUDICATED, NOT_ADJUDICATED_DUE_TO_EARLY_STOP]:
        problems.append("content.not_adjudicated_values must be exactly the frozen approved list")
    if content.get("priority_tiers") != {"both_rule": PRIORITY_TIER_BOTH_RULE, "standard": PRIORITY_TIER_STANDARD}:
        problems.append("content.priority_tiers must be exactly the frozen approved mapping")
    if not is_strict_int(content.get("max_pairs_per_session")) or content.get("max_pairs_per_session") != MAX_PAIRS_PER_SESSION:
        problems.append(f"content.max_pairs_per_session must be strict int {MAX_PAIRS_PER_SESSION}")
    if content.get("recommended_session_minutes") != list(RECOMMENDED_SESSION_MINUTES):
        problems.append("content.recommended_session_minutes must be exactly the frozen approved value")
    if content.get("queue_row_required_fields") != sorted(QUEUE_ROW_REQUIRED_FIELDS):
        problems.append("content.queue_row_required_fields must be exactly the frozen approved list")
    if content.get("adjudication_labels") != sorted(ADJUDICATION_LABELS):
        problems.append("content.adjudication_labels must be exactly the frozen approved list")
    if content.get("adjudication_record_required_fields") != sorted(ADJUDICATION_RECORD_REQUIRED_FIELDS):
        problems.append("content.adjudication_record_required_fields must be exactly the frozen approved list")
    if content.get("approved_output_paths") != dict(APPROVED_OUTPUT_PATHS):
        problems.append("content.approved_output_paths must be exactly the frozen approved mapping")
    if content.get("remediation_rule") != REMEDIATION_RULE:
        problems.append("content.remediation_rule must match the frozen text verbatim")

    queue_artifact = content.get("queue_artifact")
    if _check_exact_keys(problems, "content.queue_artifact", queue_artifact, QUEUE_ARTIFACT_ENTRY_KEYS):
        if queue_artifact.get("path") != QUEUE_ARTIFACT_REL_PATH:
            problems.append(f"content.queue_artifact.path must be {QUEUE_ARTIFACT_REL_PATH!r}")
        if not is_sha256_hex(queue_artifact.get("byte_sha256")):
            problems.append("content.queue_artifact.byte_sha256 is not a 64-char lowercase hex string")
        if not is_strict_int(queue_artifact.get("row_count")) or queue_artifact.get("row_count") != EXPECTED_SCOPED_TOTAL:
            problems.append(f"content.queue_artifact.row_count must be strict int {EXPECTED_SCOPED_TOTAL}")
        if not is_sha256_hex(queue_artifact.get("identity_order_sha256")):
            problems.append("content.queue_artifact.identity_order_sha256 is not a 64-char lowercase hex string")

    impl = content.get("implementation_sources")
    if not isinstance(impl, dict) or set(impl) != set(APPROVED_IMPLEMENTATION_SOURCE_PATHS):
        problems.append(f"content.implementation_sources keys must be exactly "
                        f"{sorted(APPROVED_IMPLEMENTATION_SOURCE_PATHS)} -- missing, extra, or renamed "
                        f"implementation sources are rejected")
    else:
        for name, entry in impl.items():
            if not _check_exact_keys(problems, f"content.implementation_sources[{name!r}]", entry,
                                     IMPLEMENTATION_SOURCE_ENTRY_KEYS):
                continue
            expected_path = APPROVED_IMPLEMENTATION_SOURCE_PATHS[name]
            if entry.get("path") != expected_path:
                problems.append(f"content.implementation_sources[{name!r}].path must be exactly "
                                f"{expected_path!r}, got {entry.get('path')!r} -- a path-substituted source "
                                f"is rejected even under its approved name")
            if not is_sha256_hex(entry.get("sha256")):
                problems.append(f"content.implementation_sources[{name!r}].sha256 is not a 64-char lowercase hex string")
            if not is_git_commit_hex(entry.get("generator_git_commit")):
                problems.append(f"content.implementation_sources[{name!r}].generator_git_commit is not a 40-char lowercase hex string")

    return problems
